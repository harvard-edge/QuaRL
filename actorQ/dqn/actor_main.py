from absl import app
from absl import flags
from acme import core
from acme import datasets
from acme import specs
from acme import specs
from acme import types
from acme import types
from acme import wrappers
from acme.adders import reverb as adders
from acme.agents import agent
from acme.agents.tf import actors
from acme.agents.tf import actors
from acme.agents.tf import d4pg
from acme.agents.tf.d4pg import learning
from acme.tf import networks
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import loggers
from dm_control import suite
from multiprocessing import Process
from typing import Mapping, Sequence
from utils import *
import acme
import argparse
import copy
import dm_env
import gym
import numpy as np
import numpy as np
import reverb
import reverb
import sonnet as snt
import sonnet as snt
import sys
import sys
import sys
import tensorflow as tf
import time
import trfl
from dqn_args import parser
import pytorch_actors
import torch
import pickle
import zlib
import lz4.frame
import lz4.block
import time
import zstd
import gc

torch.set_num_threads(1)
cpus = tf.config.list_physical_devices("CPU")
tf.config.set_visible_devices(cpus)

class ExternalVariableSource(core.VariableSource):

  def __init__(self, reverb_table, model_table_name, actor_id, args):
      self.reverb_table = reverb_table
      self.model_table_name = model_table_name
      self.actor_id = actor_id    
      self.prev_sample = None
      self.args = args
      self.cached = None

  def get_variables(self, names):    
      #if self.cached is not None:
      #  return self.cached

      # Pull from reverb table
      tstart = time.time()      
      sample = next(self.reverb_table.sample(self.model_table_name))[0]      
      tend = time.time()

      # Decode sample
      d = [x.tobytes() for x in sample.data]  

      try:
        #decoded = [pickle.loads(lz4.frame.decompress(x.tobytes() +  b'\x00\x00\x00\x00')) for x in sample.data]    
        #decoded = [pickle.loads(zlib.decompress(x.tobytes())) for x in sample.data]
        if self.args["compress"]:
          d = [zlib.decompress(x) for x in d]
          #d = [lz4.frame.decompress(x +  b'\x00\x00\x00\x00') for x in d]
        tdecompress = time.time()
        decoded = [pickle.loads(x) for x in d]
        #decoded = [pickle.loads(x) for x in d]    
        #decoded = [pickle.loads(zlib.decompress(x.item())) for x in sample.data]
        #decoded = [pickle.loads(x.tobytes()) for x in sample.data]
        tdecode = time.time()
        print("Pull time: %f, Decompress/tobytes time: %f, Deserialize time: %f" % (tend-tstart, tdecompress-tend, tdecode-tdecompress))
        return decoded
      except:
        #  traceback.print_exc()
        pass
      return []

class IntraProcessTFToPyTorchVariableSource(core.VariableSource):
  def __init__(self, tf_model):
    self.tf_model = tf_model

  def get_variables(self, name):
      res = [tf2_utils.to_numpy(v) for v in self.tf_model.variables]
      return res

def get_shutdown_status(client, shutdown_table_name):
    sample = next(client.sample(shutdown_table_name))[0]
    return int(sample.data[0])

def actor_main(actor_id, args):
    print("Starting actor %d" % actor_id)
  
    address = "localhost:%d" % args["port"]
    client = reverb.Client(address)
    actor_device_placement = args["actor_device_placement"]
    actor_device_placement = "%s:0" % (actor_device_placement)

    model_sizes = tuple([int(x) for x in args["model_str"].split(",")])

    # Create network / env
    environment = make_environment(args["taskstr"])
    environment_spec = specs.make_environment_spec(environment)

    act_spec = environment_spec.actions
    obs_spec = environment_spec.observations

    policy_network = make_networks(environment_spec.actions, placement=actor_device_placement,
                                   policy_layer_sizes=model_sizes)["policy"]

    with tf.device(actor_device_placement):
      # Create the behavior policy.
      emb_spec = tf2_utils.create_variables(tf.identity, [obs_spec])
      tf2_utils.create_variables(policy_network, [emb_spec])
      epsilon = tf.Variable(get_actor_sigma(args["sigma"], args["actor_id"], args["n_actors"]), trainable=False)
      behavior_network = snt.Sequential([
        policy_network,
        lambda q: trfl.epsilon_greedy(q, epsilon=epsilon).sample(),
      ])

      # Set up actor
      adder = adders.NStepTransitionAdder(
          priority_fns={args["replay_table_name"]: lambda x: 1.},
          client=client,
          n_step=args["n_step"],
          discount=args["discount"])
    
      variable_source = ExternalVariableSource(client, args["model_table_name"], actor_id, args)
      variable_client = tf2_variable_utils.VariableClient(
          variable_source, {'policy': policy_network.variables}, update_period=args["actor_update_period"])

      # Create Feed actor      
      actor = actors.FeedForwardActor(behavior_network, adder=adder, variable_client=variable_client)

      # Create pytorch actor
      pytorch_adder = adders.NStepTransitionAdder(
          priority_fns={args["replay_table_name"]: lambda x: 1.},
          client=client,
          n_step=args["n_step"],
          discount=args["discount"])
    
      pytorch_model = pytorch_actors.create_model(environment_spec.observations.shape[0], 
                                                  environment_spec.actions.num_values,
                                                  policy_layer_sizes=model_sizes)
      pytorch_variable_source = ExternalVariableSource(client, args["model_table_name"], actor_id, args)
      pytorch_variable_client = pytorch_actors.TFToPyTorchVariableClient(
        pytorch_variable_source, pytorch_model, update_period=args["actor_update_period"])
      pytorch_actor = pytorch_actors.FeedForwardActor(pytorch_model,
                                                      adder=pytorch_adder, 
                                                      variable_client=pytorch_variable_client,
                                                      q=args["quantize"],
                                                      args=args)    

    actor = {
      "tensorflow" : actor,
      "pytorch" : pytorch_actor,
    }[args["inference_mode"]]

    # Main actor loop
    t_start = time.time()
    n_total_steps = 0
    while True:
        should_shutdown = get_shutdown_status(client, args["shutdown_table_name"])
        sys.stdout.flush()
        if should_shutdown:
          break
        timestep = environment.reset()
        episode_return = 0
        
        actor.observe_first(timestep)
        
        t_start_local = time.time()
        local_steps = 0
        
        while not timestep.last():
            local_steps += 1
            tstart = time.time()
          
            # Generate an action from the agent's policy and step the environment.
            with tf.device(actor_device_placement):
              action = actor.select_action(timestep.observation)

            timestep = environment.step(action)
            
            # Have the agent observe the timestep and let the actor update itself.
            actor.observe(action, next_timestep=timestep)
            
            episode_return += timestep.reward
            n_total_steps += 1

            tend = time.time()

            print("Step time: %f" % (tend-tstart))

            # Update the actor
            if n_total_steps*args["n_actors"] >= args["min_replay_size"]:
              actor.update()
        
        steps_per_second = n_total_steps/(time.time()-t_start)
        local_steps_per_second = local_steps / (time.time()-t_start_local)
        print("Actor %d finished timestep (r=%f) (steps_per_second=%f) (local_steps_per_second=%f)" % (actor_id, float(episode_return), steps_per_second, local_steps_per_second))
    print("Actor %d shutting down" % (actor_id))


if __name__=="__main__":
  
  args = parser.parse_args()
  print(vars(args))
  actor_main(args.actor_id, vars(args))
