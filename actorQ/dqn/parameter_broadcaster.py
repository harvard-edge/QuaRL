import gc
from absl import app
from absl import flags
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.tf import actors
from acme.agents.tf import dqn
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
from dqn_learner import DQN_learner
from dm_control import suite
from dqn_args import parser
from dqn_learner import DQN_learner
from multiprocessing import Process
from typing import Mapping, Sequence
from utils import make_environment, make_networks, Q, input_size_from_obs_spec, Q_opt, Q_decompress, Q_compress
import acme
import argparse
import dm_env
import gym
import numpy as np
import reverb
import sonnet as snt
import sys
import tensorflow as tf
import threading
from concurrent import futures
import time
import trfl
from custom_environment_loop import CustomEnvironmentLoop
import zlib
import pickle
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import pytorch_actors
import torch
import zstd
import msgpack

# For debugging
#tf.debugging.set_log_device_placement(True)

def get_shutdown_status(client, shutdown_table_name):
    sample = next(client.sample(shutdown_table_name))[0]
    return int(sample.data[0])

def get_weights_to_broadcast(client, broadcast_table_name):
    print("Waiting for weights", time.time())
    sample = next(client.sample(broadcast_table_name))[0]
    print("Received weights", time.time())
    return sample.data
    
if __name__ == "__main__":
    args = parser.parse_args()

    print(vars(args))

    # Initialize environment
    environment = make_environment(args.taskstr)
    environment_spec = specs.make_environment_spec(environment)
    model_sizes = tuple([int(x) for x in args.model_str.split(",")])
    print(model_sizes)
    agent_networks = make_networks(environment_spec.actions, placement=args.learner_device_placement,
                                   policy_layer_sizes=model_sizes)

    # Create pytorch model (we send the state dict to the actors)
    pytorch_model = pytorch_actors.create_model(environment_spec.observations.shape[0], 
                                                environment_spec.actions.num_values,
                                                policy_layer_sizes=model_sizes)
    
    address = "localhost:%d" % args.port
    client = reverb.Client(address)

    def quantize_and_broadcast_weights(weights, id):
        print("Broadcasting weights", time.time())
        
        # Quantize weights artificially
        tstart = time.time()
        weights = [Q(x, args.quantize_communication) for x in weights]
        #weights = [Q_opt(x, args.quantize_communication) for x in weights]
        print("Quantized (t=%f)" % (time.time()-tstart))

        # Load weights into pytorch model
        tstart = time.time()
        pytorch_actors.pytorch_model_load_state_dict(pytorch_model, weights)
        print("Load weights (t=%f)" % (time.time()-tstart))

        # Quantize
        tstart = time.time()
        quantized_actor = pytorch_actors.pytorch_quantize(pytorch_model, args.quantize)
        print("Pytorch quantized (t=%f)" % (time.time()-tstart))

        # State dict
        state_dict = quantized_actor.state_dict()
        if args.weight_compress != 0:
            for k,v in state_dict.items():
                state_dict[k] = Q_compress(state_dict[k], args.weight_compress)
        state_dict["id"] = id

        # Send over packed params to avoid overhead
        tstart = time.time()
        weights = [pickle.dumps(state_dict)]
        if args.compress:
            weights = [zlib.compress(x) for x in weights]
        weights = [np.fromstring(x, dtype=np.uint8) for x in weights]
        print("Compress %f" % (time.time()-tstart))


        client.insert(weights, 
                      {args.model_table_name : 1.0})

        print("Done broadcasting", time.time())

    c = 0
    while True:        
        should_shutdown = get_shutdown_status(client, args.shutdown_table_name)
        sys.stdout.flush()
        if should_shutdown:
            break

        weights = get_weights_to_broadcast(client, args.broadcaster_table_name)
        quantize_and_broadcast_weights(weights, c)
        
        c += 1
