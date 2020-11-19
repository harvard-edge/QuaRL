import gc
from absl import app
from absl import flags
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.tf import actors
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
from dqn_learner import DQN_learner
from dm_control import suite
from dqn_args import parser
from dqn_learner import DQN_learner
from multiprocessing import Process
from typing import Mapping, Sequence
from utils import make_environment, make_networks, Q
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
import lz4.frame
import lz4.block
import pytorch_actors
import torch
import zstd
import msgpack
import signal, psutil

class PeriodicBroadcaster(object):
  """A variable client for updating variables from a remote source."""

  def __init__(self, f, update_period=1):
    self._call_counter = 0
    self._update_period = update_period
    self._request = lambda x: f(x)

    self.updated_callbacks = []

    # Create a single background thread to fetch variables without necessarily
    # blocking the actor.
    self._executor = futures.ThreadPoolExecutor(max_workers=1)
    self._async_request = lambda x: self._executor.submit(self._request, x)

    # Initialize this client's future to None to indicate to the `update()`
    # method that there is no pending/running request.
    self._future: Optional[futures.Future] = None

  def add_updated_callback(self, cb):
    self.updated_callbacks.append(cb)

  def update(self, weights):
    """Periodically updates the variables with the latest copy from the source.
    Unlike `update_and_wait()`, this method makes an asynchronous request for
    variables and returns. Unless the request is immediately fulfilled, the
    variables are only copied _within a subsequent call to_ `update()`, whenever
    the request is fulfilled by the `VariableSource`.
    This stateful update method keeps track of the number of calls to it and,
    every `update_period` call, sends an asynchronous request to its server to
    retrieve the latest variables. It does so as long as there are no existing
    requests.
    If there is an existing fulfilled request when this method is called,
    the resulting variables are immediately copied.
    """

    # Track the number of calls (we only update periodically).
    if self._call_counter < self._update_period:
      self._call_counter += 1

    period_reached: bool = self._call_counter >= self._update_period
    has_active_request: bool = self._future is not None

    if period_reached and not has_active_request:
      # The update period has been reached and no request has been sent yet, so
      # making an asynchronous request now.
      self._future = self._async_request(weights)    # todo uncomment
      self._call_counter = 0

    if has_active_request and self._future.done():
      # The active request is done so copy the result and remove the future.
      self._future: Optional[futures.Future] = None
    else:
      # There is either a pending/running request or we're between update
      # periods, so just carry on.
      return
    
    
# For debugging
#tf.debugging.set_log_device_placement(True)

if __name__ == "__main__":
    args = parser.parse_args()

    print(vars(args))

    # Initialize environment
    environment = make_environment(args.taskstr)
    environment_spec = specs.make_environment_spec(environment)
    model_sizes = tuple([int(x) for x in args.model_str.split(",")])
    agent_networks = make_networks(environment_spec.actions, placement=args.learner_device_placement,
                                   policy_layer_sizes=model_sizes)

    # Create D4PG learner
    learner = DQN_learner(environment_spec=environment_spec,
                          policy_network=agent_networks['policy'],
                          logger=loggers.make_default_logger('%s/learner' % args.logpath),
                          port=args.port,
                          replay_table_name=args.replay_table_name,
                          model_table_name=args.model_table_name,
                          replay_table_max_times_sampled=args.replay_table_max_times_sampled,
                          max_replay_size=args.replay_table_max_replay_size,
                          min_replay_size=args.min_replay_size,
                          shutdown_table_name=args.shutdown_table_name,
                          device_placement=args.learner_device_placement,
                          batch_size=args.batch_size,
                          broadcaster_table_name=args.broadcaster_table_name)

    
    # Create the evaluation policy.
    with tf.device(args.learner_device_placement):

        # Create the behavior policy.
        epsilon = tf.Variable(0.00, trainable=False)
        eval_policy= snt.Sequential([
            agent_networks["policy"],
            lambda q: trfl.epsilon_greedy(q, epsilon=epsilon).sample(),
        ])
        eval_actor = actors.FeedForwardActor(policy_network=eval_policy)

    eval_env = make_environment(args.taskstr, mode="evaluate")
    eval_loop = CustomEnvironmentLoop(eval_env, eval_actor, label='%s/' % (args.logpath))
    
    def broadcast_shutdown(should_shutdown):        
        learner.client.insert(np.array(should_shutdown), {args.shutdown_table_name : 1.0})

    steps = 0
        
    def submit_parameters_to_broadcaster(weights):
        if weights is None:
          weights = [tf2_utils.to_numpy(v) for v in learner.learner._network.variables]
        learner.client.insert(weights, {args.broadcaster_table_name: 1.0})
        
    broadcast_shutdown(0)
    variable_broadcaster = PeriodicBroadcaster(submit_parameters_to_broadcaster)

    # Main learner loop
    for i in range(args.num_episodes):
        sys.stdout.flush()
        with tf.device(args.learner_device_placement):
            steps += 1
            learner.learner.step()
            #variable_broadcaster.update([tf2_utils.to_numpy(v) for v in agent_networks["policy"].variables])
            if i  % 100 == 0:
              #variable_broadcaster.update([tf2_utils.to_numpy(v) for v in learner.learner._network.variables])
              #submit_parameters_to_broadcaster([tf2_utils.to_numpy(v) for v in learner.learner._network.variables])
              #submit_parameters_to_broadcaster([tf2_utils.to_numpy(v) for v in agent_networks["policy"].variables])
              pass
            variable_broadcaster.update(None)
            
            if i % 1000 == 0:
                eval_loop.run(num_episodes=1)
            
    broadcast_shutdown(1)

    print("Shutting down")
