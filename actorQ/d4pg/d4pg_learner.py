# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""D4PG agent implementation."""

import copy


import numpy as np
from acme import core
from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as adders
from acme.tf import variable_utils as tf2_variable_utils
from acme.agents import agent
from acme.agents.tf import actors
from acme.agents.tf.d4pg import learning
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import reverb
import sonnet as snt
import tensorflow as tf
import sys
import reverb
import sonnet as snt
import tensorflow as tf
import trfl
import tree
from acme import types
from acme import specs
from typing import Optional

class simple_dataset:
  def __init__(self, client, address, table_name, batch_size, environment_spec):
    shapes, dtypes = _spec_to_shapes_and_dtypes(
      True,
      environment_spec,
      extra_spec=None,
      sequence_length=None,
      convert_zero_size_to_none=False,
      using_deprecated_adder=False)

    self.dataset = reverb.ReplayDataset(address, table_name, dtypes, shapes, batch_size)
    self.dataset = self.dataset.batch(batch_size, drop_remainder=True)
    self.dataset = iter(self.dataset)    
    
  def __iter__(self):
    return self

  def __next__(self):
    while True:
      sample = next(self.dataset)
      return sample

def _spec_to_shapes_and_dtypes(transition_adder: bool,
                               environment_spec: specs.EnvironmentSpec,
                               extra_spec: Optional[types.NestedSpec],
                               sequence_length: Optional[int],
                               convert_zero_size_to_none: bool,
                               using_deprecated_adder: bool):
  """Creates the shapes and dtypes needed to describe the Reverb dataset.
  This takes a `environment_spec`, `extra_spec`, and additional information and
  returns a tuple (shapes, dtypes) that describe the data contained in Reverb.
  Args:
    transition_adder: A boolean, describing if a `TransitionAdder` was used to
      add data.
    environment_spec: A `specs.EnvironmentSpec`, describing the shapes and
      dtypes of the data produced by the environment (and the action).
    extra_spec: A nested structure of objects with a `.shape` and `.dtype`
      property. This describes any additional data the Actor adds into Reverb.
    sequence_length: An optional integer for how long the added sequences are,
      only used with `SequenceAdder`.
    convert_zero_size_to_none: If True, then all shape dimensions that are 0 are
      converted to None. A None dimension is only set at runtime.
    using_deprecated_adder: True if the adder used to generate the data is
      from acme/adders/reverb/deprecated.
  Returns:
    A tuple (dtypes, shapes) that describes the data that has been added into
    Reverb.
  """
  # The *transition* adder is special in that it also adds an arrival state.
  if transition_adder:
    # Use the environment spec but convert it to a plain tuple.
    adder_spec = tuple(environment_spec) + (environment_spec.observations,)
    # Any 'extra' data that is passed to the adder is put on the end.
    if extra_spec:
      adder_spec += (extra_spec,)
  elif using_deprecated_adder and deprecated_base is not None:
    adder_spec = deprecated_base.Step(
        observation=environment_spec.observations,
        action=environment_spec.actions,
        reward=environment_spec.rewards,
        discount=environment_spec.discounts,
        extras=() if not extra_spec else extra_spec)
  else:
    adder_spec = adders.Step(
        observation=environment_spec.observations,
        action=environment_spec.actions,
        reward=environment_spec.rewards,
        discount=environment_spec.discounts,
        start_of_episode=specs.Array(shape=(), dtype=bool),
        extras=() if not extra_spec else extra_spec)

  # Extract the shapes and dtypes from these specs.
  get_dtype = lambda x: tf.as_dtype(x.dtype)
  get_shape = lambda x: tf.TensorShape(x.shape)
  if sequence_length:
    get_shape = lambda x: tf.TensorShape([sequence_length, *x.shape])

  if convert_zero_size_to_none:
    # TODO(b/143692455): Consider making this default behaviour.
    get_shape = lambda x: tf.TensorShape([s if s else None for s in x.shape])
  shapes = tree.map_structure(get_shape, adder_spec)
  dtypes = tree.map_structure(get_dtype, adder_spec)
  return shapes, dtypes

class D4PG_learner(object):

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               policy_network: snt.Module,
               critic_network: snt.Module,
               observation_network: types.TensorTransformation = tf.identity,
               discount: float = 0.99,
               batch_size: int = 256,
               prefetch_size: int = 4,
               target_update_period: int = 100,
               max_replay_size: int = 1000000,
               min_replay_size: int = 10000,
               clipping: bool = True,
               logger: loggers.Logger = None,
               counter: counting.Counter = None,
               checkpoint: bool = False,
               replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE,
               model_table_name: str = "model_reverb_table",
               port: int = 8000,
               replay_table_max_times_sampled: int = 1,
               shutdown_table_name: str="shutdown_table",
               broadcaster_table_name: str="broadcaster_table",
               device_placement: str="CPU"):
    """Initialize the agent.

    Args:
      environment_spec: description of the actions, observations, etc.
      policy_network: the online (optimized) policy.
      critic_network: the online critic.
      observation_network: optional network to transform the observations before
        they are fed into any network.
      discount: discount to use for TD updates.
      batch_size: batch size for updates.
      prefetch_size: size to prefetch from replay.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      max_replay_size: maximum replay size.
      n_step: number of steps to squash into a single transition.
      clipping: whether to clip gradients by global norm.
      logger: logger object to be used by learner.
      counter: counter object used to keep track of steps.
      checkpoint: boolean indicating whether to checkpoint the learner.
      replay_table_name: string indicating what name to give the replay table.
    """
    
    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.
    replay_table = reverb.Table(
        name=replay_table_name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        max_times_sampled=replay_table_max_times_sampled,
        rate_limiter=reverb.rate_limiters.MinSize(min_replay_size))

    model_table = reverb.Table(
        name=model_table_name,
        sampler=reverb.selectors.Lifo(),
        remover=reverb.selectors.Fifo(),
        max_size=1,
        rate_limiter=reverb.rate_limiters.MinSize(1))

    broadcaster_table = reverb.Table(
      name=broadcaster_table_name,
      sampler=reverb.selectors.Fifo(),
      remover=reverb.selectors.Fifo(),
      max_size=1,
      max_times_sampled=1,
      rate_limiter=reverb.rate_limiters.MinSize(1))

    shutdown_table = reverb.Table(name=shutdown_table_name,
                                  sampler=reverb.selectors.Lifo(),
                                  remover=reverb.selectors.Fifo(),
                                  max_size=1,
                                  rate_limiter=reverb.rate_limiters.MinSize(1))

    self._server = reverb.Server([replay_table, model_table, shutdown_table, broadcaster_table], port=port)

    # The adder is used to insert observations into replay.
    address = f'localhost:{self._server.port}'
    self.client = reverb.Client(address)

    with tf.device(device_placement):
      # The dataset provides an interface to sample from replay.
      #dataset = datasets.make_reverb_dataset(
      #    table=replay_table_name,
      #    client=reverb.TFClient(address),
      #    batch_size=batch_size,
      #    prefetch_size=prefetch_size,
      #    environment_spec=environment_spec,
      #    transition_adder=True)

      dataset = simple_dataset(reverb.Client(address), address, replay_table_name, batch_size, environment_spec)

      # Make sure observation network is a Sonnet Module.
      observation_network = tf2_utils.to_sonnet_module(observation_network)

      # Create target networks.
      target_policy_network = copy.deepcopy(policy_network)
      target_critic_network = copy.deepcopy(critic_network)
      target_observation_network = copy.deepcopy(observation_network)

      # Get observation and action specs.
      act_spec = environment_spec.actions
      obs_spec = environment_spec.observations
      emb_spec = tf2_utils.create_variables(observation_network, [obs_spec])

      # Create variables.
      tf2_utils.create_variables(policy_network, [emb_spec])
      tf2_utils.create_variables(critic_network, [emb_spec, act_spec])
      tf2_utils.create_variables(target_policy_network, [emb_spec])
      tf2_utils.create_variables(target_critic_network, [emb_spec, act_spec])
      tf2_utils.create_variables(target_observation_network, [obs_spec])

      # Create optimizers.
      policy_optimizer = snt.optimizers.Adam(learning_rate=1e-4)
      critic_optimizer = snt.optimizers.Adam(learning_rate=1e-4)

      # The learner updates the parameters (and initializes them).
      self.learner = learning.D4PGLearner(
          policy_network=policy_network,
          critic_network=critic_network,
          observation_network=observation_network,
          target_policy_network=target_policy_network,
          target_critic_network=target_critic_network,
          target_observation_network=target_observation_network,
          policy_optimizer=policy_optimizer,
          critic_optimizer=critic_optimizer,
          clipping=clipping,
          discount=discount,
          target_update_period=target_update_period,
          dataset=dataset,
          counter=counter,
          logger=logger,
          checkpoint=checkpoint,
      )

