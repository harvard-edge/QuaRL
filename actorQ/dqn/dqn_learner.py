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

import copy

from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents import agent
from acme.agents.tf import actors
from acme.agents.tf.dqn import learning
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import tree_utils
from acme.utils import loggers
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

class DQN_learner(object):
  """DQN agent.

  This implements a single-process DQN agent. This is a simple Q-learning
  algorithm that inserts N-step transitions into a replay buffer, and
  periodically updates its policy by sampling these transitions using
  prioritization.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy_network: snt.Module,
      batch_size: int = 256,
      port:int = 8000,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      replay_table_max_times_sampled: float = 32.0,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      importance_sampling_exponent: float = 0.2,
      priority_exponent: float = 0.6,
      n_step: int = 5,
      epsilon: tf.Tensor = None,
      learning_rate: float = 1e-3,
      discount: float = 0.95,
      logger: loggers.Logger = None,
      checkpoint: bool = True,
      checkpoint_subpath: str = '~/acme/',
      model_table_name: str="model_table_name",
      replay_table_name: str="replay_table_name",
      shutdown_table_name: str="shutdown_table_name",
     device_placement: str="CPU",
      broadcaster_table_name: str="broadcast_table_name",
  ):
    """Initialize the agent.

    Args:
      environment_spec: description of the actions, observations, etc.
      network: the online Q network (the one being optimized)
      batch_size: batch size for updates.
      prefetch_size: size to prefetch from replay.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      samples_per_insert: number of samples to take from replay for every insert
        that is made.
      min_replay_size: minimum replay size before updating. This and all
        following arguments are related to dataset construction and will be
        ignored if a dataset argument is passed.
      max_replay_size: maximum replay size.
      importance_sampling_exponent: power to which importance weights are raised
        before normalizing.
      priority_exponent: exponent used in prioritized sampling.
      n_step: number of steps to squash into a single transition.
      epsilon: probability of taking a random action; ignored if a policy
        network is given.
      learning_rate: learning rate for the q-network update.
      discount: discount to use for TD updates.
      logger: logger object to be used by learner.
      checkpoint: boolean indicating whether to checkpoint the learner.
      checkpoint_subpath: directory for the checkpoint.
    """

    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.
    replay_table = reverb.Table(
        name=replay_table_name,
        sampler=reverb.selectors.Prioritized(priority_exponent),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        max_times_sampled=replay_table_max_times_sampled,
        rate_limiter=reverb.rate_limiters.MinSize(min_replay_size))

    model_table = reverb.Table(
        name=model_table_name,
        sampler=reverb.selectors.Fifo(),
        remover=reverb.selectors.Fifo(),
        max_size=1,
        rate_limiter=reverb.rate_limiters.MinSize(1))

    broadcaster_table = reverb.Table(
      name=broadcaster_table_name,
      sampler=reverb.selectors.Fifo(),
      remover=reverb.selectors.Fifo(),
      max_size=1,
      rate_limiter=reverb.rate_limiters.MinSize(1))

    shutdown_table = reverb.Table(name=shutdown_table_name,
                                  sampler=reverb.selectors.Lifo(),
                                  remover=reverb.selectors.Fifo(),
                                  max_size=1,
                                  rate_limiter=reverb.rate_limiters.MinSize(1))

    self._server = reverb.Server([replay_table, model_table, shutdown_table, broadcaster_table], port=port)

    # The adder is used to insert observations into replay.
    address = f'localhost:{self._server.port}'
    adder = adders.NStepTransitionAdder(
        client=reverb.Client(address),
        n_step=n_step,
        discount=discount)

    # The dataset provides an interface to sample from replay.
    self.client = reverb.Client(address)

    with tf.device(device_placement):

      #dataset = datasets.make_reverb_dataset(
      #  client=reverb.TFClient(address),
      #  environment_spec=environment_spec,
      #  batch_size=batch_size,
      #  prefetch_size=prefetch_size,
      #  transition_adder=True)
      dataset = simple_dataset(reverb.Client(address), address, replay_table_name, batch_size, environment_spec)

      # Use constant 0.05 epsilon greedy policy by default.
      if epsilon is None:
        epsilon = tf.Variable(0.05, trainable=False)

      # Create a target network.
      target_network = copy.deepcopy(policy_network) 

      # Ensure that we create the variables before proceeding (maybe not needed).
      tf2_utils.create_variables(policy_network, [environment_spec.observations])
      tf2_utils.create_variables(target_network, [environment_spec.observations])

      checkpoint = False

      # The learner updates the parameters (and initializes them).
      self.learner = learning.DQNLearner(
          network=policy_network,
          target_network=target_network,
          discount=discount,
          importance_sampling_exponent=importance_sampling_exponent,
          learning_rate=learning_rate,
          target_update_period=target_update_period,
          dataset=dataset,
          logger=logger,
          checkpoint=checkpoint)

      if checkpoint:
        self._checkpointer = tf2_savers.Checkpointer(
            directory=checkpoint_subpath,
            objects_to_save=self.learner.state,
            subdirectory='dqn_learner',
            time_delta_minutes=60.)
      else:
        self._checkpointer = None

