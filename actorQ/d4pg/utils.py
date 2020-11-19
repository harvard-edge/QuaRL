import sys
import torch
import pprint
from collections import OrderedDict
from typing import Mapping, Sequence
import sys
from acme.wrappers import base
import tree

from absl import app
from absl import flags
import acme
import reverb
import tensorflow as tf
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.tf import actors
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf.networks import duelling
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import dm_env
import gym
import numpy as np
import sonnet as snt
from dm_control import suite

from multiprocessing import Process
import threading
import argparse

from acme.wrappers import gym_wrapper
from acme.wrappers import atari_wrapper
from acme.tf.networks import base

import bsuite

Images = tf.Tensor
QValues = tf.Tensor
Logits = tf.Tensor
Value = tf.Tensor

def get_actor_sigma(sigma_max, actor_id, n_actors):
    sigmas = list(np.arange(0, sigma_max, (sigma_max-0)/n_actors))
    print(sigmas)
    return sigmas[actor_id-1]

def Q_compress(W, n):
    assert(n == 8)

    W = W.numpy()
    W_orig = W
    if n >= 32:
        return W
    assert(len(W.shape) <= 2)
    range = np.abs(np.min(W)) + np.abs(np.max(W))
    d = range / (2**(n-1))
    if d == 0:
        return W
    z = -np.min(W, 0) // d
    W = np.rint(W / d)

    W_q = torch.from_numpy(W).char()
    return d, W_q

def Q_decompress(V, n):
    return V[0]*V[1].float()

def Q(W, n):
    if n >= 32:
        return W
    assert(len(W.shape) <= 2)
    range = np.abs(np.min(W)) + np.abs(np.max(W))
    d = range / (2**(n))
    if d == 0:
        return W
    z = -np.min(W, 0) // d
    W = np.rint(W / d)
    W = d * (W)
    return W

def Q_opt(W, n, intervals=100):
    return Q(W, n)
    if n == 32:
        return Q(W, n)
    best = Q(W,n)
    best_err = np.mean(np.abs((best - W)).flatten())
    first_err = best_err
    minW, maxW = np.min(W), np.max(W)
    max_abs = max(abs(minW), abs(maxW))
    for lim in np.arange(0, max_abs, max_abs/intervals):
        W_clipped = np.clip(W, -lim, lim)
        W_clipped_Q = Q(W_clipped, n)
        mse = np.mean(np.abs((W_clipped_Q-W)).flatten())
        if mse < best_err:
            #print("New best err: (%f->%f) at clip %f (W_min=%f, W_max=%f)" % (best_err, mse, lim, minW, maxW))
            best_err = mse
            best = W_clipped_Q
    print("Opted: %f->%f err" % (first_err, best_err))
    return best

def make_environment(
    taskstr) -> dm_env.Environment:
  
  """Creates an OpenAI Gym environment."""

  # Load the gym environment.
  module, task = taskstr.split(",")

  if module == "gym":
    environment = gym.make(task)
    environment = wrappers.GymWrapper(environment)  
  elif module == "atari":
    environment = gym.make(task, full_action_space=True)
    environment = gym_wrapper.GymAtariAdapter(environment)
    environment = atari_wrapper.AtariWrapper(environment, to_float=True, max_episode_len=5000, zero_discount_on_life_loss=True,
)
  elif module == "dm_control":
    t1,t2 = task.split("_")
    environment = suite.load(t1, t2)
  elif module == "bsuite":
    environment = bsuite.load_and_record_to_csv(
      bsuite_id=task,
      results_dir="./bsuite_results"
    )
    

  # Make sure the environment obeys the dm_env.Environment interface.
  environment = wrappers.SinglePrecisionWrapper(environment)

  return environment

def input_size_from_obs_spec(env_spec):
    if hasattr(env_spec, "shape"):
        return int(np.prod(env_spec.shape))
    if type(env_spec) == OrderedDict:
        return int(sum([input_size_from_obs_spec(x) for x in env_spec.values()]))
    try:
        return int(sum([input_size_from_obs_spec(x) for x in env_spec]))
    except:
        assert(0)

def input_from_obs(observation):
    observation = tf2_utils.add_batch_dim(observation)
    observation = tf2_utils.batch_concat(observation)
    return tf2_utils.to_numpy(observation)

# The default settings in this network factory will work well for the
# MountainCarContinuous-v0 task but may need to be tuned for others. In
# particular, the vmin/vmax and num_atoms hyperparameters should be set to
# give the distributional critic a good dynamic range over possible discounted
# returns. Note that this is very different than the scale of immediate rewards.
def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
    placement: str="CPU",
) -> Mapping[str, types.TensorTransformation]:
  """Creates the networks used by the agent."""

  with tf.device(placement):

    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(action_spec.shape, dtype=int)    

    # Create the shared observation network; here simply a state-less operation.
    observation_network = tf2_utils.batch_concat

    uniform_initializer = tf.initializers.VarianceScaling(
      distribution='uniform', mode='fan_out', scale=0.333)
    network = snt.Sequential([
      snt.nets.MLP(
          policy_layer_sizes,
          w_init=uniform_initializer,
          activation=tf.nn.tanh,
          activate_final=False
      )])

    # Create the policy network.
    policy_network = snt.Sequential([
      network,
      #networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
      networks.NearZeroInitializedLinear(num_dimensions),
      networks.TanhToSpec(action_spec)])

    # Create the critic network.
    critic_network = snt.Sequential([
      # The multiplexer concatenates the observations/actions.
      networks.CriticMultiplexer(),
      networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
      networks.DiscreteValuedHead(vmin, vmax, num_atoms),
    ])

    return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': observation_network,
    }

