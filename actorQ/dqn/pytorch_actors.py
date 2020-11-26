# Internal imports.
import time
import pickle
import traceback
import numpy as np
import gc
from acme import adders
from acme import core
import random
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from typing import Mapping, Sequence
import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import torch
import tree
import sys
from concurrent import futures
from typing import Mapping, Optional, Sequence
from acme import core
import tensorflow as tf
import tree
import msgpack

class TFToPyTorchVariableClient:
  """A variable client for updating variables from a remote source."""

  def __init__(self,
               client: core.VariableSource,
               model,
               update_period: int = 1):
    self.m = model
    self._call_counter = 0
    self._update_period = update_period
    self._client = client
    self._request = lambda: client.get_variables(None)

    self.updated_callbacks = []

    # Create a single background thread to fetch variables without necessarily
    # blocking the actor.
    self._executor = futures.ThreadPoolExecutor(max_workers=1)
    self._async_request = lambda: self._executor.submit(self._request)

    # Initialize this client's future to None to indicate to the `update()`
    # method that there is no pending/running request.
    self._future: Optional[futures.Future] = None

  def add_updated_callback(self, cb):
    self.updated_callbacks.append(cb)

  def update(self):
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
      #self._future = self._async_request()    # todo uncomment
      self._copy(self._request())
      self._call_counter = 0

    if has_active_request and self._future.done():
      # The active request is done so copy the result and remove the future.
      self._copy(self._future.result())
      self._future: Optional[futures.Future] = None
    else:
      # There is either a pending/running request or we're between update
      # periods, so just carry on.
      return

  def update_and_wait(self):
    """Immediately update and block until we get the result."""
    self._copy(self._request())

  def _copy(self, new_variables: Sequence[Sequence[tf.Variable]]):
    """Copies the new variables to the old ones."""

    for cb in self.updated_callbacks:
      cb(new_variables)

def pytorch_model_load_state_dict(model, new_variables):
    if len(new_variables) == 0:
      return
    pytorch_keys = list(model.state_dict().keys())
    # Switch ordering of weight + bias
    new_pytorch_keys = []
    for i in range(0, len(pytorch_keys), 2):
      new_pytorch_keys.append(pytorch_keys[i+1])
      new_pytorch_keys.append(pytorch_keys[i])
    pytorch_keys = new_pytorch_keys
    new_state_dict = {k:torch.from_numpy(v.T) for k,v in zip(pytorch_keys, new_variables)}
    model.load_state_dict(new_state_dict)

def pytorch_quantize(m, q):
    assert(q in [8,16,32])
    if q == 8:
      return torch.quantization.quantize_dynamic(
        m, {torch.nn.Linear}, dtype=torch.qint8)
    if q == 16:
      return torch.quantization.quantize_dynamic(
        m, {torch.nn.Linear}, dtype=torch.float16)
    if q == 32:
      return m

def create_model(input_size, output_size, policy_layer_sizes=(2048,2048,2048)):
    # Create policy network
    # Pytorch equivalent of: https://github.com/deepmind/acme/blob/master/acme/tf/networks/continuous.py
    # First layer

    sizes = [input_size] + list(policy_layer_sizes) + [output_size]
    layers = []
    for i in range(len(sizes)-1):
        in_size, out_size = sizes[i], sizes[i+1]
        layers.append(torch.nn.Linear(in_size, out_size))
        layers.append(torch.nn.Tanh())

    return torch.nn.Sequential(*(layers[:-1]))

class FeedForwardActor(core.Actor):
  """A feed-forward actor.
  An actor based on a feed-forward policy which takes non-batched observations
  and outputs non-batched actions. It also allows adding experiences to replay
  and updating the weights from the policy on the learner.
  """

  def __init__(
      self,
      m,
      epsilon=.05,
      q=32,
      args=None,
      adder: adders.Adder = None,
      variable_client: tf2_variable_utils.VariableClient = None,
  ):
    """Initializes the actor.
    Args:
      policy_network: the policy to run.
      adder: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      variable_client: object which allows to copy weights from the learner copy
        of the policy to the actor copy (in case they are separate).
    """

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._variable_client.add_updated_callback(self.updated)
    self.eps = epsilon
    self.m = m    
    self.q = q
    self.q_m = pytorch_quantize(self.m, self.q)
    self.q_m_state_dict = self.q_m.state_dict()
    self.args = args

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
      pred = self.q_m(torch.from_numpy(observation).reshape(1, -1))
      pred = pred.detach().numpy().flatten()
      if random.random() <= self.eps:
        return np.int32(random.randint(0, pred.shape[0]-1))
      return np.argmax(pred)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    if self._adder:
      self._adder.add(action.astype(np.int32), next_timestep)

  def update(self):
    if self._variable_client:
      self._variable_client.update()

  def updated(self, new_variables):
    t1 = time.time()
    try:
      state_dict = new_variables[0]    
    except:
      return

    if "id" in state_dict:
      del state_dict["id"]

    if self.args["weight_compress"] != 0:
      t_q_decompress_start = time.time()
      for k,v in state_dict.items():
        state_dict[k] = Q_decompress(state_dict[k], self.args["weight_compress"])
      print("Q_decompress time: %f" % (time.time()-t_q_decompress_start))



    if self.q == 32:
      self.q_m.load_state_dict(state_dict)
    else:
      """
      with torch.no_grad():
        for name, child in self.q_m._modules.items():
          print("Loading: ", child, type(child))
          if type(child) == torch.nn.quantized.dynamic.modules.linear.Linear:
            scale = state_dict[name + ".scale"]
            zero_point = state_dict[name + ".zero_point"]
            packed_params = state_dict[name + "._packed_params._packed_params"]
            packed_params_dumped = state_dict[name + "._packed_params._packed_params.dumped"]
            child.scale = scale
            child.zero_point = zero_point
            #child._packed_params._packed_params = torch.ops.quantized.linear_prepack_fp16(packed_params[0], packed_params[1])
            child._packed_params._packed_params = pickle.loads(packed_params_dumped)        

      """
      self.q_m.load_state_dict(state_dict)
      pass


    print("Load state dict time: %f" % (time.time()-t1))
    
