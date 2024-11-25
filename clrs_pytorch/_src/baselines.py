"""JAX implementation of CLRS baseline models."""

import functools
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import chex

from clrs_pytorch._src import decoders
from clrs_pytorch._src import losses
from clrs_pytorch._src import model
from clrs_pytorch._src import nets
from clrs_pytorch._src import probing
from clrs_pytorch._src import processors
from clrs_pytorch._src import samplers
from clrs_pytorch._src import specs

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Union, List, Optional, Dict, Tuple

_Array = chex.Array
_DataPoint = probing.DataPoint
_Features = samplers.Features
_FeaturesChunked = samplers.FeaturesChunked
_Feedback = samplers.Feedback
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = samplers.Trajectory
_Type = specs.Type
_OutputClass = specs.OutputClass


class BaselineModel(model.Model):
  """Model implementation with selectable message passing algorithm."""

  def __init__(
      self,
      spec: Union[_Spec, List[_Spec]],
      dummy_trajectory: Union[List[_Feedback], _Feedback],
      processor_factory: processors.ProcessorFactory,
      hidden_dim: int = 32,
      encode_hints: bool = False,
      decode_hints: bool = True,
      encoder_init: str = 'default',
      use_lstm: bool = False,
      learning_rate: float = 0.005,
      grad_clip_max_norm: float = 0.0,
      checkpoint_path: str = '/tmp/clrs3',
      freeze_processor: bool = False,
      dropout_prob: float = 0.0,
      hint_teacher_forcing: float = 0.0,
      hint_repred_mode: str = 'soft',
      name: str = 'base_model',
      nb_msg_passing_steps: int = 1,
  ):
    """Constructor for BaselineModel.

    The model consists of encoders, processor and decoders. It can train
    and evaluate either a single algorithm or a set of algorithms; in the
    latter case, a single processor is shared among all the algorithms, while
    the encoders and decoders are separate for each algorithm.

    Args:
      spec: Either a single spec for one algorithm, or a list of specs for
        multiple algorithms to be trained and evaluated.
      dummy_trajectory: Either a single feedback batch, in the single-algorithm
        case, or a list of feedback batches, in the multi-algorithm case, that
        comply with the `spec` (or list of specs), to initialize network size.
      processor_factory: A callable that takes an `out_size` parameter
        and returns a processor (see `processors.py`).
      hidden_dim: Size of the hidden state of the model, i.e., size of the
        message-passing vectors.
      encode_hints: Whether to provide hints as model inputs.
      decode_hints: Whether to provide hints as model outputs.
      encoder_init: The initialiser type to use for the encoders.
      use_lstm: Whether to insert an LSTM after message passing.
      learning_rate: Learning rate for training.
      grad_clip_max_norm: if greater than 0, the maximum norm of the gradients.
      checkpoint_path: Path for loading/saving checkpoints.
      freeze_processor: If True, the processor weights will be frozen and
        only encoders and decoders (and, if used, the lstm) will be trained.
      dropout_prob: Dropout rate in the message-passing stage.
      hint_teacher_forcing: Probability of using ground-truth hints instead
        of predicted hints as inputs during training (only relevant if
        `encode_hints`=True)
      hint_repred_mode: How to process predicted hints when fed back as inputs.
        Only meaningful when `encode_hints` and `decode_hints` are True.
        Options are:
          - 'soft', where we use softmaxes for categoricals, pointers
              and mask_one, and sigmoids for masks. This will allow gradients
              to flow through hints during training.
          - 'hard', where we use argmax instead of softmax, and hard
              thresholding of masks. No gradients will go through the hints
              during training; even for scalar hints, which don't have any
              kind of post-processing, gradients will be stopped.
          - 'hard_on_eval', which is soft for training and hard for evaluation.
      name: Model name.
      nb_msg_passing_steps: Number of message passing steps per hint.

    Raises:
      ValueError: if `encode_hints=True` and `decode_hints=False`.
    """
    super(BaselineModel, self).__init__(spec=spec)

    if encode_hints and not decode_hints:
      raise ValueError('`encode_hints=True`, `decode_hints=False` is invalid.')

    assert hint_repred_mode in ['soft', 'hard', 'hard_on_eval']

    self.decode_hints = decode_hints
    self.checkpoint_path = checkpoint_path
    self.name = name
    self._freeze_processor = freeze_processor
    self.learning_rate = learning_rate
    self.nb_msg_passing_steps = nb_msg_passing_steps

    self.nb_dims = []
    if isinstance(dummy_trajectory, _Feedback):
      assert len(self._spec) == 1
      dummy_trajectory = [dummy_trajectory]
    for traj in dummy_trajectory:
      nb_dims = {}
      for inp in traj.features.inputs:
        nb_dims[inp.name] = inp.data.shape[-1]
      for hint in traj.features.hints:
        nb_dims[hint.name] = hint.data.shape[-1]
      for outp in traj.outputs:
        nb_dims[outp.name] = outp.data.shape[-1]
      self.nb_dims.append(nb_dims)


    self.net_fn = nets.Net(self._spec, hidden_dim, encode_hints, self.decode_hints,
                      processor_factory, use_lstm, encoder_init,
                      dropout_prob, hint_teacher_forcing,
                      hint_repred_mode,
                      self.nb_dims, self.nb_msg_passing_steps)

    self._device_params = None
    self._device_opt_state = None
    self.opt_state_skeleton = None



  def init(self, features: Union[_Features, List[_Features]]):
    if not isinstance(features, list):
      assert len(self._spec) == 1
      features = [features]
    self.params = self.net_fn(features, True,  # pytype: disable=wrong-arg-types 
                                   algorithm_index=-1,
                                   return_hints=False,
                                   return_all_outputs=False)
    self.opt = optim.Adam(self.net_fn.parameters(),lr=self.learning_rate)


  def _compute_grad(self, feedback, algorithm_index):
    self.net_fn.zero_grad()
    lss = self._loss(feedback, algorithm_index)
    lss.backward()
    grads = {name: param.grad for name, param in self.net_fn.named_parameters()}
    return self._maybe_pmean(lss), self._maybe_pmean(grads)

  def _feedback(self, feedback, algorithm_index):
    lss = self._loss(feedback, algorithm_index)
    self.opt.zero_grad()
    lss.backward()
    before_update = [param.clone() for param in self.net_fn.parameters()]
    self.opt.step()
    after_update = [param.clone() for param in self.net_fn.parameters()]
    for i, (before, after) in enumerate(zip(before_update, after_update)):
        print(f"Param {i} change: {(after - before).abs().max().item()}")
    return lss 

  def _predict(self,  features: _Features,
               algorithm_index: int, return_hints: bool,
               return_all_outputs: bool):
    outs, hint_preds = self.net_fn([features],
        repred=True, algorithm_index=algorithm_index,
        return_hints=return_hints,
        return_all_outputs=return_all_outputs)
    outs = decoders.postprocess(self._spec[algorithm_index],
                                outs,
                                sinkhorn_temperature=0.1,
                                sinkhorn_steps=50,
                                hard=True,
                                )
    return outs, hint_preds

  def compute_grad(
      self,
      feedback: _Feedback,
      algorithm_index: Optional[int] = None,
  ) -> Tuple[float, _Array]:
    """Compute gradients."""

    if algorithm_index is None:
      assert len(self._spec) == 1
      algorithm_index = 0
    assert algorithm_index >= 0

    # Calculate gradients.
    loss, grads = self._compute_grad( feedback, algorithm_index)

    return  loss, grads

  def feedback(self, feedback: _Feedback,
               algorithm_index=None) -> float:
    if algorithm_index is None:
      assert len(self._spec) == 1
      algorithm_index = 0
    # Calculate and apply gradients.
    loss = self._feedback( feedback,algorithm_index)
    return loss

  def predict(self,  features: _Features,
              algorithm_index: Optional[int] = None,
              return_hints: bool = False,
              return_all_outputs: bool = False):
    """Model inference step."""
    if algorithm_index is None:
      assert len(self._spec) == 1
      algorithm_index = 0

    return self._predict(features,
            algorithm_index,
            return_hints,
            return_all_outputs)

  def _loss(self, feedback, algorithm_index):
    """Calculates model loss f(feedback; params)."""
    output_preds, hint_preds = self.net_fn( [feedback.features],
        repred=False,
        algorithm_index=algorithm_index,
        return_hints=True,
        return_all_outputs=False)
    nb_nodes = _nb_nodes(feedback, is_chunked=False)
    lengths = feedback.features.lengths
    total_loss = 0.0  # Start with a float to accumulate loss

    # Calculate output loss.
    for truth in feedback.outputs:
      loss = losses.output_loss(
          truth=truth,
          pred=output_preds[truth.name],
          nb_nodes=nb_nodes,
      )
      total_loss += loss  # Accumulate loss
    print("total loss after output loss", total_loss)
    # Optionally accumulate hint losses.
    if self.decode_hints:
      for truth in feedback.features.hints:
        loss = losses.hint_loss(
            truth=truth,
            preds=[x[truth.name] for x in hint_preds],
            lengths=lengths,
            nb_nodes=nb_nodes,
        )
        total_loss += loss  # Accumulate loss
    print("final total loss: ", total_loss)

    return total_loss

  def verbose_loss(self, feedback: _Feedback, extra_info) -> Dict[str, _Array]:
    """Gets verbose loss information."""
    hint_preds = extra_info

    nb_nodes = _nb_nodes(feedback, is_chunked=False)
    lengths = feedback.features.lengths
    losses_ = {}

    # Optionally accumulate hint losses.
    if self.decode_hints:
      for truth in feedback.features.hints:
        losses_.update(
            losses.hint_loss(
                truth=truth,
                preds=[x[truth.name] for x in hint_preds],
                lengths=lengths,
                nb_nodes=nb_nodes,
                verbose=True,
            ))

    return losses_

  def restore_model(self, file_name: str, only_load_processor: bool = False):
    """Restore model from `file_name`."""
    path = os.path.join(self.checkpoint_path, file_name)
    with open(path, 'rb') as f:
      restored_state = pickle.load(f)
      self.load_state_dict(restored_state['params'])

  def save_model(self, file_name: str):
    """Save model (processor weights only) to `file_name`"""
    os.makedirs(self.checkpoint_path, exist_ok=True)
    to_save = {'params': self.net_fn.parameters()}
    path = os.path.join(self.checkpoint_path, file_name)
    with open(path, 'wb') as f:
      pickle.dump(to_save, f)


def _nb_nodes(feedback: _Feedback, is_chunked) -> int:
  for inp in feedback.features.inputs:
    if inp.location in [_Location.NODE, _Location.EDGE]:
      if is_chunked:
        return inp.data.shape[2]  # inputs are time x batch x nodes x ...
      else:
        return inp.data.shape[1]  # inputs are batch x nodes x ...
  assert False
