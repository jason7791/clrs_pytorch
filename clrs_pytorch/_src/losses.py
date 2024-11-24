# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Utilities for calculating losses."""

from typing import Dict, List, Tuple
import chex
from clrs_pytorch._src import probing
from clrs_pytorch._src import specs

import torch

_Array = chex.Array
_DataPoint = probing.DataPoint
_Location = specs.Location
_OutputClass = specs.OutputClass
_PredTrajectory = Dict[str, _Array]
_PredTrajectories = List[_PredTrajectory]
_Type = specs.Type

EPS = 1e-12


def _expand_to(x: _Array, y: _Array) -> _Array:
  while len(y.shape) > len(x.shape):
    x = torch.unsqueeze(x, -1)
  return x


def _expand_and_broadcast_to(x: _Array, y: _Array) -> _Array:
  return torch.broadcast_to(_expand_to(x, y), y.shape)


def output_loss_chunked(truth: _DataPoint, pred: _Array,
                        is_last: _Array, nb_nodes: int) -> float:
  """Output loss for time-chunked training."""

  mask = None

  if truth.type_ == _Type.SCALAR:
    loss = (pred - truth.data)**2

  elif truth.type_ == _Type.MASK:
    loss = (
        torch.maximum(pred, 0) - pred * truth.data +
        torch.log1p(torch.exp(-torch.abs(pred))))
    mask = (truth.data != _OutputClass.MASKED)

  elif truth.type_ in [_Type.MASK_ONE, _Type.CATEGORICAL]:
    mask = torch.any(truth.data == _OutputClass.POSITIVE, axis=-1)
    masked_truth = truth.data * (truth.data != _OutputClass.MASKED).float()
    loss = -torch.sum(masked_truth * torch.nn.functional.log_softmax(pred), axis=-1)

  elif truth.type_ == _Type.POINTER:
    loss = -torch.sum(
        torch.nn.functional.one_hot(truth.data, nb_nodes) * torch.nn.functional.log_softmax(pred), axis=-1)

  elif truth.type_ == _Type.PERMUTATION_POINTER:
    # Predictions are NxN logits aiming to represent a doubly stochastic matrix.
    # Compute the cross entropy between doubly stochastic pred and truth_data
    loss = -torch.sum(truth.data * pred, axis=-1)

  if mask is not None:
    mask = mask * _expand_and_broadcast_to(is_last, loss)
  else:
    mask = _expand_and_broadcast_to(is_last, loss)
  total_mask = torch.maximum(torch.sum(mask), EPS)
  return torch.sum(torch.where(mask, loss, 0.0)) / total_mask  # pytype: disable=bad-return-type  


def output_loss(truth: _DataPoint, pred: _Array, nb_nodes: int) -> float:
  """Output loss for full-sample training."""

  if truth.type_ == _Type.SCALAR:
    total_loss = torch.mean((pred - truth.data)**2)

  elif truth.type_ == _Type.MASK:
    loss = (
        torch.maximum(pred, 0) - pred * truth.data +
        torch.log1p(torch.exp(-torch.abs(pred))))
    mask = (truth.data != _OutputClass.MASKED).float()
    total_loss = torch.sum(loss * mask) / torch.sum(mask)

  elif truth.type_ in [_Type.MASK_ONE, _Type.CATEGORICAL]:
    masked_truth = truth.data * (truth.data != _OutputClass.MASKED).float()
    total_loss = (-torch.sum(masked_truth * torch.nn.functional.log_softmax(pred)) /
                  torch.sum(truth.data == _OutputClass.POSITIVE))

  elif truth.type_ == _Type.POINTER:
    total_loss = (
        torch.mean(-torch.sum(
            torch.nn.functional.one_hot(torch.tensor(truth.data), nb_nodes) * torch.nn.functional.log_softmax(pred),
            axis=-1)))

  elif truth.type_ == _Type.PERMUTATION_POINTER:
    # Predictions are NxN logits aiming to represent a doubly stochastic matrix.
    # Compute the cross entropy between doubly stochastic pred and truth_data
    total_loss = torch.mean(-torch.sum(truth.data * pred, axis=-1))

  return total_loss  # pytype: disable=bad-return-type  


def hint_loss(
    truth: _DataPoint,
    preds: List[_Array],
    lengths: _Array,
    nb_nodes: int,
    verbose: bool = False,
):
  """Hint loss for full-sample training."""
  total_loss = 0.
  verbose_loss = {}
  length = truth.data.shape[0] - 1

  loss, mask = _hint_loss(
      truth_data=truth.data[1:],
      truth_type=truth.type_,
      pred=torch.stack(preds),
      nb_nodes=nb_nodes,
  )
  mask *= _is_not_done_broadcast(lengths, torch.arange(length)[:, None], loss)
  loss = torch.sum(loss * mask) / torch.maximum(torch.sum(mask), torch.tensor(EPS))
  if verbose:
    verbose_loss['loss_' + truth.name] = loss
  else:
    total_loss += loss

  return verbose_loss if verbose else total_loss


def _hint_loss(
    truth_data: _Array,
    truth_type: str,
    pred: _Array,
    nb_nodes: int,
) -> Tuple[_Array, _Array]:
  """Hint loss helper."""

  truth_data = torch.tensor(truth_data)

  mask = None
  if truth_type == _Type.SCALAR:
    loss = (pred - truth_data)**2

  elif truth_type == _Type.MASK:
    loss = (torch.maximum(pred, torch.tensor(0)) - pred * truth_data +
            torch.log1p(torch.exp(-torch.abs(pred))))
    mask = torch.tensor(truth_data != _OutputClass.MASKED).float()  # pytype: disable=attribute-error  # numpy-scalars

  elif truth_type == _Type.MASK_ONE:
    loss = -torch.sum(truth_data * torch.nn.functional.log_softmax(pred), axis=-1,
                    keepdims=True)

  elif truth_type == _Type.CATEGORICAL:
    loss = -torch.sum(truth_data * torch.nn.functional.log_softmax(pred), axis=-1)
    mask = torch.any(truth_data == _OutputClass.POSITIVE, axis=-1).float()

  elif truth_type == _Type.POINTER:
    loss = -torch.sum(
        torch.nn.functional.one_hot(torch.tensor(truth_data).long(), nb_nodes) * torch.nn.functional.log_softmax(pred),
        axis=-1)

  elif truth_type == _Type.PERMUTATION_POINTER:
    # Predictions are NxN logits aiming to represent a doubly stochastic matrix.
    # Compute the cross entropy between doubly stochastic pred and truth_data
    loss = -torch.sum(truth_data * pred, axis=-1)

  if mask is None:
    mask = torch.ones_like(loss)
  return loss, mask


def _is_not_done_broadcast(lengths, i, tensor):
  is_not_done = (torch.tensor(lengths) > i + 1) * 1.0
  while len(is_not_done.shape) < len(tensor.shape):  # pytype: disable=attribute-error  # numpy-scalars
    is_not_done = torch.unsqueeze(is_not_done, -1)
  return is_not_done
