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
import numpy as np
import torch

_Array = chex.Array
_DataPoint = probing.DataPoint
_OutputClass = specs.OutputClass
_Type = specs.Type

EPS = 1e-12


def output_loss(truth: _DataPoint, pred: _Array, nb_nodes: int, device) -> float:
  """Output loss for full-sample training."""
  if isinstance(truth.data, np.ndarray):
      truth_data = torch.tensor(truth.data,dtype=torch.float32, device=device)
  else:
      truth_data = truth.data.detach().to(device)

  if truth.type_ == _Type.SCALAR:
    total_loss = torch.mean((pred - truth_data)**2)

  elif truth.type_ == _Type.MASK:
    loss = (
        torch.maximum(pred, torch.tensor(0.0, device=device)) - pred * truth_data +
        torch.log1p(torch.exp(-torch.abs(pred))))
    mask = (truth_data != _OutputClass.MASKED).float()
    total_loss = torch.sum(loss * mask) / torch.sum(mask)

  elif truth.type_ in [_Type.MASK_ONE, _Type.CATEGORICAL]:
    masked_truth = truth_data * (truth_data != _OutputClass.MASKED).float()
    total_loss = (-torch.sum(masked_truth * torch.nn.functional.log_softmax(pred, dim = -1)) /
                  torch.sum(truth_data == _OutputClass.POSITIVE))

  elif truth.type_ == _Type.POINTER:
    one_hot_truth = torch.nn.functional.one_hot(truth_data.to(dtype=torch.long), nb_nodes).float()
    total_loss = torch.mean(-torch.sum(one_hot_truth * torch.nn.functional.log_softmax(pred, dim=-1), dim=-1))

  elif truth.type_ == _Type.PERMUTATION_POINTER:
    # Predictions are NxN logits aiming to represent a doubly stochastic matrix.
    # Compute the cross entropy between doubly stochastic pred and truth_data
    total_loss = torch.mean(-torch.sum(truth_data * pred, dim=-1))

  return total_loss  # pytype: disable=bad-return-type  


def hint_loss(
    truth: _DataPoint,
    preds: List[_Array],
    lengths: _Array,
    nb_nodes: int,
    device
):
  """Hint loss for full-sample training."""
  total_loss = 0.
  length = truth.data.shape[0] - 1

  loss, mask = _hint_loss(
      truth_data=truth.data[1:],
      truth_type=truth.type_,
      pred=torch.stack(preds),
      nb_nodes=nb_nodes,
      device=device
  )
  mask *= _is_not_done_broadcast(lengths, torch.arange(length, device=device)[:, None], loss, device=device)
  loss = torch.sum(loss * mask) / torch.maximum(torch.sum(mask), torch.tensor(EPS, device=device))

  total_loss += loss

  return total_loss


def _hint_loss(
    truth_data: _Array,
    truth_type: str,
    pred: _Array,
    nb_nodes: int,
    device
) -> Tuple[_Array, _Array]:
  """Hint loss helper."""

  truth_data = torch.tensor(truth_data, device=device)

  mask = None
  if truth_type == _Type.SCALAR:
    loss = (pred - truth_data)**2

  elif truth_type == _Type.MASK:
    loss = (torch.maximum(pred, torch.tensor(0, device=device)) - pred * truth_data +
            torch.log1p(torch.exp(-torch.abs(pred))))
    mask = torch.tensor(truth_data != _OutputClass.MASKED, device=device).float()  # pytype: disable=attribute-error  # numpy-scalars

  elif truth_type == _Type.MASK_ONE:
    loss = -torch.sum(truth_data * torch.nn.functional.log_softmax(pred), dim=-1,
                    keepdims=True)

  elif truth_type == _Type.CATEGORICAL:
    loss = -torch.sum(truth_data * torch.nn.functional.log_softmax(pred), dim=-1)
    mask = torch.any(truth_data == _OutputClass.POSITIVE, dim=-1).float()

  elif truth_type == _Type.POINTER:
    loss = -torch.sum(
        torch.nn.functional.one_hot(torch.tensor(truth_data).long(), nb_nodes) * torch.nn.functional.log_softmax(pred, dim = -1),
        dim=-1)

  elif truth_type == _Type.PERMUTATION_POINTER:
    # Predictions are NxN logits aiming to represent a doubly stochastic matrix.
    # Compute the cross entropy between doubly stochastic pred and truth_data
    loss = -torch.sum(truth_data * pred, dim=-1)

  if mask is None:
    mask = torch.ones_like(loss, device=device)
  return loss, mask


def _is_not_done_broadcast(lengths, i, tensor, device):
  is_not_done = (torch.tensor(lengths, device=device, dtype=torch.float32) > i + 1) * 1.0
  while len(is_not_done.shape) < len(tensor.shape):  # pytype: disable=attribute-error  # numpy-scalars
    is_not_done = torch.unsqueeze(is_not_done, -1)
  return is_not_done
