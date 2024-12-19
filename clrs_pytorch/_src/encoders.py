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
"""Encoder utilities."""

import functools
from clrs_pytorch._src import probing
from clrs_pytorch._src import specs
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
import numpy as np

_Array = torch.Tensor
_DataPoint = probing.DataPoint
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Type = specs.Type


def construct_encoders(stage: str, loc: str, t: str,
                       hidden_dim: int, init: str, name: str):
    """Constructs encoders in PyTorch."""
    
    # Define initializer
    def truncated_normal_(tensor, stddev):
        with torch.no_grad():
            # Fill with values from a normal distribution
            tensor.normal_(mean=0.0, std=stddev)
            # Truncate values within 2 standard deviations
            tensor.clamp_(-2 * stddev, 2 * stddev)

    if init == 'xavier_on_scalars' and stage == '_Stage.HINT' and t == '_Type.SCALAR':
        stddev = 1.0 / math.sqrt(hidden_dim)
        initialiser = lambda tensor: truncated_normal_(tensor, stddev)
    elif init in ['default', 'xavier_on_scalars']:
        initialiser = None
    else:
        raise ValueError(f'Encoder initializer {init} not supported.')

    def create_linear(out_features: int) -> nn.Linear:
        layer = nn.Linear(1, out_features)
        if initialiser:
            initialiser(layer.weight) 
        nn.init.zeros_(layer.bias)  
        return layer
    
    def create_lazy_linear(out_features):
       return nn.LazyLinear(out_features)
    
    encoders = nn.ModuleList([create_lazy_linear(hidden_dim)])
    if loc == _Location.EDGE and t == _Type.POINTER:
        # Edge pointers need two-way encoders
        encoders.append(create_lazy_linear(hidden_dim))

    return encoders

def preprocess(dp: _DataPoint, nb_nodes: int) -> _DataPoint:
    """Pre-process data point.

    Make sure that the data is ready to be encoded into features.
    If the data is of POINTER type, we expand the compressed index representation
    to a full one-hot. But if the data is a SOFT_POINTER, the representation
    is already expanded and we just overwrite the type as POINTER so that
    it is treated as such for encoding.

    Args:
        dp: A DataPoint to prepare for encoding.
        nb_nodes: Number of nodes in the graph, necessary to expand pointers to
        the right dimension.
        
    Returns:
        The datapoint, with data and possibly type modified.
    """
      
    if isinstance(dp.data, np.ndarray):
      data = torch.tensor(dp.data,dtype=torch.float32)
    else:
      data = dp.data.detach()
    new_type = dp.type_

    if dp.type_ == _Type.POINTER:
        data = F.one_hot(data.long(), num_classes=nb_nodes).float()
    else:
        data = data.float()
        if dp.type_ == _Type.SOFT_POINTER:
          new_type = _Type.POINTER

    dp = probing.DataPoint(
        name=dp.name, location=dp.location, type_=new_type, data=data
    )

    return dp


def accum_adj_mat(dp: _DataPoint, adj_mat: _Array) -> _Array:
  """Accumulates adjacency matrix."""
  if dp.location == _Location.NODE and dp.type_ in [_Type.POINTER,
                                                    _Type.PERMUTATION_POINTER]:
    adj_mat += ((dp.data + dp.data.permute(0,2,1)) > 0.5)
  elif dp.location == _Location.EDGE and dp.type_ == _Type.MASK:
    adj_mat += ((dp.data + dp.data.permute(0,2,1)) > 0.0)

  return (adj_mat > 0.).float() 


def accum_edge_fts(encoders, dp: _DataPoint, edge_fts: _Array) -> _Array:
  """Encodes and accumulates edge features."""
  if dp.location == _Location.NODE and dp.type_ in [_Type.POINTER,
                                                    _Type.PERMUTATION_POINTER]:
    encoding = _encode_inputs(encoders, dp)
    edge_fts += encoding

  elif dp.location == _Location.EDGE:
    encoding = _encode_inputs(encoders, dp)
    if dp.type_ == _Type.POINTER:
      # Aggregate pointer contributions across sender and receiver nodes.
      encoding_2 = encoders[1](dp.data.unsqueeze(-1))
      edge_fts += torch.mean(encoding, dim=1) + torch.mean(encoding_2, dim=2)
    else:
      edge_fts += encoding

  return edge_fts


def accum_node_fts(encoders, dp: _DataPoint, node_fts: _Array) -> _Array:
  """Encodes and accumulates node features."""
  is_pointer = (dp.type_ in [_Type.POINTER, _Type.PERMUTATION_POINTER])
  if ((dp.location == _Location.NODE and not is_pointer) or
      (dp.location == _Location.GRAPH and dp.type_ == _Type.POINTER)):
    encoding = _encode_inputs(encoders, dp)
    node_fts = node_fts + encoding

  return node_fts


def accum_graph_fts(encoders, dp: _DataPoint,
                    graph_fts: _Array) -> _Array:
  """Encodes and accumulates graph features."""
  if dp.location == _Location.GRAPH and dp.type_ != _Type.POINTER:
    encoding = _encode_inputs(encoders, dp)
    graph_fts += encoding

  return graph_fts


def _encode_inputs(encoders, dp: _DataPoint) -> _Array:
  if dp.type_ == _Type.CATEGORICAL:
    encoding = encoders[0](dp.data)
  else:
    encoding = encoders[0](torch.unsqueeze(dp.data, -1))
  return encoding
