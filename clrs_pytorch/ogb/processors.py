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

"""PyTorch implementation of baseline processor networks."""

import abc
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Type aliases for better readability
_Array = torch.Tensor
BIG_NUMBER = 1e6
PROCESSOR_TAG = 'clrs_processor'

class Processor(nn.Module):
  """Processor abstract base class."""

  def __init__(self, name: str):
    if not name.endswith(PROCESSOR_TAG):
      name = name + '_' + PROCESSOR_TAG
    super().__init__()
    self.name = name

  @abc.abstractmethod
  def forward(
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **kwargs,
  ) -> Tuple[_Array, Optional[_Array]]:
    """Processor inference step.

    Args:
      node_fts: Node features.
      edge_fts: Edge features.
      graph_fts: Graph features.
      adj_mat: Graph adjacency matrix.
      hidden: Hidden features.
      **kwargs: Extra kwargs.

    Returns:
      Output of processor inference step as a 2-tuple of (node, edge)
      embeddings. The edge embeddings can be None.
    """
    pass

  @property
  def inf_bias(self):
    return False

  @property
  def inf_bias_edge(self):
    return False


class PGN(Processor):
    """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

    def __init__(
        self,
        out_size: int,
        mid_size: Optional[int] = None,
        mid_act: Optional[Callable] = None,
        activation: Optional[Callable] = F.relu,
        reduction: Callable = torch.max,
        msgs_mlp_sizes: Optional[List[int]] = None,
        use_ln: bool = False,
        use_triplets: bool = False,
        nb_triplet_fts: int = 8,
        gated: bool = False,
        name: str = 'mpnn_aggr',
    ):
        super().__init__(name=name)
        if mid_size is None:
            self.mid_size = out_size
        else:
            self.mid_size = mid_size
        self.out_size = out_size
        self.mid_act = mid_act
        self.activation = activation
        self.reduction = reduction
        self._msgs_mlp_sizes = msgs_mlp_sizes
        self.use_ln = use_ln
        self.use_triplets = use_triplets
        self.nb_triplet_fts = nb_triplet_fts
        self.gated = gated

        self.m_1 = nn.Linear( 2* self.mid_size, self.mid_size)
        self.m_2 = nn.Linear(2* self.mid_size,self.mid_size)
        self.m_e = nn.Linear(self.mid_size,self.mid_size)
        self.m_g = nn.Linear(self.mid_size, self.mid_size)

        self.o1 = nn.Linear(2* self.mid_size, self.out_size)
        self.o2 = nn.Linear(self.mid_size, self.out_size)

        if self._msgs_mlp_sizes is not None:
            mlp_layers = []
            input_size = self.mid_size
            for size in self._msgs_mlp_sizes:
                mlp_layers.append(nn.Linear(input_size, size))
                mlp_layers.append(nn.ReLU())
                input_size = size
            self.mlp = nn.Sequential(*mlp_layers)
            
        self.ln = None  # Placeholder for LayerNorm

    def forward(
        self,
        node_fts: _Array,
        edge_fts: _Array,
        graph_fts: _Array,
        adj_mat: _Array,
        hidden: _Array,
        **unused_kwargs,
    ) -> Tuple[_Array, Optional[_Array]]:
        """PGN inference step."""

        b, n, _ = node_fts.shape
        assert edge_fts.shape[:-1] == (b, n, n)
        assert graph_fts.shape[:-1] == (b,)
        assert adj_mat.shape == (b, n, n)

        # Concatenate node features and hidden features
        z = torch.cat([node_fts, hidden], dim=-1)  # Shape: (B, N, F_z)

        msg_1 = self.m_1(z)
        msg_2 = self.m_2(z)
        msg_e = self.m_e(edge_fts)
        msg_g = self.m_g(graph_fts)

        tri_msgs = None

        # Message aggregation
        msgs = (
            msg_1.unsqueeze(1) + msg_2.unsqueeze(2) +
            msg_e + msg_g.unsqueeze(1).unsqueeze(2)
        )  # Shape: (B, N, N, F_mid)
        assert not torch.isnan(msgs).any(), "NaN detected in aggregated messages"
        if self._msgs_mlp_sizes is not None:
            msgs = self.mlp(F.relu(msgs))

        if self.mid_act is not None:
            msgs = self.mid_act(msgs)

        # Reduction
        if self.reduction == torch.mean:
            # Perform mean reduction
            msgs = torch.sum(msgs * adj_mat.unsqueeze(-1), dim=1)
            msgs = msgs / torch.sum(adj_mat, dim=-1, keepdim=True)
        elif self.reduction == torch.max:
            # Perform max reduction
            maxarg = torch.where(adj_mat.unsqueeze(-1).bool(),
                                msgs,
                                -BIG_NUMBER)
            msgs, _ = torch.max(maxarg, dim=1)
        else:
            # Perform custom reduction
            msgs = self.reduction(msgs * adj_mat.unsqueeze(-1), dim=1)
        assert not torch.isnan(msgs).any(), "NaN detected after reduction"

        # Compute output
        h_1 = self.o1(z)  # Shape: (B, N, F_out)
        h_2 = self.o2(msgs)  # Shape: (B, N, F_out)
        ret = h_1 + h_2  # Shape: (B, N, F_out)
        if self.activation is not None:
            ret = self.activation(ret)
        if self.use_ln and self.ln is None:
            self.ln = nn.LayerNorm(ret.size(-1)).to(ret.device)  # Create LayerNorm and move to the correct device
        if self.use_ln:
            ret = self.ln(ret)  # Apply LayerNorm

        return ret, tri_msgs  # Updated node embeddings and triplet messages
    

class MPNN(PGN):
    """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

    def forward(
        self,
        node_fts: _Array,
        edge_fts: _Array,
        graph_fts: _Array,
        adj_mat: _Array,
        hidden: _Array,
        **unused_kwargs,
    ) -> Tuple[_Array, Optional[_Array]]:
        adj_mat = torch.ones_like(adj_mat)
        return super().forward(node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused_kwargs)
    