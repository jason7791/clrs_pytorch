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
import numpy as np

# Type aliases for better readability
_Array = torch.Tensor
_Fn = Callable[..., Any]
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

        self.m_1 = nn.Linear(2*self.mid_size, self.mid_size)
        self.m_2 = nn.Linear(2*self.mid_size,self.mid_size)
        self.m_e = nn.Linear(self.mid_size,self.mid_size)
        self.m_g = nn.Linear(self.mid_size, self.mid_size)

        self.o1 = nn.Linear(2*self.mid_size, self.out_size)
        self.o2 = nn.Linear(self.mid_size, self.out_size)

        if self.use_ln:
            self.ln = nn.LayerNorm(self.out_size, elementwise_affine=True)

        if self.gated:
            self.gate1 = nn.Linear(2*self.mid_size, self.out_size)
            self.gate2 = nn.Linear(self.mid_size, self.out_size)
            self.gate3 = nn.Linear(self.out_size, self.out_size)
            self.gate3.bias.data.fill_(-3)  # Initialize bias to -3


        if self._msgs_mlp_sizes is not None:
            mlp_layers = []
            input_size = self.mid_size
            for size in self._msgs_mlp_sizes:
                mlp_layers.append(nn.Linear(input_size, size))
                mlp_layers.append(nn.ReLU())
                input_size = size
            self.mlp = nn.Sequential(*mlp_layers)
            
        if self.use_triplets:
            self.t_1 = nn.Linear(2*self.mid_size,self.nb_triplet_fts)
            self.t_2 = nn.Linear(2*self.mid_size,self.nb_triplet_fts)
            self.t_3 = nn.Linear(2*self.mid_size,self.nb_triplet_fts)
            self.t_e_1 = nn.Linear(self.mid_size,self.nb_triplet_fts)
            self.t_e_2 = nn.Linear(self.mid_size,self.nb_triplet_fts)
            self.t_e_3 = nn.Linear(self.mid_size,self.nb_triplet_fts)
            self.t_g = nn.Linear(self.mid_size,self.nb_triplet_fts)
            self.o3 = nn.Linear(self.mid_size, self.out_size)
            
        self.ln = None  # Placeholder for LayerNorm

    def get_triplet_msgs(self, z, edge_fts, graph_fts, nb_triplet_fts):
        """Triplet messages, as done by Dudzik and Velickovic (2022)."""


        tri_1 = self.t_1(z)
        tri_2 = self.t_2(z)
        tri_3 = self.t_3(z)
        tri_e_1 = self.t_e_1(edge_fts)
        tri_e_2 = self.t_e_2(edge_fts)
        tri_e_3 = self.t_e_3(edge_fts)
        tri_g = self.t_g(graph_fts)

        return (
            tri_1.unsqueeze(2).unsqueeze(3)      +  # (B, N, 1, 1, H)
            tri_2.unsqueeze(1).unsqueeze(3)      +  # + (B, 1, N, 1, H)
            tri_3.unsqueeze(1).unsqueeze(2)      +  # + (B, 1, 1, N, H)
            tri_e_1.unsqueeze(3)                 +  # + (B, N, N, 1, H)
            tri_e_2.unsqueeze(2)                 +  # + (B, N, 1, N, H)
            tri_e_3.unsqueeze(1)                 +  # + (B, 1, N, N, H)
            tri_g.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # + (B, 1, 1, 1, H)
        )  # = (B, N, N, N, H)

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
        z = torch.cat([node_fts, hidden], dim=-1)  # Shape: (B, N, 2 * F_z)

        msg_1 = self.m_1(z)
        msg_2 = self.m_2(z)
        msg_e = self.m_e(edge_fts)
        msg_g = self.m_g(graph_fts)

        tri_msgs = None

        if self.use_triplets:
            # Triplet messages, as done by Dudzik and Velickovic (2022)
            triplets = self.get_triplet_msgs(z, edge_fts, graph_fts, self.nb_triplet_fts)
            tri_msgs = self.o3(torch.max(triplets, dim=1))  # (B, N, N, H)

            if self.activation is not None:
                tri_msgs = self.activation(tri_msgs)


        # Message aggregation
        msgs = (
            msg_1.unsqueeze(1) + msg_2.unsqueeze(2) +
            msg_e + msg_g.unsqueeze(1).unsqueeze(2)
        )  # Shape: (B, N, N, F_mid)

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
        # Compute output
        h_1 = self.o1(z)  # Shape: (B, N, F_out)
        h_2 = self.o2(msgs)  # Shape: (B, N, F_out)
        ret = h_1 + h_2  # Shape: (B, N, F_out)

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            ret = self.ln(ret)  

        if self.gated:
            gate = torch.sigmoid(self.gate3(torch.relu(self.gate1(z) + self.gate2(msgs))))
            ret = ret * gate + hidden * (1 - gate)

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


ProcessorFactory = Callable[[int], Processor]


def get_processor_factory(
    kind: str,
    use_ln: bool,
    nb_triplet_fts: int,
    nb_heads: Optional[int] = None
) -> ProcessorFactory:
    """Returns a processor factory.

    Args:
      kind: One of the available types of processor.
      use_ln: Whether the processor passes the output through a layernorm layer.
      nb_triplet_fts: How many triplet features to compute.
      nb_heads: Number of attention heads for GAT processors.
    Returns:
      A callable that takes an `out_size` parameter (equal to the hidden
      dimension of the network) and returns a processor instance.
    """
    def _factory(out_size: int) -> Processor:
        if kind == 'mpnn':
            processor = MPNN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=False,
                nb_triplet_fts=0,
            )
        elif kind == 'pgn':
            processor = PGN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=False,
                nb_triplet_fts=0,
            )
        elif kind == 'triplet_mpnn':
            processor = MPNN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=True,
                nb_triplet_fts=nb_triplet_fts,
            )
        elif kind == 'triplet_pgn':
            processor = PGN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=True,
                nb_triplet_fts=nb_triplet_fts,
            )
        elif kind == 'gpgn':
            processor = PGN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=False,
                nb_triplet_fts=nb_triplet_fts,
                gated=True,
            )
        elif kind == 'gmpnn':
            processor = MPNN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=False,
                nb_triplet_fts=nb_triplet_fts,
                gated=True,
            )
        elif kind == 'triplet_gpgn':
            processor = PGN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=True,
                nb_triplet_fts=nb_triplet_fts,
                gated=True,
            )
        elif kind == 'triplet_gmpnn':
            processor = MPNN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=True,
                nb_triplet_fts=nb_triplet_fts,
                gated=True,
            )
        else:
            raise ValueError(f'Unexpected processor kind {kind}')

        return processor

    return _factory