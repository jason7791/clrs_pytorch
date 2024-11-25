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


class Processor(nn.Module, abc.ABC):
    """Processor abstract base class."""

    def __init__(self, name: str):
        super().__init__()
        if not name.endswith(PROCESSOR_TAG):
            name = name + '_' + PROCESSOR_TAG
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
          node_fts: Node features. Shape: (B, N, F_node)
          edge_fts: Edge features. Shape: (B, N, N, F_edge)
          graph_fts: Graph features. Shape: (B, F_graph)
          adj_mat: Graph adjacency matrix. Shape: (B, N, N)
          hidden: Hidden features. Shape: (B, N, F_hidden)
          **kwargs: Extra kwargs.

        Returns:
          A tuple containing:
            - Updated node embeddings: Shape (B, N, F_out)
            - Updated edge embeddings or None: Shape (B, N, N, F_edge_out) or None
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
        super(PGN, self).__init__(name=name)
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

        self.m_1 = nn.LazyLinear(self.mid_size)
        self. m_2 = nn.LazyLinear(self.mid_size)
        self.m_e = nn.LazyLinear(self.mid_size)
        self.m_g = nn.LazyLinear(self.mid_size)

        self.o1 = nn.LazyLinear(self.out_size)
        self.o2 = nn.LazyLinear(self.out_size)

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
        if self._msgs_mlp_sizes is not None:
            layers = []
            for i in range(len(self._msgs_mlp_sizes)):
                layers.append(nn.LazyLinear(self._msgs_mlp_sizes[i]))
                if i < len(self._msgs_mlp_sizes) - 1:  # Add ReLU for all except the last layer
                    layers.append(nn.ReLU())
            mlp = nn.Sequential(*layers)
            msgs = mlp(F.relu(msgs))  # Apply ReLU before passing to MLP

        if self.mid_act is not None:
            msgs = self.mid_act(msgs)

        # Apply adjacency matrix
        # adj_mat_expanded = adj_mat.unsqueeze(-1)  # Shape: (B, N, N, 1)
        # msgs = msgs * adj_mat_expanded  # Zero out messages where adj_mat is 0

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
            # Define LayerNorm
            self.ln = nn.LayerNorm(ret.size(-1))  # Normalize across the last dimension
            ret = self.ln(ret)  # Apply LayerNorm to `ret`

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


def _position_encoding(sentence_size: int, embedding_size: int) -> np.ndarray:
    """Position Encoding described in section 4.1 [1]."""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return encoding.T
