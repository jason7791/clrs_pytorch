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

"""PyTorch Tests for the processor networks."""

import unittest
import torch
import numpy as np
from processors import MPNN, PGN

class ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64
        self.num_nodes = 11
        self.node_feature_size = 32
        self.edge_feature_size = 32
        self.graph_feature_size = 32
        self.hidden_size = 32
        self.out_size = 32

        # Initialize mock inputs for the processor
        self.node_features = torch.randn(self.batch_size, self.num_nodes, self.node_feature_size)
        self.edge_features = torch.randn(self.batch_size, self.num_nodes, self.num_nodes, self.edge_feature_size)
        self.graph_features = torch.randn(self.batch_size, self.graph_feature_size)
        self.adj_matrix = torch.randint(0, 2, (self.batch_size, self.num_nodes, self.num_nodes))
        self.hidden_features = torch.randn(self.batch_size, self.num_nodes, self.hidden_size)

    def test_pgn_shapes(self):
        """Test the shape and output of the PGN processor."""
        model = PGN(
            out_size=self.out_size,
            mid_size=self.hidden_size,
            activation=torch.relu,
            reduction=torch.max,
            msgs_mlp_sizes=[self.hidden_size, self.out_size],
            use_ln=True,
        )

        output_node_features, _ = model(
            self.node_features,
            self.edge_features,
            self.graph_features,
            self.adj_matrix,
            self.hidden_features,
        )

        # Check the shape of the output node features
        self.assertEqual(output_node_features.shape, (self.batch_size, self.num_nodes, self.out_size))

    def test_mpnn_shapes(self):
        """Test the shape and output of the MPNN processor."""
        model = MPNN(
            out_size=self.out_size,
            mid_size=self.hidden_size,
            activation=torch.relu,
            reduction=torch.sum,  # Using sum as reduction
            msgs_mlp_sizes=[self.hidden_size, self.out_size],
            use_ln=False,
        )

        output_node_features, _ = model(
            self.node_features,
            self.edge_features,
            self.graph_features,
            self.adj_matrix,
            self.hidden_features,
        )

        # Check the shape of the output node features
        self.assertEqual(output_node_features.shape, (self.batch_size, self.num_nodes, self.out_size))


if __name__ == '__main__':
    unittest.main()
