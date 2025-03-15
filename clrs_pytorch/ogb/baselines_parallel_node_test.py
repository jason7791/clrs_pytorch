import os
import unittest
import torch
import tempfile
from unittest.mock import patch

# Depending on your environment, you might need to import SparseTensor from torch_sparse
# or from torch_geometric.typing. For example:
# from torch_sparse import SparseTensor
from torch_sparse import SparseTensor

from baselines_parallel_node import (
    ParallelNodeModel, rename_keys
)

class TestParallelNodeModel(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        self.device = torch.device("cpu")
        # Create a dummy node feature matrix for a graph with 10 nodes and 8 features.
        self.num_nodes = 10
        self.in_features = 8
        # Use random floats as node features.
        self.x = torch.randn(self.num_nodes, self.in_features)
        
        # Create a dummy cyclic graph using SparseTensor.
        # This creates edges: 0->1, 1->2, ..., 9->0.
        row = torch.arange(0, self.num_nodes)
        col = torch.cat([torch.arange(1, self.num_nodes), torch.tensor([0])])
        self.adj_t = SparseTensor(row=row, col=col, sparse_sizes=(self.num_nodes, self.num_nodes))

        # Create a temporary directory for saving checkpoints.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(self.temp_dir.name, "checkpoint_parallel.pth")

        # Default arguments for initializing the ParallelNodeModel.
        self.default_model_args = {
            "out_dim": 3,         # e.g., 3 classes for node classification.
            "hidden_dim": 16,
            "num_layers": 2,
            "use_pretrain_weights": False,
            "pretrained_weights_path": "",
            "use_triplets": True,   # Test with triplet messages enabled.
            "gated": False,
            "nb_triplet_fts": 8,
        }

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_parallel_node_model_initialization(self):
        """Test that the ParallelNodeModel initializes correctly."""
        model = ParallelNodeModel(**self.default_model_args)
        self.assertIsInstance(model, torch.nn.Module, "Model should be a PyTorch Module")

    def test_parallel_node_model_forward(self):
        """Test the forward pass of ParallelNodeModel."""
        model = ParallelNodeModel(**self.default_model_args).to(self.device)
        output = model(self.x, self.adj_t)
        # Since this is a node-level model, expect output shape: (num_nodes, out_dim)
        self.assertEqual(
            output.shape,
            (self.num_nodes, self.default_model_args["out_dim"]),
            "Output shape should be (num_nodes, out_dim)"
        )

    def test_rename_keys(self):
        """Test renaming keys in a state dictionary."""
        state_dict = {
            "net_fn.processor.layer1.weight": torch.rand(3, 3),
            "net_fn.processor.layer2.bias": torch.rand(3),
        }
        updated_state_dict = rename_keys(state_dict, old_prefix="net_fn.processor")
        self.assertIn("layer1.weight", updated_state_dict)
        self.assertIn("layer2.bias", updated_state_dict)
        self.assertNotIn("net_fn.processor.layer1.weight", updated_state_dict)


if __name__ == "__main__":
    unittest.main()
