import os
import unittest
import torch
import tempfile
from unittest.mock import patch
from torch_sparse import SparseTensor

from baselines_node import (
    BaselineNodeModel, restore_model, rename_keys, print_weight_norms
)

class TestBaselineNodeModel(unittest.TestCase):

    def setUp(self):
        """Set up test environment before each test."""
        self.device = torch.device("cpu")
        # Create a dummy node feature matrix:
        self.num_nodes = 10
        self.in_features = 8
        # Use random floats as node features.
        self.x = torch.randn(self.num_nodes, self.in_features)

        # Create a dummy sparse adjacency.
        # For example, a cyclic graph: 0->1, 1->2, ..., 9->0.
        row = torch.arange(0, self.num_nodes)
        col = torch.cat([torch.arange(1, self.num_nodes), torch.tensor([0])])
        self.adj_t = SparseTensor(row=row, col=col, sparse_sizes=(self.num_nodes, self.num_nodes))

        # Create a temporary directory for checkpoints.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(self.temp_dir.name, "checkpoint_node.pth")

        # Default arguments for initializing the node model.
        self.default_model_args = {
            "out_dim": 3,         # e.g. three classes.
            "hidden_dim": 16,
            "num_layers": 2,
            "use_pretrain_weights": False,
            "pretrained_weights_path": "",
            "use_triplets": False,
            "gated": False
        }

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_baseline_node_model_initialization(self):
        """Test that the BaselineNodeModel initializes correctly."""
        model = BaselineNodeModel(**self.default_model_args)
        self.assertIsInstance(model, torch.nn.Module, "Model should be a PyTorch Module")

    def test_baseline_node_model_forward(self):
        """Test the forward pass of BaselineNodeModel."""
        model = BaselineNodeModel(**self.default_model_args).to(self.device)
        output = model(self.x, self.adj_t)
        # Expect output shape: (num_nodes, out_dim)
        self.assertEqual(output.shape, (self.num_nodes, self.default_model_args["out_dim"]),
                         "Output shape should be (num_nodes, out_dim)")

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
