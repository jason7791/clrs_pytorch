import os
import json
import unittest
import tempfile
import torch
from unittest.mock import patch

# Import the main function from run_node.py
import run_node

# Create a dummy dataset for node classification.
# This dummy dataset mimics a PygNodePropPredDataset with a single graph.
class DummyDataset:
    def __init__(self, name, transform=None):
        self.name = name
        self.transform = transform

    def __getitem__(self, idx):
        from torch_geometric.data import Data
        # Create dummy node features (5 nodes, 10 features)
        x = torch.randn(5, 10)
        # Create a dummy sparse adjacency (identity matrix for simplicity)
        row = torch.arange(5)
        col = torch.arange(5)
        try:
            # Try to import SparseTensor from torch_sparse (if available)
            from torch_sparse import SparseTensor
        except ImportError:
            # Fallback to torch_geometric.typing if necessary
            from torch_geometric.typing import SparseTensor
        adj_t = SparseTensor(row=row, col=col, sparse_sizes=(5, 5))
        # Create dummy binary labels for 5 nodes (shape: [5, 1])
        y = torch.randint(0, 2, (5, 1)).float()
        return Data(x=x, y=y, adj_t=adj_t)

    def __len__(self):
        return 1

    def get_idx_split(self):
        # Provide dummy splits: first 3 nodes for training, next one for validation, last one for testing.
        return {'train': torch.arange(0, 3), 'valid': torch.arange(3, 4), 'test': torch.arange(4, 5)}


class TestRunNode(unittest.TestCase):

    def setUp(self):
        """Set up the test environment for run_node.py tests."""
        # Create a temporary directory for checkpoint and performance files.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(self.temp_dir.name, "model.pth")
        self.performance_path = os.path.join(self.temp_dir.name, "performance.json")
        
        # Prepare dummy command-line arguments.
        self.test_args = [
            "run_node.py",
            "--dataset", "dummy",  # we'll patch PygNodePropPredDataset to return our DummyDataset
            "--device", "cpu",
            "--epochs", "2",  # Keep epochs small for testing.
            "--batch_size", "32",
            "--hidden_dim", "16",
            "--num_layers", "2",
            "--checkpoint_path", self.checkpoint_path,
            "--performance_path", self.performance_path,
            "--early_stop_patience", "10",
            "--seed", "42",
            "--model", "serial",
            # Use defaults for gated, use_triplets, use_pretrain_weights.
        ]
        
    def tearDown(self):
        """Clean up temporary files after tests."""
        self.temp_dir.cleanup()

    def test_set_seed_consistency(self):
        """Test that set_seed produces consistent random numbers."""
        run_node.set_seed(42)
        rand1 = torch.rand(1).item()
        run_node.set_seed(42)
        rand2 = torch.rand(1).item()
        self.assertEqual(rand1, rand2, "Random values should be identical with fixed seed.")

    @patch("run_node.logging.info")
    def test_save_results(self, mock_log):
        """Test saving performance results."""
        results = {
            "train_accuracies": [0.9],
            "valid_accuracies": [0.85],
            "test_accuracies": [0.8]
        }
        # Call the utility function from run_node.py.
        run_node.save_results(results, self.performance_path)
        self.assertTrue(os.path.exists(self.performance_path), "Performance file should be created.")
        with open(self.performance_path, "r") as f:
            loaded_results = json.load(f)
        self.assertEqual(results, loaded_results, "Saved and loaded results should match.")
        self.assertTrue(mock_log.called, "Logging should be called during save_results.")

    @patch("run_node.logging.warning")
    def test_load_model_no_checkpoint(self, mock_warn):
        """Test load_model when the checkpoint does not exist."""
        dummy_model = run_node.BaselineNodeModel(
            out_dim=1,
            hidden_dim=16,
            num_layers=2,
            use_pretrain_weights=False,
            pretrained_weights_path="",
            use_triplets=False,
            gated=False
        )
        # Ensure the checkpoint does not exist.
        if os.path.exists("non_existent_checkpoint.pth"):
            os.remove("non_existent_checkpoint.pth")
        run_node.load_model(dummy_model, "non_existent_checkpoint.pth")
        mock_warn.assert_called_once_with("No checkpoint found at non_existent_checkpoint.pth, starting from scratch.")


if __name__ == "__main__":
    unittest.main()
