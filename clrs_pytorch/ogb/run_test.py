import os
import json
import unittest
import torch
import tempfile
from unittest.mock import patch

from run import (
    BaselineModel, ParallelMPNNModel, train, evaluate, save_results, load_model, set_seed
)
from ogb.graphproppred import Evaluator
from torch_geometric.data import Data, DataLoader


class TestRun(unittest.TestCase):

    def setUp(self):
        """Set up test environment before each test."""
        self.device = torch.device("cpu")  # Use CPU for unit tests

        # Mock dataset with categorical (integer) features for embedding layer
        self.dummy_batch = Data(
            x=torch.randint(0, 100, (5, 10)),  # Ensure integer values for embedding layers
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),  # Edges
            y=torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32)  # Labels
        )
        self.dataset = [self.dummy_batch] * 10  # Mock dataset with multiple copies
        self.loader = DataLoader(self.dataset, batch_size=2, shuffle=False)

        # Create a temporary directory for saving files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(self.temp_dir.name, "checkpoint.pth")
        self.performance_path = os.path.join(self.temp_dir.name, "performance.json")

        # Default arguments for model initialization
        self.default_model_args = {
            "out_dim": 1,
            "hidden_dim": 16,
            "num_layers": 2,
            "reduction": torch.max,
            "use_pretrain_weights": False,
            "pretrained_weights_path": "",
            "use_triplets": False,
            "gated": False
        }

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_set_seed(self):
        """Test that setting the seed produces consistent results."""
        set_seed(42)
        rand1 = torch.rand(1).item()
        set_seed(42)
        rand2 = torch.rand(1).item()
        self.assertEqual(rand1, rand2, "Random values should match when seed is fixed")

    def test_baseline_model_initialization(self):
        """Test initialization of the BaselineModel."""
        model = BaselineModel(**self.default_model_args)
        self.assertIsInstance(model, torch.nn.Module, "Model should be a PyTorch Module")

    def test_parallel_model_initialization(self):
        """Test initialization of the ParallelMPNNModel."""
        model = ParallelMPNNModel(**self.default_model_args)
        self.assertIsInstance(model, torch.nn.Module, "Parallel model should be a PyTorch Module")

    def test_save_results(self):
        """Test saving and loading performance results."""
        results = {"train_accuracies": [0.85], "valid_accuracies": [0.88], "test_accuracies": [0.90]}
        save_results(results, self.performance_path)

        # Check that the file was created
        self.assertTrue(os.path.exists(self.performance_path), "Performance file should be created")

        # Verify JSON content
        with open(self.performance_path, "r") as f:
            loaded_results = json.load(f)
        self.assertEqual(results, loaded_results, "Saved and loaded results should match")

    def test_load_model(self):
        """Test loading a model checkpoint."""
        model = BaselineModel(**self.default_model_args)
        torch.save(model.state_dict(), self.checkpoint_path)

        new_model = BaselineModel(**self.default_model_args)
        load_model(new_model, self.checkpoint_path)  # Load weights

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2), "Model parameters should match after loading checkpoint")

    @patch("torch.save")
    def test_load_model_no_checkpoint(self, mock_save):
        """Test load_model when checkpoint does not exist."""
        model = BaselineModel(**self.default_model_args)
        with self.assertLogs(level="WARNING") as log:
            load_model(model, "non_existent_checkpoint.pth")
            self.assertIn("No checkpoint found", log.output[0], "Warning should be logged if checkpoint is missing")


if __name__ == "__main__":
    unittest.main()
