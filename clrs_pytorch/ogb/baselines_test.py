import os
import unittest
import torch
import tempfile
from unittest.mock import patch
from torch_geometric.data import Data

from baselines import (
    BaselineModel, restore_model, rename_keys, print_weight_norms
)
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class TestBaselines(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        self.device = torch.device("cpu")  # Use CPU for tests

        # Create a dummy AtomEncoder to get the expected number of atom features.
        dummy_atom_encoder = AtomEncoder(emb_dim=16)
        num_atom_features = len(dummy_atom_encoder.atom_embedding_list)

        # Create a dummy BondEncoder to determine valid bond feature dimensions.
        dummy_bond_encoder = BondEncoder(emb_dim=16)
        bond_feature_dim = len(dummy_bond_encoder.bond_embedding_list)
        # For each bond feature column, generate valid indices based on that embedding's vocab size.
        edge_attr_list = [
            torch.randint(0, dummy_bond_encoder.bond_embedding_list[i].weight.shape[0], (3,), dtype=torch.long)
            for i in range(bond_feature_dim)
        ]
        # Stack to create a (num_edges x bond_feature_dim) tensor.
        edge_attr = torch.stack(edge_attr_list, dim=1)

        # Create a dummy batch with valid node and bond indices.
        self.dummy_batch = Data(
            x=torch.zeros((5, num_atom_features), dtype=torch.long),  # Valid indices for AtomEncoder.
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),           # Example edge indices.
            edge_attr=edge_attr,                                         # Valid bond indices.
            batch=torch.zeros(5, dtype=torch.long)                      # All nodes belong to graph 0                   
        )
        

        # Create a temporary directory for saving checkpoints.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(self.temp_dir.name, "checkpoint.pth")

        # Default arguments for BaselineModel initialization.
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

    def test_baseline_model_initialization(self):
        """Test if the BaselineModel initializes correctly."""
        model = BaselineModel(**self.default_model_args)
        self.assertIsInstance(model, torch.nn.Module, "Model should be a PyTorch Module")

    def test_baseline_model_forward(self):
        """Test the forward pass of BaselineModel."""
        model = BaselineModel(**self.default_model_args).to(self.device)
        output = model(self.dummy_batch)
        # Since the model produces graph-level predictions, we expect a tensor of shape (B, out_dim)
        self.assertEqual(output.shape[0], 1, "Output should have batch size 1")
        self.assertEqual(output.shape[1], self.default_model_args["out_dim"], "Output should match out_dim")

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

    @patch("torch.save")
    def test_restore_model_no_checkpoint(self, mock_save):
        """Test restore_model when checkpoint does not exist."""
        model = BaselineModel(**self.default_model_args)
        with self.assertRaises(FileNotFoundError):
            restore_model(model, "non_existent_checkpoint.pth")

    def test_restore_model(self):
        """Test restoring a model from a checkpoint."""
        model = BaselineModel(**self.default_model_args)
        torch.save({"model_state_dict": model.state_dict()}, self.checkpoint_path)
        new_model = BaselineModel(**self.default_model_args)
        restore_model(new_model, self.checkpoint_path)
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2), "Model parameters should match after loading checkpoint")

    @patch("baselines.logging.info")
    def test_print_weight_norms(self, mock_log):
        """Test printing of weight norms."""
        model = BaselineModel(**self.default_model_args)
        print_weight_norms(model, prefix="Test: ")
        self.assertTrue(mock_log.called, "Logging should be called when printing weight norms")


if __name__ == "__main__":
    unittest.main()
