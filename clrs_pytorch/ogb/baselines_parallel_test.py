import os
import unittest
import torch
import tempfile
from unittest.mock import patch
from torch_geometric.data import Data

from baselines_parallel import ParallelMPNNModel, print_weight_norms, rename_keys, restore_model
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class TestParallelMPNNModel(unittest.TestCase):

    def setUp(self):
        """Set up test environment and create dummy data for tests."""
        self.device = torch.device("cpu")
        
        # Create a dummy AtomEncoder to determine valid node feature size.
        dummy_atom_encoder = AtomEncoder(emb_dim=16)
        num_atom_features = len(dummy_atom_encoder.atom_embedding_list)
        
        # Create a dummy BondEncoder to determine valid bond feature dimensions.
        dummy_bond_encoder = BondEncoder(emb_dim=16)
        bond_feature_dim = len(dummy_bond_encoder.bond_embedding_list)
        
        # For each bond feature column, generate a tensor of valid random indices.
        edge_attr_list = [
            torch.randint(
                0, dummy_bond_encoder.bond_embedding_list[i].weight.shape[0],
                (3,), dtype=torch.long
            )
            for i in range(bond_feature_dim)
        ]
        # Stack the list to create a (num_edges x bond_feature_dim) tensor.
        edge_attr = torch.stack(edge_attr_list, dim=1)
        
        # Create dummy data: single graph (all nodes belong to graph 0).
        self.dummy_batch = Data(
            x=torch.zeros((5, num_atom_features), dtype=torch.long),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
            edge_attr=edge_attr,
            batch=torch.zeros(5, dtype=torch.long)
        )
        
        # Create a temporary directory for checkpoint tests.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(self.temp_dir.name, "checkpoint.pth")
        
        # Default arguments for ParallelMPNNModel initialization.
        self.default_model_args = {
            "out_dim": 1,
            "hidden_dim": 16,
            "num_layers": 2,
            "reduction": torch.max,
            "use_pretrain_weights": False,
            "pretrained_weights_path": "",
            "use_triplets": False,
            "gated": False,
            "nb_triplet_fts": 8,
        }

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_model_initialization(self):
        """Test that the ParallelMPNNModel initializes correctly."""
        model = ParallelMPNNModel(**self.default_model_args)
        self.assertIsInstance(model, torch.nn.Module, "Model should be a PyTorch Module")

    def test_forward_pass(self):
        """Test the forward pass of ParallelMPNNModel."""
        model = ParallelMPNNModel(**self.default_model_args).to(self.device)
        output = model(self.dummy_batch)
        # Since our dummy batch represents one graph, expect output shape (1, out_dim)
        self.assertEqual(output.shape[0], 1, "Output should have batch size 1")
        self.assertEqual(output.shape[1], self.default_model_args["out_dim"],
                         "Output's second dimension should match out_dim")


if __name__ == "__main__":
    unittest.main()
