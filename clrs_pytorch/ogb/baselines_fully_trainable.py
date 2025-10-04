import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from clrs_pytorch._src.processors import MPNN


# ---------------------- Utility Functions ---------------------- #

def print_weight_norms(model, prefix=""):
    """
    Prints the weight norms of a model's parameters.

    Args:
        model (nn.Module): The PyTorch model.
        prefix (str, optional): Prefix for logging. Defaults to "".
    """
    for name, param in model.named_parameters():
        logging.info(f"{prefix}{name}: {torch.norm(param).item()}")


def rename_keys(state_dict, old_prefix, new_prefix=""):
    """
    Renames keys in a state_dict to match the current model's expected keys.

    Args:
        state_dict (dict): The saved state dictionary.
        old_prefix (str): Prefix to remove from the keys.
        new_prefix (str, optional): Prefix to add to the keys. Defaults to "".

    Returns:
        dict: Updated state dictionary with renamed keys.
    """
    return {
        (new_prefix + k[len(old_prefix) + 1:] if k.startswith(old_prefix) else k): v
        for k, v in state_dict.items()
    }


def restore_model(model, pretrained_weights_path):
    """
    Restores the model from a checkpoint.

    Args:
        model (nn.Module): The model to restore.
        pretrained_weights_path (str): Path to the checkpoint file.

    Raises:
        FileNotFoundError: If the checkpoint file is missing.
    """
    if not os.path.exists(pretrained_weights_path):
        raise FileNotFoundError(f"Checkpoint file not found: {pretrained_weights_path}")

    checkpoint = torch.load(pretrained_weights_path, weights_only=True)
    updated_state_dict = rename_keys(checkpoint['model_state_dict'], old_prefix="net_fn.processor")

    model.load_state_dict(updated_state_dict, strict=False)
    logging.info(f"Model restored from {pretrained_weights_path}")

    print_weight_norms(model, prefix="Restored Param - ")

# ---------------------- Baseline Model ---------------------- #

class BaselineFullyTrainableModel(nn.Module):
    """
    Baseline Graph Neural Network model that is fully trainable.

    Args:
        out_dim (int): Output dimension.
        hidden_dim (int): Hidden dimension size.
        num_layers (int): Number of MPNN layers.
        reduction (function): Aggregation function (e.g., torch.max).
        use_pretrain_weights (bool): Whether to load pre-trained weights.
        pretrained_weights_path (str): Path to pre-trained weights.
        use_triplets (bool): Whether to use triplet features.
        gated (bool): Whether to use gated message passing.
        nb_triplet_fts (int, optional): Number of triplet features. Defaults to 8.
    """

    def __init__(self, out_dim, hidden_dim, num_layers, reduction, use_pretrain_weights, pretrained_weights_path,
                 use_triplets, gated, nb_triplet_fts=8):
        super(BaselineFullyTrainableModel, self).__init__()

        self.num_layers = num_layers
        self.use_triplets = use_triplets
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            mpnn_layer = MPNN(
                out_size=hidden_dim,
                mid_size=hidden_dim,
                reduction=reduction,
                activation=F.relu,
                use_ln=True,
                msgs_mlp_sizes=[hidden_dim, hidden_dim],
                use_triplets=use_triplets,
                nb_triplet_fts=nb_triplet_fts,
                gated=gated
            )

            if use_pretrain_weights: 
                logging.info(f"Freezing pretrained weights for layer {i}")
                print_weight_norms(mpnn_layer, prefix=f"Layer {i} - BEFORE LOADING: ")
                
                restore_model(mpnn_layer, pretrained_weights_path)
                
                print_weight_norms(mpnn_layer, prefix=f"Layer {i} - AFTER LOADING: ")
            else:
                logging.info(f"Layer {i}: {'Using random initialization' if not use_pretrain_weights else 'No pretrained weights applied'}")

            self.layers.append(mpnn_layer)

        self.graph_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Input encoders
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

        if use_triplets:
            self.edge_reducers = nn.ModuleList([
                nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(num_layers)
            ])

    def _graph_embedding(self, batch):
        """
        Returns graph-level embeddings (before final prediction MLP).
        """
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        device = x.device

        # Encode node/edge features
        node_fts = self.atom_encoder(x).to(device)
        edge_fts = self.bond_encoder(edge_attr).to(device)

        # Dense node batch
        node_fts_dense, node_mask = to_dense_batch(node_fts, batch=batch_idx)  # (B, N, F)

        # Dense adjacency & dense edge features (use PyG util; removes your manual loop/ones)
        # Shapes: adj: (B, N, N), edge_fts_dense: (B, N, N, F_e)
        adj_mat = torch.ones((num_graphs, max_nodes, max_nodes), device=device)
        edge_fts_dense = to_dense_adj(edge_index, batch=batch_idx, edge_attr=edge_fts).to(device)

        # Graph-level features placeholder (match your processor sig)
        edge_feature_dim = edge_fts.size(-1)
        num_graphs, max_nodes = node_fts_dense.size(0), node_fts_dense.size(1)
        graph_fts = torch.zeros((num_graphs, edge_feature_dim), device=device)

        hidden = torch.zeros_like(node_fts_dense, device=device)
        triplet_msgs = None

        for i, layer in enumerate(self.layers):
            hidden, triplet_msgs = layer(
                node_fts=node_fts_dense,
                edge_fts=edge_fts_dense,
                graph_fts=graph_fts,
                adj_mat=adj_mat,
                hidden=hidden,
                triplet_msgs=triplet_msgs
            )

        # Aggregate node features into graph-level embeddings
        graph_emb = hidden.mean(dim=1)  # (B, F)
        return graph_emb

    def forward(self, batch):
        graph_emb = self._graph_embedding(batch)
        return self.graph_pred(graph_emb)

    @torch.no_grad()
    def get_graph_embeddings(self, batch):
        self.eval()
        return self._graph_embedding(batch)

       