import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from clrs_pytorch._src.processors import MPNN


# Function to compute weight norms
def print_weight_norms(model, prefix=""):
    for name, param in model.named_parameters():
        print(f"{prefix}{name}: {torch.norm(param).item()}")


def rename_keys(state_dict, old_prefix, new_prefix=""):
    """
    Rename keys in a state_dict to match the current model's expected keys.
    Args:
        state_dict (dict): The saved state dictionary.
        old_prefix (str): The prefix to remove from the keys.
        new_prefix (str): The prefix to add to the keys (default: "").
    Returns:
        dict: Updated state dictionary with renamed keys.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(old_prefix):
            new_key = new_prefix + k[len(old_prefix) + 1:]  # Remove old prefix
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def restore_model(model, pretrained_weights_path):
    """Restore model from `file_name`."""
    if not os.path.exists(pretrained_weights_path):
        raise FileNotFoundError(f"Checkpoint file not found: {pretrained_weights_path}")

    checkpoint = torch.load(pretrained_weights_path, weights_only=True)
    updated_state_dict = rename_keys(checkpoint['model_state_dict'], old_prefix="net_fn.processor")
    model.load_state_dict(updated_state_dict, strict=False)
    
class ParallelMPNNModel(nn.Module):
    def __init__(self, out_dim, hidden_dim, num_layers, reduction, use_pretrain_weights, pretrained_weights_path,
                  use_triplets,  gated, nb_triplet_fts=8 ):
        super(ParallelMPNNModel, self).__init__()
        self.num_layers = num_layers
        self.use_pretrain_weights = use_pretrain_weights
        self.use_triplets = use_triplets
        self.layers = nn.ModuleList()
        self.reduction_layer = nn.ModuleList()

        for i in range(num_layers):
            # Initialize two MPNNs
            random_mpnn = MPNN(
                out_size=hidden_dim,
                mid_size=hidden_dim,
                reduction=reduction,
                activation=F.relu,
                use_ln=True,
                msgs_mlp_sizes=[hidden_dim, hidden_dim],
                use_triplets=use_triplets,
                nb_triplet_fts=nb_triplet_fts,
                gated=gated,
            )

            pretrained_mpnn = MPNN(
                out_size=hidden_dim,
                mid_size=hidden_dim,
                reduction=reduction,
                activation=F.relu,
                use_ln=True,
                msgs_mlp_sizes=[hidden_dim, hidden_dim],
                use_triplets=use_triplets,
                nb_triplet_fts=nb_triplet_fts,
                gated=gated,
            )

            if use_pretrain_weights:
                print(f"Initializing pretrained MPNN for layer {i}")
                restore_model(pretrained_mpnn, pretrained_weights_path)
                for param in pretrained_mpnn.parameters():
                    param.requires_grad = False

            self.layers.append(nn.ModuleDict({
                'random': random_mpnn,
                'pretrained': pretrained_mpnn
            }))

            # Add a linear layer to reduce concatenated output from 2F to F
            self.reduction_layer.append(nn.Linear(2 * hidden_dim, hidden_dim))

        self.graph_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Input encoders
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

        if(use_triplets):
            self.edge_reducers = nn.ModuleList([
            nn.Linear(3 * hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        device = x.device

        node_fts = self.atom_encoder(x).to(device)
        edge_fts = self.bond_encoder(edge_attr).to(device)
        node_fts_dense, mask = to_dense_batch(node_fts, batch=batch_idx) # B x N x F

        num_graphs = batch_idx.max().item() + 1
        max_nodes = node_fts_dense.size(1)
        edge_feature_dim = edge_fts.size(-1)

        adj_mat = torch.ones((num_graphs, max_nodes, max_nodes), device=device)  # B x N x N

        edge_fts_dense = torch.zeros((num_graphs, max_nodes, max_nodes, edge_feature_dim), device=device)  # B x N x N x F

        for i in range(num_graphs):
            mask = batch_idx[edge_index[0]] == i
            local_edge_index = edge_index[:, mask]
            local_edge_fts = edge_fts[mask]
            local_edge_index = local_edge_index - local_edge_index.min()
            edge_fts_dense[i, local_edge_index[0], local_edge_index[1]] = local_edge_fts

        graph_fts = torch.zeros((num_graphs, edge_feature_dim), device=device)  # B x F

        hidden = torch.zeros_like(node_fts_dense, device=device)  # B x N x F

        current_edge_fts = edge_fts_dense

        triplet_msgs = None

        for i, layer_dict in enumerate(self.layers):
            random_mpnn = layer_dict['random']
            pretrained_mpnn = layer_dict['pretrained']
            if self.use_triplets:
                random_output, triplet_msgs = random_mpnn(
                    node_fts=node_fts_dense,
                    edge_fts=current_edge_fts,
                    graph_fts=graph_fts,
                    adj_mat=adj_mat,
                    hidden=hidden,
                    triplet_msgs=triplet_msgs
                )
                pretrained_output, triplet_msgs = pretrained_mpnn(
                    node_fts=node_fts_dense,
                    edge_fts=current_edge_fts,
                    graph_fts=graph_fts,
                    adj_mat=adj_mat,
                    hidden=hidden,
                    triplet_msgs=triplet_msgs
                )
            else:
                random_output, _ = random_mpnn(
                    node_fts=node_fts_dense,
                    edge_fts=current_edge_fts,
                    graph_fts=graph_fts,
                    adj_mat=adj_mat,
                    hidden=hidden
                )
                pretrained_output, _ = pretrained_mpnn(
                    node_fts=node_fts_dense,
                    edge_fts=current_edge_fts,
                    graph_fts=graph_fts,
                    adj_mat=adj_mat,
                    hidden=hidden
                )
            concatenated_output = torch.cat([random_output, pretrained_output], dim=-1)  # B x N x 2F
            hidden = F.relu(self.reduction_layer[i](concatenated_output))  # B x N x F


        # Compute graph embeddings by mean pooling over nodes
        graph_emb = hidden.mean(dim=1)  # B x F

        # Final prediction
        res = self.graph_pred(graph_emb.to(device))  # B x out_dim

        return res
