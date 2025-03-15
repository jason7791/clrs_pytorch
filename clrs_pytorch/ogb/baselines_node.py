import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj  # For dense conversion

from clrs_pytorch._src.processors import MPNN

# ---------------------- Utility Functions ---------------------- #

def print_weight_norms(model, prefix=""):
    """
    Prints the weight norms of a model's parameters.
    """
    for name, param in model.named_parameters():
        logging.info(f"{prefix}{name}: {torch.norm(param).item()}")

def rename_keys(state_dict, old_prefix, new_prefix=""):
    """
    Renames keys in a state_dict to match the current model's expected keys.
    """
    return {
        (new_prefix + k[len(old_prefix) + 1:] if k.startswith(old_prefix) else k): v
        for k, v in state_dict.items()
    }

def restore_model(model, pretrained_weights_path):
    """
    Restores the model from a checkpoint.
    """
    if not os.path.exists(pretrained_weights_path):
        raise FileNotFoundError(f"Checkpoint file not found: {pretrained_weights_path}")
    checkpoint = torch.load(pretrained_weights_path, weights_only=True)
    updated_state_dict = rename_keys(checkpoint['model_state_dict'], old_prefix="net_fn.processor")
    model.load_state_dict(updated_state_dict, strict=False)
    logging.info(f"Model restored from {pretrained_weights_path}")

# ---------------------- Baseline Node Model ---------------------- #

class BaselineNodeModel(nn.Module):
    """
    Baseline model for node classification tasks using MPNN layers.
    
    This model adapts the standard MPNN layer (which expects dense inputs) to 
    work on node-level tasks. It assumes that the input is a single graph with:
      - x: Node features of shape [num_nodes, in_features]
      - adj_t: A sparse adjacency (e.g. produced by T.ToSparseTensor) which 
               will be converted to dense.
               
    Note: Converting a large graph to dense form can be very memory intensive.
    
    Args:
        out_dim (int): Number of output classes.
        hidden_dim (int): Hidden dimension size.
        num_layers (int): Number of MPNN layers.
        use_pretrain_weights (bool): Whether to load pretrained weights.
        pretrained_weights_path (str): Path to pretrained weights.
        use_triplets (bool): Whether to use triplet features.
        gated (bool): Whether to use gated message passing.
        nb_triplet_fts (int, optional): Number of triplet features. Defaults to 8.
    """
    
    def __init__(self, out_dim, hidden_dim, num_layers,
                 use_pretrain_weights=False, pretrained_weights_path="",
                 use_triplets=False, gated=False, nb_triplet_fts=8):
        super(BaselineNodeModel, self).__init__()
        self.num_layers = num_layers
        self.use_triplets = use_triplets
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        
        # Build MPNN layers.
        for i in range(num_layers):
            mpnn_layer = MPNN(
                out_size=hidden_dim,
                mid_size=hidden_dim,
                reduction=torch.max,
                activation=F.relu,
                use_ln=True,
                msgs_mlp_sizes=[hidden_dim, hidden_dim],
                use_triplets=use_triplets,
                nb_triplet_fts=nb_triplet_fts,
                gated=gated
            )
            # Optionally load and freeze pretrained weights on every even layer.
            if use_pretrain_weights and i % 2 == 1:
                logging.info(f"Freezing pretrained weights for layer {i}")
                print_weight_norms(mpnn_layer, prefix=f"Layer {i} - BEFORE LOADING: ")
                restore_model(mpnn_layer, pretrained_weights_path)
                for param in mpnn_layer.parameters():
                    param.requires_grad = False
                print_weight_norms(mpnn_layer, prefix=f"Layer {i} - AFTER LOADING: ")
            else:
                logging.info(f"Layer {i}: {'Using random initialization' if not use_pretrain_weights else 'No pretrained weights applied'}")
            self.layers.append(mpnn_layer)
        
        # Lazy creation of a node projector to map raw node features to hidden_dim.
        self.node_projector = nn.LazyLinear(hidden_dim)
        
        # Prediction head for node classification.
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    
    def forward(self, x, adj_t):
        """
        Node-level forward pass.
        
        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, in_features].
            adj_t (SparseTensor): Sparse adjacency matrix.
        
        Returns:
            Tensor: Log softmax outputs for node classification (shape [num_nodes, out_dim]).
        """
        device = x.device

        # Project raw node features.
        node_fts = self.node_projector(x)  # [num_nodes, hidden_dim]
        # Add a batch dimension: assume a single graph.
        node_fts = node_fts.unsqueeze(0)  # [1, N, hidden_dim]
        
        # Convert the sparse adjacency to dense.
        # Note: This conversion may be infeasible for large graphs.
        dense_adj = adj_t.to_dense()  # [N, N]
        dense_adj = dense_adj.unsqueeze(0)  # [1, N, N]
        
        # Create dummy edge features.
        # Here, we use the dense adjacency indicator and repeat it to have feature dimension = hidden_dim.
        edge_fts = dense_adj.unsqueeze(-1).repeat(1, 1, 1, self.hidden_dim)  # [1, N, N, hidden_dim]
        
        # Create dummy graph features for the single graph.
        graph_fts = torch.zeros((1, self.hidden_dim), device=device)  # [1, hidden_dim]
        hidden = torch.zeros_like(node_fts, device=device)
        triplet_msgs = None

        # Run message passing layers.
        for layer in self.layers:
            hidden, triplet_msgs = layer(
                node_fts=node_fts,
                edge_fts=edge_fts,
                graph_fts=graph_fts,
                adj_mat=dense_adj,
                hidden=hidden,
                triplet_msgs=triplet_msgs
            )
        
        # Remove the batch dimension.
        hidden = hidden.squeeze(0)  # [N, hidden_dim]
        out = self.prediction_head(hidden)  # [N, out_dim]
        return F.log_softmax(out, dim=-1)
