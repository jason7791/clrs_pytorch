import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj  # For dense conversion

from processors_node import MPNNEfficient
from clrs_pytorch.ogb.baselines import print_weight_norms, rename_keys, restore_model

# ---------------------- Parallel Node Model ---------------------- #

class ParallelNodeModel(nn.Module):
    """
    Parallel Message Passing Neural Network for node classification.
    
    This model implements two parallel streams of MPNN layers (a randomly initialized stream
    and a pretrained stream) that operate on node features. Their outputs are concatenated 
    and reduced to form the final hidden representation for each node, which is then used for 
    node-level classification.
    
    Args:
        out_dim (int): Number of output classes.
        hidden_dim (int): Hidden dimension size.
        num_layers (int): Number of MPNN layers.
        use_pretrain_weights (bool): If True, load pretrained weights for the pretrained stream.
        pretrained_weights_path (str): Path to the pretrained weights file.
        use_triplets (bool): Whether to use triplet message passing.
        gated (bool): Whether to use gated message passing.
        nb_triplet_fts (int, optional): Number of triplet features. Defaults to 8.
    """
    def __init__(self, out_dim, hidden_dim, num_layers, 
                 use_pretrain_weights, pretrained_weights_path,
                 use_triplets, gated, nb_triplet_fts=8):
        super(ParallelNodeModel, self).__init__()
        self.num_layers = num_layers
        self.use_pretrain_weights = use_pretrain_weights
        self.use_triplets = use_triplets
        self.hidden_dim = hidden_dim
        
        # Create parallel streams for each layer.
        self.layers = nn.ModuleList()
        self.reduction_layers = nn.ModuleList()
        if self.use_triplets:
            self.triplet_reduction_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Create two MPNN streams: one randomly initialized, one pretrained.
            random_mpnn = MPNNEfficient(
                out_size=hidden_dim,
                mid_size=hidden_dim,
                reduction=torch.max,
                activation=F.relu,
                use_ln=True,
                msgs_mlp_sizes=[hidden_dim, hidden_dim],
                use_triplets=use_triplets,
                nb_triplet_fts=nb_triplet_fts,
                gated=gated,
            )
            pretrained_mpnn = MPNNEfficient(
                out_size=hidden_dim,
                mid_size=hidden_dim,
                reduction=torch.max,
                activation=F.relu,
                use_ln=True,
                msgs_mlp_sizes=[hidden_dim, hidden_dim],
                use_triplets=use_triplets,
                nb_triplet_fts=nb_triplet_fts,
                gated=gated,
            )
            if use_pretrain_weights:
                logging.info(f"Initializing pretrained MPNN for layer {i}")
                restore_model(pretrained_mpnn, pretrained_weights_path)
                for param in pretrained_mpnn.parameters():
                    param.requires_grad = False
            
            self.layers.append(nn.ModuleDict({
                'random': random_mpnn,
                'pretrained': pretrained_mpnn
            }))
            # Reduction layer: maps concatenated output (2*hidden_dim) to hidden_dim.
            self.reduction_layers.append(nn.Linear(2 * hidden_dim, hidden_dim))
            if self.use_triplets:
                self.triplet_reduction_layers.append(nn.Linear(2 * hidden_dim, hidden_dim))
        
        # Node projector: projects raw node features to hidden_dim.
        self.node_projector = nn.LazyLinear(hidden_dim)
        
        # Prediction head for node classification (applied to each node separately).
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
            Tensor: Log softmax predictions for node classification of shape [num_nodes, out_dim].
        """
        device = x.device
        
        # Project raw node features.
        node_fts = self.node_projector(x)  # [N, hidden_dim]
        # Add a batch dimension (assume single graph).
        node_fts = node_fts.unsqueeze(0)  # [1, N, hidden_dim]

        hidden = torch.zeros_like(node_fts, device=device)
        triplet_msgs = None
        
        # Process each layer.
        for i, layer_dict in enumerate(self.layers):
            random_mpnn = layer_dict['random']
            pretrained_mpnn = layer_dict['pretrained']

            # Process the two streams separately with triplet messages.
            random_out, random_triplet = random_mpnn(
                node_fts=node_fts,
                hidden=hidden,
                triplet_msgs=triplet_msgs
            )
            pretrained_out, pretrained_triplet = pretrained_mpnn(
                node_fts=node_fts,
                hidden=hidden,
                triplet_msgs=triplet_msgs
            )
            # Concatenate the main outputs from both streams.
            concatenated = torch.cat([random_out, pretrained_out], dim=-1)  # [1, N, 2*hidden_dim]
            hidden = self.reduction_layers[i](concatenated)  # [1, N, hidden_dim]
            
            # Similarly, concatenate the triplet messages.
            if(self.use_triplets):
                concatenated_triplet = torch.cat([random_triplet, pretrained_triplet], dim=-1)  # [1, N, N, 2*hidden_dim]
                triplet_msgs = self.triplet_reduction_layers[i](concatenated_triplet)  # [1, N, N, hidden_dim]

        
        # Remove the batch dimension.
        hidden = hidden.squeeze(0)  # [N, hidden_dim]
        # Compute per-node predictions.
        out = self.prediction_head(hidden)  # [N, out_dim]
        return F.log_softmax(out, dim=-1)
