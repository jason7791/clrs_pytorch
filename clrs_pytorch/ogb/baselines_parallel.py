import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from clrs_pytorch._src.processors import MPNN
from clrs_pytorch.ogb.baselines import print_weight_norms, rename_keys, restore_model

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ParallelMPNNModel(nn.Module):
    """
    Parallel Message Passing Neural Network (MPNN) that leverages two streams:
    a randomly initialized MPNN and a pretrained MPNN. Their outputs are concatenated
    and reduced to form the final hidden representation.

    Args:
        out_dim (int): The output dimension.
        hidden_dim (int): The hidden layer dimension.
        num_layers (int): The number of MPNN layers.
        reduction (callable): Reduction function for message aggregation (e.g., torch.max).
        use_pretrain_weights (bool): If True, load pretrained weights for one MPNN stream.
        pretrained_weights_path (str): Path to the pretrained weights file.
        use_triplets (bool): Whether to include triplet message passing.
        gated (bool): Whether to use gated message passing.
        nb_triplet_fts (int, optional): Number of triplet features. Defaults to 8.
    """

    def __init__(
        self,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        reduction,
        use_pretrain_weights: bool,
        pretrained_weights_path: str,
        use_triplets: bool,
        gated: bool,
        nb_triplet_fts: int = 8,
    ):
        super(ParallelMPNNModel, self).__init__()
        self.num_layers = num_layers
        self.use_pretrain_weights = use_pretrain_weights
        self.use_triplets = use_triplets
        self.layers = nn.ModuleList()
        self.reduction_layer = nn.ModuleList()

        for i in range(num_layers):
            # Initialize two MPNN streams: random and pretrained.
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
                logging.info(f"Initializing pretrained MPNN for layer {i}")
                restore_model(pretrained_mpnn, pretrained_weights_path)
                for param in pretrained_mpnn.parameters():
                    param.requires_grad = False

            self.layers.append(nn.ModuleDict({
                'random': random_mpnn,
                'pretrained': pretrained_mpnn
            }))

            # Linear layer to reduce concatenated output (2 * hidden_dim) to hidden_dim.
            self.reduction_layer.append(nn.Linear(2 * hidden_dim, hidden_dim))

        self.graph_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Input encoders for nodes and bonds.
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

        if use_triplets:
            self.edge_reducers = nn.ModuleList([
                nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(num_layers)
            ])

    def forward(self, batch) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            batch (Data): A graph batch containing:
                - x: Node features.
                - edge_index: Edge indices.
                - edge_attr: Edge attributes.
                - batch: Batch vector mapping nodes to graphs.

        Returns:
            torch.Tensor: Graph-level predictions of shape (B, out_dim).
        """
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        device = x.device

        # Encode node and edge features.
        node_fts = self.atom_encoder(x).to(device)
        edge_fts = self.bond_encoder(edge_attr).to(device)
        node_fts_dense, _ = to_dense_batch(node_fts, batch=batch_idx)  # Shape: (B, N, F)

        num_graphs = batch_idx.max().item() + 1
        max_nodes = node_fts_dense.size(1)
        edge_feature_dim = edge_fts.size(-1)

        # Build dense adjacency and edge feature tensors.
        adj_mat = torch.ones((num_graphs, max_nodes, max_nodes), device=device)  # (B, N, N)
        edge_fts_dense = torch.zeros((num_graphs, max_nodes, max_nodes, edge_feature_dim), device=device)  # (B, N, N, F)

        for i in range(num_graphs):
            mask = batch_idx[edge_index[0]] == i
            local_edge_index = edge_index[:, mask]
            local_edge_fts = edge_fts[mask]
            # Normalize local edge indices to start at 0.
            local_edge_index = local_edge_index - local_edge_index.min()
            edge_fts_dense[i, local_edge_index[0], local_edge_index[1]] = local_edge_fts

        # Initialize graph features and hidden representations.
        graph_fts = torch.zeros((num_graphs, edge_feature_dim), device=device)  # (B, F)
        hidden = torch.zeros_like(node_fts_dense, device=device)  # (B, N, F)
        current_edge_fts = edge_fts_dense

        triplet_msgs = None

        # Process each layer.
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

            # Concatenate outputs from both streams and reduce dimensionality.
            concatenated_output = torch.cat([random_output, pretrained_output], dim=-1)  # (B, N, 2F)
            hidden = self.reduction_layer[i](concatenated_output)  # (B, N, F)

        # Aggregate node representations via mean pooling to form graph embeddings.
        graph_emb = hidden.mean(dim=1)  # (B, F)

        # Return final graph-level prediction.
        return self.graph_pred(graph_emb.to(device))
