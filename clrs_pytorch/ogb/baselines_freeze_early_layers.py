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
    for name, param in model.named_parameters():
        logging.info(f"{prefix}{name}: {param.norm().item():.6f}")


def rename_keys(state_dict, old_prefix, new_prefix=""):
    op = old_prefix + "."
    out = {}
    for k, v in state_dict.items():
        if k.startswith(op):
            out[new_prefix + k[len(op):]] = v
        else:
            out[k] = v
    return out


def restore_model(module: nn.Module, pretrained_weights_path: str, verbose: bool = False):
    if not os.path.exists(pretrained_weights_path):
        raise FileNotFoundError(f"Checkpoint file not found: {pretrained_weights_path}")

    checkpoint = torch.load(pretrained_weights_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    state = rename_keys(state, old_prefix="net_fn.processor", new_prefix="")

    missing, unexpected = module.load_state_dict(state, strict=False)
    logging.info(f"Restored from {pretrained_weights_path} "
                 f"(missing={len(missing)}, unexpected={len(unexpected)})")
    if verbose:
        if missing: logging.info(f"Missing keys: {missing}")
        if unexpected: logging.info(f"Unexpected keys: {unexpected}")
        print_weight_norms(module, prefix="Restored Param - ")


# ---------------------- Baseline Model ---------------------- #

class BaselineModel(nn.Module):
    """
    Baseline Graph Neural Network model.

    Args:
        out_dim (int): Output dimension.
        hidden_dim (int): Hidden dimension size.
        num_layers (int): Number of MPNN layers.
        reduction (Callable): Aggregation function (e.g., torch.max).
        use_pretrain_weights (bool): Whether to load pre-trained weights.
        pretrained_weights_path (str): Path to pre-trained weights.
        use_triplets (bool): Whether to use triplet features.
        gated (bool): Whether to use gated message passing.
        nb_triplet_fts (int, optional): Number of triplet features. Defaults to 8.
        freeze_every_other (bool): If True, freeze every other layer when using pretrain (layers 1,3,5,...).
        verbose_restore (bool): Print norms/missing keys during restore.
    """

    def __init__(
        self,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        reduction,
        use_pretrain_weights: bool,
        pretrained_weights_path: str | None,
        use_triplets: bool,
        gated: bool,
        nb_triplet_fts: int = 8,
        freeze_every_other: bool = True,
        verbose_restore: bool = False,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.use_triplets = use_triplets
        self.layers = nn.ModuleList()
        self.verbose_restore = verbose_restore

        # Input encoders
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

        for i in range(num_layers):
            layer = MPNN(
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
            self.layers.append(layer)

            # Load + (optionally) freeze alternating layers
            if use_pretrain_weights and pretrained_weights_path:
                if freeze_every_other and (i % 2 == 1):  # freeze layers 1,3,5,...
                    logging.info(f"Loading + freezing pretrained weights for layer {i}")
                    restore_model(layer, pretrained_weights_path, verbose=self.verbose_restore)
                    for p in layer.parameters():
                        p.requires_grad = False
                else:
                    logging.info(f"Layer {i}: randomly init (no frozen pretrained weights)")
            else:
                logging.info(f"Layer {i}: randomly init (no pretrained path provided)")

        self.graph_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch):
        """
        Args:
            batch (torch_geometric.data.Batch): expects batch.x, batch.edge_index, batch.edge_attr, batch.batch
        Returns:
            torch.Tensor: (B, out_dim)
        """
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        device = x.device

        # Encode node/edge features (already on correct device via input)
        node_fts = self.atom_encoder(x)                # (N, hidden_dim)
        edge_fts = self.bond_encoder(edge_attr)        # (E, hidden_dim)

        # Dense nodes with mask
        node_fts_dense, mask = to_dense_batch(node_fts, batch=batch_idx)  # (B, Vmax, H), (B, Vmax)

        # Dense adjacency and edge features in one pass (no Python loops)
        # adj_mat: (B, Vmax, Vmax), edge_fts_dense: (B, Vmax, Vmax, H)
        adj_mat = to_dense_adj(edge_index, batch=batch_idx).to(device)
        edge_fts_dense = to_dense_adj(edge_index, batch=batch_idx, edge_attr=edge_fts).to(device)

        num_graphs = node_fts_dense.size(0)
        hidden = torch.zeros_like(node_fts_dense)      # (B, Vmax, H)
        graph_fts = torch.zeros((num_graphs, node_fts_dense.size(-1)), device=device)  # (B, H)
        triplet_msgs = None

        # Message passing
        for i, layer in enumerate(self.layers):
            hidden, triplet_msgs = layer(
                node_fts=node_fts_dense,
                edge_fts=edge_fts_dense,
                graph_fts=graph_fts,
                adj_mat=adj_mat,
                hidden=hidden,
                triplet_msgs=triplet_msgs,
                node_mask=mask,  # if your MPNN supports masks; otherwise remove
            )

        # Graph pooling: mask-aware mean
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)  # (B,1)
        graph_emb = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B,H)

        return self.graph_pred(graph_emb)
