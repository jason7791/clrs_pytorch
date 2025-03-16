import os
import json
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_sparse import SparseTensor  # Import SparseTensor
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import NeighborLoader

# Import your custom models:
from baselines_node import BaselineNodeModel
from baselines_parallel_node import ParallelNodeModel

# ---------------------- Utility Functions ---------------------- #

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_epoch(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.adj_t)
        # Use the first batch.batch_size nodes as the seed nodes.
        seed_size = batch.batch_size  
        loss = criterion(out[:seed_size], batch.y[:seed_size].squeeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seed_size
        total_examples += seed_size
    return total_loss / total_examples

@torch.no_grad()
def evaluate_epoch(model, device, loader, evaluator, metric_key):
    model.eval()
    y_true_list = []
    y_pred_list = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.adj_t)
        pred = out.argmax(dim=-1, keepdim=True)
        seed_size = batch.batch_size  # first batch.batch_size nodes are seed nodes.
        y_true_list.append(batch.y[:seed_size])
        y_pred_list.append(pred[:seed_size])
    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})[metric_key]

def save_results(results, path):
    """Save training results as a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {path}")

def load_model(model, checkpoint_path):
    """Load model parameters from the checkpoint file."""
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint)
    else:
        logging.warning(f"No checkpoint found at {checkpoint_path}, starting from scratch.")

# ---------------------- Main Training Pipeline ---------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Node Classification on OGBN-Arxiv with Mini-Batch Training"
    )
    parser.add_argument("--dataset", type=str, choices=["ogbn-arxiv", "ogbn-proteins"],
                        default="ogbn-arxiv", help="Dataset name")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of GNN layers")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model.pth",
                        help="Path to save the best model")
    parser.add_argument("--performance_path", type=str, default="performance.json",
                        help="Path to save performance metrics")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, choices=["serial", "parallel"], default="serial",
                        help="Model type: serial or parallel")
    parser.add_argument("--gated", action="store_true", help="Use gated message passing")
    parser.add_argument("--use_triplets", action="store_true", help="Use triplet reasoning")
    parser.add_argument("--use_pretrain_weights", action="store_true",
                        help="Use pre-trained weights")
    parser.add_argument("--pretrained_weights_path", type=str, default="checkpoints/pretrained.pth",
                        help="Path to pre-trained weights")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the last checkpoint and performance results")
    # New flag for neighbor sampling: comma-separated list, e.g., "32,32"
    parser.add_argument("--num_neighbors", type=str, default="32,32",
                        help="Comma-separated list of neighbor counts for each layer")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    # Parse num_neighbors flag into a list of integers.
    num_neighbors = [int(x.strip()) for x in args.num_neighbors.split(",")]

    # Load dataset with a sparse tensor transform.
    # Do not move the entire graph to GPU, as mini-batches will be sampled.
    dataset = PygNodePropPredDataset(name=args.dataset, transform=T.ToSparseTensor())
    data = dataset[0]
    # Ensure the adjacency matrix is symmetric.
    # if hasattr(data.adj_t, 'to_symmetric'):
    #     data.adj_t = data.adj_t.to_symmetric()
    # else:
    #     if data.adj_t.layout == torch.sparse_csc:
    #         data.adj_t = data.adj_t.to_sparse_csr()
    #     data.adj_t = data.adj_t + data.adj_t.transpose(0, 1).to_sparse_csr()
    if not isinstance(data.adj_t, SparseTensor):
        # If it's a dense tensor, convert it:
        data.adj_t = SparseTensor.from_dense(data.adj_t)
    # Now, make the adjacency symmetric.
    data.adj_t = data.adj_t.to_symmetric()

    split_idx = dataset.get_idx_split()
    out_dim = dataset.num_tasks if hasattr(dataset, "num_tasks") else dataset.num_classes

    # Create NeighborLoaders with the specified neighbor counts.
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=split_idx['train'],
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=split_idx['valid'],
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=split_idx['test'],
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Initialize the model.
    ModelClass = BaselineNodeModel if args.model == "serial" else ParallelNodeModel
    model = ModelClass(
        out_dim=out_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_pretrain_weights=args.use_pretrain_weights,
        pretrained_weights_path=args.pretrained_weights_path,
        gated=args.gated,
        use_triplets=args.use_triplets
    ).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    evaluator = Evaluator(name=args.dataset)
    metric_key = "acc" if args.dataset == "ogbn-arxiv" else "ap"

    results = {"train_accuracies": [], "valid_accuracies": [], "test_accuracies": []}
    start_epoch = 0
    best_valid_acc = 0.0
    early_stop_counter = 0

    # Resume logic.
    if args.resume:
        if os.path.exists(args.performance_path):
            try:
                with open(args.performance_path, 'r') as f:
                    results = json.load(f)
                start_epoch = len(results["train_accuracies"])
                if results["valid_accuracies"]:
                    best_valid_acc = max(results["valid_accuracies"])
                    best_epoch = results["valid_accuracies"].index(best_valid_acc) + 1
                    early_stop_counter = start_epoch - best_epoch
                logging.info(f"Resuming training from epoch {start_epoch+1} with best valid acc: {best_valid_acc:.4f}")
                load_model(model, args.checkpoint_path)
            except Exception as e:
                logging.error(f"Failed to load previous results: {str(e)}")
        else:
            logging.info("No previous performance results found. Starting fresh training.")

    train_acc_list = results["train_accuracies"]
    valid_acc_list = results["valid_accuracies"]
    test_acc_list = results["test_accuracies"]

    for epoch in range(start_epoch + 1, args.epochs + 1):
        loss_val = train_epoch(model, device, train_loader, optimizer, criterion)
        train_acc = evaluate_epoch(model, device, train_loader, evaluator, metric_key)
        valid_acc = evaluate_epoch(model, device, valid_loader, evaluator, metric_key)
        logging.info(f"Epoch {epoch}/{args.epochs}, Loss: {loss_val:.4f}, "
                     f"Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}")
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), args.checkpoint_path)
            logging.info(f"New best model saved at epoch {epoch} with Valid Acc: {best_valid_acc:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        results = {
            "train_accuracies": train_acc_list,
            "valid_accuracies": valid_acc_list,
            "test_accuracies": test_acc_list
        }
        save_results(results, args.performance_path)

        if early_stop_counter >= args.early_stop_patience:
            logging.info(f"Early stopping triggered at epoch {epoch}.")
            break

    load_model(model, args.checkpoint_path)
    test_acc = evaluate_epoch(model, device, test_loader, evaluator, metric_key)
    logging.info(f"Final Test Accuracy: {test_acc:.4f}")
    test_acc_list.append(test_acc)

    results = {
        "train_accuracies": train_acc_list,
        "valid_accuracies": valid_acc_list,
        "test_accuracies": test_acc_list
    }
    save_results(results, args.performance_path)

if __name__ == "__main__":
    main()
