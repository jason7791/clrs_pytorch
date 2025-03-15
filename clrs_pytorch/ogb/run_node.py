import os
import json
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

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

def train(model, data, train_idx, optimizer, criterion):
    """Train the model for one epoch on the training nodes."""
    model.train()
    optimizer.zero_grad()
    # Forward pass: only compute predictions for training nodes.
    out = model(data.x, data.adj_t)[train_idx]
    loss = criterion(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, split_idx, evaluator, metric_key):
    """Evaluate the model on train, validation, and test splits."""
    model.eval()
    out = model(data.x, data.adj_t)
    # For node classification, predictions are given by argmax.
    y_pred = out.argmax(dim=-1, keepdim=True)
    
    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })[metric_key]
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })[metric_key]
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })[metric_key]
    
    return train_acc, valid_acc, test_acc

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
    parser = argparse.ArgumentParser(description="Node Classification on OGBN-Arxiv with Custom Models")
    parser.add_argument("--dataset", type=str, choices=["ogbn-arxiv", "ogbn-proteins"], default="ogbn-arxiv", help="Dataset name")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of GNN layers")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model.pth", help="Path to save the best model")
    parser.add_argument("--performance_path", type=str, default="performance.json", help="Path to save performance metrics")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, choices=["serial", "parallel"], default="serial", help="Model type: serial or parallel")
    parser.add_argument("--gated", action="store_true", help="Use gated message passing")
    parser.add_argument("--use_triplets", action="store_true", help="Use triplet reasoning")
    parser.add_argument("--use_pretrain_weights", action="store_true", help="Use pre-trained weights")
    parser.add_argument("--pretrained_weights_path", type=str, default="checkpoints/pretrained.pth", help="Path to pre-trained weights")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint and performance results")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Load OGBN-Arxiv dataset with a sparse tensor transform.
    dataset = PygNodePropPredDataset(name=args.dataset, transform=T.ToSparseTensor())
    data = dataset[0]
    # Ensure the adjacency matrix is symmetric.
    if hasattr(data.adj_t, 'to_symmetric'):
        data.adj_t = data.adj_t.to_symmetric()
    else:
        # Manually symmetrize assuming a dense tensor representation.
        data.adj_t = data.adj_t + data.adj_t.transpose(0, 1)

    data = data.to(device)

    # Get train/validation/test splits.
    split_idx = dataset.get_idx_split()
    out_dim = dataset.num_tasks if hasattr(dataset, "num_tasks") else dataset.num_classes
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

    # Set up loss and optimizer.
    # Here we use NLLLoss assuming the model outputs log_softmax probabilities.
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    evaluator = Evaluator(name=args.dataset)
    metric_key = "acc" if args.dataset == "ogbn-arxiv" else "ap"

    # Initialize performance metrics and resume variables.
    results = {"train_accuracies": [], "valid_accuracies": [], "test_accuracies": []}
    start_epoch = 0
    best_valid_acc = 0.0
    early_stop_counter = 0

    # Resume logic: load previous results and checkpoint if the resume flag is set.
    if args.resume:
        if os.path.exists(args.performance_path):
            try:
                with open(args.performance_path, 'r') as f:
                    results = json.load(f)
                start_epoch = len(results["train_accuracies"])
                if results["valid_accuracies"]:
                    best_valid_acc = max(results["valid_accuracies"])
                    best_epoch = results["valid_accuracies"].index(best_valid_acc)+1
                    early_stop_counter = start_epoch - best_epoch
                logging.info(f"Resuming training from epoch {start_epoch+1} with best valid acc: {best_valid_acc:.4f}")
                # Load model checkpoint
                load_model(model, args.checkpoint_path)
            except Exception as e:
                logging.error(f"Failed to load previous results: {str(e)}")
        else:
            logging.info("No previous performance results found. Starting fresh training.")

    train_acc_list = results["train_accuracies"]
    valid_acc_list = results["valid_accuracies"]
    test_acc_list = results["test_accuracies"]

    # Training loop with early stopping, starting from the next epoch.
    for epoch in range(start_epoch + 1, args.epochs + 1):
        loss_val = train(model, data, split_idx['train'], optimizer, criterion)
        train_acc, valid_acc, _ = evaluate(model, data, split_idx, evaluator, metric_key)
        logging.info(f"Epoch {epoch}/{args.epochs}, Loss: {loss_val:.4f}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}")

        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        # Save best model and reset early stopping counter if validation accuracy improves.
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), args.checkpoint_path)
            logging.info(f"New best model saved at epoch {epoch} with Valid Acc: {best_valid_acc:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Save performance metrics at the end of this epoch.
        results = {
            "train_accuracies": train_acc_list,
            "valid_accuracies": valid_acc_list,
            "test_accuracies": test_acc_list
        }
        save_results(results, args.performance_path)

        # Check early stopping condition.
        if early_stop_counter >= args.early_stop_patience:
            logging.info(f"Early stopping triggered at epoch {epoch}.")
            break

    # Load the best model and evaluate on the test split.
    load_model(model, args.checkpoint_path)
    _, _, best_test_acc = evaluate(model, data, split_idx, evaluator, metric_key)
    logging.info(f"Final Test Accuracy: {best_test_acc:.4f}")
    test_acc_list.append(best_test_acc)

    # Save final performance metrics.
    results = {
        "train_accuracies": train_acc_list,
        "valid_accuracies": valid_acc_list,
        "test_accuracies": test_acc_list
    }
    save_results(results, args.performance_path)

if __name__ == "__main__":
    main()
