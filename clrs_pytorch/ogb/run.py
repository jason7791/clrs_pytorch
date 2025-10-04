import os
import json
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from baselines_freeze_early_layers import BaselineModel
from baselines_parallel import ParallelMPNNModel

# ---------------------- Utility Functions ---------------------- #

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True

def train(model, device, loader, optimizer, criterion):
    """Train the model for one epoch."""
    model.train()
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        # Create a mask for valid labels
        is_labeled = batch.y == batch.y  
        loss = criterion(pred[is_labeled], batch.y[is_labeled].float())
        loss.backward()
        optimizer.step()

def evaluate(model, device, loader, evaluator):
    """Evaluate the model on the given dataset."""
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            pred = model(batch)
            y_true.append(batch.y.view(pred.shape).cpu())
            y_pred.append(pred.cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    
    return evaluator.eval({"y_true": y_true, "y_pred": y_pred})

def save_results(results, path):
    """Save training results as a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {path}")

def load_model(model, checkpoint_path):
    """Load a model from the checkpoint file."""
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint)
    else:
        logging.warning(f"No checkpoint found at {checkpoint_path}, starting from scratch.")

# ---------------------- Main Training Pipeline ---------------------- #



def main():
    """Main function to train and evaluate the PGN model."""
    
    # Argument parser setup
    parser = argparse.ArgumentParser(description="GNN on OGB datasets")
    parser.add_argument("--dataset", type=str, choices=["ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast"], default="ogbg-molclintox", help="Dataset name")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
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

    # Device configuration
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # Load dataset and evaluator based on the dataset argument
    dataset = PygGraphPropPredDataset(name=args.dataset)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name=args.dataset)
    
    # Select the appropriate metric key for logging and evaluation.
    # ogbg-molhiv uses ROC-AUC while ogbg-molpcba uses Average Precision (AP)
    if(args.dataset=="ogbg-molmuv"):
        metric_key = "ap"
    else:
        metric_key = "rocauc"
    
    # Set random seed
    set_seed(args.seed)

    # Prepare data loaders
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)

    # Initialize model
    # The output dimension is automatically set based on the dataset's number of tasks.
    ModelClass = BaselineModel if args.model == "serial" else ParallelMPNNModel
    model = ModelClass(
        out_dim=dataset.num_tasks,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        reduction=torch.max,
        use_pretrain_weights=args.use_pretrain_weights,
        pretrained_weights_path=args.pretrained_weights_path,
        gated=args.gated,
        use_triplets=args.use_triplets
    ).to(device)

    # Set up optimizer and loss function
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Initialize performance metrics and resume variables.
    results = {
        "train_accuracies": [],
        "valid_accuracies": [],
        "test_accuracies": []
    }
    start_epoch = 0
    best_valid_perf = float("-inf")
    early_stop_counter = 0

    # Resume logic: load previous results and checkpoint if the resume flag is set.
    if args.resume:
        if os.path.exists(args.performance_path):
            try:
                with open(args.performance_path, 'r') as f:
                    results = json.load(f)
                if("test_accuracies" in results and len(results["test_accuracies"])>0):
                    return
                start_epoch = len(results["train_accuracies"])
                if results["valid_accuracies"]:
                    best_valid_perf = max(results["valid_accuracies"])
                    best_epoch = results["valid_accuracies"].index(best_valid_perf) + 1
                    early_stop_counter = start_epoch - best_epoch
                logging.info(f"Resuming training from epoch {start_epoch+1} with best valid {metric_key.upper()}: {best_valid_perf:.4f} and early_stop_counter: {early_stop_counter}")
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
        logging.info(f"Epoch {epoch} / {args.epochs}")
        
        # Train and evaluate.
        train(model, device, train_loader, optimizer, criterion)
        train_perf = evaluate(model, device, train_loader, evaluator)
        valid_perf = evaluate(model, device, valid_loader, evaluator)
        
        logging.info(f"Train {metric_key.upper()}: {train_perf[metric_key]:.4f}, Valid {metric_key.upper()}: {valid_perf[metric_key]:.4f}")

        # Store performance metrics.
        train_acc_list.append(train_perf[metric_key])
        valid_acc_list.append(valid_perf[metric_key])

        # Save the model if validation performance improves.
        if valid_perf[metric_key] > best_valid_perf:
            best_valid_perf = valid_perf[metric_key]
            torch.save(model.state_dict(), args.checkpoint_path)
            logging.info(f"New best model saved at epoch {epoch} with Valid {metric_key.upper()}: {best_valid_perf:.4f}")
            early_stop_counter = 0  # Reset early stopping counter.
        else:
            early_stop_counter += 1

        # Update and save performance metrics (including early_stop_counter and best_valid_perf).
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

    # Restore best model for final evaluation.
    load_model(model, args.checkpoint_path)
    test_perf = evaluate(model, device, test_loader, evaluator)
    logging.info(f"Test {metric_key.upper()}: {test_perf[metric_key]:.4f}")
    test_acc_list.append(test_perf[metric_key])

    # Save final results.
    results = {
        "train_accuracies": train_acc_list,
        "valid_accuracies": valid_acc_list,
        "test_accuracies": test_acc_list
    }
    save_results(results, args.performance_path)

if __name__ == "__main__":
    main()
