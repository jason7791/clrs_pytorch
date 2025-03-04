import os
import json
import argparse
from tqdm import tqdm
from absl import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from baselines import BaselineModel
from baselines_parallel import ParallelMPNNModel

def train(model, device, loader, optimizer, criterion):
    model.train()
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        is_labeled = batch.y == batch.y  # Ignore unlabeled data
        loss = criterion(pred[is_labeled], batch.y[is_labeled].float())
        loss.backward()
        optimizer.step()



def eval(model, device, loader, evaluator):
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

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="PGN on ogbg-molhiv")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--pretrained_weights_path", type=str, default = "/Users/jasonwjh/Documents/clrs_pytorch/clrs_checkpoints/checkpoint.pth")
    parser.add_argument("--use_pretrain_weights", action="store_true", help="Use pre-trained weights.")
    parser.add_argument("--performance_path", type=str, default = "/Users/jasonwjh/Documents/clrs_pytorch/ogb_performance/performance.json")
    parser.add_argument("--checkpoint_path", type=str, default = "/Users/jasonwjh/Documents/clrs_pytorch/ogb_checkpoints/checkpoint.pth")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Number of epochs to wait for validation performance to improve before stopping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--model", type=str, default="serial", help="Parallel or Serial Model")
    parser.add_argument("--gated", action="store_true", help="Use gating")
    parser.add_argument("--use_triplets", action="store_true", help="Use triplet reasoning")
    args = parser.parse_args()

    # Setup
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name="ogbg-molhiv")
    set_seed(args.seed)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)

    # Model and optimizer
    if(args.model == "serial"):
        model = BaselineModel(
            out_dim=dataset.num_tasks,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            reduction=torch.max,
            use_pretrain_weights=args.use_pretrain_weights, 
            pretrained_weights_path=args.pretrained_weights_path,
            gated=args.gated,
            use_triplets=args.use_triplets
        ).to(device)
    else:
        model = ParallelMPNNModel(
            out_dim=dataset.num_tasks,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            reduction=torch.max,
            use_pretrain_weights=args.use_pretrain_weights, 
            pretrained_weights_path=args.pretrained_weights_path,
            gated=args.gated,
            use_triplets=args.use_triplets
        ).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    criterion = nn.BCEWithLogitsLoss()

    best_valid_perf = float("-inf")
    early_stop_counter = 0

    # Lists to store accuracies for plotting
    train_acc_list = []
    valid_acc_list = []
    test_acc_list = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}")
        train(model, device, train_loader, optimizer, criterion)
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        print(f"Train: {train_perf}, Valid: {valid_perf}")

        # Append accuracies to lists
        train_acc_list.append(train_perf["rocauc"])
        valid_acc_list.append(valid_perf["rocauc"])

        # Save the model if the validation performance improves
        if valid_perf["rocauc"] > best_valid_perf:
            best_valid_perf = valid_perf["rocauc"]
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"New best model saved at epoch {epoch} with valid ROC-AUC: {best_valid_perf:.4f}")
            early_stop_counter = 0  # Reset early stopping counter
        else:
            early_stop_counter += 1

        # Check if early stopping criteria is met
        if early_stop_counter >= args.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    logging.info('Restoring best model from checkpoint...')
    checkpoint = torch.load(args.checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint)
    test_perf = eval(model, device, test_loader, evaluator)
    print(f"Test: {test_perf}")
    test_acc_list.append(test_perf["rocauc"])

    results = {
        "train_accuracies": train_acc_list,
        "valid_accuracies": valid_acc_list,
        "test_accuracies": test_acc_list
    }
    # Ensure the directory for the performance path exists
    performance_dir = os.path.dirname(args.performance_path)
    os.makedirs(performance_dir, exist_ok=True)
    with open(args.performance_path, 'w') as f:
        json.dump(results, f)
    print(f"Accuracies saved to {args.performance_path}")

if __name__ == "__main__":
    main()
