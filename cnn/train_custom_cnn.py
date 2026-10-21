import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from config import CHECKPOINT_NAME, DATA_PATH, IMG_SIZE, SAVE_DIR
from data_module import SoundDataModule
from evaluator import ModelEvaluator
from model import SoundCNN
from plotter import TrainingPlotter
from utils import count_parameters, get_device, seed_everything


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(data_loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch = labels.size(0)
        loss_sum += loss.item() * batch
        correct += (outputs.argmax(1) == labels).sum().item()
        total += batch
    return loss_sum / total, correct / total


def save_checkpoint(path, epoch, model_state, class_names, eval_loss, eval_acc, eval_split):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state,
            "class_names": class_names,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "eval_split": eval_split,
            "val_loss": eval_loss,
            "val_acc": eval_acc,
            "img_size": IMG_SIZE,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train 5-block CNN for sound classification.")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    os.makedirs(SAVE_DIR, exist_ok=True)

    data = SoundDataModule(
        data_path=args.data,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        img_size=IMG_SIZE,
        device=device,
    ).setup()
    train_loader, val_loader, test_loader = data.loaders()
    eval_loader = test_loader or val_loader
    eval_name = "Test" if test_loader else "Val"
    if eval_loader is None:
        raise RuntimeError("Need a test or val split for evaluation after each epoch.")

    model = SoundCNN(num_classes=len(data.class_names)).to(device)
    total_params = count_parameters(model)
    print(f"Device: {device}")
    print(f"Classes: {data.class_names}")
    print(f"Parameters before training: {total_params:,}")

    weights = compute_class_weight("balanced", classes=np.arange(len(data.class_names)), y=np.array(data.train_targets))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    evaluator = ModelEvaluator(device)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    checkpoint_path = os.path.join(SAVE_DIR, CHECKPOINT_NAME)
    history = {"train_loss": [], "eval_loss": [], "train_acc": [], "eval_acc": []}

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        eval_loss, eval_acc = evaluator.evaluate(model, eval_loader, criterion, eval_name)
        scheduler.step(eval_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["eval_loss"].append(eval_loss)
        history["eval_acc"].append(eval_acc)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train {train_loss:.4f}/{train_acc:.4f} | "
            f"{eval_name.lower()} {eval_loss:.4f}/{eval_acc:.4f} | "
            f"{time.time() - start:.1f}s"
        )

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(checkpoint_path, epoch, best_state, data.class_names, eval_loss, eval_acc, eval_name.lower())
            print(f"Saved best checkpoint: {checkpoint_path}")

    TrainingPlotter().plot_history(history, os.path.join(SAVE_DIR, "custom_cnn_curves.png"), eval_name)
    print(f"Best {eval_name.lower()} loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
