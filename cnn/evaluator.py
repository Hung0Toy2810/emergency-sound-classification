"""Evaluation helpers used by the SoundCNN training and report scripts."""

import torch
from tqdm import tqdm


class ModelEvaluator:
    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def evaluate(self, model, data_loader, criterion, desc="Eval"):
        model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(data_loader, desc=desc):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            logits = model(inputs)
            batch = labels.size(0)
            loss_sum += criterion(logits, labels).item() * batch
            correct += (logits.argmax(1) == labels).sum().item()
            total += batch
        if not total:
            raise ValueError("Cannot evaluate an empty dataset.")
        return loss_sum / total, correct / total

    @torch.no_grad()
    def collect_predictions(self, model, data_loader):
        model.eval()
        targets, predictions = [], []
        for inputs, labels in tqdm(data_loader, desc="Predict"):
            logits = model(inputs.to(self.device))
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
        if not targets:
            raise ValueError("Cannot collect predictions from an empty dataset.")
        return targets, predictions
