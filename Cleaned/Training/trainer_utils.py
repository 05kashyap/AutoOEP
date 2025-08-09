"""
Training utilities replicated from the legacy core/trainer.py, now under Training/.
Provides generic PyTorch training loops, evaluation, plotting, and model save/load helpers.
"""
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve


@dataclass
class TrainingHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)


class ModelTrainer:
    """Generic trainer for temporal models (ported from core/trainer.py)."""

    def __init__(self, model: torch.nn.Module, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.training_history = TrainingHistory()

    def _train_epoch(self, train_loader, optimizer, criterion) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (torch.sigmoid(output.squeeze()) > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()

        avg_loss = total_loss / max(1, len(train_loader))
        accuracy = 100.0 * correct / max(1, total)
        return avg_loss, accuracy

    def _validate_epoch(self, val_loader, criterion) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output.squeeze(), target)
                total_loss += loss.item()
                predicted = (torch.sigmoid(output.squeeze()) > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = total_loss / max(1, len(val_loader))
        accuracy = 100.0 * correct / max(1, total)
        return avg_loss, accuracy

    def train(self, train_loader, val_loader, epochs: int = 50, lr: float = 1e-3, patience: int = 10) -> TrainingHistory:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        best_val_loss = float("inf")
        patience_counter = 0
        start_time = time.time()

        print(f"Starting training on {self.device}")
        print("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time")
        print("-" * 60)

        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)

            self.training_history.train_loss.append(train_loss)
            self.training_history.val_loss.append(val_loss)
            self.training_history.train_acc.append(train_acc)
            self.training_history.val_acc.append(val_acc)

            epoch_time = time.time() - epoch_start
            print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:9.2f} | {val_loss:8.4f} | {val_acc:7.2f} | {epoch_time:5.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Optional: you can save a checkpoint externally
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f} seconds")
        return self.training_history

    def evaluate(self, test_loader) -> Dict[str, Any]:
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                predictions = torch.sigmoid(output.squeeze())
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        auc_score = roc_auc_score(all_targets, all_predictions) if len(np.unique(all_targets)) > 1 else float("nan")
        binary_preds = (all_predictions > 0.5).astype(int)
        accuracy = (binary_preds == all_targets).mean() if len(all_targets) > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "auc_score": float(auc_score),
            "predictions": all_predictions,
            "targets": all_targets,
        }

    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.training_history.train_loss, label="Train Loss")
        ax1.plot(self.training_history.val_loss, label="Validation Loss")
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        ax2.plot(self.training_history.train_acc, label="Train Accuracy")
        ax2.plot(self.training_history.val_acc, label="Validation Accuracy")
        ax2.set_title("Model Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, test_results: Dict[str, Any]):
        fpr, tpr, _ = roc_curve(test_results["targets"], test_results["predictions"]) \
            if len(np.unique(test_results["targets"])) > 1 else (np.array([0, 1]), np.array([0, 1]), None)
        auc_score = test_results.get("auc_score", float("nan"))

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()


class ModelSaver:
    """Save/load helpers for temporal checkpoints (ported from core/trainer.py)."""

    @staticmethod
    def save_model(model: torch.nn.Module, scaler, model_path: str, scaler_path: Optional[str] = None):
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_size": getattr(model, "input_size", None),
                "hidden_size": getattr(model, "hidden_size", None),
                "num_layers": getattr(model, "num_layers", None),
            },
            "scaler_mean": getattr(scaler, "mean_", None),
            "scaler_scale": getattr(scaler, "scale_", None),
        }, model_path)
        print(f"Model saved to {model_path}")

        if scaler_path is not None:
            import joblib
            os.makedirs(os.path.dirname(scaler_path) or ".", exist_ok=True)
            joblib.dump(scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")

    @staticmethod
    def load_model(model_class, model_path: str, device: Optional[str] = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint.get("model_config", {})
        model = model_class(**config) if config else model_class()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        class _Scaler:
            pass
        scaler = _Scaler()
        setattr(scaler, "mean_", checkpoint.get("scaler_mean", None))
        setattr(scaler, "scale_", checkpoint.get("scaler_scale", None))
        return model, scaler
