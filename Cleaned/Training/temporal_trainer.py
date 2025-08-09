"""
Temporal Trainer (isolated) - Train LSTM/GRU temporal models for VideoProctor
This script is placed under Training/ to decouple training from inference.

Usage examples (PowerShell):
  python Training/temporal_trainer.py --csv Inputs/Temporal/sequences.csv --model_type LSTM --save_path Inputs/Models/temporal_model.pth
  # Or provide a folder containing a CSV (it will pick the first *.csv):
  python Training/temporal_trainer.py --data_dir Inputs/Temporal --model_type GRU --save_path Inputs/Models/temporal_gru.pth
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Project imports
from core.data_handler import DataProcessor, CustomScaler
from Proctor.temporal_models import LSTMModel, GRUModel
from Training.trainer_utils import ModelTrainer


def _find_csv_in_dir(data_dir: str) -> str:
    p = Path(data_dir)
    for ext in ("*.csv", "*.CSV"):  # simple glob
        matches = list(p.glob(ext))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"No CSV file found in {data_dir}")


def _save_checkpoint(model: torch.nn.Module, scaler: CustomScaler, save_path: str, model_type: str,
                     input_size: int, hidden_size1: int, hidden_size2: int, window_size: int, lr: float, batch_size: int, epochs: int):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_type': model_type,
            'input_size': input_size,
            'hidden_size1': hidden_size1,
            'hidden_size2': hidden_size2,
            'output_size': 1,
            'window_size': window_size,
        },
        'scaler_mean': getattr(scaler, 'mean_', None),
        'scaler_scale': getattr(scaler, 'scale_', None),
        'training_config': {
            'learning_rate': lr,
            'batch_size': batch_size,
            'num_epochs': epochs,
        }
    }
    torch.save(checkpoint, save_path)
    print(f"✅ Model checkpoint saved to: {save_path}")


def main():
    ap = argparse.ArgumentParser(description="Train temporal model (LSTM/GRU) using CSV sequences")
    ap.add_argument("--csv", help="Path to CSV with frame-level features + label column 'is_cheating'")
    ap.add_argument("--data_dir", help="Directory containing a CSV (first *.csv will be used)")
    ap.add_argument("--model_type", default="LSTM", choices=["LSTM", "GRU"], help="Type of temporal model")
    ap.add_argument("--window_size", type=int, default=15, help="Temporal window size")
    ap.add_argument("--hidden_size1", type=int, default=128, help="Hidden size 1")
    ap.add_argument("--hidden_size2", type=int, default=64, help="Hidden size 2")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size")
    ap.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    ap.add_argument("--save_path", default=str(ROOT / "Inputs" / "Models" / "temporal_model.pth"), help="Output .pth path")
    args = ap.parse_args()

    # Resolve CSV path
    csv_path = args.csv
    if not csv_path and args.data_dir:
        csv_path = _find_csv_in_dir(args.data_dir)
    if not csv_path:
        raise ValueError("Please provide --csv or --data_dir containing a CSV")

    # Load sequences and labels using the project's DataProcessor
    sequences, labels, scaler = DataProcessor.load_and_preprocess_data(csv_path, args.window_size, scaler=None)
    input_size = sequences.shape[-1]
    print(f"Loaded {sequences.shape[0]} sequences of window {args.window_size} with {input_size} features")

    # Train/val/test split
    labels_np = np.asarray(labels)
    X_temp, X_test, y_temp, y_test = train_test_split(sequences, labels_np, test_size=0.2, random_state=42, stratify=labels_np)
    val_size_adjusted = 0.1 / (1 - 0.2)  # keep 10% of remaining for val
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp)

    # Tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    # DataLoaders
    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_ds = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    test_ds = torch.utils.data.TensorDataset(X_test_t, y_test_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Build model
    if args.model_type.upper() == 'LSTM':
        model = LSTMModel(input_size, args.hidden_size1, args.hidden_size2, 1)
    else:
        model = GRUModel(input_size, args.hidden_size1, args.hidden_size2, 1)
    model = model.to(device)

    # Train
    generic_trainer = ModelTrainer(model, device=device)
    history = generic_trainer.train(train_loader, val_loader, epochs=args.epochs, lr=args.learning_rate, patience=10)

    # Evaluate
    results = generic_trainer.evaluate(test_loader)
    print(f"\n=== Test Results ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUC: {results['auc_score']:.4f}")

    # Save checkpoint compatible with TemporalProctor
    _save_checkpoint(model, scaler, args.save_path, args.model_type.upper(), input_size, args.hidden_size1, args.hidden_size2, args.window_size, args.learning_rate, args.batch_size, args.epochs)
    print("Training finished. Model saved at:", args.save_path)


if __name__ == "__main__":
    main()
