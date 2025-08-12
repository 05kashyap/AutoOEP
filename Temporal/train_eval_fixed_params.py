import os
import sys
import argparse
from glob import glob
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Ensure repo root is on sys.path so 'Temporal' package can be imported regardless of CWD
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Temporal.temporal_trainer import TemporalProctor, set_seeds
from sklearn.metrics import (
    classification_report as sk_classification_report,
    confusion_matrix as sk_confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

PARAMS = {
    "window_size": 128,
    "stride": 10,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 16,
    "epochs": 50,
}


def compute_metrics_sklearn(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int).flatten()
    y_prob = np.asarray(y_prob).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, Any] = {}
    metrics["threshold"] = float(threshold)
    metrics["classification_report"] = sk_classification_report(y_true, y_pred, digits=4)
    metrics["confusion_matrix"] = sk_confusion_matrix(y_true, y_pred).tolist()

    # AUROC (needs both classes)
    if np.unique(y_true).size >= 2:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["auroc"] = None
    else:
        metrics["auroc"] = None

    # AUPRC (needs at least one positive)
    if np.any(y_true == 1):
        try:
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["auprc"] = None
    else:
        metrics["auprc"] = None

    return metrics


def ensure_test_features(test_df: pd.DataFrame, feature_cols: list, scaler_means: Optional[np.ndarray]) -> pd.DataFrame:
    # Add missing columns with training means for stability, and enforce order
    if scaler_means is None or len(scaler_means) == 0:
        scaler_means = np.zeros(len(feature_cols), dtype=float)
    train_means = {col: (scaler_means[idx] if idx < len(scaler_means) else 0.0) for idx, col in enumerate(feature_cols)}
    missing_cols = set(feature_cols) - set(test_df.columns)
    for col in missing_cols:
        test_df[col] = train_means.get(col, 0.0)
    # Keep only required columns and in the proper order
    return test_df[feature_cols]


def build_sequences_from_df(df: pd.DataFrame, proctor: TemporalProctor, fit_scaler: bool) -> Tuple[np.ndarray, np.ndarray]:
    X, y = proctor.create_sequences(df, fit_scaler=fit_scaler)
    return X, y


def main():
    # Args: take directory of processed CSVs and optional patterns
    parser = argparse.ArgumentParser(description="Train and evaluate Temporal LSTM with fixed params on processed CSVs")
    parser.add_argument("--csv-dir", type=str, required=True, help="Directory containing processed CSV files")
    parser.add_argument("--train-pattern", type=str, default="Train*_processed.csv", help="Glob pattern for training CSVs inside csv-dir")
    parser.add_argument("--test-pattern", type=str, default="Test*_processed.csv", help="Glob pattern for test CSVs inside csv-dir")
    parser.add_argument("--model-out", type=str, default="Models/temporal_proctor_fixed_params.pt", help="Path to save trained model")
    args = parser.parse_args()

    set_seeds(42)

    window_size = PARAMS["window_size"]
    stride = PARAMS["stride"]
    overlap = max(1, window_size - stride)

    proctor = TemporalProctor(window_size=window_size, overlap=overlap, model_type='lstm', stride=stride)

    # Load training data from provided directory
    csv_dir = os.path.abspath(args.csv_dir)
    if not os.path.isdir(csv_dir):
        print(f"CSV directory not found: {csv_dir}")
        return

    train_files = sorted(glob(os.path.join(csv_dir, args.train_pattern)))
    print(f"Found {len(train_files)} train CSV(s) in {csv_dir} matching '{args.train_pattern}':")
    for p in train_files:
        print(f" - {os.path.basename(p)}")

    train_dfs = []
    for p in train_files:
        if os.path.exists(p):
            df = proctor.load_data(p)
            train_dfs.append(df)
        else:
            print(f"Warning: {p} not found; skipping")

    if not train_dfs:
        print("No training data found. Exiting.")
        return

    combined_train_df = pd.concat(train_dfs, ignore_index=True)
    combined_train_df = combined_train_df.sort_values('timestamp')

    # Build sequences; fit scaler on training
    X_all, y_all = build_sequences_from_df(combined_train_df, proctor, fit_scaler=True)
    n = len(X_all)
    split_idx = int(n * 0.8)
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_val, y_val = X_all[split_idx:], y_all[split_idx:]

    # Train with fixed hyperparameters
    _ = proctor.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=PARAMS["epochs"],
        batch_size=PARAMS["batch_size"],
        lr=PARAMS["lr"],
        weight_decay=PARAMS["weight_decay"],
    )

    # Prepare testing from provided directory
    test_files = sorted(glob(os.path.join(csv_dir, args.test_pattern)))
    print(f"Found {len(test_files)} test CSV(s) in {csv_dir} matching '{args.test_pattern}':")
    for p in test_files:
        print(f" - {os.path.basename(p)}")

    results = []
    feature_cols = proctor.feature_cols or []

    for p in test_files:
        if not os.path.exists(p):
            print(f"Warning: {p} not found; skipping")
            continue
        print(f"\n--- Testing on {os.path.basename(p)} ---")
        df = proctor.load_data(p)
        # Align features
        test_df_aligned = ensure_test_features(df.copy(), feature_cols, proctor.scaler.mean_)
        test_data = test_df_aligned.values
        test_target = df['is_cheating'].values

        # Scale with training scaler
        test_scaled = proctor.scaler.transform(test_data)

        # Windowing with the same stride
        X_test, y_test = [], []
        for i in range(0, len(test_scaled) - proctor.window_size + 1, proctor.step):
            X_test.append(test_scaled[i:i + proctor.window_size])
            y_test.append(test_target[i + proctor.window_size - 1])
        X_test, y_test = np.array(X_test), np.array(y_test).reshape(-1, 1)
        print(f"Test sequences: {X_test.shape}")

        # Inference
        y_prob, y_pred = proctor.evaluate(X_test, y_test, batch_size=PARAMS["batch_size"], threshold=proctor.best_threshold)
        if y_prob is None:
            continue

        # Metrics with sklearn
        m = compute_metrics_sklearn(y_test.flatten(), np.array(y_prob).flatten(), threshold=proctor.best_threshold)
        print("\n[sklearn] Classification Report:\n" + m["classification_report"]) 
        print("[sklearn] Confusion Matrix:")
        print(np.array(m["confusion_matrix"]))
        print(f"[sklearn] AUROC: {m['auroc']}")
        print(f"[sklearn] AUPRC (AP): {m['auprc']}")

        results.append({
            'file': os.path.basename(p),
            'y_true': y_test.flatten(),
            'y_prob': np.array(y_prob).flatten(),
            'y_pred': np.array(y_pred).flatten(),
            'metrics': m,
        })

    # Combined metrics (if more than one set)
    if len(results) > 1:
        print("\n--- Combined Test Results (sklearn) ---")
        y_true_all = np.concatenate([r['y_true'] for r in results])
        y_prob_all = np.concatenate([r['y_prob'] for r in results])
        m_all = compute_metrics_sklearn(y_true_all, y_prob_all, threshold=proctor.best_threshold)
        print("\n[sklearn] Classification Report (combined):\n" + m_all["classification_report"]) 
        print("[sklearn] Confusion Matrix (combined):")
        print(np.array(m_all["confusion_matrix"]))
        print(f"[sklearn] AUROC (combined): {m_all['auroc']}")
        print(f"[sklearn] AUPRC (AP, combined): {m_all['auprc']}")

    # Save the fixed-params model
    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)
    proctor.hparams = {
        'window_size': window_size,
        'stride': stride,
        'lr': PARAMS['lr'],
        'weight_decay': PARAMS['weight_decay'],
        'batch_size': PARAMS['batch_size'],
        'epochs': PARAMS['epochs'],
    }
    proctor.search_score = None
    proctor.save_model(args.model_out)


if __name__ == "__main__":
    main()
