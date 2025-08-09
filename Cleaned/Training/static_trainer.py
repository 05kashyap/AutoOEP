"""
Static Trainer - Train the static proctor model (e.g., LightGBM/XGBoost)
This script is isolated under Training/ to separate all training flows from inference.

Usage examples (PowerShell):
  python Training/static_trainer.py --data Inputs/Static/train.csv --model-out Inputs/Models/static_model.pkl --scaler-out Inputs/Models/static_scaler.pkl --metadata-out Inputs/Models/static_metadata.pkl

Notes:
- Replace the placeholder data loading with your actual dataset schema.
- Produces a joblib-persisted model, scaler, and optional metadata to be consumed by VideoProctor via Config paths.
"""
import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

try:
    from config import Config
except Exception:
    class Config:
        pass


def load_static_training_data(csv_path: str, label_col: str):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}")
    y = df[label_col].astype(int).values
    X = df.drop(columns=[label_col]).values
    return X, y


def train_static_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # You can swap in LightGBM/XGBoost here if desired
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_s, y_train)

    preds = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"Validation ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, (preds > 0.5).astype(int)))

    return model, scaler, {"feature_count": X.shape[1], "auc": float(auc)}


def main():
    ap = argparse.ArgumentParser(description="Train static proctor model")
    ap.add_argument("--data", required=True, help="Path to CSV with features + label")
    ap.add_argument("--label-col", default="is_cheating", help="Label column name in CSV")
    ap.add_argument("--model-out", required=True, help="Output path for model .pkl")
    ap.add_argument("--scaler-out", required=False, help="Output path for scaler .pkl")
    ap.add_argument("--metadata-out", required=False, help="Output path for metadata .pkl")
    args = ap.parse_args()

    X, y = load_static_training_data(args.data, args.label_col)
    model, scaler, metadata = train_static_model(X, y)

    os.makedirs(Path(args.model_out).parent, exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"Saved model to {args.model_out}")

    if args.scaler_out:
        os.makedirs(Path(args.scaler_out).parent, exist_ok=True)
        joblib.dump(scaler, args.scaler_out)
        print(f"Saved scaler to {args.scaler_out}")

    if args.metadata_out:
        os.makedirs(Path(args.metadata_out).parent, exist_ok=True)
        joblib.dump(metadata, args.metadata_out)
        print(f"Saved metadata to {args.metadata_out}")


if __name__ == "__main__":
    main()
