import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd

# Add project root to import TemporalProctor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Temporal.temporal_trainer import TemporalProctor  # noqa: E402


FIXED_FEATURE_COLS = [
    'timestamp',
    'verification_result',
    'num_faces',
    'iris_pos',
    'iris_ratio',
    'mouth_zone',
    'mouth_area',
    'x_rotation',
    'y_rotation',
    'z_rotation',
    'radial_distance',
    'gaze_direction',
    'gaze_zone',
    'watch',
    'headphone',
    'closedbook',
    'earpiece',
    'cell phone',
    'openbook',
    'chits',
    'sheet',
    'H-Distance',
    'F-Distance',
]


def _collect_test_files(dataset_path: str, pattern: str) -> list[str]:
    if os.path.isdir(dataset_path):
        # Search recursively for CSVs matching pattern
        search_glob = os.path.join(dataset_path, '**', pattern)
        files = sorted(glob.glob(search_glob, recursive=True))
        return files
    if os.path.isfile(dataset_path) and dataset_path.lower().endswith('.csv'):
        return [dataset_path]
    return []


def _ensure_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in FIXED_FEATURE_COLS if c in df.columns]
    missing = [c for c in FIXED_FEATURE_COLS if c not in df.columns]
    # Fill missing with zeros
    for c in missing:
        df[c] = 0
    # Reorder
    return df[FIXED_FEATURE_COLS]


def main():
    parser = argparse.ArgumentParser(description='Evaluate LSTM/GRU temporal model on processed CSVs.')
    parser.add_argument('--checkpoint', required=True, help='Path to saved model checkpoint (.pt) with scaler')
    parser.add_argument('--dataset', required=True, help='Path to a CSV file or a directory containing CSVs')
    parser.add_argument('--pattern', default='*Test*processed.csv', help='Glob pattern to match CSVs inside dataset dir (default: *Test*processed.csv)')
    parser.add_argument('--model-type', default='lstm', choices=['lstm', 'gru'], help='Model type used for training (default: lstm)')
    parser.add_argument('--window-size', type=int, default=15, help='Initial window size; will be overridden by checkpoint if present')
    parser.add_argument('--overlap', type=int, default=5, help='Overlap between windows')
    parser.add_argument('--threshold', type=float, default=0.4, help='Classification threshold for probabilities')
    parser.add_argument('--device', default=None, choices=[None, 'cpu', 'cuda'], help='Force device for inference (default: auto)')
    args = parser.parse_args()

    test_files = _collect_test_files(args.dataset, args.pattern)
    if not test_files:
        print(f"No CSVs found. dataset={args.dataset} pattern={args.pattern}")
        sys.exit(1)

    print('Discovered test files:')
    for f in test_files:
        print(f"  - {f}")

    # Initialize proctor and load model
    proctor = TemporalProctor(window_size=args.window_size, overlap=args.overlap, model_type=args.model_type, device=args.device)

    # Determine input size from first file using fixed feature order
    first_df = proctor.load_data(test_files[0])
    first_df_features = _ensure_feature_order(first_df.copy())
    input_size = first_df_features.shape[1]
    proctor.load_model(args.checkpoint, input_size=input_size)

    all_results = []

    for file_path in test_files:
        print(f"\n--- Testing on {os.path.basename(file_path)} ---")
        test_df = proctor.load_data(file_path)
        print(f"Test data shape: {test_df.shape}")

        test_df_features = _ensure_feature_order(test_df.copy())
        test_data = test_df_features.values
        test_target = test_df['is_cheating'].values if 'is_cheating' in test_df.columns else None

        # Ensure scaler is loaded from checkpoint
        try:
            test_data_scaled = proctor.scaler.transform(test_data)
        except Exception as e:
            print(f"ERROR: Scaler not ready or mismatch: {e}")
            sys.exit(1)

        # Create test sequences
        X_test, y_test = [], []
        step = proctor.step
        w = proctor.window_size
        for j in range(0, len(test_data_scaled) - w + 1, step):
            X_test.append(test_data_scaled[j:j + w])
            if test_target is not None:
                y_test.append(test_target[j + w - 1])

        X_test = np.array(X_test)
        y_test = np.array(y_test).reshape(-1, 1) if y_test else np.zeros((len(X_test), 1))
        print(f"Test sequences shape: {X_test.shape}")

        # Evaluate
        y_pred_proba, y_pred = proctor.evaluate(X_test, y_test, threshold=args.threshold)
        all_results.append({
            'file': os.path.basename(file_path),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
        })

    # Combined evaluation
    if len(all_results) > 1:
        print("\n--- Combined Test Results ---")
        combined_X_test = np.concatenate([r['X_test'] for r in all_results])
        combined_y_test = np.concatenate([r['y_test'] for r in all_results])
        print(f"Combined test shape: {combined_X_test.shape}")
        _ = proctor.evaluate(combined_X_test, combined_y_test, threshold=args.threshold)

    print("\nEvaluation completed.")


if __name__ == '__main__':
    main()
