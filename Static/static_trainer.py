import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import optuna
import logging
from tqdm import tqdm
import sys
import argparse

# Configure logging
def setup_logging():
    """Setup logging configuration to save detailed logs to file"""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/training_log_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) if os.getenv('VERBOSE') else logging.NullHandler()
        ]
    )
    
    # Also configure optuna to be quiet
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print(f"✓ Logging configured. Detailed logs saved to: {log_file}")
    return log_file

# Load and prepare the data
def load_and_prepare_data(train_csv_files, test_csv_files=None):
    """
    Load CSV data from multiple files and prepare features for training
    """
    logging.info("Loading training data...")
    
    # Load and combine training files
    train_dfs = []
    progress_bar = tqdm(train_csv_files, desc="Loading training files", unit="file")
    
    for csv_file in progress_bar:
        progress_bar.set_postfix_str(f"Loading {os.path.basename(csv_file)}")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            logging.info(f"  Loaded {csv_file}: {df.shape[0]} samples")
            train_dfs.append(df)
        else:
            logging.warning(f"  Warning: File not found - {csv_file}")
    
    if not train_dfs:
        raise FileNotFoundError("No training CSV files found")
    
    # Combine all training data
    train_df = pd.concat(train_dfs, ignore_index=True)
    logging.info(f"Combined training data shape: {train_df.shape}")
    
    # Load test data if provided
    test_df = None
    if test_csv_files:
        logging.info("Loading test data...")
        test_dfs = []
        progress_bar = tqdm(test_csv_files, desc="Loading test files", unit="file")
        
        for csv_file in progress_bar:
            progress_bar.set_postfix_str(f"Loading {os.path.basename(csv_file)}")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                logging.info(f"  Loaded {csv_file}: {df.shape[0]} samples")
                test_dfs.append(df)
            else:
                logging.warning(f"  Warning: File not found - {csv_file}")
        
        if test_dfs:
            test_df = pd.concat(test_dfs, ignore_index=True)
            logging.info(f"Combined test data shape: {test_df.shape}")
    
    # Display basic info about the dataset
    logging.info(f"Training dataset info:")
    logging.info(f"Shape: {train_df.shape}")
    logging.info(f"Target variable distribution:")
    train_counts = train_df['is_cheating'].value_counts()
    logging.info(train_counts)
    logging.info(f"Cheating rate: {train_df['is_cheating'].mean():.2%}")
    c1 = int(train_counts.get(1, 0))
    c0 = int(train_counts.get(0, 0))
    logging.info(f"Class imbalance ratio (cheating:non-cheating): {c1}:{c0}")
    
    if test_df is not None:
        logging.info(f"Test dataset info:")
        logging.info(f"Shape: {test_df.shape}")
        logging.info(f"Target variable distribution:")
        test_counts = test_df['is_cheating'].value_counts()
        logging.info(test_counts)
        logging.info(f"Cheating rate: {test_df['is_cheating'].mean():.2%}")
    
    # Remove columns that shouldn't be used for training
    columns_to_remove = ['timestamp', 'is_cheating', 'split', 'video']
    
    # Prepare training features and target
    train_features = train_df.drop(columns=[col for col in columns_to_remove if col in train_df.columns])
    train_target = train_df['is_cheating']

    # --- DEBUG: Print feature columns and shape ---
    print(f"Training features columns: {list(train_features.columns)}")
    print(f"Training features shape: {train_features.shape}")

    # Prepare test features and target if available
    test_features = None
    test_target = None
    if test_df is not None:
        test_features = test_df.drop(columns=[col for col in columns_to_remove if col in test_df.columns])
        test_target = test_df['is_cheating']
    
    # Check for missing values
    logging.info(f"Missing values in training data:")
    missing_values = train_features.isnull().sum()
    logging.info(missing_values[missing_values > 0])
    
    print(f"✓ Data loaded: {train_df.shape[0]} training samples, {test_df.shape[0] if test_df is not None else 0} test samples")
    
    return train_features, train_target, test_features, test_target, train_df

def handle_class_imbalance(X_train, y_train, method='smote'):
    """
    Handle class imbalance using various resampling techniques
    """
    logging.info(f"Applying class imbalance handling: {method}")
    logging.info(f"Original distribution: {np.bincount(y_train)}")
    
    # Choose safe k based on minority count to avoid errors on tiny classes
    try:
        counts = np.bincount(y_train.astype(int))
        minority_count = int(counts.min()) if counts.size > 0 else 0
    except Exception:
        minority_count = 0
    k_safe = max(1, min(3, minority_count - 1)) if minority_count > 0 else 1

    if method == 'smote':
        sampler = SMOTE(random_state=42, k_neighbors=k_safe)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=42, n_neighbors=k_safe)
    elif method == 'borderline_smote':
        sampler = BorderlineSMOTE(random_state=42, k_neighbors=k_safe)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=42, smote=SMOTE(random_state=42, k_neighbors=k_safe))
    elif method == 'smote_enn':
        sampler = SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=k_safe))
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        logging.info("No resampling applied")
        return X_train, y_train
    
    try:
        with tqdm(total=1, desc=f"Applying {method}", unit="step") as pbar:
            res = sampler.fit_resample(X_train, y_train)
            # Most samplers return a 2-tuple (X, y); fallback to originals if unexpected
            if isinstance(res, tuple) and len(res) >= 2:
                X_resampled, y_resampled = res[0], res[1]
            else:
                X_resampled, y_resampled = X_train, y_train
            pbar.update(1)
        logging.info(f"Resampled distribution: {np.bincount(np.asarray(y_resampled).astype(int))}")
        return X_resampled, y_resampled
    except Exception as e:
        logging.error(f"Resampling failed: {e}. Using original data.")
        return X_train, y_train

# Feature engineering and preprocessing
def preprocess_features(X_train, X_test, use_scaling=True):
    """
    Preprocess features - handle missing values and optionally scale
    """
    with tqdm(total=3, desc="Preprocessing features", unit="step") as pbar:
        # Fill missing values with median for numerical features
        X_train_processed = X_train.fillna(X_train.median())
        pbar.update(1)
        
        X_test_processed = X_test.fillna(X_train.median())  # Use train median for test set
        pbar.update(1)
        
        scaler = None
        if use_scaling:
            # Apply scaling
            scaler = StandardScaler()
            X_train_processed = pd.DataFrame(
                scaler.fit_transform(X_train_processed),
                columns=X_train_processed.columns,
                index=X_train_processed.index
            )
            X_test_processed = pd.DataFrame(
                scaler.transform(X_test_processed),
                columns=X_test_processed.columns,
                index=X_test_processed.index
            )
        pbar.update(1)
    
    return X_train_processed, X_test_processed, scaler

# Train XGBoost model
def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost classifier with extensive hyperparameter tuning
    """
    logging.info("Training XGBoost classifier...")
    
    # Expanded hyperparameter grid
    param_grid_extensive = {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0, 1.5],
        'scale_pos_weight': [1, 2, 3]  # For class imbalance
    }
        
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        tree_method='hist'  # Faster training
    )
    
    # Randomized search (more efficient for large parameter spaces)
    random_search = RandomizedSearchCV(
        xgb_model,
        param_grid_extensive,
        n_iter=100,  # Try 100 random combinations
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0,  # Disable verbose output
        random_state=42
    )
    
    with tqdm(total=1, desc="XGBoost hyperparameter tuning", unit="search") as pbar:
        random_search.fit(X_train, y_train)
        pbar.update(1)
    
    return random_search.best_estimator_, random_search

def create_optuna_objective(X_train, y_train, X_val, y_val, model_type='xgboost', use_gpu: bool = False):
    """
    Create Optuna objective function for hyperparameter optimization
    """
    def objective(trial):
        if model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': 42,
                'tree_method': 'gpu_hist' if use_gpu else 'hist'
            }
            model = xgb.XGBClassifier(**params)
        elif model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
                'objective': 'binary',
                'metric': 'binary_logloss',
                'random_state': 42,
                'verbosity': -1,
                'device_type': 'gpu' if use_gpu else 'cpu'
            }
            model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"Unsupported model_type for optimization: {model_type}")

        # Try GPU; if it fails, fallback to CPU inside the trial for robustness
        try:
            model.fit(X_train, y_train)
        except Exception as gpu_err:
            if use_gpu:
                logging.warning(f"GPU training failed for {model_type} trial; falling back to CPU. Error: {gpu_err}")
                if model_type == 'xgboost':
                    params['tree_method'] = 'hist'
                    model = xgb.XGBClassifier(**params)
                elif model_type == 'lightgbm':
                    params['device_type'] = 'cpu'
                    model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train)
            else:
                raise
        y_pred_proba = np.asarray(model.predict_proba(X_val))[:, 1]
        auc_score = float(roc_auc_score(y_val, y_pred_proba))

        return auc_score
    
    return objective

def train_ensemble_model(X_train, y_train, X_test, y_test, imbalance_method='smote', n_trials: int = 100, timeout: int = 1800, seed: int = 42, use_gpu: bool = False):
    """
    Train ensemble of different models with extensive hyperparameter tuning
    """
    logging.info("Training ensemble model with extensive hyperparameter optimization...")
    
    # Handle class imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train, imbalance_method)
    
    # Split balanced training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_balanced, y_train_balanced, test_size=0.2, random_state=42, stratify=y_train_balanced
    )
    
    models = {}
    best_params = {}
    
    # XGBoost optimization
    logging.info("Optimizing XGBoost...")
    study_xgb = optuna.create_study(direction='maximize', study_name='xgboost_optimization', sampler=optuna.samplers.TPESampler(seed=seed))
    
    with tqdm(total=100, desc="XGBoost optimization", unit="trial") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix_str(f"Best AUC: {study.best_value:.4f}")
        
        study_xgb.optimize(
            create_optuna_objective(X_train_split, y_train_split, X_val_split, y_val_split, 'xgboost', use_gpu=use_gpu), 
            n_trials=n_trials, 
            timeout=timeout,  # seconds
            callbacks=[callback]
        )
    
    best_xgb_params = study_xgb.best_params
    best_xgb_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': seed,
        'tree_method': 'gpu_hist' if use_gpu else 'hist'
    })
    
    models['xgboost'] = xgb.XGBClassifier(**best_xgb_params)
    best_params['xgboost'] = best_xgb_params
    
    # LightGBM optimization
    logging.info("Optimizing LightGBM...")
    study_lgb = optuna.create_study(direction='maximize', study_name='lightgbm_optimization', sampler=optuna.samplers.TPESampler(seed=seed))
    
    with tqdm(total=100, desc="LightGBM optimization", unit="trial") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix_str(f"Best AUC: {study.best_value:.4f}")
        
        study_lgb.optimize(
            create_optuna_objective(X_train_split, y_train_split, X_val_split, y_val_split, 'lightgbm', use_gpu=use_gpu), 
            n_trials=n_trials, 
            timeout=timeout,  # seconds
            callbacks=[callback]
        )
    
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update({
        'objective': 'binary',
        'metric': 'binary_logloss',
        'random_state': seed,
        'verbosity': -1,
        'device_type': 'gpu' if use_gpu else 'cpu'
    })
    
    models['lightgbm'] = lgb.LGBMClassifier(**best_lgb_params)
    best_params['lightgbm'] = best_lgb_params
    
    # Random Forest with class weighting
    logging.info("Optimizing Random Forest...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    rf_params = {
        'n_estimators': [200, 500, 800, 1000],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [class_weight_dict, 'balanced']
    }
    
    rf_model = RandomForestClassifier(random_state=seed, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf_model, rf_params, n_iter=50, cv=3, scoring='roc_auc', 
        n_jobs=-1, random_state=seed, verbose=0
    )
    
    with tqdm(total=1, desc="Random Forest optimization", unit="search") as pbar:
        rf_search.fit(X_train_split, y_train_split)
        pbar.update(1)
    
    models['random_forest'] = rf_search.best_estimator_
    best_params['random_forest'] = rf_search.best_params_
    
    # Train all models on full balanced training data
    logging.info("Training final models on full balanced training data...")
    trained_models = {}
    
    with tqdm(models.items(), desc="Training final models", unit="model") as pbar:
        for name, model in pbar:
            pbar.set_postfix_str(f"Training {name}")
            model.fit(X_train_balanced, y_train_balanced)
            trained_models[name] = model
    
    # Evaluate models and select best
    best_model = None
    best_score = 0
    best_name = ""
    
    logging.info("Evaluating models on test set:")
    with tqdm(trained_models.items(), desc="Evaluating models", unit="model") as pbar:
        for name, model in pbar:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            logging.info(f"{name}: AUC = {auc_score:.4f}")
            pbar.set_postfix_str(f"{name}: AUC = {auc_score:.4f}")
            
            if auc_score > best_score:
                best_score = auc_score
                best_model = model
                best_name = name
    
    logging.info(f"Best model: {best_name} with AUC = {best_score:.4f}")
    print(f"✓ Best model: {best_name} with AUC = {best_score:.4f}")
    
    return best_model, best_name, best_params[best_name], trained_models, study_xgb.best_value

# Enhanced model saving functionality
def save_model_and_metadata(model, feature_names, scaler=None, model_name="best_model", 
                           best_params=None, model_dir='Models_new'):
    """
    Save the trained model, scaler, and associated metadata
    """
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save paths
    model_path = os.path.join(model_dir, f'{model_name}_cheating_model_{timestamp}.pkl')
    scaler_path = os.path.join(model_dir, f'scaler_{timestamp}.pkl')
    metadata_path = os.path.join(model_dir, f'model_metadata_{timestamp}.pkl')
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save the scaler if it exists
    if scaler is not None:
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
    else:
        scaler_path = None
        print("No scaler to save (scaling was not applied)")
    
    # Save metadata (feature names, model parameters, etc.)
    metadata = {
        'feature_names': list(feature_names),
        'model_type': model_name,
        'training_timestamp': timestamp,
        'model_params': best_params if best_params else {},
        'feature_count': len(feature_names),
        'scaler_used': scaler is not None,
        'scaler_path': scaler_path
    }
    
    joblib.dump(metadata, metadata_path)
    print(f"Model metadata saved to: {metadata_path}")
    
    # --- DEBUG: Print feature names being saved ---
    print(f"Saving model with feature names: {feature_names}")
    
    return model_path, metadata_path, scaler_path

def load_model_and_metadata(model_path):
    """
    Load a saved model, scaler, and metadata
    """
    # Load the model
    model = joblib.load(model_path)
    
    # Try to load metadata and scaler robustly using timestamp at end of filename
    try:
        import re as _re
        base = os.path.basename(model_path)
        dirn = os.path.dirname(model_path)
        m = _re.search(r"_(\d{8}_\d{6})\.pkl$", base)
        if not m:
            raise ValueError("Could not extract timestamp from model filename")
        ts = m.group(1)
        metadata_path = os.path.join(dirn, f"model_metadata_{ts}.pkl")
        metadata = joblib.load(metadata_path)
        
        # Load scaler if it was used
        scaler = None
        if metadata.get('scaler_used', False):
            scaler_path = metadata.get('scaler_path')
            if scaler_path and os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                print(f"Scaler loaded successfully")
            else:
                fallback = os.path.join(dirn, f"scaler_{ts}.pkl")
                if os.path.exists(fallback):
                    scaler = joblib.load(fallback)
                    print(f"Scaler loaded from fallback path")
                else:
                    print("Warning: Scaler was expected but not found")
        
        print(f"Model and metadata loaded successfully")
        print(f"Model trained on: {metadata['training_timestamp']}")
        print(f"Feature count: {metadata['feature_count']}")
        print(f"Scaler used: {metadata.get('scaler_used', False)}")
        
        return model, metadata, scaler
        
    except Exception as e:
        print(f"Model loaded successfully, but metadata/scaler loading failed: {str(e)}")
        return model, None, None

def predict_cheating(model, sample_data, feature_names=None, metadata=None, scaler=None):
    """
    Make predictions on new sample data with proper preprocessing
    
    Parameters:
    - model: trained XGBoost model
    - sample_data: pandas DataFrame or numpy array with features
    - feature_names: list of feature names (optional, for validation)
    - metadata: model metadata (optional)
    - scaler: fitted scaler object (optional)
    
    Returns:
    - predictions: binary predictions (0/1)
    - probabilities: probability of cheating
    """
    # Convert to DataFrame if it's not already
    if not isinstance(sample_data, pd.DataFrame):
        if feature_names is not None:
            sample_data = pd.DataFrame(sample_data, columns=feature_names)
        else:
            sample_data = pd.DataFrame(sample_data)
    
    # Validate features if metadata is available
    if metadata is not None:
        expected_features = metadata['feature_names']
        current_features = list(sample_data.columns)
        
        # Check if all expected features are present
        missing_features = set(expected_features) - set(current_features)
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
        
        # Reorder columns to match training data
        sample_data = sample_data.reindex(columns=expected_features, fill_value=0)
    
    # Handle missing values (use same approach as training)
    sample_data_processed = sample_data.fillna(sample_data.median())
    
    # Apply scaling if scaler is provided
    if scaler is not None:
        sample_data_processed = pd.DataFrame(
            scaler.transform(sample_data_processed),
            columns=sample_data_processed.columns,
            index=sample_data_processed.index
        )
        print("Applied scaling to input data")
    
    # Make predictions
    predictions = model.predict(sample_data_processed)
    probabilities = model.predict_proba(sample_data_processed)[:, 1]
    
    return predictions, probabilities

def demonstrate_prediction(model_path, sample_csv_path=None):
    """
    Demonstrate how to use the saved model for predictions
    """
    print("\n" + "="*50)
    print("PREDICTION DEMONSTRATION")
    print("="*50)
    
    # Load the model, metadata, and scaler
    model, metadata, scaler = load_model_and_metadata(model_path)
    
    if sample_csv_path and os.path.exists(sample_csv_path):
        # Load sample data from CSV
        sample_data = pd.read_csv(sample_csv_path)
        
        # Remove timestamp and target if they exist
        columns_to_remove = ['timestamp', 'is_cheating']
        for col in columns_to_remove:
            if col in sample_data.columns:
                sample_data = sample_data.drop(columns=[col])
        
        print(f"Making predictions on {len(sample_data)} samples from {sample_csv_path}")
        
    else:
        # Create a sample data point for demonstration
        if metadata:
            feature_names = metadata['feature_names']
        else:
            # Default feature names if metadata not available
            feature_names = [
                'verification_result', 'num_faces', 'iris_pos', 'iris_ratio', 
                'mouth_zone', 'mouth_area', 'x_rotation', 'y_rotation', 'z_rotation',
                'radial_distance', 'gaze_direction', 'gaze_zone', 'watch', 'headphone',
                'closedbook', 'earpiece', 'cell phone', 'openbook', 'chits', 'sheet',
                'H-Distance', 'F-Distance'
            ]
        
        # Create sample data
        sample_data = pd.DataFrame({
            'verification_result': [1],
            'num_faces': [1],
            'iris_pos': [0],
            'iris_ratio': [0.5],
            'mouth_zone': [0],
            'mouth_area': [25.0],
            'x_rotation': [0],
            'y_rotation': [0],
            'z_rotation': [0],
            'radial_distance': [5000.0],
            'gaze_direction': [2],
            'gaze_zone': [2],
            'watch': [0],
            'headphone': [0],
            'closedbook': [0],
            'earpiece': [0],
            'cell phone': [1],  # Has cell phone - suspicious
            'openbook': [0],
            'chits': [0],
            'sheet': [0],
            'H-Distance': [100.0],
            'F-Distance': [1000.0]
        })
        
        print("Making predictions on sample data:")
        print(sample_data)
    
    # Make predictions with proper preprocessing
    predictions, probabilities = predict_cheating(
        model, sample_data, metadata=metadata, scaler=scaler
    )
    
    # Display results
    print(f"\nPrediction Results:")
    print("-" * 30)
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        status = "CHEATING" if pred == 1 else "NOT CHEATING"
        confidence = prob if pred == 1 else (1 - prob)
        print(f"Sample {i+1}: {status} (Confidence: {confidence:.3f})")
        print(f"  Cheating Probability: {prob:.3f}")
        
        # Risk assessment
        if prob > 0.8:
            risk = "HIGH RISK"
        elif prob > 0.5:
            risk = "MEDIUM RISK"
        elif prob > 0.3:
            risk = "LOW RISK"
        else:
            risk = "MINIMAL RISK"
        print(f"  Risk Level: {risk}")
        print()
    
    return predictions, probabilities

# Enhanced evaluation with more metrics
def evaluate_model_comprehensive(model, X_train, y_train, X_test, y_test, feature_names, model_name="Model"):
    """
    Comprehensive model evaluation with multiple metrics
    """
    logging.info(f"{model_name.upper()} EVALUATION")
    
    with tqdm(total=5, desc="Model evaluation", unit="step") as pbar:
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        pbar.update(1)
        
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        pbar.update(1)
        
        # Calculate multiple metrics
        train_auc = roc_auc_score(y_train, y_train_proba)
        test_auc = roc_auc_score(y_test, y_test_proba)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        pbar.update(1)
        
        # Find optimal threshold using PR curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
        denom = np.clip(precision + recall, 1e-12, None)
        f1_scores = np.nan_to_num(2 * (precision * recall) / denom, nan=0.0, posinf=0.0, neginf=0.0)
        optimal_idx = int(np.argmax(f1_scores)) if f1_scores.size else 0
        if thresholds.size > 0:
            thr_idx = min(optimal_idx, thresholds.size - 1)
            optimal_threshold = float(thresholds[thr_idx])
        else:
            optimal_threshold = 0.5
        pbar.update(1)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        pbar.update(1)
    
    logging.info(f"Training AUC: {train_auc:.4f}")
    logging.info(f"Test AUC: {test_auc:.4f}")
    logging.info(f"Training F1: {train_f1:.4f}")
    logging.info(f"Test F1: {test_f1:.4f}")
    logging.info(f"Optimal threshold: {optimal_threshold:.3f}")
    logging.info(f"Optimal F1 score: {f1_scores[optimal_idx]:.4f}")
    
    print(f"✓ Evaluation complete: Test AUC = {test_auc:.4f}, F1 = {test_f1:.4f}")
    
    return feature_importance, y_test_proba, optimal_threshold

# Visualization functions
def plot_results(feature_importance, y_test, y_test_proba, save_dir='plots'):
    """
    Create visualizations for model results and save to files
    """
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set matplotlib to non-interactive backend
    plt.ioff()
    
    plt.figure(figsize=(15, 10))
    
    # Feature importance plot
    plt.subplot(2, 2, 1)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importances')
    plt.gca().invert_yaxis()
    
    # ROC Curve
    plt.subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auc_score = roc_auc_score(y_test, y_test_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Prediction distribution
    plt.subplot(2, 2, 3)
    plt.hist(y_test_proba[y_test == 0], bins=30, alpha=0.7, label='Non-cheating', density=True)
    plt.hist(y_test_proba[y_test == 1], bins=30, alpha=0.7, label='Cheating', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    
    # Confusion matrix heatmap
    plt.subplot(2, 2, 4)
    y_test_pred = (y_test_proba > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'model_results_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Results plot saved to: {plot_path}")
    return plot_path

def plot_comprehensive_results(feature_importance, y_test, y_test_proba, model_name="Model", save_dir='plots', threshold: float = 0.5):
    """
    Create comprehensive visualizations for model results and save to files
    """
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set matplotlib to non-interactive backend
    plt.ioff()
    
    plt.figure(figsize=(20, 15))
    
    # Feature importance plot
    if feature_importance is not None:
        plt.subplot(2, 3, 1)
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
    
    # ROC Curve
    plt.subplot(2, 3, 2)
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auc_score = roc_auc_score(y_test, y_test_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Precision-Recall Curve
    plt.subplot(2, 3, 3)
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    # Prediction distribution
    plt.subplot(2, 3, 4)
    plt.hist(y_test_proba[y_test == 0], bins=30, alpha=0.7, label='Non-cheating', density=True)
    plt.hist(y_test_proba[y_test == 1], bins=30, alpha=0.7, label='Cheating', density=True)
    # Show decision threshold
    try:
        thr = float(threshold)
    except Exception:
        thr = 0.5
    plt.axvline(thr, color='red', linestyle='--', linewidth=1, label=f'Th={thr:.2f}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    
    # Confusion matrix heatmap
    plt.subplot(2, 3, 5)
    cm = confusion_matrix(y_test, (y_test_proba > thr).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (threshold={thr:.2f})')
    
    # Class distribution
    plt.subplot(2, 3, 6)
    class_counts = np.bincount(y_test)
    plt.bar(['Non-cheating', 'Cheating'], class_counts)
    plt.ylabel('Count')
    plt.title('Test Set Class Distribution')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'comprehensive_results_{model_name}_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Comprehensive results plot saved to: {plot_path}")
    print(f"✓ Plots saved to: {plot_path}")
    return plot_path

# Main execution function
def main(train_csv_files, test_csv_files=None, use_scaling=True, imbalance_method='smote', 
         model_type=None, optimal_threshold=None, skip_optimization=False,
         n_trials: int = 100, timeout: int = 1800, seed: int = 42, use_gpu: bool = False):
    """
    Main function to run the complete pipeline with enhanced class imbalance handling
    """
    log_file = None
    try:
        # Setup logging
        log_file = setup_logging()
        
        print("🚀 Starting Enhanced Cheating Detection Training")
        print(f"📊 Training on {len(train_csv_files)} files, Testing on {len(test_csv_files) if test_csv_files else 0} files")
        print(f"📝 Detailed logs: {log_file}")
        print(f"🖥️  Acceleration: {'GPU' if use_gpu else 'CPU'}")
        print()
        
        # Load and prepare data
        X_train, y_train, X_test_external, y_test_external, full_train_df = load_and_prepare_data(
            train_csv_files, test_csv_files
        )
        
        # If external test data is provided, use it; otherwise split training data
        if X_test_external is not None and y_test_external is not None:
            logging.info("Using provided external test data")
            X_test, y_test = X_test_external, y_test_external
            X_train_final, y_train_final = X_train, y_train
        else:
            logging.info("No external test data provided, splitting training data")
            with tqdm(total=1, desc="Splitting data", unit="step") as pbar:
                X_train_final, X_test, y_train_final, y_test = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                pbar.update(1)
        
        logging.info(f"Final training set size: {X_train_final.shape[0]}")
        logging.info(f"Final test set size: {X_test.shape[0]}")
        
        # Preprocess features
        X_train_processed, X_test_processed, scaler = preprocess_features(
            X_train_final, X_test, use_scaling=use_scaling
        )
        
        # --- NEW: Direct model selection if skip_optimization is True ---
        if skip_optimization and model_type is not None:
            # Handle class imbalance
            X_train_balanced, y_train_balanced = handle_class_imbalance(X_train_processed, y_train_final, imbalance_method)
            # Train only the specified model
            if model_type == 'lightgbm':
                import lightgbm as lgb
                model = lgb.LGBMClassifier()
            elif model_type == 'xgboost':
                import xgboost as xgb
                model = xgb.XGBClassifier()
            elif model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier()
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            model.fit(X_train_balanced, y_train_balanced)
            model_name = model_type
            best_params = model.get_params()
            all_models = {model_type: model}
            best_cv_score = None
        else:
            # Train ensemble model with extensive optimization
            model, model_name, best_params, all_models, best_cv_score = train_ensemble_model(
                X_train_processed, y_train_final, X_test_processed, y_test, imbalance_method,
                n_trials=n_trials, timeout=timeout, seed=seed, use_gpu=use_gpu
            )
        
        # Comprehensive evaluation
        feature_importance, y_test_proba, found_optimal_threshold = evaluate_model_comprehensive(
            model, X_train_processed, y_train_final, X_test_processed, y_test, 
            X_train_processed.columns, model_name
        )
        
        # Use provided optimal_threshold if given
        if optimal_threshold is not None:
            print(f"Using provided optimal threshold: {optimal_threshold}")
        else:
            optimal_threshold = found_optimal_threshold
        
        # Save model and metadata
        with tqdm(total=1, desc="Saving model", unit="step") as pbar:
            model_path, metadata_path, scaler_path = save_model_and_metadata(
                model, X_train.columns, scaler, model_name, best_params
            )
            pbar.update(1)
        
        # Create comprehensive visualizations
        with tqdm(total=1, desc="Creating visualizations", unit="step") as pbar:
            plot_path = plot_comprehensive_results(feature_importance, y_test, y_test_proba, model_name, threshold=optimal_threshold if optimal_threshold is not None else 0.5)
            pbar.update(1)
        
        final_auc = roc_auc_score(y_test, y_test_proba)
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📈 Best CV Score: {best_cv_score:.4f}" if best_cv_score is not None else "")
        print(f"🎯 Final Test AUC: {final_auc:.4f}")
        print(f"⚖️ Optimal threshold: {optimal_threshold:.3f}")
        print(f"💾 Model saved: {model_path}")
        print(f"📊 Plots saved: {plot_path}")
        print(f"📋 Logs saved: {log_file}")
        
        return model, feature_importance, best_params, model_path, scaler, all_models, final_auc
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        if log_file:
            print(f"❌ Training failed! Check logs for details: {log_file}")
        else:
            print("❌ Training failed! Check logs for details in the logs directory.")
        return None, None, None, None, None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Cheating Detection Classifier")
    parser.add_argument('--imbalance-method', type=str, default=None, help='Imbalance handling method (e.g., smote, borderline_smote)')
    parser.add_argument('--model-type', type=str, default=None, help='Model type to use directly (lightgbm, xgboost, random_forest)')
    parser.add_argument('--optimal-threshold', type=float, default=None, help='Optimal threshold to use for classification')
    parser.add_argument('--skip-optimization', action='store_true', help='Skip hyperparameter optimization and use specified model')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials for model tuning')
    parser.add_argument('--timeout', type=int, default=1800, help='Global timeout for each study in seconds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration for supported models (XGBoost/LightGBM)')
    # Configuration for your specific files
    parser.add_argument('--train-dir', type=str, required=True, help='Directory containing training CSV files')
    parser.add_argument('--test-dir', type=str, required=True, help='Directory containing test CSV files')

    def get_csv_files_from_dir(pathlike):
        """Collect CSV files from a directory, recursively, or from a single CSV path or glob pattern."""
        import glob
        if not pathlike:
            return []
        # Glob patterns
        if any(ch in pathlike for ch in ['*', '?']):
            files = glob.glob(pathlike, recursive=True)
            return sorted([f for f in files if os.path.isfile(f) and f.lower().endswith('.csv')])
        # Single file
        if os.path.isfile(pathlike) and pathlike.lower().endswith('.csv'):
            return [pathlike]
        # Directory (recursive)
        if os.path.isdir(pathlike):
            found = []
            for root, _, files in os.walk(pathlike):
                for f in files:
                    if f.lower().endswith('.csv'):
                        found.append(os.path.join(root, f))
            return sorted(found)
        return []

    args = parser.parse_args()
    train_csv_files = get_csv_files_from_dir(args.train_dir)
    test_csv_files = get_csv_files_from_dir(args.test_dir)
    print(f"Found {len(train_csv_files)} training CSV(s) in '{args.train_dir}'")
    if train_csv_files:
        print("  e.g.:", ", ".join(os.path.basename(p) for p in train_csv_files[:5]))
    print(f"Found {len(test_csv_files)} test CSV(s) in '{args.test_dir}'")
    if test_csv_files:
        print("  e.g.:", ", ".join(os.path.basename(p) for p in test_csv_files[:5]))
    
    print("Enhanced Cheating Detection Classifier with Class Imbalance Handling")
    print("="*70)
    # If user specifies skip_optimization and model_type, run only that config
    if args.skip_optimization and args.model_type:
        print(f"Running with user-specified config: imbalance_method={args.imbalance_method}, model_type={args.model_type}, optimal_threshold={args.optimal_threshold}")
        main(
            train_csv_files,
            test_csv_files,
            use_scaling=True,
            imbalance_method=args.imbalance_method or 'borderline_smote',
            model_type=args.model_type,
            optimal_threshold=args.optimal_threshold,
            skip_optimization=True,
            n_trials=args.n_trials,
            timeout=args.timeout,
            seed=args.seed,
            use_gpu=args.gpu
        )
    else:
        # Try different imbalance handling methods as before
        imbalance_methods = ['smote', 'borderline_smote', 'smote_tomek']
        best_result = None
        best_auc = 0
        best_method = None
        
        overall_progress = tqdm(imbalance_methods, desc="Overall Progress", unit="method", position=0)
        
        for method in overall_progress:
            overall_progress.set_postfix_str(f"Trying {method}")
            
            logging.info(f"TRYING IMBALANCE METHOD: {method.upper()}")
            
            result = main(
                train_csv_files,
                test_csv_files,
                use_scaling=True,
                imbalance_method=method,
                n_trials=args.n_trials,
                timeout=args.timeout,
                seed=args.seed,
                use_gpu=args.gpu
            )
            
            if result[0] is not None:
                # Prefer external test quick evaluation when files exist; otherwise use returned final AUC
                model = result[0]
                files = [f for f in test_csv_files if os.path.exists(f)]
                if files:
                    try:
                        test_df_list = [pd.read_csv(f) for f in files]
                        if test_df_list:
                            test_df = pd.concat(test_df_list, ignore_index=True)
                            X_test_quick = test_df.drop(columns=['timestamp', 'is_cheating', 'split', 'video'], errors='ignore')
                            y_test_quick = test_df['is_cheating']
                            if result[4] is not None:  # scaler
                                X_test_quick = pd.DataFrame(result[4].transform(X_test_quick), columns=X_test_quick.columns)
                            y_pred_proba = np.asarray(model.predict_proba(X_test_quick))[:, 1]
                            auc_score = roc_auc_score(y_test_quick, y_pred_proba)
                            logging.info(f"Method {method} achieved AUC (quick external): {auc_score:.4f}")
                        else:
                            auc_score = float(result[6]) if len(result) > 6 and result[6] is not None else 0.0
                            logging.info(f"No test frames after reading; using returned AUC: {auc_score:.4f}")
                    except Exception as eval_err:
                        logging.warning(f"Quick external evaluation failed: {eval_err}; using returned AUC if available.")
                        auc_score = float(result[6]) if len(result) > 6 and result[6] is not None else 0.0
                else:
                    auc_score = float(result[6]) if len(result) > 6 and result[6] is not None else 0.0
                    logging.info(f"No test CSV files found; using returned AUC: {auc_score:.4f}")

                if auc_score > best_auc:
                    best_auc = auc_score
                    best_result = result
                    best_method = method
        
        overall_progress.close()
        
        if best_result and best_result[0] is not None:
            print("\n" + "="*50)
            print("🏆 FINAL RESULTS")
            print("="*50)
            print(f"🥇 Best method: {best_method}")
            print(f"📊 Best AUC: {best_auc:.4f}")
            print(f"💾 Model saved at: {best_result[3]}")
            print("="*50)
        else:
            print("❌ Training failed for all methods!")