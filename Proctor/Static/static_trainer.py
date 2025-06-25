import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

# Load and prepare the data
def load_and_prepare_data(csv_file_path):
    """
    Load CSV data and prepare features for training
    """
    df = pd.read_csv(csv_file_path)
    
    # Display basic info about the dataset
    print("Dataset shape:", df.shape)
    print("\nDataset info:")
    print(df.info())
    print("\nTarget variable distribution:")
    print(df['is_cheating'].value_counts())
    print(f"Cheating rate: {df['is_cheating'].mean():.2%}")
    
    # Remove timestamp column and separate features from target
    features = df.drop(['timestamp', 'is_cheating'], axis=1)
    target = df['is_cheating']
    
    # Check for missing values
    print(f"\nMissing values per column:")
    missing_values = features.isnull().sum()
    print(missing_values[missing_values > 0])
    
    return features, target, df

# Feature engineering and preprocessing
def preprocess_features(X_train, X_test, use_scaling=True):
    """
    Preprocess features - handle missing values and optionally scale
    """
    # Fill missing values with median for numerical features
    X_train_processed = X_train.fillna(X_train.median())
    X_test_processed = X_test.fillna(X_train.median())  # Use train median for test set
    
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
    
    return X_train_processed, X_test_processed, scaler

# Train XGBoost model
def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost classifier with extensive hyperparameter tuning
    """
    print("Training XGBoost classifier...")
    
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
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    return random_search.best_estimator_, random_search

# Enhanced model saving functionality
def save_model_and_metadata(model, feature_names, scaler=None, model_dir='Proctor/Models'):
    """
    Save the trained model, scaler, and associated metadata
    """
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save paths
    model_path = os.path.join(model_dir, f'xgboost_cheating_model_{timestamp}.pkl')
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
        'model_type': 'XGBoost',
        'training_timestamp': timestamp,
        'model_params': model.get_params(),
        'feature_count': len(feature_names),
        'scaler_used': scaler is not None,
        'scaler_path': scaler_path
    }
    
    joblib.dump(metadata, metadata_path)
    print(f"Model metadata saved to: {metadata_path}")
    
    return model_path, metadata_path, scaler_path

def load_model_and_metadata(model_path):
    """
    Load a saved model, scaler, and metadata
    """
    # Load the model
    model = joblib.load(model_path)
    
    # Try to load metadata and scaler
    try:
        # Extract timestamp from model path
        timestamp = model_path.split('_')[-1].replace('.pkl', '')
        metadata_path = model_path.replace(f'xgboost_cheating_model_{timestamp}.pkl', 
                                         f'model_metadata_{timestamp}.pkl')
        metadata = joblib.load(metadata_path)
        
        # Load scaler if it was used
        scaler = None
        if metadata.get('scaler_used', False):
            scaler_path = metadata.get('scaler_path')
            if scaler_path and os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                print(f"Scaler loaded successfully")
            else:
                # Fallback: try to construct scaler path
                scaler_path = model_path.replace(f'xgboost_cheating_model_{timestamp}.pkl', 
                                               f'scaler_{timestamp}.pkl')
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
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

# Evaluate model performance
def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names):
    """
    Comprehensive model evaluation
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Training performance
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"Training AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Classification reports
    print("\nTraining Classification Report:")
    print(classification_report(y_train, y_train_pred))
    
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion matrices
    print("\nTest Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return feature_importance, y_test_proba

# Visualization functions
def plot_results(feature_importance, y_test, y_test_proba):
    """
    Create visualizations for model results
    """
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
    cm = confusion_matrix(y_test, (y_test_proba > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.show()

# Main execution function
def main(csv_file_path, use_scaling=True):
    """
    Main function to run the complete pipeline
    """
    try:
        # Load and prepare data
        X, y, df = load_and_prepare_data(csv_file_path)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Preprocess features (including scaling if requested)
        X_train_processed, X_test_processed, scaler = preprocess_features(
            X_train, X_test, use_scaling=use_scaling
        )
        
        if use_scaling:
            print("✓ Applied feature scaling")
        else:
            print("✓ No scaling applied")
        
        # Train model
        model, grid_search = train_xgboost(X_train_processed, y_train, X_test_processed, y_test)
        
        # Evaluate model
        feature_importance, y_test_proba = evaluate_model(
            model, X_train_processed, y_train, X_test_processed, y_test, X.columns
        )
        
        # Save model, scaler, and metadata
        model_path, metadata_path, scaler_path = save_model_and_metadata(
            model, X.columns, scaler
        )
        
        # Create visualizations
        plot_results(feature_importance, y_test, y_test_proba)
        
        # Demonstrate predictions
        demonstrate_prediction(model_path)
        
        return model, feature_importance, grid_search, model_path, scaler
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None, None, None, None

# Example usage
if __name__ == "__main__":
    csv_file_path = 'Proctor/Datasets/training_proctor_results.csv'
    
    print("XGBoost Cheating Detection Classifier")
    print("="*50)
    
    # Run the complete pipeline with scaling
    model, feature_importance, grid_search, model_path, scaler = main(
        csv_file_path, use_scaling=True  # Set to False if you don't want scaling
    )
    
    if model is not None:
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Model saved at: {model_path}")
        
        print("\nTo load and use the model later:")
        print(f"from static_trainer import load_model_and_metadata, predict_cheating")
        print(f"model, metadata, scaler = load_model_and_metadata('{model_path}')")
        print(f"predictions, probabilities = predict_cheating(model, new_data, metadata=metadata, scaler=scaler)")
        
        # Example of using the model for future predictions
        print("\n" + "="*30)
        print("TESTING MODEL LOADING...")
        
        # Test loading and prediction
        loaded_model, loaded_metadata, loaded_scaler = load_model_and_metadata(model_path)
        print("✓ Model, metadata, and scaler loaded successfully!")