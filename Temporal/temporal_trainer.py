import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import sys
import os
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Temporal.temporal_models import LSTMModel, GRUModel

class CustomScaler:
    """Simple scaler that standardizes features"""
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self
        
    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler not fitted. Call fit before using transform.")
        return (X - self.mean_) / self.scale_
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)
        
    def initialize(self, mean, scale):
        """Initialize the scaler with provided mean and scale values"""
        self.mean_ = np.array(mean)
        self.scale_ = np.array(scale)
        return self

def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    """Custom implementation of train_test_split"""
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
        
    # Get number of samples
    n_samples = len(X)
    # Calculate number of test samples
    n_test = int(n_samples * test_size)
    # Create shuffled indices
    indices = np.random.permutation(n_samples)
    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def custom_confusion_matrix(y_true, y_pred):
    """Custom implementation of confusion_matrix with label-to-index mapping"""
    # Get unique classes and create mapping
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    
    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill confusion matrix using mapped indices
    for i in range(len(y_true)):
        true_idx = class_to_index[y_true[i]]
        pred_idx = class_to_index[y_pred[i]]
        cm[true_idx, pred_idx] += 1
        
    return cm

def custom_classification_report(y_true, y_pred):
    """Custom implementation of classification_report"""
    # Get unique classes
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    # Initialize metrics
    precision = {}
    recall = {}
    f1_score = {}
    support = {}
    
    # Calculate metrics for each class
    for cls in classes:
        # True positives
        tp = np.sum((y_true == cls) & (y_pred == cls))
        # False positives
        fp = np.sum((y_true != cls) & (y_pred == cls))
        # False negatives
        fn = np.sum((y_true == cls) & (y_pred != cls))
        # True negatives
        tn = np.sum((y_true != cls) & (y_pred != cls))
        
        # Calculate precision, recall, f1
        precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[cls] = 2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls]) if (precision[cls] + recall[cls]) > 0 else 0
        
        # Calculate support (number of samples of this class)
        support[cls] = np.sum(y_true == cls)
    
    # Calculate weighted averages
    total_support = sum(support.values())
    avg_precision = sum(precision[cls] * support[cls] for cls in classes) / total_support if total_support > 0 else 0
    avg_recall = sum(recall[cls] * support[cls] for cls in classes) / total_support if total_support > 0 else 0
    avg_f1 = sum(f1_score[cls] * support[cls] for cls in classes) / total_support if total_support > 0 else 0
    
    # Create report string
    report = "Classification Report:\n"
    report += f"{'Class':>10}{'Precision':>12}{'Recall':>10}{'F1-Score':>12}{'Support':>10}\n"
    report += "-" * 52 + "\n"
    
    for cls in classes:
        report += f"{int(cls):>10}{precision[cls]:>12.4f}{recall[cls]:>10.4f}{f1_score[cls]:>12.4f}{support[cls]:>10}\n"
    
    report += "-" * 52 + "\n"
    report += f"{'avg/total':>10}{avg_precision:>12.4f}{avg_recall:>10.4f}{avg_f1:>12.4f}{total_support:>10}\n"
    
    # Calculate accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    report += f"\nAccuracy: {accuracy:.4f}\n"
    
    return report

class SequenceDataset(Dataset):
    """PyTorch Dataset for sequential proctoring data"""
    
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

class TemporalProctor:
    def __init__(self, window_size=10, overlap=4, model_type='lstm', device=None, stride=None):
        """
        Initialize the temporal proctor
        
        Args:
            window_size: Number of previous frames to consider
            overlap: Number of frames to overlap between sequences.
            model_type: 'lstm' or 'gru'
            device: 'cuda' or 'cpu'
        """
        self.window_size = window_size
        print(f"Window size: {self.window_size}")
        self.overlap = overlap
        # Prefer explicit stride if given; else derive from overlap
        self.step = stride if stride is not None else (self.window_size - self.overlap)
        if self.step <= 0:
            raise ValueError("Overlap must be smaller than window_size.")
        self.model_type = model_type.lower()
        self.model = None
        self.scaler = CustomScaler()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        # Keep track of feature order and threshold picked on validation
        self.feature_cols = None
        self.best_threshold = 0.5
        # Store tuned hparams and search score when using hyperparameter search
        self.hparams = None
        self.search_score = None
        
    def load_data(self, csv_path):
        """Load and preprocess the dataset"""
        # Load the dataset
        df = pd.read_csv(csv_path)

        # Sort by timestamp (just to be sure)
        df = df.sort_values('timestamp')

        # Calculate time differences for potential weighting
        df['time_diff'] = df['timestamp'].diff()

        # Fix: Use direct assignment instead of chained inplace operation
        df['time_diff'] = df['time_diff'].fillna(0)
        # Alternative fix: Use fillna with inplace on the entire DataFrame
        # df.fillna({'time_diff': 0}, inplace=True)

        return df
    
    def create_sequences(self, df, fit_scaler=False):
        """Create sequences for temporal modeling, with optional scaler fitting."""
        # Define the exact feature order (avoid timestamp; use time_diff instead)
        if self.feature_cols is None:
            candidate_cols = [
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
                'time_diff'
            ]
            self.feature_cols = [col for col in candidate_cols if col in df.columns]
        print(f"Temporal training feature columns (used for model): {self.feature_cols}")

        # Ensure all needed columns exist; fill missing with 0 before scaling, they will be near 0 after z-score if using means later
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        data = df[self.feature_cols].values
        target = df['is_cheating'].values

        if fit_scaler:
            data_scaled = self.scaler.fit_transform(data)
        else:
            data_scaled = self.scaler.transform(data)

        # Create sequences with overlap/stride
        X, y = [], []
        for i in range(0, len(data_scaled) - self.window_size + 1, self.step):
            X.append(data_scaled[i:i + self.window_size])
            y.append(target[i + self.window_size - 1])

        return np.array(X), np.array(y).reshape(-1, 1)
    
    
    def build_model(self, input_size):
        """Build the sequential model"""
        if self.model_type == 'lstm':
            model = LSTMModel(input_size)
        else:  # GRU
            model = GRUModel(input_size)
        
        model.to(self.device)
        # Remember input size for inference-time checks
        self.input_size = input_size
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001, weight_decay=1e-4):
        """Train the model"""
        # Get input size from data
        input_size = X_train.shape[2]
        print(f"Training data distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"Class {u}: {c} samples ({c/len(y_train)*100:.1f}%)")
        if len(unique) == 2:
            minority_ratio = min(counts) / max(counts)
            print(f"Minority class ratio: {minority_ratio:.3f}")
            if minority_ratio < 0.1:
                print("⚠️ WARNING: Severe class imbalance detected!")
                print("Consider using class weights or different sampling")
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model(input_size)
            
        # Create data loaders
        train_dataset = SequenceDataset(X_train, y_train)
        val_dataset = SequenceDataset(X_val, y_val)
        
        # Handle class imbalance via sampler
        y_train_flat = y_train.flatten()
        class_sample_counts = np.bincount(y_train_flat.astype(int)) if len(np.unique(y_train_flat)) == 2 else None
        sampler = None
        if class_sample_counts is not None and len(class_sample_counts) == 2 and all(class_sample_counts > 0):
            class_weights = 1.0 / class_sample_counts
            sample_weights = class_weights[y_train_flat.astype(int)]
            sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
            print(f"Using WeightedRandomSampler with class counts: {class_sample_counts.tolist()}")

        pin_mem = True if self.device == 'cuda' else False
        num_workers = max(0, min(4, (os.cpu_count() or 1) - 1))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, pin_memory=pin_mem, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=pin_mem, num_workers=num_workers)
        
        # 🔧 FIX: Loss function and optimizer with proper device handling
        if len(unique) == 2:
            # Compute robust pos_weight = neg/pos
            pos = float((y_train_flat == 1).sum())
            neg = float((y_train_flat == 0).sum())
            pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"Using weighted loss with pos_weight: {pos_weight.item():.3f}")
            
            # 🔧 Also need to modify your model to NOT use sigmoid in final layer
            # Since BCEWithLogitsLoss applies sigmoid internally
            print("⚠️ Make sure your model's final layer does NOT use sigmoid activation!")
            
        else:
            criterion = nn.BCELoss()
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        use_amp = self.device == 'cuda'
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                # 🔧 Ensure all tensors are on the same device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item() * inputs.size(0)
                
                # 🔧 Fix prediction logic based on loss function
                if len(unique) == 2 and isinstance(criterion, nn.BCEWithLogitsLoss):
                    # For BCEWithLogitsLoss, apply sigmoid to get probabilities
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    # For BCELoss, outputs are already probabilities
                    predicted = (outputs > 0.5).float()
                    
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
            train_loss /= len(train_dataset)
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # 🔧 Ensure all tensors are on the same device
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    
                    # 🔧 Fix prediction logic for validation too
                    if len(unique) == 2 and isinstance(criterion, nn.BCEWithLogitsLoss):
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                    else:
                        predicted = (outputs > 0.5).float()
                        
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
            val_loss /= len(val_dataset)
            val_acc = val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
            
            # LR scheduler step
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load the best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            
        # After training, tune decision threshold on validation set (maximize F1)
        try:
            with torch.no_grad():
                self.model.eval()
                val_probs = []
                val_labels = []
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
                    val_probs.append(probs)
                    val_labels.append(labels.numpy().flatten())
                val_probs = np.concatenate(val_probs)
                val_labels = np.concatenate(val_labels)
                best_f1 = -1
                best_thr = 0.5
                for thr in np.linspace(0.1, 0.9, 81):
                    preds = (val_probs >= thr).astype(int)
                    tp = np.sum((preds == 1) & (val_labels == 1))
                    fp = np.sum((preds == 1) & (val_labels == 0))
                    fn = np.sum((preds == 0) & (val_labels == 1))
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thr = float(thr)
                self.best_threshold = best_thr
                print(f"Picked best threshold on validation: {self.best_threshold:.3f} (F1={best_f1:.3f})")
        except Exception as e:
            print(f"Threshold tuning skipped due to error: {e}")

        return history
    
    def evaluate(self, X_test, y_test, batch_size=32, threshold=None):
        """Evaluate the model"""
        if self.model is None:
            print("No model to evaluate. Train a model first.")
            return None, None
        
        test_dataset = SequenceDataset(X_test, y_test)
        pin_mem = True if self.device == 'cuda' else False
        num_workers = max(0, min(4, (os.cpu_count() or 1) - 1))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=pin_mem, num_workers=num_workers)
        
        self.model.eval()
        y_true = []
        y_pred_proba = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)  # 🔧 Ensure on correct device
                outputs = self.model(inputs)
                
                # 🔧 Apply sigmoid to get probabilities from logits
                proba = torch.sigmoid(outputs)
                
                y_true.extend(labels.numpy())
                y_pred_proba.extend(proba.cpu().numpy())  # 🔧 Move to CPU for numpy
        
        y_true = np.array(y_true).flatten()  # flatten to 1D
        y_pred_proba = np.array(y_pred_proba).flatten()
        thr = self.best_threshold if threshold is None else threshold
        y_pred = (y_pred_proba >= thr).astype(int)
        
        # Debug: print unique values to check for label/prediction issues
        print(f"Unique y_true: {np.unique(y_true)}")
        print(f"Unique y_pred: {np.unique(y_pred)}")
        
        print("\nClassification Report:")
        print(custom_classification_report(y_true, y_pred))
        
        print("\nConfusion Matrix:")
        cm = custom_confusion_matrix(y_true, y_pred)
        print(cm)
        try:
            auroc = roc_auc_score(y_true, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            # Compute area under PR curve via trapezoid
            auprc = np.trapz(precision[::-1], recall[::-1])
            print(f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Threshold: {thr:.3f}")
        except Exception:
            pass
        
        return y_pred_proba, y_pred

    def predict_sequence(self, sequence, batch_size=32):
        """
        Predict on a single sequence or batch of sequences
        """
        if self.model is None:
            print("No model to use for prediction. Train or load a model first.")
            return None
            
        # Check if we have a single sequence or a batch
        if len(sequence.shape) == 2:
            # Single sequence, add batch dimension
            sequence = np.expand_dims(sequence, axis=0)
            
        # Convert to tensor and move to device
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(sequence_tensor)
            # 🔧 Apply sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(output)
            
        return probabilities.cpu().numpy()  # 🔧 Move to CPU for numpy
    
    def predict_with_sliding_window(self, features, deployment_window_size=None):
        """
        Make predictions using a sliding window approach for deployment
        
        Args:
            features: numpy array of shape (seq_len, features) - the feature sequence to predict on
            deployment_window_size: Size of window to use for deployment (defaults to self.window_size)
            
        Returns:
            Array of cheating probabilities for each step after initial window
        """
        if self.model is None:
            print("No model to use for prediction. Train or load a model first.")
            return None
            
        # Use the model's window size if deployment_window_size not specified
        window_size = deployment_window_size or self.window_size
        
        # Prepare features - apply scaling using the fitted scaler
        if self.scaler.mean_ is None:
            print("Scaler not fitted. Please train the model first or fit the scaler manually.")
            return None
            
        features_scaled = self.scaler.transform(features)
        
        # We need at least window_size frames to make a prediction
        if len(features_scaled) < window_size:
            print(f"Not enough features. Need at least {window_size} frames.")
            return None
            
        # Create sequences using sliding window
        sequences = []
        for i in range(len(features_scaled) - window_size + 1):
            sequences.append(features_scaled[i:i + window_size])
            
        sequences = np.array(sequences)
        
        # Make predictions
        predictions = self.predict_sequence(sequences)
        
        return predictions
    
    def make_realtime_prediction(self, feature_buffer):
        """
        Make a real-time prediction using the current feature buffer
        
        Args:
            feature_buffer: List or array of the latest features
            
        Returns:
            Probability of cheating for the current state
        """
        # Ensure we have enough frames
        if len(feature_buffer) < self.window_size:
            print(f"Not enough features in buffer. Need at least {self.window_size} frames, got {len(feature_buffer)}")
            return None
        
        # Take the last window_size frames
        recent_features = feature_buffer[-self.window_size:]
        
        # Convert to numpy array and ensure proper shape
        recent_features = np.array(recent_features)
        print(f"DEBUG: Feature buffer shape before scaling: {recent_features.shape}")
        
        # Check if scaler is properly initialized
        if self.scaler.mean_ is None or self.scaler.scale_ is None:
            print("ERROR: Scaler not properly initialized! Model will output garbage.")
            return None
        
        # Use all features (including timestamp) for scaling and prediction
        features_for_scaling = recent_features  # <-- FIXED: do not drop any columns
        
        # Verify that the number of features matches the model's input size
        expected_features = getattr(self, 'input_size', features_for_scaling.shape[1])
        if features_for_scaling.shape[1] != expected_features:
            print(f"ERROR: Feature mismatch. Model expects {expected_features} features, but got {features_for_scaling.shape[1]}.")
            return None
        
        # Scale features
        try:
            scaled_features = self.scaler.transform(features_for_scaling)
            print(f"DEBUG: Scaled features shape: {features_for_scaling.shape}")
        except Exception as e:
            print(f"ERROR in scaling: {e}")
            return None
        
        # Reshape for model input (add batch dimension)
        model_input = np.expand_dims(scaled_features, axis=0)  # <-- FIXED: use scaled features
        print(f"DEBUG: Model input shape: {model_input.shape}")
        
        # Predict
        try:
            prediction = self.predict_sequence(model_input)
            print(f"DEBUG: Raw prediction: {prediction}")
            return prediction[0][0] if prediction is not None else None
        except Exception as e:
            print(f"ERROR in prediction: {e}")
            return None
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'])
        plt.plot(history['val_acc'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.show()
            
    def save_model(self, path='Models/temporal_proctor_model.pt'):
        """Save the model and scaler"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'window_size': self.window_size,
                'model_type': self.model_type,
                'scaler_mean': self.scaler.mean_,
                'scaler_scale': self.scaler.scale_,
                'feature_cols': self.feature_cols,
                'step': self.step,
                'best_threshold': self.best_threshold,
                'hparams': self.hparams,
                'search_score': self.search_score
            }, path)
            print(f"Model and scaler saved to {path}")
        else:
            print("No model to save. Train a model first.")

    def load_model(self, path='Models/temporal_proctor_model.pt', input_size=None):
        """Load a saved model and scaler"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.window_size = checkpoint.get('window_size', self.window_size)
        self.model_type = checkpoint.get('model_type', self.model_type)
        self.feature_cols = checkpoint.get('feature_cols', self.feature_cols)
        self.step = checkpoint.get('step', self.step)
        self.best_threshold = checkpoint.get('best_threshold', self.best_threshold)
        self.hparams = checkpoint.get('hparams', None)
        self.search_score = checkpoint.get('search_score', None)
        
        # Load scaler parameters
        if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
            self.scaler.initialize(checkpoint['scaler_mean'], checkpoint['scaler_scale'])
            print("Scaler parameters loaded successfully")
        else:
            print("Warning: No scaler parameters found in saved model")
        
        if input_size is None:
            raise ValueError("Please provide input_size when loading a model")
            
        self.model = self.build_model(input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"Model loaded from {path}")
    
    def plot_and_save_comprehensive_results(self, history, y_true, y_pred_proba, y_pred, save_dir='plots', model_name="TemporalModel"):
        """
        Create and save comprehensive visualizations for training and test results.
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.ioff()
        plt.figure(figsize=(20, 15))

        # 1. Accuracy
        plt.subplot(2, 3, 1)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        # 2. Loss
        plt.subplot(2, 3, 2)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        # 3. ROC Curve
        plt.subplot(2, 3, 3)
        if y_true is not None and y_pred_proba is not None:
            try:
                y_true_flat = np.array(y_true).flatten()
                # ROC AUC is undefined if only one class is present
                if np.unique(y_true_flat).size < 2:
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (skipped: single-class y_true)')
                    plt.legend()
                else:
                    fpr, tpr, _ = roc_curve(y_true_flat, y_pred_proba)
                    auc_score = roc_auc_score(y_true_flat, y_pred_proba)
                    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend()
            except Exception as e:
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.title(f'ROC Curve (error: {str(e)[:40]}...)')
                plt.legend()

        # 4. Precision-Recall Curve
        plt.subplot(2, 3, 4)
        if y_true is not None and y_pred_proba is not None:
            try:
                y_true_flat = np.array(y_true).flatten()
                pos_count = int(np.sum(y_true_flat == 1))
                if pos_count == 0:
                    plt.title('PR Curve (skipped: no positive class)')
                else:
                    precision, recall, _ = precision_recall_curve(y_true_flat, y_pred_proba)
                    plt.plot(recall, precision)
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title('Precision-Recall Curve')
            except Exception as e:
                plt.title(f'PR Curve (error: {str(e)[:40]}...)')

        # 5. Prediction Probability Distribution
        plt.subplot(2, 3, 5)
        if y_true is not None and y_pred_proba is not None:
            y_true_flat = np.array(y_true).flatten()
            mask_neg = (y_true_flat == 0)
            mask_pos = (y_true_flat == 1)
            if np.any(mask_neg):
                plt.hist(np.array(y_pred_proba)[mask_neg], bins=30, alpha=0.7, label='Non-cheating', density=True)
            if np.any(mask_pos):
                plt.hist(np.array(y_pred_proba)[mask_pos], bins=30, alpha=0.7, label='Cheating', density=True)
            plt.xlabel('Predicted Probability')
            plt.ylabel('Density')
            plt.title('Prediction Probability Distribution')
            plt.legend()

        # 6. Confusion Matrix
        plt.subplot(2, 3, 6)
        if y_true is not None and y_pred is not None:
            cm = custom_confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')

        plt.tight_layout()
        plot_path = os.path.join(save_dir, f'comprehensive_results_{model_name}_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Plots saved to: {plot_path}")
        return plot_path

# --------- Utilities for reproducibility and search ---------
def set_seeds(seed: int = 42):
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    # Trapz expects x increasing; recall is increasing, precision corresponds
    return float(np.trapz(precision, recall))

def grid_iter(param_grid: dict):
    keys = list(param_grid.keys())
    for values in itertools.product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, values))

def run_hparam_search(train_df: pd.DataFrame, search_params: dict, search_epochs: int = 20, seed: int = 42):
    """Run a lightweight hyperparameter search. Returns (best_proctor, best_params, best_score)."""
    set_seeds(seed)
    best = {
        'score': -1.0,
        'params': None,
        'proctor': None
    }
    trial = 0
    for params in grid_iter(search_params):
        trial += 1
        w = params['window_size']
        s = params['stride']
        bs = params['batch_size']
        lr = params['lr']
        wd = params['weight_decay']
        overlap = max(1, w - s)

        print(f"\n[Search] Trial {trial}: window_size={w}, stride={s}, lr={lr}, wd={wd}, batch_size={bs}")
        proctor = TemporalProctor(window_size=w, overlap=overlap, model_type='lstm', stride=s)
        # Build sequences and split by time (no leakage)
        X_all, y_all = proctor.create_sequences(train_df.copy(), fit_scaler=True)
        if len(X_all) < 5:
            print("Too few sequences for this config; skipping.")
            continue
        split_idx = int(len(X_all) * 0.8)
        X_tr, y_tr = X_all[:split_idx], y_all[:split_idx]
        X_va, y_va = X_all[split_idx:], y_all[split_idx:]

        # Train
        proctor.train(X_tr, y_tr, X_va, y_va, epochs=search_epochs, batch_size=bs, lr=lr, weight_decay=wd)

        # Evaluate on validation for selection
        y_prob, y_pred = proctor.evaluate(X_va, y_va, batch_size=bs, threshold=proctor.best_threshold)
        if y_prob is None:
            continue
        score = compute_auprc(y_va.flatten(), np.array(y_prob).flatten())
        print(f"[Search] Validation AUPRC: {score:.4f}")

        if score > best['score']:
            # attach tuned params and score to the proctor for later persistence
            proctor.hparams = params
            proctor.search_score = score
            best = {'score': score, 'params': params, 'proctor': proctor}

    return best['proctor'], best['params'], best['score']

# Main execution code
if __name__ == "__main__":
    set_seeds(42)
    # Initialize the temporal proctor
    proctor = TemporalProctor(window_size=150, overlap=140, model_type='lstm', stride=10)
    
    # Load and combine training data
    print("Loading training data...")
    train_files = [
        r'New_Processed_Csv/Train_Video1_processed.csv',
        r'New_Processed_Csv/Train_Video2_processed.csv'
    ]
    
    train_dfs = []
    for file_path in train_files:
        if os.path.exists(file_path):
            df = proctor.load_data(file_path)
            print(f"Loaded {file_path}: {df.shape}")
            train_dfs.append(df)
        else:
            print(f"Warning: {file_path} not found!")
    
    if not train_dfs:
        print("No training files found!")
        exit(1)
    
    # Combine training data
    combined_train_df = pd.concat(train_dfs, ignore_index=True)
    combined_train_df = combined_train_df.sort_values('timestamp')
    print(f"Combined training data shape: {combined_train_df.shape}")
    
    # Optional: run a small hyperparameter search to pick a good config
    run_search = True
    if run_search:
        param_grid = {
            'window_size': [96, 128, 150],
            'stride': [5, 10],
            'lr': [1e-3, 3e-4],
            'weight_decay': [1e-4, 1e-3],
            'batch_size': [16, 32]
        }
        best_proctor, best_params, best_score = run_hparam_search(combined_train_df, param_grid, search_epochs=20, seed=42)
        if best_proctor is not None:
            print(f"\n[Search] Best params: {best_params} with AUPRC={best_score:.4f}")
            proctor = best_proctor
            # Recreate sequences for final training using best window/stride and re-fit scaler
            X_full, y_full = proctor.create_sequences(combined_train_df, fit_scaler=True)
            n = len(X_full)
            split_idx = int(n * 0.8)
            X_train, y_train = X_full[:split_idx], y_full[:split_idx]
            X_val, y_val = X_full[split_idx:], y_full[split_idx:]
            # Optionally increase epochs for final training
            history = proctor.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=best_params['batch_size'], lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        else:
            print("[Search] No valid configuration found, proceeding with default config.")
            X_full, y_full = proctor.create_sequences(combined_train_df, fit_scaler=True)
            n = len(X_full)
            split_idx = int(n * 0.8)
            X_train, y_train = X_full[:split_idx], y_full[:split_idx]
            X_val, y_val = X_full[split_idx:], y_full[split_idx:]
            history = proctor.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    else:
        # Create sequences from training data; fit scaler here
        X_full, y_full = proctor.create_sequences(combined_train_df, fit_scaler=True)
        # Leakage-safe split: split by time into blocks of sequences (80% earliest timestamps for train)
        n = len(X_full)
        split_idx = int(n * 0.8)
        X_train, y_train = X_full[:split_idx], y_full[:split_idx]
        X_val, y_val = X_full[split_idx:], y_full[split_idx:]
        history = proctor.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Load and test on test data
    print("\nLoading test data...")
    test_files = [
        r'New_Processed_Csv/Test_Video1_processed.csv',
        r'New_Processed_Csv/Test_Video2_processed.csv'
    ]
    
    all_test_results = []
    
    # Use the feature order discovered during training
    fixed_feature_cols = proctor.feature_cols if proctor.feature_cols is not None else []
    
    for i, file_path in enumerate(test_files, 1):
        if os.path.exists(file_path):
            print(f"\n--- Testing on {os.path.basename(file_path)} ---")
            test_df = proctor.load_data(file_path)
            print(f"Test data shape: {test_df.shape}")

            # Use the same feature columns and order as training
            test_feature_cols = [col for col in fixed_feature_cols if col in test_df.columns]
            missing_cols = set(fixed_feature_cols) - set(test_feature_cols)
            if missing_cols:
                print(f"Warning: Test data missing columns: {missing_cols}")
            # Add missing columns with training means for stability
            scaler_means = proctor.scaler.mean_ if hasattr(proctor.scaler, 'mean_') and proctor.scaler.mean_ is not None else np.zeros(len(fixed_feature_cols))
            train_means = {col: (scaler_means[idx] if idx < len(scaler_means) else 0.0)
                           for idx, col in enumerate(fixed_feature_cols)}
            for col in missing_cols:
                test_df[col] = train_means.get(col, 0.0)
            # Ensure correct order
            test_data = test_df[fixed_feature_cols].values
            print(f"Temporal test feature columns: {fixed_feature_cols}")
            test_target = test_df['is_cheating'].values

            # Use the fitted scaler to transform test data
            test_data_scaled = proctor.scaler.transform(test_data)
            
            # Create test sequences
            X_test, y_test = [], []
            for j in range(0, len(test_data_scaled) - proctor.window_size + 1, proctor.step):
                X_test.append(test_data_scaled[j:j + proctor.window_size])
                y_test.append(test_target[j + proctor.window_size - 1])
            
            X_test, y_test = np.array(X_test), np.array(y_test).reshape(-1, 1)
            print(f"Test sequences shape: {X_test.shape}")
            
            # Evaluate on this test file
            y_pred_proba, y_pred = proctor.evaluate(X_test, y_test, threshold=proctor.best_threshold)
            
            all_test_results.append({
                'file': os.path.basename(file_path),
                'X_test': X_test,
                'y_test': y_test,
                'y_pred_proba': y_pred_proba,
                'y_pred': y_pred
            })
        else:
            print(f"Warning: {file_path} not found!")
    
    # Combined test evaluation
    if len(all_test_results) > 1:
        print(f"\n--- Combined Test Results ---")
        combined_X_test = np.concatenate([result['X_test'] for result in all_test_results])
        combined_y_test = np.concatenate([result['y_test'] for result in all_test_results])
        
        print(f"Combined test shape: {combined_X_test.shape}")
        y_pred_proba_combined, y_pred_combined = proctor.evaluate(combined_X_test, combined_y_test, threshold=proctor.best_threshold)
    
    # Plot training history
    proctor.plot_training_history(history)
    
    # Save the model
    os.makedirs('Models', exist_ok=True)
    proctor.save_model('Models/temporal_proctor_trained_on_processed.pt')
    
    # Save comprehensive results plot for last test file's results
    if all_test_results and 'y_pred_proba' in all_test_results[-1]:
        # Use last test file's results
        last = all_test_results[-1]
        proctor.plot_and_save_comprehensive_results(
            history,
            last['y_test'].flatten(),
            last['y_pred_proba'].flatten(),
            last['y_pred'].flatten(),
            save_dir='plots',
            model_name="TemporalModel"
        )
    
    print("\nTraining and evaluation completed!")