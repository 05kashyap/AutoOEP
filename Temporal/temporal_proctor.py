import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from temporal_models import LSTMModel, GRUModel
import matplotlib.pyplot as plt
import time

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
    """Custom implementation of confusion_matrix"""
    # Get unique classes
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    
    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill confusion matrix
    for i in range(len(y_true)):
        cm[int(y_true[i]), int(y_pred[i])] += 1
        
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
    def __init__(self, window_size=10, overlap=4, model_type='lstm', device=None):
        """
        Initialize the temporal proctor
        
        Args:
            window_size: Number of previous frames to consider
            overlap: Number of frames to overlap between sequences.
            model_type: 'lstm' or 'gru'
            device: 'cuda' or 'cpu'
        """
        self.window_size = window_size
        self.overlap = overlap
        self.step = self.window_size - self.overlap
        if self.step <= 0:
            raise ValueError("Overlap must be smaller than window_size.")
        self.model_type = model_type.lower()
        self.model = None
        self.scaler = CustomScaler()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
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
    
    def create_sequences(self, df):
        """Create sequences for temporal modeling"""
        # Features we'll use (exclude timestamp and target variable)
        feature_cols = df.columns.tolist()
        feature_cols.remove('timestamp')
        feature_cols.remove('is_cheating')
        
        # Get the data
        data = df[feature_cols].values
        target = df['is_cheating'].values
        
        # Scale the features
        data_scaled = self.scaler.fit_transform(data)
        
        # Create sequences with overlap
        X, y = [], []
        # The step is window_size - overlap
        for i in range(0, len(data_scaled) - self.window_size + 1, self.step):
            X.append(data_scaled[i:i + self.window_size])
            # The label corresponds to the last frame in the window
            y.append(target[i + self.window_size - 1])
        
        return np.array(X), np.array(y).reshape(-1, 1)
    
    
    def build_model(self, input_size):
        """Build the sequential model"""
        if self.model_type == 'lstm':
            model = LSTMModel(input_size)
        else:  # GRU
            model = GRUModel(input_size)
        
        model.to(self.device)
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001):
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
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 🔧 FIX: Loss function and optimizer with proper device handling
        if len(unique) == 2:
            # 🔧 Make sure pos_weight is on the correct device
            pos_weight = torch.tensor([counts[0] / counts[1]], dtype=torch.float32).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"Using weighted loss with pos_weight: {pos_weight.item():.3f}")
            
            # 🔧 Also need to modify your model to NOT use sigmoid in final layer
            # Since BCEWithLogitsLoss applies sigmoid internally
            print("⚠️ Make sure your model's final layer does NOT use sigmoid activation!")
            
        else:
            criterion = nn.BCELoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
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
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
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
                
            train_loss /= len(train_loader.dataset)
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
                    
            val_loss /= len(val_loader.dataset)
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
            
        return history
    
    def evaluate(self, X_test, y_test, batch_size=32):
        """Evaluate the model"""
        if self.model is None:
            print("No model to evaluate. Train a model first.")
            return None, None
        
        test_dataset = SequenceDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
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
        
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        print("\nClassification Report:")
        print(custom_classification_report(y_true, y_pred))
        
        print("\nConfusion Matrix:")
        cm = custom_confusion_matrix(y_true, y_pred)
        print(cm)
        
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
            print("ERROR: Scaler not properly initialized!")
            return None
        
        # The scaler was trained on features without the timestamp.
        features_for_scaling = recent_features[:, 1:]
        
        # Verify that the number of features matches the model's input size
        expected_features = self.model.lstm1.input_size if self.model_type == 'lstm' else self.model.gru1.input_size
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
        model_input = np.expand_dims(features_for_scaling, axis=0)
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
                'scaler_scale': self.scaler.scale_
            }, path)
            print(f"Model and scaler saved to {path}")
        else:
            print("No model to save. Train a model first.")

    def load_model(self, path='Models/temporal_proctor_model.pt', input_size=None):
        """Load a saved model and scaler"""
        checkpoint = torch.load(path, map_location=self.device)
        self.window_size = checkpoint.get('window_size', self.window_size)
        self.model_type = checkpoint.get('model_type', self.model_type)
        
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

# Main execution code
if __name__ == "__main__":
    # Initialize the temporal proctor
    proctor = TemporalProctor(window_size=15, overlap=4, model_type='lstm')
    
    # Load and preprocess the data
    df = proctor.load_data('Datasets/training_proctor_results.csv')
    
    # Create sequences
    X, y = proctor.create_sequences(df)
    
    # Split the data using custom train_test_split
    X_train, X_temp, y_train, y_temp = custom_train_test_split(X, y, test_size=0.8, random_state=42)
    X_val, X_test, y_val, y_test = custom_train_test_split(X_temp, y_temp, test_size=0.66, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Build and train the model
    history = proctor.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # Evaluate the model
    y_pred_proba, y_pred = proctor.evaluate(X_test, y_test)
    
    # Plot training history
    proctor.plot_training_history(history)
    
    # Save the model
    proctor.save_model()