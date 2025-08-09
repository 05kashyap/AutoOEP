"""
Temporal Trainer - Dedicated script for training temporal models
Separate from inference to allow independent training runs
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the advanced models - REQUIRED
try:
    from .temporal_models import LSTMModel, GRUModel
    HAS_ADVANCED_MODELS = True
except ImportError:
    try:
        # Fallback to absolute import
        from temporal_models import LSTMModel, GRUModel
        HAS_ADVANCED_MODELS = True
    except ImportError:
        raise ImportError("PyTorch temporal models (LSTMModel, GRUModel) are required but not available. Please ensure temporal_models.py is present.")

# Import feature extractor and data handler
try:
    from core.feature_extractor import FeatureExtractor
    from core.data_handler import DataHandler
    HAS_CORE_MODULES = True
except ImportError:
    HAS_CORE_MODULES = False
    raise ImportError("Core modules (FeatureExtractor, DataHandler) are required but not available.")

# Custom scaler for compatibility
class SimpleScaler:
    """Simple scaler for temporal features"""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.fitted = False
    
    def fit(self, X):
        """Fit scaler to data"""
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # Avoid division by zero
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform data using fitted parameters"""
        if not self.fitted:
            return np.array(X)  # Return as-is if not fitted
        X = np.array(X)
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """Fit and transform data"""
        return self.fit(X).transform(X)


class TemporalModelTrainer:
    """
    Dedicated trainer for temporal models
    Handles data loading, preprocessing, training, and model saving
    """
    
    def __init__(self, model_type='LSTM', window_size=15, device=None, 
                 input_size=23, hidden_size1=128, hidden_size2=64):
        """
        Initialize temporal model trainer
        
        Args:
            model_type: Type of model ('LSTM' or 'GRU')
            window_size: Number of frames to analyze in sequence
            device: Device preference (auto-detected if None)
            input_size: Number of input features (default 23 for full feature set)
            hidden_size1: First hidden layer size
            hidden_size2: Second hidden layer size
        """
        self.model_type = model_type.upper()
        self.window_size = window_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        
        # Initialize components
        self.model = None
        self.scaler = SimpleScaler()
        self.feature_extractor = FeatureExtractor()
        self.data_handler = DataHandler()
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 50
        self.patience = 10  # Early stopping patience
        
        print(f"Temporal Model Trainer initialized:")
        print(f"  Model: {self.model_type}")
        print(f"  Window Size: {self.window_size}")
        print(f"  Device: {self.device}")
        print(f"  Input Features: {self.input_size}")
    
    def create_model(self):
        """Create new model instance"""
        try:
            if self.model_type == 'LSTM':
                self.model = LSTMModel(self.input_size, self.hidden_size1, self.hidden_size2, 1)
            elif self.model_type == 'GRU':
                self.model = GRUModel(self.input_size, self.hidden_size1, self.hidden_size2, 1)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.model.to(self.device)
            print(f"✅ Created {self.model_type} model")
            
        except Exception as e:
            raise RuntimeError(f"Failed to create model: {e}")
    
    def load_training_data(self, data_dir):
        """
        Load and preprocess training data
        
        Args:
            data_dir: Directory containing training data
            
        Returns:
            Tuple of (sequences, labels) ready for training
        """
        print(f"Loading training data from: {data_dir}")
        
        # Load raw training data
        raw_sequences, raw_labels = self.data_handler.load_training_data(data_dir)
        
        if len(raw_sequences) == 0:
            raise ValueError("No training data found")
        
        print(f"Loaded {len(raw_sequences)} raw sequences")
        
        # Process sequences into proper feature vectors
        processed_sequences = []
        processed_labels = []
        
        for seq, label in zip(raw_sequences, raw_labels):
            try:
                # Convert each frame in sequence to feature vector
                sequence_features = []
                for frame_data in seq:
                    features = self.feature_extractor.extract_features_from_results(frame_data)
                    if len(features) == self.input_size:
                        sequence_features.append(features)
                
                # Only keep sequences with enough frames
                if len(sequence_features) >= self.window_size:
                    # Take sliding windows from the sequence
                    for i in range(len(sequence_features) - self.window_size + 1):
                        window = sequence_features[i:i + self.window_size]
                        processed_sequences.append(window)
                        processed_labels.append(label)
                
            except Exception as e:
                print(f"Warning: Skipping sequence due to error: {e}")
                continue
        
        print(f"Processed into {len(processed_sequences)} training windows")
        
        if len(processed_sequences) == 0:
            raise ValueError("No valid training sequences after processing")
        
        return np.array(processed_sequences), np.array(processed_labels)
    
    def prepare_data_loaders(self, sequences, labels, test_size=0.2, validation_size=0.1):
        """
        Prepare data loaders for training
        
        Args:
            sequences: Training sequences
            labels: Training labels
            test_size: Fraction for test set
            validation_size: Fraction for validation set
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Fit scaler on training data
        X_train_flat = X_train.reshape(-1, self.input_size)
        self.scaler.fit(X_train_flat)
        
        # Scale all data
        X_train_scaled = self._scale_sequences(X_train)
        X_val_scaled = self._scale_sequences(X_val)
        X_test_scaled = self._scale_sequences(X_test)
        
        # Convert to PyTorch tensors
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.FloatTensor(y_test)
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def _scale_sequences(self, sequences):
        """Scale sequence data using fitted scaler"""
        scaled_sequences = []
        for seq in sequences:
            scaled_seq = self.scaler.transform(seq)
            scaled_sequences.append(scaled_seq)
        return np.array(scaled_sequences)
    
    def train_model(self, train_loader, val_loader):
        """
        Train the temporal model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary with training history
        """
        if self.model is None:
            raise RuntimeError("Model not created. Call create_model() first.")
        
        # Setup training
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predictions = torch.sigmoid(outputs) > 0.5
                train_correct += (predictions == batch_y).sum().item()
                train_total += batch_y.size(0)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    predictions = torch.sigmoid(outputs) > 0.5
                    val_correct += (predictions == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}] - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping check
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print("✅ Training completed!")
        return history
    
    def evaluate_model(self, test_loader):
        """
        Evaluate the trained model on test data
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Train model first.")
        
        self.model.eval()
        test_loss = 0.0
        all_predictions = []
        all_labels = []
        
        criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                
                predictions = torch.sigmoid(outputs) > 0.5
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        print(f"\n=== Test Results ===")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_predictions))
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def save_model(self, save_path):
        """
        Save the trained model and scaler
        
        Args:
            save_path: Path to save the model
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train model first.")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'model_type': self.model_type,
                'input_size': self.input_size,
                'hidden_size1': self.hidden_size1,
                'hidden_size2': self.hidden_size2,
                'window_size': self.window_size
            },
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'training_config': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs
            }
        }
        
        torch.save(save_dict, save_path)
        print(f"✅ Model saved to: {save_path}")
    
    def train_complete_pipeline(self, data_dir, save_path):
        """
        Complete training pipeline from data loading to model saving
        
        Args:
            data_dir: Directory containing training data
            save_path: Path to save the trained model
            
        Returns:
            Dictionary with training results
        """
        try:
            # 1. Load and prepare data
            sequences, labels = self.load_training_data(data_dir)
            train_loader, val_loader, test_loader = self.prepare_data_loaders(sequences, labels)
            
            # 2. Create and train model
            self.create_model()
            history = self.train_model(train_loader, val_loader)
            
            # 3. Evaluate model
            test_results = self.evaluate_model(test_loader)
            
            # 4. Save model
            self.save_model(save_path)
            
            results = {
                'training_history': history,
                'test_results': test_results,
                'model_path': save_path
            }
            
            print(f"\n✅ Complete training pipeline finished successfully!")
            print(f"Model saved to: {save_path}")
            print(f"Final test accuracy: {test_results['test_accuracy']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Training pipeline failed: {e}")
            raise


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Temporal Model for Video Proctoring')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--model_type', type=str, default='LSTM', choices=['LSTM', 'GRU'], help='Type of model to train')
    parser.add_argument('--window_size', type=int, default=15, help='Sequence window size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='models/temporal_model.pth', help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = TemporalModelTrainer(
        model_type=args.model_type,
        window_size=args.window_size
    )
    
    # Set training parameters
    trainer.num_epochs = args.epochs
    trainer.batch_size = args.batch_size
    trainer.learning_rate = args.learning_rate
    
    # Run complete training pipeline
    try:
        results = trainer.train_complete_pipeline(args.data_dir, args.save_path)
        print("\n🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
