"""
Enhanced Temporal Trainer - Combines PyTorch model loading with simplified interface
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the advanced models if available
try:
    from .temporal_models import LSTMModel, GRUModel
    HAS_ADVANCED_MODELS = True
except ImportError:
    try:
        # Fallback to absolute import
        from temporal_models import LSTMModel, GRUModel
        HAS_ADVANCED_MODELS = True
    except ImportError:
        HAS_ADVANCED_MODELS = False
        print("Warning: Advanced temporal models not available, using fallback")

# Import feature extractor for proper feature processing
try:
    from core.feature_extractor import FeatureExtractor
    HAS_FEATURE_EXTRACTOR = True
except ImportError:
    HAS_FEATURE_EXTRACTOR = False
    print("Warning: FeatureExtractor not available, using simple features")

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


class TemporalTrainerEnhanced:
    """
    Enhanced temporal trainer that can use both PyTorch models and simple rules
    Maintains compatibility with the existing interface while supporting advanced models
    """
    
    def __init__(self, window_size=15, device=None):
        """
        Initialize enhanced temporal trainer
        
        Args:
            window_size: Number of frames to analyze in sequence
            device: Device preference (auto-detected if None)
        """
        self.window_size = window_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_history = []
        self.threshold = 0.5
        
        # PyTorch model components
        self.model = None
        self.scaler = SimpleScaler()
        self.model_type = None
        self.input_size = 23  # Full feature set expected by trained models
        
        # Fallback to rule-based if no models available
        self.use_pytorch_model = False
        
        # Initialize feature extractor for proper feature processing
        if HAS_FEATURE_EXTRACTOR:
            self.feature_extractor = FeatureExtractor()
        else:
            self.feature_extractor = None
        
        print(f"Enhanced TemporalTrainer initialized with window_size={window_size}, device={self.device}")
        print(f"Advanced models available: {HAS_ADVANCED_MODELS}")
        
        # Try to auto-load models from default locations
        self._try_auto_load_models()
    
    def _try_auto_load_models(self):
        """Try to automatically load PyTorch models from default locations"""
        default_paths = [
            'Inputs/Models',
            'Models',
            '../Models',
            '../../Models'
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                try:
                    self.load_models(path)
                    if self.use_pytorch_model:
                        print(f"✅ Auto-loaded PyTorch models from {path}")
                        break
                except Exception as e:
                    print(f"⚠️ Failed to auto-load from {path}: {e}")
                    continue
    
    def add_frame_features(self, features_dict):
        """
        Add features from current frame to temporal sequence
        
        Args:
            features_dict: Dictionary containing frame analysis results
        """
        # Extract key features for temporal analysis
        feature_vector = self._extract_feature_vector(features_dict)
        
        # Add to history
        self.feature_history.append(feature_vector)
        
        # Maintain window size
        if len(self.feature_history) > self.window_size:
            self.feature_history.pop(0)
    
    def _extract_feature_vector(self, features_dict):
        """
        Extract numerical feature vector from features dictionary using proper FeatureExtractor
        
        Args:
            features_dict: Dictionary with detection results
            
        Returns:
            List of numerical features (23 features expected by trained models)
        """
        if self.feature_extractor is not None:
            # Use proper feature extractor for full 23-feature vector
            try:
                features = self.feature_extractor.extract_features_from_results(features_dict)
                if len(features) == 23:
                    return features
                else:
                    print(f"Warning: FeatureExtractor returned {len(features)} features, expected 23")
            except Exception as e:
                print(f"Warning: FeatureExtractor failed: {e}, using fallback")
        
        # Fallback to simple feature extraction (pad to 23 features)
        features = []
        
        # Extract key features (convert boolean to 0/1)
        key_features = [
            'H-Hand Detected', 'F-Hand Detected',
            'H-Prohibited Item', 'F-Prohibited Item',
            'verification_result', 'num_faces', 'Cheat Score'
        ]
        
        for feature in key_features:
            if feature in features_dict:
                value = features_dict[feature]
                if isinstance(value, bool):
                    features.append(1.0 if value else 0.0)
                elif isinstance(value, (int, float)):
                    features.append(float(value))
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        # Pad with zeros to reach 23 features (to match trained model expectation)
        while len(features) < 23:
            features.append(0.0)
        
        return features[:23]  # Ensure exactly 23 features
    
    def get_temporal_prediction(self):
        """
        Get temporal prediction using the best available method
        
        Returns:
            Temporal cheating probability (0-1)
        """
        if len(self.feature_history) < 3:  # Need minimum frames
            return 0.0
        
        try:
            # Use PyTorch model if available and loaded
            if self.use_pytorch_model and self.model is not None:
                return self._pytorch_prediction()
            else:
                return self._rule_based_prediction()
                
        except Exception as e:
            print(f"Warning: Temporal prediction failed: {e}")
            return self._rule_based_prediction()
    
    def _pytorch_prediction(self):
        """
        Make prediction using loaded PyTorch model (original logic preserved)
        
        Returns:
            Temporal cheating probability (0-1)
        """
        if len(self.feature_history) < self.window_size:
            # Use available frames, pad if necessary
            sequence = self.feature_history.copy()
            while len(sequence) < self.window_size:
                sequence.insert(0, [0.0] * self.input_size)
        else:
            # Take the last window_size frames
            sequence = self.feature_history[-self.window_size:]
        
        # Convert to numpy array and scale
        sequence_array = np.array(sequence)
        if self.scaler.fitted:
            scaled_sequence = self.scaler.transform(sequence_array)
        else:
            scaled_sequence = sequence_array
        
        # Convert to tensor and add batch dimension
        tensor_sequence = torch.FloatTensor(scaled_sequence).unsqueeze(0).to(self.device)
        
        # Make prediction (original logic preserved)
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor_sequence)
            probability = torch.sigmoid(output).item()
        
        return probability
    
    def _rule_based_prediction(self):
        """
        Simple rule-based temporal analysis (fallback method)
        
        Returns:
            Temporal cheating probability (0-1)
        """
        # Simple temporal analysis
        recent_frames = self.feature_history[-5:]  # Last 5 frames
        
        # Calculate average suspicious activity
        suspicious_scores = []
        for frame_features in recent_frames:
            if len(frame_features) > 6:  # Ensure we have cheat score
                suspicious_scores.append(frame_features[6])  # Cheat Score index
        
        if suspicious_scores:
            avg_score = np.mean(suspicious_scores)
            
            # Simple temporal pattern detection
            if len(suspicious_scores) >= 3:
                # Check for increasing trend
                recent_trend = np.mean(suspicious_scores[-3:]) - np.mean(suspicious_scores[:-3])
                if recent_trend > 0.1:  # Increasing suspicion
                    avg_score += 0.2
            
            return min(avg_score, 1.0)
        
        return 0.0
    
    def load_models(self, save_dir):
        """
        Load temporal models from directory
        
        Args:
            save_dir: Directory containing saved models
        """
        if not os.path.exists(save_dir):
            print(f"Model directory not found: {save_dir}")
            return
        
        # Try to load PyTorch model first
        model_found = False
        
        # Check for the specific trained model
        from config import Config
        if hasattr(Config, 'DEFAULT_TEMPORAL_MODEL'):
            pytorch_model_path = Config.DEFAULT_TEMPORAL_MODEL
            if os.path.exists(pytorch_model_path):
                print(f"Found configured temporal model: {pytorch_model_path}")
                model_found = self._load_pytorch_model(pytorch_model_path)
                if model_found:
                    print(f"Successfully loaded {getattr(Config, 'DEFAULT_TEMPORAL_MODEL_TYPE', 'LSTM')} model")
        
        # Check for models in the save directory
        if not model_found:
            for filename in os.listdir(save_dir):
                if filename.endswith('.pt') or filename.endswith('.pth'):
                    model_path = os.path.join(save_dir, filename)
                    if self._load_pytorch_model(model_path):
                        model_found = True
                        break
        
        # Fallback to config file
        if not model_found:
            self._load_config_file(save_dir)
    
    def _load_pytorch_model(self, model_path):
        """
        Load PyTorch model from file
        
        Args:
            model_path: Path to model file
            
        Returns:
            bool: True if successfully loaded
        """
        if not HAS_ADVANCED_MODELS:
            print("Advanced models not available, skipping PyTorch model loading")
            return False
        
        try:
            print(f"Loading PyTorch temporal model from: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract model configuration
            if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
                config = checkpoint['model_config']
                model_type = config.get('model_type', 'LSTM')
                input_size = config.get('input_size', 23)
                hidden_size1 = config.get('hidden_size1', 128)
                hidden_size2 = config.get('hidden_size2', 64)
                output_size = config.get('output_size', 1)
            else:
                # Default configuration
                model_type = 'LSTM'
                input_size = 23
                hidden_size1 = 128
                hidden_size2 = 64
                output_size = 1
            
            # Create model
            if model_type.upper() == 'LSTM':
                self.model = LSTMModel(input_size, hidden_size1, hidden_size2, output_size)
            elif model_type.upper() == 'GRU':
                self.model = GRUModel(input_size, hidden_size1, hidden_size2, output_size)
            else:
                print(f"Unknown model type: {model_type}")
                return False
            
            self.model.to(self.device)
            self.model_type = model_type
            self.input_size = input_size
            
            # Load model state
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                # Load scaler if available
                if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
                    self.scaler.mean_ = checkpoint['scaler_mean']
                    self.scaler.scale_ = checkpoint['scaler_scale']
                    self.scaler.fitted = True
            else:
                self.model.load_state_dict(checkpoint)
            
            self.use_pytorch_model = True
            print(f"Successfully loaded {model_type} model with {input_size} features")
            return True
            
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            return False
    
    def _load_config_file(self, save_dir):
        """
        Load simple configuration file (fallback)
        
        Args:
            save_dir: Directory containing config file
        """
        config_path = os.path.join(save_dir, 'temporal_config.txt')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            if key == 'window_size':
                                self.window_size = int(value)
                            elif key == 'threshold':
                                self.threshold = float(value)
                            elif key == 'device':
                                self.device = value
                
                print(f"Temporal trainer configuration loaded from {save_dir}")
            except Exception as e:
                print(f"Warning: Failed to load temporal config: {e}")
    
    def save_models(self, save_dir):
        """
        Save model state
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if self.use_pytorch_model and self.model is not None:
            # Save PyTorch model
            model_path = os.path.join(save_dir, 'temporal_model.pth')
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'model_type': self.model_type,
                    'input_size': self.input_size,
                    'window_size': self.window_size
                }
            }
            
            if self.scaler.fitted:
                save_dict['scaler_mean'] = self.scaler.mean_
                save_dict['scaler_scale'] = self.scaler.scale_
            
            torch.save(save_dict, model_path)
            print(f"PyTorch temporal model saved to {model_path}")
        
        # Always save basic configuration
        config_path = os.path.join(save_dir, 'temporal_config.txt')
        with open(config_path, 'w') as f:
            f.write(f"window_size={self.window_size}\n")
            f.write(f"threshold={self.threshold}\n")
            f.write(f"device={self.device}\n")
            f.write(f"use_pytorch_model={self.use_pytorch_model}\n")
        
        print(f"Temporal trainer configuration saved to {save_dir}")
    
    def reset_history(self):
        """Reset feature history"""
        self.feature_history = []
    
    def get_statistics(self):
        """
        Get current statistics about the temporal analysis
        
        Returns:
            Dictionary with statistics
        """
        return {
            'history_length': len(self.feature_history),
            'window_size': self.window_size,
            'device': self.device,
            'threshold': self.threshold,
            'use_pytorch_model': self.use_pytorch_model,
            'model_type': self.model_type,
            'input_size': self.input_size
        }
    
    def train_models(self, training_sequences, labels):
        """
        Train models (simplified version - just adjusts threshold)
        
        Args:
            training_sequences: List of feature sequences
            labels: Corresponding binary labels
        """
        if len(training_sequences) == 0:
            print("Warning: No training data provided")
            return
        
        # Simple threshold adjustment based on training data
        try:
            # Calculate optimal threshold based on training data
            scores = []
            for seq, label in zip(training_sequences, labels):
                if len(seq) > 0:
                    avg_score = np.mean([np.mean(frame) for frame in seq])
                    scores.append((avg_score, label))
            
            if scores:
                # Find threshold that best separates classes
                scores.sort()
                best_threshold = 0.5
                best_accuracy = 0.0
                
                for threshold in np.linspace(0.1, 0.9, 9):
                    correct = sum(1 for score, label in scores 
                                if (score > threshold) == bool(label))
                    accuracy = correct / len(scores)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_threshold = threshold
                
                self.threshold = best_threshold
                print(f"Trained temporal model: threshold={self.threshold:.3f}, accuracy={best_accuracy:.3f}")
            
        except Exception as e:
            print(f"Warning: Training failed, using default threshold: {e}")


if __name__ == "__main__":
    # Test the Enhanced TemporalTrainer
    print("Testing Enhanced TemporalTrainer...")
    
    trainer = TemporalTrainerEnhanced(window_size=10)
    
    # Test with some sample features
    sample_features = {
        'H-Hand Detected': True,
        'F-Hand Detected': False,
        'H-Prohibited Item': False,
        'F-Prohibited Item': False,
        'verification_result': True,
        'num_faces': 1,
        'Cheat Score': 0.3
    }
    
    trainer.add_frame_features(sample_features)
    prediction = trainer.get_temporal_prediction()
    stats = trainer.get_statistics()
    
    print(f"Sample prediction: {prediction}")
    print(f"Statistics: {stats}")
    print("Enhanced TemporalTrainer test completed!")
