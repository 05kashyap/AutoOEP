"""
Enhanced Temporal Proctor - Inference-only version for VideoProctor
Focuses on real-time temporal analysis without training logic
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from collections import deque

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

# Import feature extractor for proper feature processing
try:
    from core.feature_extractor import FeatureExtractor
    HAS_FEATURE_EXTRACTOR = True
except ImportError:
    HAS_FEATURE_EXTRACTOR = False
    raise ImportError("FeatureExtractor is required but not available. Please ensure core/feature_extractor.py exists.")

# Custom scaler for compatibility
class SimpleScaler:
    """Simple scaler for temporal features"""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.fitted = False
    
    def transform(self, X):
        """Transform data using fitted parameters"""
        if not self.fitted:
            return np.array(X)  # Return as-is if not fitted
        X = np.array(X)
        return (X - self.mean_) / self.scale_


class TemporalProctor:
    """
    Inference-only temporal proctor optimized for real-time video proctoring
    """
    
    def __init__(self, window_size=15, device=None):
        self.window_size = window_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_history = deque(maxlen=self.window_size)
        self.threshold = 0.5
        
        # PyTorch model components
        self.model = None
        self.scaler = SimpleScaler()
        self.model_type = None
        self.input_size = 23  # Full feature set expected by trained models
        self.use_pytorch_model = False
        
        # Initialize feature extractor - REQUIRED
        self.feature_extractor = FeatureExtractor()
        
        print("Enhanced TemporalProctor (Inference) initialized:")
        print(f"  Window Size: {window_size}")
        print(f"  Device: {self.device}")
        print(f"  Input Features: {self.input_size}")
        
        # Try to auto-load models from default locations
        self._try_auto_load_models()
    
    def _try_auto_load_models(self):
        """Try to automatically load temporal model from Config - no fallbacks"""
        try:
            from config import Config
            if hasattr(Config, 'DEFAULT_TEMPORAL_MODEL') and os.path.exists(Config.DEFAULT_TEMPORAL_MODEL):
                self.load_models()
                print("✅ Auto-loaded temporal model from config")
            else:
                print("⚠️ No temporal model configured - using rule-based fallback")
        except Exception as e:
            print(f"⚠️ Auto-load failed: {e} - using rule-based fallback")
    
    def add_frame_features(self, features_dict):
        """Add features from current frame to temporal sequence"""
        # Extract key features for temporal analysis
        feature_vector = self._extract_feature_vector(features_dict)
        
        # Add to history and maintain window size
        self.feature_history.append(feature_vector)
    
    def _extract_feature_vector(self, features_dict):
        """Extract numerical feature vector (exactly 23 features required)"""
        features = self.feature_extractor.extract_features_from_results(features_dict)
        if len(features) != 23:
            raise RuntimeError(f"FeatureExtractor returned {len(features)} features, expected exactly 23")
        return features
    
    def get_temporal_prediction(self):
        """Get temporal prediction using the best available method (0-1)"""
        if len(self.feature_history) < 3:  # Need minimum frames
            return 0.0
        try:
            if self.use_pytorch_model and self.model is not None:
                return self._pytorch_prediction()
            return self._rule_based_prediction()
        except Exception as e:
            print(f"Warning: Temporal prediction failed: {e}")
            return self._rule_based_prediction()
    
    def _pytorch_prediction(self):
        """Make prediction using loaded PyTorch model"""
        if self.model is None:
            raise RuntimeError("No PyTorch model loaded - cannot make prediction")
        
        # Prepare sequence window
        hist_list = list(self.feature_history)
        if len(hist_list) < self.window_size:
            sequence = hist_list.copy()
            while len(sequence) < self.window_size:
                sequence.insert(0, [0.0] * self.input_size)
        else:
            sequence = hist_list[-self.window_size:]
        
        # Scale if scaler is fitted
        sequence_array = np.array(sequence)
        scaled_sequence = self.scaler.transform(sequence_array) if self.scaler.fitted else sequence_array
        
        # To tensor and forward
        tensor_sequence = torch.FloatTensor(scaled_sequence).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor_sequence)
            probability = torch.sigmoid(output).item()
        return probability
    
    def _rule_based_prediction(self):
        """Simple rule-based temporal analysis (fallback method)"""
        hist_list = list(self.feature_history)
        recent_frames = hist_list[-5:]
        suspicious_scores = []
        for frame_features in recent_frames:
            if len(frame_features) > 6:  # Ensure we have cheat score
                suspicious_scores.append(frame_features[6])  # Cheat Score index
        if suspicious_scores:
            avg_score = np.mean(suspicious_scores)
            if len(suspicious_scores) >= 3:
                recent_trend = np.mean(suspicious_scores[-3:]) - np.mean(suspicious_scores[:-3])
                if recent_trend > 0.1:
                    avg_score += 0.2
            return min(avg_score, 1.0)
        return 0.0
    
    def load_models(self, model_path=None):
        """Load temporal model from checkpoint path (or Config.DEFAULT_TEMPORAL_MODEL)"""
        if model_path is None:
            from config import Config
            if not hasattr(Config, 'DEFAULT_TEMPORAL_MODEL'):
                raise RuntimeError("DEFAULT_TEMPORAL_MODEL not configured and no model_path provided")
            model_path = Config.DEFAULT_TEMPORAL_MODEL
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Temporal model not found: {model_path}")
        print(f"Loading temporal model from: {model_path}")
        self._load_pytorch_model(model_path)
        print("✅ Successfully loaded temporal model")
    
    def _load_pytorch_model(self, model_path):
        """Load PyTorch model from checkpoint file"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
                config = checkpoint['model_config']
                model_type = config.get('model_type', 'LSTM')
                input_size = config.get('input_size', 23)
                hidden_size1 = config.get('hidden_size1', 128)
                hidden_size2 = config.get('hidden_size2', 64)
                output_size = config.get('output_size', 1)
            else:
                model_type = 'LSTM'; input_size = 23; hidden_size1 = 128; hidden_size2 = 64; output_size = 1
                print("⚠️ No model config in checkpoint, using default LSTM configuration")
            
            if model_type.upper() == 'LSTM':
                self.model = LSTMModel(input_size, hidden_size1, hidden_size2, output_size)
            elif model_type.upper() == 'GRU':
                self.model = GRUModel(input_size, hidden_size1, hidden_size2, output_size)
            else:
                raise RuntimeError(f"Unsupported model type: {model_type}")
            
            self.model.to(self.device)
            self.model_type = model_type
            self.input_size = input_size
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
                    self.scaler.mean_ = checkpoint['scaler_mean']
                    self.scaler.scale_ = checkpoint['scaler_scale']
                    self.scaler.fitted = True
                    print("✅ Scaler parameters loaded from checkpoint")
            else:
                self.model.load_state_dict(checkpoint)
                print("⚠️ No scaler parameters in checkpoint")
            
            self.use_pytorch_model = True
            print(f"✅ Loaded {model_type} model with {input_size} input features")
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model from {model_path}: {e}")
    
    def reset_history(self):
        """Reset feature history"""
        self.feature_history = deque(maxlen=self.window_size)
    
    def get_statistics(self):
        """Get current statistics about the temporal analysis"""
        return {
            'history_length': len(self.feature_history),
            'window_size': self.window_size,
            'device': self.device,
            'threshold': self.threshold,
            'use_pytorch_model': self.use_pytorch_model,
            'model_type': self.model_type,
            'input_size': self.input_size,
        }


# Backward-compatibility alias
TemporalTrainerEnhanced = TemporalProctor

if __name__ == "__main__":
    # Simple self-test for TemporalProctor
    print("Testing Enhanced TemporalProctor (Inference)...")
    trainer = TemporalProctor(window_size=10)
    sample_features = {
        'H-Hand Detected': True,
        'F-Hand Detected': False,
        'H-Prohibited Item': False,
        'F-Prohibited Item': False,
        'verification_result': True,
        'num_faces': 1,
        'Cheat Score': 0.3,
    }
    trainer.add_frame_features(sample_features)
    prediction = trainer.get_temporal_prediction()
    stats = trainer.get_statistics()
    print(f"Sample prediction: {prediction}")
    print(f"Statistics: {stats}")
    print("Enhanced TemporalProctor (Inference) test completed!")
