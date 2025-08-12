"""
Model management utilities for video proctor
"""
import joblib
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class ModelManager:
    """Handles loading and managing various ML models"""
    
    def __init__(self):
        self.static_model = None
        self.static_scaler = None
        self.static_metadata = None
        self.xgboost_model = None
        self.xgboost_scaler = None
    
    def load_static_model(self, model_path, scaler_path=None, metadata_path=None):
        """Load generic static model with optional scaler and metadata"""
        try:
            self.static_model = joblib.load(model_path)
            print(f"Static model loaded from {model_path}")
            
            if scaler_path:
                self.static_scaler = joblib.load(scaler_path)
                print(f"Static scaler loaded from {scaler_path}")
            
            if metadata_path:
                self.static_metadata = joblib.load(metadata_path)
                print(f"Static metadata loaded from {metadata_path}")
                
        except Exception as e:
            print(f"Error loading static model/scaler/metadata: {e}")
    
    def load_xgboost_model(self, model_path, scaler_path=None):
        """Load XGBoost model with optional scaler"""
        try:
            self.xgboost_model = joblib.load(model_path)
            print(f"XGBoost model loaded successfully from {model_path}")
            
            if scaler_path:
                try:
                    self.xgboost_scaler = joblib.load(scaler_path)
                    print("XGBoost scaler loaded successfully")
                except Exception as e:
                    print(f"Error loading XGBoost scaler: {e}")
                    self.xgboost_scaler = StandardScaler()
                    self._initialize_xgboost_scaler()
            else:
                self.xgboost_scaler = StandardScaler()
                self._initialize_xgboost_scaler()
                
        except Exception as e:
            print(f"Error loading XGBoost model: {e}")
    
    def _initialize_xgboost_scaler(self):
        """Initialize XGBoost scaler with default values"""
        if self.xgboost_scaler is None:
            self.xgboost_scaler = StandardScaler()
            
        print("Initializing XGBoost scaler with default values...")
        sample_features = np.array([Config.XGBOOST_SAMPLE_FEATURES])
        self.xgboost_scaler.fit(sample_features)
        print("XGBoost scaler initialized with default values")
    
    def predict_static(self, features):
        """Make prediction using static model"""
        if self.static_model is None:
            return None
        
        try:
            static_features = np.array(features[1:]).reshape(1, -1)  # Exclude timestamp
            
            if self.static_scaler is not None:
                static_features = self.static_scaler.transform(static_features)
            
            prediction = self.static_model.predict_proba(static_features)[:, 1][0]
            return prediction
            
        except Exception as e:
            print(f"Error in static model prediction: {e}")
            return 0.0
    
    def predict_xgboost(self, features):
        """Make prediction using XGBoost model"""
        if self.xgboost_model is None or self.xgboost_scaler is None:
            return None
        
        try:
            xgb_features = np.array(features[1:]).reshape(1, -1)  # Skip timestamp
            xgb_features_scaled = self.xgboost_scaler.transform(xgb_features)
            prediction = self.xgboost_model.predict_proba(xgb_features_scaled)[0][1]
            return prediction
            
        except Exception as e:
            print(f"Error in XGBoost prediction: {e}")
            return 0.0
    
    def load_detection_models(self):
        """Load YOLO and MediaPipe models for detection - STRICT mode, all required"""
        try:
            from ultralytics import YOLO
            import mediapipe as mp
            
            # STRICT: Load YOLO model from Config path only
            yolo_model_path = Config.DEFAULT_YOLO_MODEL
            if not os.path.exists(yolo_model_path):
                raise FileNotFoundError(f"YOLO model not found at configured path: {yolo_model_path}")
            
            print(f"Loading YOLO model from configured path: {yolo_model_path}")
            yolo_model = YOLO(yolo_model_path)
            
            # Setup MediaPipe using Config
            media_pipe_dict = Config.get_mediapipe_config()
            
            print("Detection models loaded successfully")
            return yolo_model, media_pipe_dict
            
        except Exception as e:
            print(f"CRITICAL ERROR loading detection models: {e}")
            print("Ensure all required model files are present:")
            print(f"   - YOLO model: {Config.DEFAULT_YOLO_MODEL}")
            raise RuntimeError(f"Failed to load detection models: {e}")
            raise
