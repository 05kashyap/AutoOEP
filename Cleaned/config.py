"""
Configuration settings for the video proctor system
"""
import os


class Config:
    """Central configuration for all components"""
    
    # Model paths
    DEFAULT_YOLO_MODEL = 'Inputs/Models/OEP_YOLOv11n.pt'
    DEFAULT_MEDIAPIPE_MODEL = 'Inputs/Models/face_landmarker.task'
    DEFAULT_MODEL_DIR = 'Inputs/Models'
    DEFAULT_TEMPORAL_MODEL = 'Inputs/Models/temporal_proctor_trained_on_processed.pt'
    DEFAULT_TEMPORAL_MODEL_TYPE = 'LSTM'
    
    # Static model files
    DEFAULT_STATIC_MODEL = 'Inputs/Models/lightgbm_cheating_model_20250801_200619.pkl'
    DEFAULT_STATIC_SCALER = 'Inputs/Models/scaler_20250801_200619.pkl'
    DEFAULT_STATIC_METADATA = 'Inputs/Models/model_metadata_20250801_200619.pkl'
    
    # Training parameters
    DEFAULT_WINDOW_SIZE = 15
    DEFAULT_INPUT_SIZE = 23
    DEFAULT_HIDDEN_SIZE = 128
    DEFAULT_NUM_LAYERS = 2
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_EPOCHS = 50
    DEFAULT_PATIENCE = 10
    
    # Data parameters
    DEFAULT_BUFFER_SIZE = 30
    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_RANDOM_STATE = 42
    
    # MediaPipe parameters
    MAX_NUM_HANDS = 2
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Image processing parameters
    CONTRAST_ALPHA = 1.3
    BRIGHTNESS_BETA = 30
    
    # Cheating detection thresholds
    PROHIBITED_ITEM_SCORE = 0.3
    IDENTITY_FAILURE_SCORE = 0.4
    SUSPICIOUS_GAZE_SCORE = 0.2
    MULTIPLE_FACES_SCORE = 0.1
    CHEATING_THRESHOLD = 0.5
    ALERT_THRESHOLD = 0.7
    
    # Model weighting for final score
    STATIC_WEIGHT = 0.6
    TEMPORAL_WEIGHT = 0.4
    
    # Logging configuration
    ENABLE_CHEATING_LOGS = True
    LOG_DIRECTORY = "logs"
    LOG_REAL_TIME_ALERTS = True
    
    # Feature names (in order)
    FEATURE_NAMES = [
        'timestamp', 'verification_result', 'num_faces', 'iris_pos', 
        'iris_ratio', 'mouth_zone', 'mouth_area', 'x_rotation', 
        'y_rotation', 'z_rotation', 'radial_distance', 'gaze_direction', 
        'gaze_zone', 'watch', 'headphone', 'closedbook', 'earpiece', 
        'cell phone', 'openbook', 'chits', 'sheet', 'H-Distance', 'F-Distance'
    ]
    
    # LSTM scaler default values
    LSTM_SCALER_MEANS = [
        0.0, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 500.0, 500.0
    ]
    
    LSTM_SCALER_STDS = [
        1.0, 0.5, 0.5, 1.0, 0.2, 1.0, 0.2, 0.2, 0.2, 0.2, 0.5, 1.5, 1.0,
        0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 200.0, 200.0
    ]
    
    # XGBoost scaler default values  
    XGBOOST_SAMPLE_FEATURES = [
        0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 500.0, 500.0
    ]
    
    # Expected feature ranges for validation
    FEATURE_RANGES = {
        'verification_result': (0, 1),
        'num_faces': (0, 10),
        'iris_pos': (-1, 2),
        'mouth_zone': (-1, 3),
        'gaze_direction': (-1, 4),
        'gaze_zone': (-1, 2),
        'H-Distance': (0, 10000),
        'F-Distance': (0, 10000)
    }
    
    # Encoding mappings
    IRIS_POSITION_MAPPING = {'center': 0, 'left': 1, 'right': 2}
    MOUTH_ZONE_MAPPING = {'GREEN': 0, 'YELLOW': 1, 'ORANGE': 2, 'RED': 3}
    GAZE_DIRECTION_MAPPING = {'forward': 0, 'left': 1, 'right': 2, 'up': 3, 'down': 4}
    GAZE_ZONE_MAPPING = {'white': 0, 'yellow': 1, 'red': 2}
    
    # Prohibited items list
    PROHIBITED_ITEMS = [
        'watch', 'headphone', 'closedbook', 'earpiece',
        'cell phone', 'openbook', 'chits', 'sheet'
    ]
    
    @classmethod
    def get_mediapipe_config(cls):
        """Get MediaPipe configuration dictionary"""
        import mediapipe as mp
        
        mpHands = mp.solutions.hands
        return {
            'mpHands': mpHands,
            'hands': mpHands.Hands(
                static_image_mode=False,
                max_num_hands=cls.MAX_NUM_HANDS,
                min_detection_confidence=cls.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=cls.MIN_TRACKING_CONFIDENCE
            ),
            'mpdraw': mp.solutions.drawing_utils
        }
    
    @classmethod
    def validate_paths(cls, **paths):
        """Validate that required file paths exist"""
        missing_paths = []
        for name, path in paths.items():
            if path and not os.path.exists(path):
                missing_paths.append(f"{name}: {path}")
        
        if missing_paths:
            print("Warning: The following paths do not exist:")
            for path in missing_paths:
                print(f"  - {path}")
        
        return len(missing_paths) == 0
    
    @classmethod
    def validate_mediapipe_model(cls):
        """Validate that MediaPipe face landmarker model exists"""
        if not cls.DEFAULT_MEDIAPIPE_MODEL:
            raise ValueError("DEFAULT_MEDIAPIPE_MODEL is not configured")
        
        if not os.path.exists(cls.DEFAULT_MEDIAPIPE_MODEL):
            raise FileNotFoundError(f"MediaPipe face landmarker model not found: {cls.DEFAULT_MEDIAPIPE_MODEL}")
        
        return True


class DebugConfig:
    """Configuration for debugging and validation"""
    
    ENABLE_FEATURE_LOGGING = False
    ENABLE_RANGE_VALIDATION = True
    ENABLE_CONSISTENCY_CHECK = True
    LOG_CSV_PATH = "realtime_features.csv"
    REFERENCE_CSV_PATH = "Datasets/training_proctor_results.csv"
    
    # Debug message templates
    DEBUG_MESSAGES = {
        'feature_count': "DEBUG: Extracted {count} features",
        'buffer_size': "DEBUG: Buffer size: {current}/{max}",
        'temporal_prediction': "DEBUG: Making temporal prediction...",
        'prediction_result': "DEBUG: Temporal prediction result: {result}",
        'insufficient_frames': "DEBUG: Not enough frames for temporal prediction ({current}/{required})"
    }
