import cv2
import numpy as np
import torch
import mediapipe as mp
import time
import argparse
from collections import deque
from ultralytics import YOLO
from Proctor.proctor import StaticProctor
from Temporal.temporal_trainer import TemporalProctor
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
import joblib  # Add for loading static models
from sklearn.preprocessing import StandardScaler 


class VideoProctor:
    def __init__(self, lstm_model_path, yolo_model_path, 
                 static_model_path=None, static_scaler_path=None, static_metadata_path=None,
                 xgboost_model_path=None, xgboost_scaler_path=None, mediapipe_model_path=None, window_size=15, 
                 input_size=None, buffer_size=30, device=None, debug_features=False):
        """
        Initialize the video proctor that combines frame-by-frame analysis with temporal analysis
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.debug_features = debug_features
        self.log_features = debug_features  # Enable feature logging

        if debug_features:
            print("🐛 DEBUG MODE: Feature verification enabled")
            # Don't call verify_feature_consistency here - we don't have features yet
            # Instead, just verify that the training CSV exists and show expected features
            self.show_expected_features()
        
        # Load YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Setup MediaPipe
        self.mpHands = mp.solutions.hands
        self.media_pipe_dict = {
            'mpHands': self.mpHands,
            'hands': self.mpHands.Hands(static_image_mode=False,  # For video streams
                                    max_num_hands=2,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5),
            'mpdraw': mp.solutions.drawing_utils
        }
        
        # Initialize static proctor
        self.static_proctor = StaticProctor(self.yolo_model, self.media_pipe_dict, mediapipe_model_path=mediapipe_model_path)

        # Initialize temporal proctor
        self.temporal_proctor = TemporalProctor(window_size=window_size, device=self.device)
        
        # Load the LSTM model
        if input_size is None:
            raise ValueError("input_size must be provided for loading the LSTM model")
        self.temporal_proctor.load_model(lstm_model_path, input_size)
        if self.temporal_proctor.scaler.mean_ is None:
            print("Warning: LSTM scaler not loaded from model file, initializing with default values")
            self.initialize_scaler()
        else:
            print("LSTM scaler loaded successfully from model file")

        # --- Static Model Loading (generic, not just XGBoost) ---
        self.static_model = None
        self.static_scaler = None
        self.static_metadata = None
        if static_model_path:
            try:
                self.static_model = joblib.load(static_model_path)
                print(f"Static model loaded from {static_model_path}")
                # Try to load scaler and metadata if provided
                if static_scaler_path:
                    self.static_scaler = joblib.load(static_scaler_path)
                    print(f"Static scaler loaded from {static_scaler_path}")
                if static_metadata_path:
                    self.static_metadata = joblib.load(static_metadata_path)
                    print(f"Static metadata loaded from {static_metadata_path}")
            except Exception as e:
                print(f"Error loading static model/scaler/metadata: {e}")
        
        # --- XGBoost model loading (legacy, can be removed if not needed) ---
        self.xgboost_model = None
        self.xgboost_scaler = None
        print(f"Loading XGBoost model from {xgboost_model_path} and scaler from {xgboost_scaler_path}")
        if xgboost_model_path:
            try:
                # Load the XGBoost model
                self.xgboost_model = joblib.load(xgboost_model_path)
                print(f"XGBoost model loaded successfully from {xgboost_model_path}")
                
                # Load the scaler that was used during XGBoost training
                try:
                    self.xgboost_scaler = joblib.load(xgboost_scaler_path)
                    print("XGBoost scaler loaded successfully")
                except Exception as e:
                    print(f"Error loading XGBoost scaler: {e}")
                    # Create a fallback scaler if loading fails
                    self.xgboost_scaler = StandardScaler()
                    self.initialize_xgboost_scaler()
            except Exception as e:
                print(f"Error loading XGBoost model: {e}")
        
        # Feature buffer for temporal analysis
        self.feature_buffer = deque(maxlen=buffer_size)
        
        # Store prediction history for visualization
        self.timestamps = deque(maxlen=100)
        self.predictions = deque(maxlen=100)
        self.static_scores = deque(maxlen=100)
        self.xgboost_scores = deque(maxlen=100)  # For XGBoost predictions
        
        # For visualization
        self.plot_initialized = False
        self.fig = None
        self.ax = None
        self.line1 = None
        self.line2 = None
        self.line3 = None
        
    def initialize_scaler(self):
        """Initialize the scaler with pre-calculated mean and std values from training data"""
        # These values should match the statistics of the training data
        # Approximating values for the 23 features based on typical ranges
        means = np.array([
            0.0,                # timestamp (normalized to 0)
            0.5,                # verification_result (0 or 1)
            1.0,                # num_faces
            0.0,                # iris_pos (-1, 0, 1, 2)
            1.0,                # iris_ratio
            1.0,                # mouth_zone (0-3)
            0.0,                # mouth_area
            0.0,                # x_rotation
            0.0,                # y_rotation
            0.0,                # z_rotation
            0.0,                # radial_distance
            0.0,                # gaze_direction (-1, 0-4)
            0.0,                # gaze_zone (-1, 0-2)
            0.1,                # watch (0 or 1)
            0.1,                # headphone (0 or 1)
            0.1,                # closedbook (0 or 1)
            0.1,                # earpiece (0 or 1)
            0.1,                # cell phone (0 or 1)
            0.1,                # openbook (0 or 1)
            0.1,                # chits (0 or 1)
            0.1,                # sheet (0 or 1)
            500.0,              # H-Distance
            500.0               # F-Distance
        ])
        
        stds = np.array([
            1.0,                # timestamp (normalized)
            0.5,                # verification_result
            0.5,                # num_faces
            1.0,                # iris_pos
            0.2,                # iris_ratio
            1.0,                # mouth_zone
            0.2,                # mouth_area
            0.2,                # x_rotation
            0.2,                # y_rotation
            0.2,                # z_rotation
            0.5,                # radial_distance
            1.5,                # gaze_direction
            1.0,                # gaze_zone
            0.3,                # watch
            0.3,                # headphone
            0.3,                # closedbook
            0.3,                # earpiece
            0.3,                # cell phone
            0.3,                # openbook
            0.3,                # chits
            0.3,                # sheet
            200.0,              # H-Distance
            200.0               # F-Distance
        ])
        
        # Set these values in the temporal_proctor's scaler
        self.temporal_proctor.scaler.mean_ = means
        self.temporal_proctor.scaler.scale_ = stds
        print("Scaler initialized with pre-calculated mean and std values")
    def initialize_xgboost_scaler(self):
        """Initialize the XGBoost scaler with default values if loading fails"""
        print("Initializing XGBoost scaler with default values...")
        
        # Create sample data to fit the scaler (excluding timestamp)
        sample_features = np.array([
            [0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # verification through gaze_zone
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # prohibited items
            500.0, 500.0]  # H-Distance, F-Distance
        ])
        
        # Fit the scaler with sample data
        self.xgboost_scaler.fit(sample_features)
        print("XGBoost scaler initialized with default values")
    def process_frame_pair(self, target_frame, face_frame, hand_frame):
        """
        Process a pair of frames (face and hand) to extract features and make predictions
        
        Args:
            target_frame: The reference face image for identity verification
            face_frame: Current face camera frame
            hand_frame: Current hand/desk camera frame
            
        Returns:
            Dictionary with static analysis results and temporal prediction
        """
        # Process frames with static proctor
        static_results = self.static_proctor.process_frames(target_frame, face_frame, hand_frame)
        
        # Extract features for temporal analysis
        features = self.extract_features_from_results(static_results)
        
        if self.debug_features:
            print(f"DEBUG: Extracted {len(features)} features")
        
        # Add features to buffer
        self.feature_buffer.append(features)
        
        if self.debug_features:
            print(f"DEBUG: Buffer size: {len(self.feature_buffer)}/{self.temporal_proctor.window_size}")
        
        # Make temporal prediction if buffer has enough frames
        temporal_prediction = None
        if len(self.feature_buffer) >= self.temporal_proctor.window_size:
            if self.debug_features:
                print("DEBUG: Making temporal prediction...")
            temporal_prediction = self.temporal_proctor.make_realtime_prediction(list(self.feature_buffer))
            if self.debug_features:
                print(f"DEBUG: Temporal prediction result: {temporal_prediction}")
        else:
            if self.debug_features:
                print(f"DEBUG: Not enough frames for temporal prediction ({len(self.feature_buffer)}/{self.temporal_proctor.window_size})")
        
        # Make XGBoost prediction on current frame if model is available
        xgboost_prediction = None
        if self.xgboost_model is not None:
            # Convert features to the format expected by XGBoost (exclude timestamp)
            xgb_features = np.array(features[1:]).reshape(1, -1)  # Skip timestamp (first feature)
            
            try:
                # Scale features using StandardScaler
                xgb_features_scaled = self.xgboost_scaler.transform(xgb_features)
                
                # Get prediction probability
                xgboost_prediction = self.xgboost_model.predict_proba(xgb_features_scaled)[0][1]
            except Exception as e:
                print(f"Error in XGBoost prediction: {e}")
                xgboost_prediction = 0.0
        
        # --- Static Model Prediction (generic) ---
        static_model_prediction = None
        if self.static_model is not None:
            # Extract features (excluding timestamp)
            static_features = np.array(self.extract_features_from_results(static_results)[1:]).reshape(1, -1)
            try:
                # Apply scaler if available
                if self.static_scaler is not None:
                    static_features = self.static_scaler.transform(static_features)
                # Predict probability
                static_model_prediction = self.static_model.predict_proba(static_features)[:, 1][0]
            except Exception as e:
                print(f"Error in static model prediction: {e}")
                static_model_prediction = 0.0
        
        # Add predictions to history for visualization
        current_time = time.time()
        self.timestamps.append(current_time)
        self.predictions.append(temporal_prediction if temporal_prediction is not None else 0)
        # Replace static_scores with static_model_prediction if available
        self.static_scores.append(static_model_prediction if static_model_prediction is not None else static_results.get('Cheat Score', 0))
        self.xgboost_scores.append(xgboost_prediction if xgboost_prediction is not None else 0)
        
        # Combine results
        results = {
            'static_results': static_results,
            'static_model_prediction': static_model_prediction,
            'temporal_prediction': temporal_prediction,
            'xgboost_prediction': xgboost_prediction,
            'timestamp': current_time
        }
        
        return results
    
    def extract_features_from_results(self, results):
        """
        Extract numerical features from static proctor results for temporal analysis
        
        Args:
            results: Dictionary of results from static proctor
            
        Returns:
            List of features matching the format expected by the LSTM model
        """
        # Define feature mapping based on process_csv.py processing
        features = []
        
        # Add timestamp - using relative timestamp for feature extraction
        current_time = time.time()
        if not hasattr(self, 'start_time'):
            self.start_time = current_time
        features.append(current_time - self.start_time)
        
        # Add verification result (Identity)
        features.append(int(results.get('verification_result', False)))
        
        # Add number of faces
        features.append(float(results.get('num_faces', 0)))
        
        # Eye/iris position
        eye_dir = results.get('iris_pos', 'center')
        if isinstance(eye_dir, str):
            if eye_dir.lower() == 'center':
                features.append(0)
            elif eye_dir.lower() == 'left':
                features.append(1)
            elif eye_dir.lower() == 'right':
                features.append(2)
            else:
                features.append(-1)  # Unknown
        else:
            features.append(-1)  # Unknown
        
        # Add iris ratio
        iris_ratio = results.get('iris_ratio', 1.0)
        features.append(float(iris_ratio) if iris_ratio is not None else 1.0)
        
        # Mouth zone
        mouth_zone = results.get('mouth_zone', 'GREEN')
        if isinstance(mouth_zone, str):
            if mouth_zone == 'GREEN':
                features.append(0)
            elif mouth_zone == 'YELLOW':
                features.append(1)
            elif mouth_zone == 'ORANGE':
                features.append(2)
            elif mouth_zone == 'RED':
                features.append(3)
            else:
                features.append(-1)  # Unknown
        else:
            features.append(-1)  # Unknown
        
        # Mouth area
        mouth_area = results.get('mouth_area', 0.0)
        features.append(float(mouth_area) if mouth_area is not None else 0.0)
        
        # Face rotation (x, y, z)
        x_rotation = results.get('x_rotation', 0.0)
        features.append(float(x_rotation) if x_rotation is not None else 0.0)
        
        y_rotation = results.get('y_rotation', 0.0)
        features.append(float(y_rotation) if y_rotation is not None else 0.0)
        
        z_rotation = results.get('z_rotation', 0.0)
        features.append(float(z_rotation) if z_rotation is not None else 0.0)
        
        # Radial distance
        radial_distance = results.get('radial_distance', 0.0)
        features.append(float(radial_distance) if radial_distance is not None else 0.0)
        
        # Gaze direction
        gaze_dir = results.get('gaze_direction', 'forward')
        if isinstance(gaze_dir, str):
            if gaze_dir.lower() == 'forward':
                features.append(0)
            elif gaze_dir.lower() == 'left':
                features.append(1)
            elif gaze_dir.lower() == 'right':
                features.append(2)
            elif gaze_dir.lower() == 'up':
                features.append(3)
            elif gaze_dir.lower() == 'down':
                features.append(4)
            else:
                features.append(-1)  # Unknown
        else:
            features.append(-1)  # Unknown
        
        # Gaze zone
        gaze_zone = results.get('gaze_zone', 'white')
        if isinstance(gaze_zone, str):
            if gaze_zone.lower() == 'white':
                features.append(0)
            elif gaze_zone.lower() == 'yellow':
                features.append(1)
            elif gaze_zone.lower() == 'red':
                features.append(2)
            else:
                features.append(-1)  # Unknown
        else:
            features.append(-1)  # Unknown
        
        # Process prohibited items - one-hot encoding for common objects
        # The items mentioned in process_csv.py
        prohibited_items = {
            'watch': 0,
            'headphone': 0,
            'closedbook': 0,
            'earpiece': 0,
            'cell phone': 0,
            'openbook': 0,
            'chits': 0,
            'sheet': 0
        }
        
        # Check for prohibited items in both face and hand frames
        face_prohibited = results.get('F-Prohibited Item', [])
        hand_prohibited = results.get('H-Prohibited Item', [])
        
        # Handle case when values are lists
        if isinstance(face_prohibited, list):
            for item in face_prohibited:
                if item in prohibited_items:
                    prohibited_items[item] = 1
        
        if isinstance(hand_prohibited, list):
            for item in hand_prohibited:
                if item in prohibited_items:
                    prohibited_items[item] = 1
        
        # Add all prohibited item indicators to features
        for item in ['watch', 'headphone', 'closedbook', 'earpiece', 
                     'cell phone', 'openbook', 'chits', 'sheet']:
            features.append(prohibited_items[item])
        
        # Add distance values - Handle None values by using default value
        h_distance = results.get('H-Distance')
        features.append(float(h_distance) if h_distance is not None else 1000)
        
        f_distance = results.get('F-Distance')
        features.append(float(f_distance) if f_distance is not None else 1000)
        
        # --- DEBUG: Compare with expected feature count ---
        expected_feature_count = 23  # Should match training
        if len(features) != expected_feature_count:
            print(f"WARNING: Extracted {len(features)} features, expected {expected_feature_count}")
        
        if hasattr(self, 'log_features') and self.log_features:
            feature_names = [
                'timestamp', 'verification_result', 'num_faces', 'iris_pos', 
                'iris_ratio', 'mouth_zone', 'mouth_area', 'x_rotation', 
                'y_rotation', 'z_rotation', 'radial_distance', 'gaze_direction', 
                'gaze_zone', 'watch', 'headphone', 'closedbook', 'earpiece', 
                'cell phone', 'openbook', 'chits', 'sheet', 'H-Distance', 'F-Distance'
            ]
            
            print("\n=== FEATURE VERIFICATION ===")
            print(f"Expected features: {len(feature_names)}")
            print(f"Extracted features: {len(features)}")
            
            for i, (name, value) in enumerate(zip(feature_names, features)):
                print(f"{i:2d}. {name:15s}: {value}")
            
            # Verify against expected ranges
            self.verify_feature_ranges(features, feature_names)
            
            # Now call feature consistency check with actual features
            self.verify_feature_consistency(features)
        
        return features
        
    def verify_feature_ranges(self, features, feature_names):
        """Verify that features are within expected ranges"""
        expected_ranges = {
            'verification_result': (0, 1),
            'num_faces': (0, 10),
            'iris_pos': (-1, 2),
            'mouth_zone': (-1, 3),
            'gaze_direction': (-1, 4),
            'gaze_zone': (-1, 2),
            'H-Distance': (0, 10000),
            'F-Distance': (0, 10000)
        }
        
        warnings = []
        for i, (name, value) in enumerate(zip(feature_names, features)):
            if name in expected_ranges:
                min_val, max_val = expected_ranges[name]
                if not (min_val <= value <= max_val):
                    warnings.append(f"WARNING: {name} = {value} outside range [{min_val}, {max_val}]")
        
        if warnings:
            print("\n⚠️  FEATURE RANGE WARNINGS:")
            for warning in warnings:
                print(f"   {warning}")
        else:
            print("✅ All features within expected ranges")

    def show_expected_features(self, reference_csv_path=None):
        """
        Show expected features from training data during initialization
        """
        if reference_csv_path is None:
            reference_csv_path = 'Datasets/training_proctor_results.csv'
        
        try:
            import pandas as pd
            # Load reference data
            ref_df = pd.read_csv(reference_csv_path)
            ref_features = ref_df.drop(['timestamp', 'is_cheating'], axis=1, errors='ignore')
            
            expected_feature_names = ref_features.columns.tolist();
            
            print("\n=== EXPECTED FEATURES FROM TRAINING DATA ===")
            print(f"Total expected features: {len(expected_feature_names)}")
            for i, feat in enumerate(expected_feature_names):
                print(f"  {i:2d}. {feat}")
            
            print(f"\nTraining data shape: {ref_df.shape}")
            print(f"Features will be extracted in this order during processing.")
            
        except Exception as e:
            print(f"Warning: Could not load training data for feature verification: {e}")

    def verify_feature_consistency(self, current_features=None, reference_csv_path=None):
        """
        Verify that current features match the format of training data
        """
        if reference_csv_path is None:
            reference_csv_path = 'Datasets/training_proctor_results.csv'
        
        try:
            import pandas as pd
            # Load reference data
            ref_df = pd.read_csv(reference_csv_path)
            ref_features = ref_df.drop(['timestamp', 'is_cheating'], axis=1, errors='ignore')
            
            expected_feature_names = ref_features.columns.tolist();
            
            current_feature_names = [
                'verification_result', 'num_faces', 'iris_pos', 'iris_ratio', 
                'mouth_zone', 'mouth_area', 'x_rotation', 'y_rotation', 'z_rotation',
                'radial_distance', 'gaze_direction', 'gaze_zone', 'watch', 'headphone',
                'closedbook', 'earpiece', 'cell phone', 'openbook', 'chits', 'sheet',
                'H-Distance', 'F-Distance'
            ]
            
            print("\n=== FEATURE CONSISTENCY CHECK ===")
            print(f"Expected features (from training): {len(expected_feature_names)}")
            print(f"Current feature names: {len(current_feature_names)}")
            
            if current_features is not None:
                print(f"Current feature values: {len(current_features) - 1}")  # -1 for timestamp
            
            # Check for missing features
            missing = set(expected_feature_names) - set(current_feature_names)
            extra = set(current_feature_names) - set(expected_feature_names)
            
            if missing:
                print(f"❌ Missing features: {missing}")
            if extra:
                print(f"⚠️  Extra features: {extra}")
            if not missing and not extra:
                print("✅ Feature names match perfectly")
            
            # Check feature order
            if current_feature_names == expected_feature_names:
                print("✅ Feature order matches")
            else:
                print("⚠️  Feature order differs:")
                for i, (curr, exp) in enumerate(zip(current_feature_names, expected_feature_names)):
                    if curr != exp:
                        print(f"   Position {i}: got '{curr}', expected '{exp}'")
            
            return missing, extra
            
        except Exception as e:
            print(f"Error in feature consistency check: {e}")
            return None, None

    
    def process_videos(self, face_video_path, hand_video_path, target_frame_path, 
                  output_path=None, display=True, fps=30, test_duration=None):
        """
        Process video streams from face and hand cameras
        
        Args:
            face_video_path: Path to face camera video
            hand_video_path: Path to hand/desk camera video
            target_frame_path: Path to reference face image
            output_path: Path to save output video (optional)
            display: Whether to display the processed video
            fps: Frames per second for output video
            test_duration: Duration (in seconds) to process for testing (optional)
            
        Returns:
            DataFrame with analysis results
        """
        # Load target frame
        target_frame = cv2.imread(target_frame_path)
        if target_frame is None:
            raise FileNotFoundError(f"Target frame not found at {target_frame_path}")
        
        # Open video captures
        face_cap = cv2.VideoCapture(face_video_path)
        hand_cap = cv2.VideoCapture(hand_video_path)
        
        if not face_cap.isOpened() or not hand_cap.isOpened():
            raise ValueError("Error opening video streams")
        
        # Get video properties
        face_width = int(face_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        face_height = int(face_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        hand_width = int(hand_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hand_height = int(hand_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = None  # Delay video writer initialization until the first frame is processed

        # Initialize plot if we're going to need it
        if output_path or display:
            self.initialize_plot()

        results = []
        frame_count = 0
        start_time = time.time()
        
        # Calculate the maximum number of frames to process for testing
        max_frames = None
        if test_duration:
            max_frames = int(test_duration * fps)
        
        # Start processing frames
        while True:
            face_ret, face_frame = face_cap.read()
            hand_ret, hand_frame = hand_cap.read()
            
            # Break loop if either video ends
            if not face_ret or not hand_ret:
                break
            
            # Process frame pair
            result = self.process_frame_pair(target_frame, face_frame, hand_frame)
            results.append(result)
            
            # Create display frame
            display_frame = self.create_display_frame(face_frame, hand_frame, result)
            
            # Initialize video writer after the first frame is read
            if out is None and output_path:
                # Combined width is max of face and hand width * 2 to place them side by side
                # Height will be max height + space for the prediction graph
                out_width = display_frame.shape[1]
                out_height = display_frame.shape[0] + 200  # Extra space for visualization
                
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
                print(f"Initialized video writer with dimensions: {out_width}x{out_height}")
            
            # Update plot
            self.update_plot()
                
            if output_path and out is not None:
                # Get plot frame
                plot_frame = self.get_plot_frame()
                
                # Resize plot_frame to match the width of display_frame
                plot_frame = cv2.resize(plot_frame, (display_frame.shape[1], 200))
                
                # Create combined frame
                combined_frame = np.zeros((display_frame.shape[0] + plot_frame.shape[0], 
                                        display_frame.shape[1], 3), dtype=np.uint8)
                combined_frame[:display_frame.shape[0], :] = display_frame
                combined_frame[display_frame.shape[0]:, :] = plot_frame
                
                # Write the combined frame to the output video
                out.write(combined_frame)
            
            if display:
                cv2.imshow('Video Proctor', display_frame)
                # Break loop on ESC key
                if cv2.waitKey(1) == 27:
                    break
            
            frame_count += 1
            
            # Break loop if max_frames is reached during testing
            if max_frames and frame_count >= max_frames:
                break
            
        # Calculate and display processing speed
        elapsed_time = time.time() - start_time
        processed_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({processed_fps:.2f} FPS)")
        
        # Release resources
        face_cap.release()
        hand_cap.release()
        if out:
            out.release()
            print(f"Video output saved to: {output_path}")
        if display:
            cv2.destroyAllWindows()
        
        # Close the plot
        plt.close(self.fig)
        
        return results
    
    def create_display_frame(self, face_frame, hand_frame, result):
        """
        Create a display frame with annotations for visualization
        
        Args:
            face_frame: Face camera frame
            hand_frame: Hand camera frame
            result: Processing result dictionary
            
        Returns:
            Combined and annotated frame for display
        """
        # Resize frames to the same height if necessary
        max_height = max(face_frame.shape[0], hand_frame.shape[0])
        
        # Resize frames if necessary
        if face_frame.shape[0] != max_height:
            scale = max_height / face_frame.shape[0]
            face_frame = cv2.resize(face_frame, (int(face_frame.shape[1] * scale), max_height))
            
        if hand_frame.shape[0] != max_height:
            scale = max_height / hand_frame.shape[0]
            hand_frame = cv2.resize(hand_frame, (int(hand_frame.shape[1] * scale), max_height))
        
        # Create a combined frame
        combined_width = face_frame.shape[1] + hand_frame.shape[1]
        combined_frame = np.zeros((max_height, combined_width, 3), dtype=np.uint8)
        
        # Place frames side by side
        combined_frame[:, :face_frame.shape[1]] = face_frame
        combined_frame[:, face_frame.shape[1]:] = hand_frame
        
        # Add static results as text
        static_results = result['static_results']
        
        # Add temporal prediction
        temporal_pred = result.get('temporal_prediction')
        
        # Add XGBoost prediction
        xgboost_pred = result.get('xgboost_prediction')
        
        # Draw static cheat score
        static_score = static_results.get('Cheat Score', 0)
        cv2.putText(combined_frame, f"Static Score: {static_score:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw temporal prediction
        if temporal_pred is not None:
            # Color based on prediction (red for likely cheating)
            color = (0, 0, 255) if temporal_pred > 0.5 else (0, 255, 0)
            cv2.putText(combined_frame, f"Temporal Prediction: {temporal_pred:.2f}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw XGBoost prediction
        if xgboost_pred is not None:
            # Color based on prediction (red for likely cheating)
            color = (0, 0, 255) if xgboost_pred > 0.5 else (0, 255, 0)
            cv2.putText(combined_frame, f"XGBoost Prediction: {xgboost_pred:.2f}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add warning text for high probability of either model
            warning_threshold = 0.7
            if temporal_pred is not None and temporal_pred > warning_threshold or xgboost_pred > warning_threshold:
                cv2.putText(combined_frame, "WARNING: Likely Cheating Detected", 
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return combined_frame
    
    def initialize_plot(self):
        """Initialize the matplotlib plot for displaying predictions over time"""
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.line1, = self.ax.plot([], [], 'r-', label='Temporal Prediction')
        self.line2, = self.ax.plot([], [], 'b-', label='Static Score')
        self.line3, = self.ax.plot([], [], 'g-', label='XGBoost Prediction')  # New line for XGBoost
        
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Cheat Probability')
        self.ax.set_title('Cheating Detection')
        self.ax.legend()
        self.ax.grid(True)
        
        self.plot_initialized = True
        plt.show(block=False)
    
    def update_plot(self):
        """Update the plot with new data"""
        if not self.plot_initialized or len(self.timestamps) < 2:
            return
        
        # Convert timestamps to relative time (seconds from start)
        if len(self.timestamps) > 0:
            rel_times = [t - self.timestamps[0] for t in self.timestamps]
            
            self.line1.set_data(rel_times, self.predictions)
            self.line2.set_data(rel_times, self.static_scores)
            self.line3.set_data(rel_times, self.xgboost_scores)  # Add XGBoost data
            
            self.ax.set_xlim(0, rel_times[-1] + 0.5)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
    
    def get_plot_frame(self):
        """Render the matplotlib plot as an image for the video output"""
        self.fig.canvas.draw()
        plot_image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process videos with VideoProctor')
    parser.add_argument('--face', type=str, required=True, help='Path to face camera video')
    parser.add_argument('--hand', type=str, required=True, help='Path to hand camera video')
    parser.add_argument('--target', type=str, required=True, help='Path to target/reference face image')
    parser.add_argument('--output', type=str, default=None, help='Path to save output video')
    parser.add_argument('--lstm-model', type=str, required=True, help='Path to trained LSTM model')
    parser.add_argument('--xgboost-model', type=str, default=None, help='Path to trained XGBoost model')
    parser.add_argument('--xgboost-scaler', type=str, default=None, help='Path to XGBoost scaler')
    parser.add_argument('--yolo-model', type=str, default='Models/OEP_YOLOv11n.pt', help='Path to YOLO model')
    parser.add_argument('--mediapipe-task', type=str, required=True, help='Path to mediapipe face_landmarker.task')
    parser.add_argument('--input-size', type=int, default=23, help='Number of features for LSTM input')
    parser.add_argument('--window-size', type=int, default=15, help='Window size for temporal analysis')
    parser.add_argument('--buffer-size', type=int, default=30, help='Size of feature buffer')
    parser.add_argument('--display', action='store_true', help='Display processed video')
    parser.add_argument('--test-duration', type=int, default=None, 
                        help='Duration (in seconds) to process for testing')
    parser.add_argument('--static-model', type=str, default=None, help='Path to saved static model (LightGBM/XGBoost/RandomForest)')
    parser.add_argument('--static-scaler', type=str, default=None, help='Path to static model scaler (optional)')
    parser.add_argument('--static-metadata', type=str, default=None, help='Path to static model metadata (optional)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Initialize VideoProctor
    proctor = VideoProctor(
        lstm_model_path=args.lstm_model,
        yolo_model_path=args.yolo_model,
        static_model_path=args.static_model,
        static_scaler_path=args.static_scaler,
        static_metadata_path=args.static_metadata,
        xgboost_model_path=args.xgboost_model,
        xgboost_scaler_path=args.xgboost_scaler,
        mediapipe_model_path=args.mediapipe_task,
        window_size=args.window_size,
        input_size=args.input_size,
        buffer_size=args.buffer_size,
        debug_features=True,  # Enable debug features for verification
    )
    
    # Process videos
    results = proctor.process_videos(
        face_video_path=args.face,
        hand_video_path=args.hand,
        target_frame_path=args.target,
        output_path=args.output,
        display=args.display,
        test_duration=args.test_duration
    )
    
    print(f"Processed {len(results)} frames")
    
    # Calculate overall statistics
    temporal_predictions = [r['temporal_prediction'] for r in results if r['temporal_prediction'] is not None]
    xgboost_predictions = [r['xgboost_prediction'] for r in results if r['xgboost_prediction'] is not None]
    
    if temporal_predictions:
        avg_prediction = sum(temporal_predictions) / len(temporal_predictions)
        max_prediction = max(temporal_predictions)
        
        print(f"Average temporal cheating probability: {avg_prediction:.4f}")
        print(f"Maximum temporal cheating probability: {max_prediction:.4f}")
        print(f"Percentage of frames above threshold (0.5): "
              f"{sum(p > 0.5 for p in temporal_predictions) / len(temporal_predictions) * 100:.2f}%")
    
    if xgboost_predictions:
        avg_xgb_prediction = sum(xgboost_predictions) / len(xgboost_predictions)
        max_xgb_prediction = max(xgboost_predictions)
        
        print(f"Average XGBoost cheating probability: {avg_xgb_prediction:.4f}")
        print(f"Maximum XGBoost cheating probability: {max_xgb_prediction:.4f}")
        print(f"Percentage of frames above threshold (0.5): "
              f"{sum(p > 0.5 for p in xgboost_predictions) / len(xgboost_predictions) * 100:.2f}%")
        print(f"Maximum XGBoost cheating probability: {max_xgb_prediction:.4f}")
        print(f"Percentage of frames above threshold (0.5): "
              f"{sum(p > 0.5 for p in xgboost_predictions) / len(xgboost_predictions) * 100:.2f}%")
