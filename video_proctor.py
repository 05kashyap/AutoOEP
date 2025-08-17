import cv2
import numpy as np
import torch
import time
import argparse
import os
from collections import deque
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
from typing import cast

# Import the new FeatureExtractor
from Proctor.feature_extractor import FeatureExtractor
from Temporal.temporal_trainer import TemporalProctor

class VideoProctor:
    def __init__(self, lstm_model_path, target_frame_path, mediapipe_model_path,
                 yolo_model_path=None,
                 static_model_path=None, static_scaler_path=None, static_metadata_path=None,
                 window_size=15, input_size=None, buffer_size=30, device=None, debug_features=False):
        """
        Initialize the video proctor that combines frame-by-frame analysis with temporal analysis.
        This version uses the centralized FeatureExtractor.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.debug_features = debug_features

        # Initialize the new FeatureExtractor
        # This now handles all raw feature extraction (YOLO, MediaPipe, StaticProctor logic)
        # Pass suppress flag: if debug enabled, do NOT suppress internal prints
        self.feature_extractor = FeatureExtractor(
            target_frame_path,
            face_landmarker_path=mediapipe_model_path,
            yolo_model_path=(yolo_model_path if yolo_model_path else os.path.join(os.path.dirname(__file__), 'Models', 'OEP_YOLOv11n.pt')),
            suppress_runtime_output=(not debug_features)
        )
        
        # The list of features our models expect, in the correct order.
        self.feature_order = [
            'timestamp', 'verification_result', 'num_faces', 'iris_pos', 'iris_ratio',
            'mouth_zone', 'mouth_area', 'x_rotation', 'y_rotation', 'z_rotation',
            'radial_distance', 'gaze_direction', 'gaze_zone', 'watch', 'headphone',
            'closedbook', 'earpiece', 'cell phone', 'openbook', 'chits', 'sheet',
            'H-Distance', 'F-Distance'
        ]
        
        if debug_features:
            print("🐛 DEBUG MODE: Feature verification enabled")
            print(f"Expecting {len(self.feature_order)} features in this order: {self.feature_order}")

        # Initialize temporal proctor
        self.temporal_proctor = TemporalProctor(window_size=window_size, device=self.device)
        
        # Load the LSTM model with the exact feature size used at inference (prevents mismatches)
        expected_input_size = len(self.feature_order)
        if input_size is not None and input_size != expected_input_size:
            print(f"Warning: Overriding provided --input-size={input_size} with expected {expected_input_size} based on feature order.")
        self.temporal_proctor.load_model(lstm_model_path, expected_input_size)
        # Scaler provenance: prefer loading from checkpoint; otherwise warm-up on first frames
        if self.temporal_proctor.scaler.mean_ is None:
            print("Info: LSTM scaler not found in model file. Will fit scaler on the first few frames during runtime warm-up.")
        else:
            print("LSTM scaler loaded successfully from model file")

        # --- Static Model Loading (e.g., LightGBM, XGBoost) ---
        self.static_model = None
        self.static_scaler = None
        self.static_metadata = None
        if static_model_path:
            try:
                self.static_model = joblib.load(static_model_path)
                print(f"Static model loaded from {static_model_path}")
                if static_scaler_path:
                    self.static_scaler = joblib.load(static_scaler_path)
                    print(f"Static scaler loaded from {static_scaler_path}")
                if static_metadata_path:
                    self.static_metadata = joblib.load(static_metadata_path)
                    print(f"Static metadata loaded from {static_metadata_path}")
            except Exception as e:
                print(f"Error loading static model/scaler/metadata: {e}")

        # Feature buffer for temporal analysis
        self.feature_buffer = deque(maxlen=buffer_size)
        # Warm-up control for temporal scaler
        self._temporal_scaler_fitted = (self.temporal_proctor.scaler.mean_ is not None)
        self._temporal_scaler_warned = False
        self._temporal_scaler_warmup_needed = window_size  # require at least one window to estimate mean/std
        
        # Store prediction history for visualization
        self.timestamps = deque(maxlen=100)
        self.predictions = deque(maxlen=100)
        self.static_scores = deque(maxlen=100)
        
        # For visualization
        self.plot_initialized = False
        self.fig, self.ax, self.line1, self.line2 = None, None, None, None
        
    def initialize_scaler(self):
        """Initialize the scaler with pre-calculated mean and std values from training data"""
        # These values should match the statistics of the training data
        # Approximating values for the 23 features based on typical ranges
        means = np.array([
            0.0, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 500.0, 500.0
        ])
        stds = np.array([
            1.0, 0.5, 0.5, 1.0, 0.2, 1.0, 0.2, 0.2, 0.2, 0.2, 0.5, 1.5, 1.0,
            0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 200.0, 200.0
        ])
        
        self.temporal_proctor.scaler.mean_ = means
        self.temporal_proctor.scaler.scale_ = stds
        print("Scaler initialized with pre-calculated mean and std values")

    def _get_features_as_list(self, feature_dict: dict) -> list:
        """Converts the feature dictionary to a list in the correct order for the models."""
        # Add a relative timestamp
        current_time = time.time()
        if not hasattr(self, 'start_time'):
            self.start_time = current_time
        feature_dict['timestamp'] = current_time - self.start_time
        
        # Create the list, defaulting missing features to 0
        return [feature_dict.get(key, 0) for key in self.feature_order]

    def process_frame_pair(self, face_frame, hand_frame):
        """
        Process a pair of frames (face and hand) to extract features and make predictions.
        This version uses the new FeatureExtractor.
        """
        # 1. Get processed features from the FeatureExtractor
        # This single call replaces all the logic from the old StaticProctor and feature extraction.
        processed_features_dict = self.feature_extractor.process_frame_pair(face_frame, hand_frame)
        
        # 2. Convert feature dictionary to an ordered list for the models
        features_list = self._get_features_as_list(processed_features_dict)
        # Ensure numeric list (floats) for downstream models
        try:
            features_list = [float(x) for x in features_list]
        except Exception:
            # Fallback: coerce via numpy
            features_list = list(np.asarray(features_list, dtype=float))
        
        if self.debug_features:
            print(f"DEBUG: Extracted {len(features_list)} features. Buffer size: {len(self.feature_buffer)}/{self.temporal_proctor.window_size}")
        
        # 3. Add features to buffer for temporal analysis
        self.feature_buffer.append(features_list)
        
        # 4. Make temporal prediction if buffer is ready
        temporal_prediction = None
        # Fit temporal scaler on first window if not loaded from checkpoint
        if not self._temporal_scaler_fitted:
            if len(self.feature_buffer) >= self._temporal_scaler_warmup_needed:
                try:
                    buf_arr = np.array(list(self.feature_buffer), dtype=float)
                    self.temporal_proctor.scaler.fit(buf_arr)
                    self._temporal_scaler_fitted = True
                    if self.debug_features:
                        print(f"Temporal scaler fitted on warm-up buffer of shape {buf_arr.shape}.")
                except Exception as e:
                    if not self._temporal_scaler_warned:
                        print(f"Warning: Failed to warm-up temporal scaler: {e}")
                        self._temporal_scaler_warned = True
        # Only predict when we have enough frames and scaler is ready
        if len(self.feature_buffer) >= self.temporal_proctor.window_size and self._temporal_scaler_fitted:
            temporal_prediction = self.temporal_proctor.make_realtime_prediction(list(self.feature_buffer))
        
        # 5. Make static model prediction on the current frame's features
        static_model_prediction = None
        if self.static_model is not None:
            # Convert features to the format expected by the model (exclude timestamp)
            # Ensure numeric dtype for scaler/model compatibility
            static_features = np.asarray(features_list[1:], dtype=float).reshape(1, -1)
            try:
                if self.static_scaler is not None:
                    static_features = self.static_scaler.transform(static_features)
                static_model_prediction = self.static_model.predict_proba(static_features)[:, 1][0]
            except Exception as e:
                print(f"Error in static model prediction: {e}")
                static_model_prediction = 0.0
        
        # 6. Store results for visualization
        current_time = time.time()
        self.timestamps.append(current_time)
        self.predictions.append(temporal_prediction if temporal_prediction is not None else 0)
        self.static_scores.append(static_model_prediction if static_model_prediction is not None else 0)
        
        # 7. Combine results into a dictionary to return
        results = {
            'static_results': processed_features_dict, # The raw dict from the extractor
            'static_model_prediction': static_model_prediction,
            'temporal_prediction': temporal_prediction,
            'timestamp': current_time
        }
        
        return results
    
    def process_videos(self, face_video_path, hand_video_path,
                  output_path=None, display=True, fps=30, test_duration=None):
        """
        Process video streams from face and hand cameras.
        """
        # Open video captures
        face_cap = cv2.VideoCapture(face_video_path)
        hand_cap = cv2.VideoCapture(hand_video_path)
        
        if not face_cap.isOpened() or not hand_cap.isOpened():
            raise ValueError("Error opening video streams")
        
        out = None
        if output_path or display:
            self.initialize_plot()

        results = []
        frame_count = 0
        start_time = time.time()
        
        max_frames = int(test_duration * fps) if test_duration else None
        
        while True:
            face_ret, face_frame = face_cap.read()
            hand_ret, hand_frame = hand_cap.read()
            
            if not face_ret or not hand_ret:
                break
            
            result = self.process_frame_pair(face_frame, hand_frame)
            results.append(result)
            
            display_frame = self.create_display_frame(face_frame, hand_frame, result)
            
            if out is None and output_path:
                out_height = display_frame.shape[0] + 200  # Extra space for plot
                out_width = display_frame.shape[1]
                fourcc_fn = getattr(cv2, 'VideoWriter_fourcc', None)
                fourcc: int = 0
                if callable(fourcc_fn):
                    try:
                        fourcc = cast(int, fourcc_fn(*'XVID'))  # type: ignore[call-arg]
                    except Exception:
                        fourcc = 0
                out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
            
            self.update_plot()
                
            if out:
                plot_frame = self.get_plot_frame()
                plot_frame = cv2.resize(plot_frame, (display_frame.shape[1], 200))
                combined_frame = np.vstack([display_frame, plot_frame])
                out.write(combined_frame)
            
            if display:
                cv2.imshow('Video Proctor', display_frame)
                if cv2.waitKey(1) & 0xFF == 27: # ESC key
                    break
            
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
            
        elapsed_time = time.time() - start_time
        processed_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({processed_fps:.2f} FPS)")
        
        face_cap.release()
        hand_cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()
        if self.fig is not None:
            plt.close(self.fig)
        
        return results
    
    def create_display_frame(self, face_frame, hand_frame, result):
        """Create a display frame with annotations for visualization."""
        # Resize frames to the same height
        max_height = max(face_frame.shape[0], hand_frame.shape[0])
        if face_frame.shape[0] != max_height:
            face_frame = cv2.resize(face_frame, (int(face_frame.shape[1] * max_height / face_frame.shape[0]), max_height))
        if hand_frame.shape[0] != max_height:
            hand_frame = cv2.resize(hand_frame, (int(hand_frame.shape[1] * max_height / hand_frame.shape[0]), max_height))
        
        # Combine frames side-by-side
        combined_frame = np.hstack([face_frame, hand_frame])
        
        # Get predictions
        static_pred = result.get('static_model_prediction')
        temporal_pred = result.get('temporal_prediction')
        
        # Display Static Prediction
        if static_pred is not None:
            color = (0, 0, 255) if static_pred > 0.5 else (0, 255, 0)
            cv2.putText(combined_frame, f"Static Pred: {static_pred:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display Temporal Prediction
        if temporal_pred is not None:
            color = (0, 0, 255) if temporal_pred > 0.5 else (0, 255, 0)
            cv2.putText(combined_frame, f"Temporal Pred: {temporal_pred:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # Display overall warning
        warning_threshold = 0.7
        if (static_pred and static_pred > warning_threshold) or \
           (temporal_pred and temporal_pred > warning_threshold):
            cv2.putText(combined_frame, "WARNING: CHEATING DETECTED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return combined_frame
    
    def initialize_plot(self):
        """Initialize the matplotlib plot."""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.line1, = self.ax.plot([], [], 'r-', label='Temporal Prediction')
        self.line2, = self.ax.plot([], [], 'b-', label='Static Prediction')
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Cheat Probability')
        self.ax.set_title('Cheating Detection Over Time')
        self.ax.legend()
        self.ax.grid(True)
        self.plot_initialized = True
        plt.show(block=False)
    
    def update_plot(self):
        """Update the plot with new data."""
        if not self.plot_initialized or len(self.timestamps) < 2:
            return
        if self.fig is None or self.ax is None or self.line1 is None or self.line2 is None:
            return
        rel_times = [t - self.timestamps[0] for t in list(self.timestamps)]
        self.line1.set_data(rel_times, list(self.predictions))
        self.line2.set_data(rel_times, list(self.static_scores))
        self.ax.set_xlim(0, rel_times[-1] + 1)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def get_plot_frame(self):
        """Render the matplotlib plot as an image."""
        if self.fig is None:
            # Return a small black strip if plot not ready
            return np.zeros((200, 400, 3), dtype=np.uint8)
        # Render figure to an in-memory PNG and decode with OpenCV to avoid backend-specific methods
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png')
        buf.seek(0)
        data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback to a blank strip on failure
            return np.zeros((200, 400, 3), dtype=np.uint8)
        return img

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process videos with VideoProctor using FeatureExtractor')
    parser.add_argument('--face', type=str, required=True, help='Path to face camera video')
    parser.add_argument('--hand', type=str, required=True, help='Path to hand camera video')
    parser.add_argument('--target', type=str, required=True, help='Path to target/reference face image for identity verification')
    parser.add_argument('--output', type=str, default=None, help='Path to save output video')
    parser.add_argument('--lstm-model', type=str, required=True, help='Path to trained LSTM model')
    parser.add_argument('--mediapipe-task', type=str, required=True, help='Path to mediapipe face_landmarker.task')
    parser.add_argument('--yolo-model', type=str, default=None, help='Path to YOLO model weights (optional; defaults to Models/OEP_YOLOv11n.pt)')
    parser.add_argument('--input-size', type=int, default=23, help='Number of features for LSTM input')
    parser.add_argument('--window-size', type=int, default=15, help='Window size for temporal analysis')
    parser.add_argument('--buffer-size', type=int, default=30, help='Size of feature buffer')
    parser.add_argument('--display', action='store_true', help='Display processed video in real-time')
    parser.add_argument('--test-duration', type=int, default=None, help='Duration in seconds to process for testing')
    parser.add_argument('--static-model', type=str, default=None, help='Path to saved static model (e.g., LightGBM/XGBoost)')
    parser.add_argument('--static-scaler', type=str, default=None, help='Path to static model scaler (optional)')
    parser.add_argument('--static-metadata', type=str, default=None, help='Path to static model metadata (optional)')
    parser.add_argument('--debug', action='store_true', help='Enable debug feature logging')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Initialize VideoProctor
    proctor = VideoProctor(
        lstm_model_path=args.lstm_model,
        target_frame_path=args.target,
        mediapipe_model_path=args.mediapipe_task,
    yolo_model_path=args.yolo_model,
        static_model_path=args.static_model,
        static_scaler_path=args.static_scaler,
        static_metadata_path=args.static_metadata,
        window_size=args.window_size,
        input_size=args.input_size,
        buffer_size=args.buffer_size,
        debug_features=args.debug,
    )
    
    # Process videos
    results = proctor.process_videos(
        face_video_path=args.face,
        hand_video_path=args.hand,
        output_path=args.output,
        display=args.display,
        test_duration=args.test_duration
    )
    
    print(f"Processed {len(results)} frames.")
    
    # Calculate and print overall statistics
    temporal_predictions = [r['temporal_prediction'] for r in results if r.get('temporal_prediction') is not None]
    static_predictions = [r['static_model_prediction'] for r in results if r.get('static_model_prediction') is not None]
    
    if temporal_predictions:
        avg_temporal = np.mean(temporal_predictions)
        max_temporal = np.max(temporal_predictions)
        print(f"\n--- Temporal Model Stats ---")
        print(f"Average cheat probability: {avg_temporal:.4f}")
        print(f"Maximum cheat probability: {max_temporal:.4f}")
        print(f"Frames with >0.5 probability: {np.sum(np.array(temporal_predictions) > 0.5) / len(temporal_predictions) * 100:.2f}%")

    if static_predictions:
        avg_static = np.mean(static_predictions)
        max_static = np.max(static_predictions)
        print(f"\n--- Static Model Stats ---")
        print(f"Average cheat probability: {avg_static:.4f}")
        print(f"Maximum cheat probability: {max_static:.4f}")
        print(f"Frames with >0.5 probability: {np.sum(np.array(static_predictions) > 0.5) / len(static_predictions) * 100:.2f}%")

