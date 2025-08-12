"""
Refactored Static Proctor - Clean implementation for frame-by-frame analysis
"""
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VisionUtils.handpose import inference
from VisionUtils.face_inference import get_face_inference
from core.image_processor import ImageProcessor

try:
    from config import Config
    CONFIG_LOADED = True
except ImportError:
    print("Warning: Could not load config module")
    CONFIG_LOADED = False
    Config = None  # Ensure symbol exists for type checkers

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class StaticProctor:
    """
    Handles frame-by-frame analysis for cheating detection
    Combines hand detection, face analysis, and identity verification
    """
    
    def __init__(self, yolo_model, media_pipe_dict, model_path):
        """
        Initialize static proctor with required models
        
        Args:
            yolo_model: YOLO model for object detection
            media_pipe_dict: MediaPipe configuration dictionary
            model_path: Path to MediaPipe face landmark model
        """
        self.yolo_model = yolo_model
        self.media_pipe = media_pipe_dict
        self.image_processor = ImageProcessor()
        
        # Initialize MediaPipe FaceLandmarker
        self._setup_face_landmarker(model_path)
    
    def _setup_face_landmarker(self, model_path):
        """Setup MediaPipe face landmarker with validation"""
        # Validate that model path exists
        if not model_path:
            raise ValueError("MediaPipe face landmarker model path is required")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"MediaPipe face landmarker model not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as file:
                model_data = file.read()
            
            self.options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=model_data),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=3
            )
            
            self.landmarker = FaceLandmarker.create_from_options(self.options)
            print(f"MediaPipe face landmarker initialized with model: {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe face landmarker: {e}")
    
    def process_frames(self, target_frame, face_frame, hand_frame):
        """
        Process a set of frames to extract cheating detection features
        
        Args:
            target_frame: Reference image for identity verification
            face_frame: Current face camera frame
            hand_frame: Current hand/desk camera frame
            
        Returns:
            Dictionary with analysis results
        """
        output = {}
        
        # Process hand detection on hand camera
        hand_results = self._process_hand_detection(hand_frame, 'H')
        output.update(hand_results)
        
        # Process hand detection on face camera (for objects near face)
        face_hand_results = self._process_hand_detection(face_frame, 'F')
        output.update(face_hand_results)
        
        # Process face analysis and identity verification
        face_results = self._process_face_analysis(face_frame, target_frame)
        output.update(face_results)
        
        # Calculate overall cheat score
        output['Cheat Score'] = self._calculate_cheat_score(output)
        
        return output
    
    def _process_hand_detection(self, frame, prefix):
        """
        Process hand detection and object recognition
        
        Args:
            frame: Input frame for analysis
            prefix: Prefix for output keys ('H' for hand cam, 'F' for face cam)
            
        Returns:
            Dictionary with hand detection results
        """
        results = {}
        
        # Preprocess frame for better detection
        processed_frame = self.image_processor.preprocess_for_object_detection(frame)
        
        # Run hand detection inference
        hand_dict = inference(processed_frame, self.yolo_model, self.media_pipe)
        
        if hand_dict:
            # Map detection results to output format
            key_mapping = {
                'hand_detected': f'{prefix}-Hand Detected',
                'prohibited_item_use': f'{prefix}-Prohibited Item Use', 
                'distance': f'{prefix}-Distance',
                'illegal_objects': f'{prefix}-Illegal Objects',
                'prohibited_item': f'{prefix}-Prohibited Item'
            }
            
            for src_key, dst_key in key_mapping.items():
                if src_key in hand_dict:
                    results[dst_key] = hand_dict[src_key]
        
        return results
    
    def _process_face_analysis(self, face_frame, target_frame):
        """
        Process face analysis including landmarks and identity verification
        
        Args:
            face_frame: Current face frame
            target_frame: Reference frame for identity verification
            
        Returns:
            Dictionary with face analysis results
        """
        # Get comprehensive face analysis
        face_details = get_face_inference(face_frame, target_frame, self.landmarker)
        return face_details
    
    def _calculate_cheat_score(self, output):
        """
        Calculate overall cheating probability score
        
        Args:
            output: Dictionary with all detection results
            
        Returns:
            Float between 0 and 1 indicating cheating probability
        """
        # Scoring aligned with legacy VideoProctor; weights come from Config
        score = 0.0

        # Resolve weights (fallback to defaults if Config not loaded)
        prohibited_item_w = getattr(Config, 'PROHIBITED_ITEM_SCORE', 0.3) if CONFIG_LOADED else 0.3
        identity_fail_w = getattr(Config, 'IDENTITY_FAILURE_SCORE', 0.4) if CONFIG_LOADED else 0.4
        suspicious_gaze_w = getattr(Config, 'SUSPICIOUS_GAZE_SCORE', 0.2) if CONFIG_LOADED else 0.2
        multiple_faces_w = getattr(Config, 'MULTIPLE_FACES_SCORE', 0.1) if CONFIG_LOADED else 0.1

        # Prohibited items detected on either stream additively
        for item_key in ('H-Prohibited Item', 'F-Prohibited Item'):
            if output.get(item_key):
                score += prohibited_item_w

        # Identity verification failure
        if 'verification_result' in output and not bool(output['verification_result']):
            score += identity_fail_w

        # Suspicious gaze zone
        if output.get('gaze_zone') == 'red':
            score += suspicious_gaze_w

        # Multiple faces present
        try:
            if float(output.get('num_faces', 0)) > 1:
                score += multiple_faces_w
        except Exception:
            pass

        return min(score, 1.0)
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'landmarker'):
            try:
                self.landmarker.close()
            except:
                pass


def create_test_proctor():
    """Create a test instance of StaticProctor for development"""
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use Config for YOLO model path
    if CONFIG_LOADED:
        from config import Config
        yolo_model_path = Config.DEFAULT_YOLO_MODEL
    else:
        raise RuntimeError("Config not loaded - cannot determine YOLO model path")
    
    if not Path(yolo_model_path).exists():
        raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")
    
    model = YOLO(yolo_model_path)
    
    # Build MediaPipe config via central Config helper to avoid direct mp.solutions references
    media_pipe_dict = Config.get_mediapipe_config()
    
    # Use Config path for MediaPipe model
    if CONFIG_LOADED:
        from config import Config
        model_path = Config.DEFAULT_MEDIAPIPE_MODEL
    else:
        raise RuntimeError("Config not loaded - cannot determine MediaPipe model path")
    
    return StaticProctor(model, media_pipe_dict, model_path)


if __name__ == '__main__':
    # Test the refactored StaticProctor
    try:
        proctor = create_test_proctor()
        
        # Load test frames
        target_frame = cv2.imread('Images/identity.jpeg')
        face_frame = cv2.imread('Images/facecam1.png')
        hand_frame = cv2.imread('Images/test.jpg')
        
        if all(frame is not None for frame in [target_frame, face_frame, hand_frame]):
            result = proctor.process_frames(target_frame, face_frame, hand_frame)
            print("Analysis Results:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print("Test images not found. Please ensure test images are available.")
            
    except Exception as e:
        print(f"Test failed: {e}")
        print("This is expected if test images or models are not available.")
