"""
Clean Video Proctor - Final version
Main orchestrator for the video proctoring system
"""
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add necessary paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "core"))
sys.path.append(str(current_dir / "Proctor"))

# Core module imports
from core.feature_extractor import FeatureExtractor
from core.model_manager import ModelManager
from core.debug_validator import DebugValidator
from core.visualizer import Visualizer
from core.video_processor import VideoProcessor
from core.statistics import Statistics
from core.image_processor import ImageProcessor
from core.data_handler import DataHandler
from core.trainer import Trainer

# Proctor component imports
from Proctor.static_proctor import StaticProctor
from Proctor.temporal_trainer_enhanced import TemporalTrainerEnhanced

# Configuration
from config import Config


class VideoProctor:
    """
    Main video proctoring system that orchestrates all components
    Combines static frame analysis with temporal sequence learning
    """
    
    def __init__(self, target_image_path=None, model_save_dir=None, debug_mode=False):
        """
        Initialize the video proctoring system
        
        Args:
            target_image_path: Path to reference image for identity verification
            model_save_dir: Directory to save/load trained models
            debug_mode: Enable detailed debugging output
        """
        self.debug_mode = debug_mode
        self.model_save_dir = model_save_dir or Config.DEFAULT_MODEL_DIR
        
        # Initialize core components
        self.feature_extractor = FeatureExtractor()
        self.model_manager = ModelManager()
        self.debug_validator = DebugValidator()
        self.visualizer = Visualizer()
        self.video_processor = VideoProcessor()
        self.statistics = Statistics()
        self.image_processor = ImageProcessor()
        self.data_handler = DataHandler()
        self.trainer = Trainer()
        
        # Initialize proctor components
        self.static_proctor = None
        self.temporal_trainer = None
        
        # Target image for identity verification
        self.target_image = None
        if target_image_path and os.path.exists(target_image_path):
            self.target_image = cv2.imread(target_image_path)
        
        # Setup models
        self._setup_models()
        
        print("VideoProctor initialized successfully")
    
    def _setup_models(self):
        """Setup YOLO and MediaPipe models"""
        try:
            # Load YOLO and MediaPipe
            yolo_model, media_pipe_dict = self._setup_yolo_and_mediapipe()
            
            # Initialize static proctor
            self.static_proctor = StaticProctor(
                yolo_model, 
                media_pipe_dict, 
                Config.DEFAULT_MEDIAPIPE_MODEL
            )
            
            # Initialize temporal trainer with STRICT requirements
            print("📦 Initializing enhanced temporal trainer...")
            self.temporal_trainer = TemporalTrainerEnhanced()
            
            # Load pre-trained temporal models - REQUIRED, no fallbacks
            if not os.path.exists(self.model_save_dir):
                raise FileNotFoundError(f"Model directory is required but not found: {self.model_save_dir}")
            
            print(f"📥 Loading temporal models from: {self.model_save_dir}")
            self.temporal_trainer.load_models(self.model_save_dir)
            print("✅ Temporal models loaded successfully")
            
            # Load static models (LightGBM, scaler, metadata) - REQUIRED
            self._load_static_models()
            
            print("✅ All models setup completed successfully")
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR in model setup: {e}")
            print("💡 Ensure all required model files are present:")
            print(f"   - Temporal models in: {self.model_save_dir}")
            print(f"   - Static model: {Config.DEFAULT_STATIC_MODEL}")
            print(f"   - MediaPipe model: {Config.DEFAULT_MEDIAPIPE_MODEL}")
            raise RuntimeError(f"Failed to initialize VideoProctor: {e}")
            raise
    
    def _setup_yolo_and_mediapipe(self):
        """Setup YOLO and MediaPipe models using model manager"""
        return self.model_manager.load_detection_models()
    
    def _load_static_models(self):
        """Load static models (LightGBM, scaler, metadata) - STRICT mode, all required"""
        try:
            # Verify all static model components exist
            if not hasattr(Config, 'DEFAULT_STATIC_MODEL'):
                raise RuntimeError("DEFAULT_STATIC_MODEL not configured")
            
            if not os.path.exists(Config.DEFAULT_STATIC_MODEL):
                raise FileNotFoundError(f"Static model file not found: {Config.DEFAULT_STATIC_MODEL}")
            
            scaler_path = getattr(Config, 'DEFAULT_STATIC_SCALER', None)
            metadata_path = getattr(Config, 'DEFAULT_STATIC_METADATA', None)
            
            if scaler_path and not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            if metadata_path and not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            # Load static model components
            self.model_manager.load_static_model(
                Config.DEFAULT_STATIC_MODEL,
                scaler_path,
                metadata_path
            )
            print("✅ Static models (LightGBM, scaler, metadata) loaded successfully")
                
        except Exception as e:
            print(f"❌ CRITICAL ERROR loading static models: {e}")
            print("💡 Required static model files:")
            print(f"   - Model: {getattr(Config, 'DEFAULT_STATIC_MODEL', 'NOT_CONFIGURED')}")
            print(f"   - Scaler: {getattr(Config, 'DEFAULT_STATIC_SCALER', 'NOT_CONFIGURED')}")
            print(f"   - Metadata: {getattr(Config, 'DEFAULT_STATIC_METADATA', 'NOT_CONFIGURED')}")
            raise RuntimeError(f"Failed to load required static models: {e}")
    
    def process_frame_pair(self, face_frame, hand_frame):
        """
        Process a pair of frames (face camera + hand/desk camera)
        
        Args:
            face_frame: Frame from face camera
            hand_frame: Frame from hand/desk camera
            
        Returns:
            Dictionary with complete analysis results
        """
        try:
            # Validate frames
            if not self.debug_validator.validate_frame_pair(face_frame, hand_frame):
                return {'error': 'Invalid frame pair'}
            
            # Get target frame (use first face frame if no target image)
            target_frame = self.target_image if self.target_image is not None else face_frame
            
            # Static analysis
            static_results = self.static_proctor.process_frames(
                target_frame, face_frame, hand_frame
            )
            
            # Add to temporal sequence
            self.temporal_trainer.add_frame_features(static_results)
            
            # Get temporal prediction
            temporal_score = self.temporal_trainer.get_temporal_prediction()
            
            # Combine results
            combined_results = static_results.copy()
            combined_results['Temporal Score'] = temporal_score
            combined_results['Final Score'] = self._calculate_final_score(
                static_results.get('Cheat Score', 0.0),
                temporal_score
            )
            
            # Update statistics
            self.statistics.update_frame_stats(combined_results)
            
            # Debug validation
            if self.debug_mode:
                self.debug_validator.validate_feature_ranges(combined_results)
            
            return combined_results
            
        except Exception as e:
            error_msg = f"Error processing frame pair: {e}"
            print(error_msg)
            return {'error': error_msg}
    
    def _calculate_final_score(self, static_score, temporal_score):
        """
        Calculate final cheating probability score
        
        Args:
            static_score: Score from static frame analysis
            temporal_score: Score from temporal sequence analysis
            
        Returns:
            Combined final score (0-1)
        """
        # Weighted combination
        final_score = (
            Config.STATIC_WEIGHT * static_score + 
            Config.TEMPORAL_WEIGHT * temporal_score
        )
        
        return min(final_score, 1.0)  # Cap at 1.0
    
    def process_video_stream(self, face_camera_id=0, hand_camera_id=1, display=True):
        """
        Process live video streams from two cameras
        
        Args:
            face_camera_id: Camera ID for face camera
            hand_camera_id: Camera ID for hand/desk camera
            display: Whether to display video feed with annotations
        """
        try:
            # Initialize video captures
            face_cap = cv2.VideoCapture(face_camera_id)
            hand_cap = cv2.VideoCapture(hand_camera_id)
            
            if not face_cap.isOpened() or not hand_cap.isOpened():
                raise ValueError("Could not open one or both cameras")
            
            print("Starting video stream processing...")
            print("Press 'q' to quit, 's' to save current model state")
            
            while True:
                # Read frames
                face_ret, face_frame = face_cap.read()
                hand_ret, hand_frame = hand_cap.read()
                
                if not face_ret or not hand_ret:
                    print("Failed to read from cameras")
                    break
                
                # Process frames
                results = self.process_frame_pair(face_frame, hand_frame)
                
                if 'error' not in results:
                    # Display results if enabled
                    if display:
                        annotated_frames = self.visualizer.create_annotated_display(
                            face_frame, hand_frame, results
                        )
                        cv2.imshow('Video Proctoring - Face Camera', annotated_frames[0])
                        cv2.imshow('Video Proctoring - Hand Camera', annotated_frames[1])
                    
                    # Print key metrics
                    final_score = results.get('Final Score', 0.0)
                    if final_score > Config.ALERT_THRESHOLD:
                        print(f"ALERT: High cheating probability detected: {final_score:.3f}")
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_models()
                    print("Models saved successfully")
            
            # Cleanup
            face_cap.release()
            hand_cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            stats = self.get_session_statistics()
            print("\nSession Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"Error in video stream processing: {e}")
            raise
    
    def process_video_files(self, face_video_path, hand_video_path, output_path=None):
        """
        Process pre-recorded video files
        
        Args:
            face_video_path: Path to face camera video
            hand_video_path: Path to hand camera video  
            output_path: Optional path to save annotated output video
            
        Returns:
            List of frame analysis results
        """
        return self.video_processor.process_video_files(
            face_video_path, hand_video_path, output_path,
            process_callback=self.process_frame_pair
        )
    
    def train_temporal_models(self, training_data_dir):
        """
        Train temporal models with labeled data
        
        Args:
            training_data_dir: Directory containing training data
        """
        # Load training data
        training_sequences, labels = self.data_handler.load_training_data(training_data_dir)
        
        if len(training_sequences) > 0:
            # Train models
            self.temporal_trainer.train_models(training_sequences, labels)
            print(f"Trained models with {len(training_sequences)} sequences")
        else:
            print("No training data found")
    
    def save_models(self):
        """Save trained temporal models"""
        try:
            self.temporal_trainer.save_models(self.model_save_dir)
            print(f"Models saved to {self.model_save_dir}")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def get_session_statistics(self):
        """Get comprehensive session statistics"""
        stats = self.statistics.get_comprehensive_stats()
        temporal_stats = self.temporal_trainer.get_statistics()
        stats.update(temporal_stats)
        return stats
    
    def reset_session(self):
        """Reset session data and statistics"""
        self.temporal_trainer.reset_history()
        self.statistics.reset()
        print("Session reset completed")
    
    def set_target_image(self, image_path):
        """
        Set new target image for identity verification
        
        Args:
            image_path: Path to new target image
        """
        if os.path.exists(image_path):
            self.target_image = cv2.imread(image_path)
            print(f"Target image updated: {image_path}")
        else:
            print(f"Target image not found: {image_path}")
    
    def __del__(self):
        """Cleanup resources"""
        try:
            cv2.destroyAllWindows()
        except:
            pass


# Convenience function for quick setup
def create_video_proctor(target_image_path=None, debug_mode=False):
    """
    Create and return a configured VideoProctor instance
    
    Args:
        target_image_path: Path to reference image
        debug_mode: Enable debug output
        
    Returns:
        Configured VideoProctor instance
    """
    return VideoProctor(target_image_path=target_image_path, debug_mode=debug_mode)


if __name__ == "__main__":
    # Example usage
    try:
        # Create video proctor
        proctor = create_video_proctor(debug_mode=True)
        
        # Start live monitoring
        proctor.process_video_stream(display=True)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
