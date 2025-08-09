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
from core.statistics import StatisticsCalculator
from core.image_processor import ImageProcessor
from core.data_handler import DataHandler

# Proctor component imports
from Proctor.static_proctor import StaticProctor
from Proctor.temporal_proctor import TemporalProctor

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
        self.statistics = StatisticsCalculator()
        self.image_processor = ImageProcessor()
        self.data_handler = DataHandler()
        
        # Initialize proctor components
        self.static_proctor = None
        self.temporal_proctor = None
        
        # Target image for identity verification
        self.target_image = None
        try:
            if target_image_path and os.path.exists(target_image_path):
                self.target_image = cv2.imread(target_image_path)
            else:
                print(f"Warning: Target image not found at {target_image_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading target image: {e}")

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
            
            # Initialize temporal proctor (inference-only)
            print("📦 Initializing temporal proctor (inference)...")
            self.temporal_proctor = TemporalProctor()
            
            # Load pre-trained temporal models - REQUIRED, no fallbacks
            if not os.path.exists(self.model_save_dir):
                raise FileNotFoundError(f"Model directory is required but not found: {self.model_save_dir}")
            
            print(f"📥 Loading temporal models from: {self.model_save_dir}")
            self.temporal_proctor.load_models(None)  # Use Config path instead of directory
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

    def _safe_float(self, value, default=0.0):
        """Safely convert a value to float, returning default on failure."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
    
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
            
            # Ensure proctor components are initialized
            if self.static_proctor is None or self.temporal_proctor is None:
                raise RuntimeError("Proctor components not initialized")
            
            # Get target frame (use first face frame if no target image)
            target_frame = self.target_image if self.target_image is not None else face_frame
            
            # Static analysis
            static_results = self.static_proctor.process_frames(target_frame, face_frame, hand_frame)

            # Add to temporal sequence
            self.temporal_proctor.add_frame_features(static_results)
            
            # Get temporal prediction
            temporal_score = self.temporal_proctor.get_temporal_prediction()
            
            # Combine results
            combined_results = static_results.copy()
            combined_results['Temporal Score'] = temporal_score
            # Ensure compatibility keys for statistics/visualizer
            combined_results['temporal_prediction'] = temporal_score
            combined_results['xgboost_prediction'] = combined_results.get('xgboost_prediction', None)
            combined_results['static_model_prediction'] = combined_results.get('static_model_prediction', None)
            
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
            print("Press 'q' to quit, 'r' to reset session, 's' for training info")
            
            while True:
                # Read frames
                face_ret, face_frame = face_cap.read()
                hand_ret, hand_frame = hand_cap.read()
                
                if not face_ret or not hand_ret:
                    print("Failed to read from cameras")
                    break
                
                # Process frames
                results = self.process_frame_pair(face_frame, hand_frame)
                
                if 'error' not in results and display:
                    # Prepare visualization payload compatible with Visualizer
                    viz_payload = {
                        'static_results': results,
                        'temporal_prediction': results.get('Temporal Score'),
                        'xgboost_prediction': None,
                        'static_model_prediction': None,
                    }
                    
                    # Create a single combined annotated frame for display
                    combined_frame = self.visualizer.create_display_frame(
                        face_frame, hand_frame, viz_payload
                    )
                    cv2.imshow('Video Proctoring', combined_frame)
                
                    # Print key metrics (separate scores)
                    static_score_val = self._safe_float(results.get('Cheat Score', 0.0), 0.0)
                    temporal_score_val = self._safe_float(results.get('Temporal Score', 0.0), 0.0)
                    if static_score_val > Config.ALERT_THRESHOLD or temporal_score_val > Config.ALERT_THRESHOLD:
                        print(f"ALERT: High probability detected - Static: {static_score_val:.3f}, Temporal: {temporal_score_val:.3f}")
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_session()
                elif key == ord('s'):
                    self.save_models()
            
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
        results = []
        
        # Open video captures
        face_cap = cv2.VideoCapture(face_video_path)
        hand_cap = cv2.VideoCapture(hand_video_path)
        
        if not face_cap.isOpened():
            raise ValueError(f"Could not open face video: {face_video_path}")
        if not hand_cap.isOpened():
            raise ValueError(f"Could not open hand video: {hand_video_path}")
        
        try:
            frame_count = 0
            max_frames = min(
                int(face_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(hand_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            )
            
            print(f"Processing {max_frames} frames...")
            
            while True:
                face_ret, face_frame = face_cap.read()
                hand_ret, hand_frame = hand_cap.read()
                
                if not face_ret or not hand_ret:
                    break
                
                # Process frame pair
                try:
                    frame_result = self.process_frame_pair(face_frame, hand_frame)
                    frame_result['frame_number'] = str(frame_count)
                    results.append(frame_result)
                except Exception as e:
                    results.append({
                        'frame_number': str(frame_count),
                        'error': str(e),
                        'Cheat Score': 0.0,
                        'Temporal Score': 0.0
                    })
                
                frame_count += 1
                
                # Progress update
                if frame_count % 25 == 0:
                    print(f"  Progress: {frame_count}/{max_frames} frames processed")
                
        finally:
            face_cap.release()
            hand_cap.release()
        
        print(f"✅ Completed processing {len(results)} frames")
        return results
    
    def train_temporal_models(self, training_data_dir):
        """
        Training is now handled by separate training script
        Use Training/temporal_trainer.py for temporal training
        and Training/static_trainer.py for static model training
        
        Args:
            training_data_dir: Directory containing training data
        """
        print("⚠️  Training has been moved to separate scripts for better organization!")
        print("To train temporal models, run:")
        print("  python Training/temporal_trainer.py --data_dir <path> --model_type LSTM")
        print("To train the static model, run:")
        print("  python Training/static_trainer.py --data <csv> --model-out <pkl> --scaler-out <pkl> --metadata-out <pkl>")
        print(f"Provided training data directory: {training_data_dir}")
        print("\nThis keeps the VideoProctor focused on inference and real-time processing.")
        print("After training, update Config paths so VideoProctor can load the new models.")
    
    def save_models(self):
        """Model saving is now handled by training scripts"""
        print("⚠️  Model saving has been moved to training scripts!")
        print("Use the scripts under Training/:")
        print("  - Training/temporal_trainer.py (saves .pth)")
        print("  - Training/static_trainer.py (saves model/scaler/metadata .pkl)")
        print("Update config.py to point to the newly saved artifacts.")
    
    def get_session_statistics(self):
        """Get comprehensive session statistics"""
        stats = self.statistics.get_comprehensive_stats()
        if self.temporal_proctor is not None:
            temporal_stats = self.temporal_proctor.get_statistics()
            stats.update(temporal_stats)
        return stats
    
    def reset_session(self):
        """Reset session data and statistics"""
        if self.temporal_proctor is not None:
            self.temporal_proctor.reset_history()
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
