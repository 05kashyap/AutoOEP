"""
Main entry point for the refactored video proctor application with full functionality
"""
import os
import sys
import argparse
import cv2
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from config import Config
    CONFIG_LOADED = True
except Exception as e:
    print(f"Warning: Could not load config: {e}")
    CONFIG_LOADED = False


def setup_cheating_logger():
    """Setup logging for cheating detection events"""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"cheating_detection_{timestamp}.log"
    
    # Setup logger
    logger = logging.getLogger('cheating_detector')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Video Proctoring System Test')
    parser.add_argument('--debug', action='store_true', help='Enable debug features')
    parser.add_argument('--demo', action='store_true', help='Run camera demo')
    parser.add_argument('--test-imports', action='store_true', help='Test import functionality')
    parser.add_argument('--process-videos', action='store_true', help='Process video files from Inputs folder')
    parser.add_argument('--no-display', action='store_true', help='Disable video display for faster processing')
    return parser.parse_args()


def test_imports():
    """Test all import functionality"""
    print("🧪 Testing System Imports")
    print("=" * 50)
    
    # Test core Python libraries
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
    except AttributeError:
        print("✅ OpenCV: Available (version info unavailable)")
        
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe: {mp.__version__}")
    except ImportError as e:
        print(f"❌ MediaPipe: {e}")
        
    try:
        from ultralytics import YOLO
        print("✅ YOLO (ultralytics)")
    except ImportError as e:
        print(f"❌ YOLO: {e}")
    
    # Test our modules
    print("\n📦 Testing Project Modules:")
    
    if CONFIG_LOADED:
        print("✅ Config module")
    else:
        print("❌ Config module")
    
    try:
        from VisionUtils.handpose import inference
        print("✅ VisionUtils.handpose")
    except ImportError as e:
        print(f"❌ VisionUtils.handpose: {e}")
        
    try:
        from VisionUtils.face_inference import get_face_inference
        print("✅ VisionUtils.face_inference")
    except ImportError as e:
        print(f"❌ VisionUtils.face_inference: {e}")
        
    try:
        from Proctor.static_proctor import StaticProctor
        print("✅ Proctor.StaticProctor")
    except ImportError as e:
        print(f"❌ Proctor.StaticProctor: {e}")
        
    try:
        from Proctor.temporal_trainer_enhanced import TemporalTrainerEnhanced
        print("✅ Proctor.TemporalTrainer")
    except ImportError as e:
        print(f"❌ Proctor.TemporalTrainer: {e}")


def test_camera():
    """Test camera functionality"""
    print("\n📹 Testing Camera Access")
    print("=" * 30)
    
    for cam_id in range(3):  # Test first 3 camera IDs
        try:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Camera {cam_id}: Available ({frame.shape})")
                else:
                    print(f"⚠️ Camera {cam_id}: Opens but no frame")
                cap.release()
            else:
                print(f"❌ Camera {cam_id}: Not available")
        except Exception as e:
            print(f"❌ Camera {cam_id}: Error - {e}")


def process_input_videos():
    """Process video files from the Inputs folder with full proctor functionality"""
    print("\n🎬 Processing Input Videos with Full Proctor Analysis")
    print("=" * 60)
    
    # Setup cheating detection logger
    cheat_logger, log_file = setup_cheating_logger()
    cheat_logger.info("=== CHEATING DETECTION SESSION STARTED ===")
    print(f"📝 Logging cheating detections to: {log_file}")
    
    # Define input paths
    inputs_dir = Path("Inputs")
    face_video = inputs_dir / "front_test.avi"
    hand_video = inputs_dir / "side_test.avi"
    target_image = inputs_dir / "ID.png"
    
    # Verify input files exist
    missing_files = []
    for file_path, name in [(face_video, "Face video"), (hand_video, "Hand video"), (target_image, "Target image")]:
        if not file_path.exists():
            missing_files.append(f"❌ {name}: {file_path}")
        else:
            print(f"✅ {name}: {file_path}")
    
    if missing_files:
        print("Missing required files:")
        for missing in missing_files:
            print(f"  {missing}")
        return False
    
    try:
        # Import the full proctor components
        print("\n📦 Importing proctor components...")
        from Proctor.static_proctor import StaticProctor
        from Proctor.temporal_trainer_enhanced import TemporalTrainerEnhanced
        from ultralytics import YOLO
        import mediapipe as mp
        import numpy as np
        print("✅ All components imported successfully")
        
        # Load target image
        target_frame = cv2.imread(str(target_image))
        if target_frame is None:
            print(f"❌ Could not load target image: {target_image}")
            return False
        print(f"✅ Target image loaded: {target_frame.shape}")
        
        # Initialize YOLO model
        print("\n🤖 Initializing AI models...")
        yolo_model = None
        try:
            yolo_model_path = inputs_dir / "Models" / "OEP_YOLOv11n.pt"
            if yolo_model_path.exists():
                print(f"✅ Found YOLO model at: {yolo_model_path}")
                yolo_model = YOLO(str(yolo_model_path))
                print("✅ YOLO model loaded successfully")
            else:
                print(f"❌ YOLO model not found at: {yolo_model_path}")
                # Try alternative paths
                alt_paths = [
                    "Models/OEP_YOLOv11n.pt",
                    "OEP_YOLOv11n.pt",
                    inputs_dir / "OEP_YOLOv11n.pt"
                ]
                for alt_path in alt_paths:
                    if Path(alt_path).exists():
                        print(f"✅ Found YOLO model at alternative path: {alt_path}")
                        yolo_model = YOLO(str(alt_path))
                        print("✅ YOLO model loaded successfully")
                        break
                else:
                    print("⚠️ YOLO model not found, using basic detection")
                    yolo_model = None
        except Exception as e:
            print(f"⚠️ YOLO model failed to load: {e}")
            yolo_model = None
        
        # Initialize MediaPipe
        media_pipe_dict = None
        try:
            # Check for MediaPipe API structure
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands') and hasattr(mp.solutions, 'drawing_utils'):
                mp_hands = mp.solutions.hands
                hands = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                mp_drawing = mp.solutions.drawing_utils
                media_pipe_dict = {
                    'mpHands': mp_hands,
                    'hands': hands,
                    'mpdraw': mp_drawing
                }
                print("✅ MediaPipe initialized successfully (solutions API)")
            else:
                print("⚠️ MediaPipe solutions API not available - continuing without MediaPipe")
                media_pipe_dict = None
        except Exception as e:
            print(f"⚠️ MediaPipe initialization failed: {e}")
            media_pipe_dict = None
        
        # Initialize MediaPipe Face Landmarker
        face_landmarker_path = None
        try:
            face_landmarker_path = inputs_dir / "Models" / "face_landmarker.task"
            if face_landmarker_path.exists():
                print(f"✅ Face landmarker model found at: {face_landmarker_path}")
            else:
                print(f"❌ Face landmarker model not found at: {face_landmarker_path}")
                # Try alternative paths
                alt_paths = [
                    "face_landmarker.task",
                    "Models/face_landmarker.task",
                    inputs_dir / "face_landmarker.task"
                ]
                for alt_path in alt_paths:
                    if Path(alt_path).exists():
                        face_landmarker_path = alt_path
                        print(f"✅ Found face landmarker at alternative path: {face_landmarker_path}")
                        break
                else:
                    print("⚠️ Face landmarker model not found")
                    face_landmarker_path = None
        except Exception as e:
            print(f"⚠️ Face landmarker setup issue: {e}")
            face_landmarker_path = None
        
        # Initialize Static Proctor
        print("\n🔍 Initializing Static Proctor...")
        static_proctor = None
        if yolo_model:
            try:
                # Initialize with available components
                face_landmarker_str = str(face_landmarker_path) if face_landmarker_path else "face_landmarker.task"
                static_proctor = StaticProctor(yolo_model, media_pipe_dict, face_landmarker_str)
                print("✅ Static Proctor initialized successfully")
            except Exception as e:
                print(f"⚠️ Static Proctor initialization failed: {e}")
                static_proctor = None
        else:
            print("⚠️ Static Proctor skipped due to missing YOLO model")
        
        # Initialize Temporal Trainer
        print("\n⏱️ Initializing Temporal Trainer...")
        temporal_trainer = None
        try:
            temporal_trainer = TemporalTrainerEnhanced(window_size=15)
            print("✅ Temporal Trainer initialized successfully")
        except Exception as e:
            print(f"❌ Temporal Trainer initialization failed: {e}")
            temporal_trainer = None
        
        # Open video files
        print("\n📹 Opening video files...")
        face_cap = cv2.VideoCapture(str(face_video))
        hand_cap = cv2.VideoCapture(str(hand_video))
        
        if not face_cap.isOpened():
            print(f"❌ Could not open face video: {face_video}")
            return False
            
        if not hand_cap.isOpened():
            print(f"❌ Could not open hand video: {hand_video}")
            face_cap.release()
            return False
        
        # Get video properties
        face_fps = face_cap.get(cv2.CAP_PROP_FPS)
        hand_fps = hand_cap.get(cv2.CAP_PROP_FPS)
        face_frames = int(face_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        hand_frames = int(hand_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Face video: {face_frames} frames @ {face_fps:.1f} FPS")
        print(f"📊 Hand video: {hand_frames} frames @ {hand_fps:.1f} FPS")
        
        # Process frames with full proctor analysis
        frame_count = 0
        max_frames = min(face_frames, hand_frames, 300)  # Process up to 300 frames for testing
        
        print(f"🎯 Processing {max_frames} frames with proctor analysis...")
        
        cheating_detections = []
        temporal_features = []
        total_cheat_count = 0  # Counter for total cheating instances
        
        cheat_logger.info(f"Starting analysis of {max_frames} frames")
        cheat_logger.info(f"Face video: {face_frames} frames @ {face_fps:.1f} FPS")
        cheat_logger.info(f"Hand video: {hand_frames} frames @ {hand_fps:.1f} FPS")
        
        while frame_count < max_frames:
            face_ret, face_frame = face_cap.read()
            hand_ret, hand_frame = hand_cap.read()
            
            if not face_ret or not hand_ret:
                break
            
            frame_count += 1
            
            # Progress update every 25 frames
            if frame_count % 25 == 0:
                print(f"  📹 Processed {frame_count}/{max_frames} frames...")
            
            try:
                # Static proctor analysis
                static_result = None
                if static_proctor:
                    try:
                        static_result = static_proctor.process_frames(target_frame, face_frame, hand_frame)
                        cheat_score = static_result.get('Cheat Score', 0) if static_result else 0
                        
                        # Use configurable threshold
                        cheat_threshold = Config.CHEATING_THRESHOLD if CONFIG_LOADED else 0.5
                        
                        if cheat_score > cheat_threshold:  # Cheating detected!
                            total_cheat_count += 1
                            timestamp_str = f"{frame_count / face_fps:.2f}s" if face_fps > 0 else f"frame_{frame_count}"
                            
                            # Real-time cheating alert
                            print(f"  🚨 CHEAT DETECTED #{total_cheat_count} at {timestamp_str} (Score: {cheat_score:.2f})")
                            
                            # Log to file with detailed information
                            cheat_details = []
                            if static_result.get('H-Hand Detected'):
                                cheat_details.append("Hand in hand camera")
                            if static_result.get('F-Hand Detected'):
                                cheat_details.append("Hand in face camera")
                            if static_result.get('H-Prohibited Item'):
                                cheat_details.append("Prohibited item in hand camera")
                            if static_result.get('F-Prohibited Item'):
                                cheat_details.append("Prohibited item in face camera")
                            if not static_result.get('verification_result', True):
                                cheat_details.append("Identity verification failed")
                            
                            details_str = ", ".join(cheat_details) if cheat_details else "General suspicious behavior"
                            
                            cheat_logger.warning(f"CHEAT #{total_cheat_count} - Frame {frame_count} ({timestamp_str}) - Score: {cheat_score:.2f} - Details: {details_str}")
                            
                            cheating_detections.append({
                                'frame': frame_count,
                                'timestamp': frame_count / face_fps if face_fps > 0 else frame_count,
                                'type': 'static',
                                'details': static_result,
                                'cheat_number': total_cheat_count,
                                'cheat_score': cheat_score
                            })
                    except Exception as e:
                        if frame_count % 50 == 0:  # Only print occasionally to avoid spam
                            print(f"    ⚠️ Static proctor error on frame {frame_count}: {e}")
                
                # Temporal trainer analysis
                if temporal_trainer and static_result:
                    try:
                        # Add frame features to temporal sequence
                        temporal_trainer.add_frame_features(static_result)
                        
                        # Get temporal prediction if we have enough frames
                        temporal_score = temporal_trainer.get_temporal_prediction()
                        
                        # Create temporal result record
                        temporal_result = {
                            'frame': frame_count,
                            'temporal_score': temporal_score,
                            'sequence_length': len(temporal_trainer.feature_history)
                        }
                        temporal_features.append(temporal_result)
                        
                        # Log significant temporal patterns
                        if temporal_score > 0.7:
                            cheat_logger.info(f"High temporal risk at frame {frame_count}: {temporal_score:.2f}")
                            
                    except Exception as e:
                        if frame_count % 50 == 0:
                            print(f"    ⚠️ Temporal trainer error on frame {frame_count}: {e}")
                
            except Exception as e:
                if frame_count % 50 == 0:
                    print(f"    ⚠️ Frame processing error on frame {frame_count}: {e}")
        
        # Cleanup
        face_cap.release()
        hand_cap.release()
        
        # Results summary
        print(f"\n✅ Processing completed!")
        print(f"📈 Analysis Results:")
        print(f"  - Total frames processed: {frame_count}")
        print(f"  - Video duration: {frame_count / face_fps if face_fps > 0 else 0:.1f} seconds")
        print(f"  - 🚨 TOTAL CHEATING INSTANCES: {total_cheat_count}")
        print(f"  - Static cheating detections: {len(cheating_detections)}")
        print(f"  - Temporal features extracted: {len(temporal_features)}")
        
        # Show temporal analysis summary if available
        if temporal_features:
            avg_temporal_score = sum(t.get('temporal_score', 0) for t in temporal_features) / len(temporal_features)
            high_temporal_risk = sum(1 for t in temporal_features if t.get('temporal_score', 0) > 0.7)
            print(f"  - Average temporal risk score: {avg_temporal_score:.2f}")
            print(f"  - High temporal risk frames: {high_temporal_risk}")
            cheat_logger.info(f"Temporal analysis: {len(temporal_features)} frames, avg score: {avg_temporal_score:.2f}")
        
        # Log final summary
        cheat_logger.info(f"=== ANALYSIS COMPLETED ===")
        cheat_logger.info(f"Total frames processed: {frame_count}")
        cheat_logger.info(f"Video duration: {frame_count / face_fps if face_fps > 0 else 0:.1f} seconds")
        cheat_logger.info(f"TOTAL CHEATING INSTANCES: {total_cheat_count}")
        cheat_logger.info(f"Cheating detection rate: {total_cheat_count/frame_count*100:.1f}%" if frame_count > 0 else "No frames processed")
        
        if cheating_detections:
            print(f"\n🚨 Cheating Detection Summary:")
            cheat_logger.info(f"=== CHEATING INCIDENTS SUMMARY ===")
            for i, detection in enumerate(cheating_detections[:10]):  # Show first 10 detections
                details = detection['details']
                cheat_info = []
                if details.get('H-Hand Detected'):
                    cheat_info.append("Hand-Cam")
                if details.get('F-Hand Detected'):
                    cheat_info.append("Face-Cam") 
                if details.get('H-Prohibited Item') or details.get('F-Prohibited Item'):
                    cheat_info.append("Prohibited-Item")
                if not details.get('verification_result', True):
                    cheat_info.append("Identity-Fail")
                
                info_str = ", ".join(cheat_info) if cheat_info else "General"
                print(f"    ⚠️ CHEAT #{detection.get('cheat_number', i+1)}: Frame {detection['frame']} ({detection['timestamp']:.1f}s) - {info_str}")
                
            if len(cheating_detections) > 10:
                print(f"    ... and {len(cheating_detections) - 10} more detections")
                cheat_logger.info(f"Plus {len(cheating_detections) - 10} additional cheating incidents")
        else:
            print(f"\n✅ No cheating detected in processed frames")
            cheat_logger.info("No cheating detected during analysis")
        
        # Final log entry
        cheat_logger.info("=== CHEATING DETECTION SESSION ENDED ===")
        print(f"📝 Complete log saved to: {log_file}")
        
        if static_proctor:
            print(f"✅ Full proctor analysis completed successfully!")
        else:
            print(f"⚠️ Basic analysis completed (limited due to missing models)")
        
        return True
        
    except Exception as e:
        print(f"❌ Video processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_demo():
    """Run a simple camera demo"""
    print("\n🎥 Camera Demo")
    print("Press 'q' to quit")
    print("=" * 30)
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera 0")
            return
            
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
                
            frame_count += 1
            
            # Add overlay text
            cv2.putText(frame, f"Video Proctoring Demo - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Video Proctoring Demo', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Demo completed - processed {frame_count} frames")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def main():
    """Main function"""
    args = parse_arguments()
    
    print("🎬 Video Proctoring System - Full Functionality Test")
    print("=" * 60)
    
    if args.test_imports or args.debug:
        test_imports()
    
    if args.debug:
        test_camera()
    
    if args.demo:
        run_demo()
    
    if args.process_videos:
        print("🎥 Video processing mode enabled")
        if args.no_display:
            print("📺 Display disabled for faster processing")
        success = process_input_videos()
        if success:
            print("🎉 Video processing completed successfully!")
        else:
            print("❌ Video processing failed!")
    
    if not any([args.test_imports, args.debug, args.demo, args.process_videos]):
        print("📋 Available options:")
        print("  --debug           : Run debug tests")
        print("  --demo            : Run camera demo")
        print("  --test-imports    : Test imports only")
        print("  --process-videos  : Process videos from Inputs folder")
        print("  --no-display      : Disable video display (use with --process-videos)")
        print("\nExample usage:")
        print("  python main_fixed.py --debug")
        print("  python main_fixed.py --demo")
        print("  python main_fixed.py --process-videos --no-display")
    
    print("\n✨ Main execution completed!")


if __name__ == "__main__":
    main()
