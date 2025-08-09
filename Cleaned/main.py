"""
Main entry point for the video proctor application - Uses VideoProctor class
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

# Import the main VideoProctor class
try:
    from video_proctor import VideoProctor, create_video_proctor
    VIDEOPROCTOR_LOADED = True
except Exception as e:
    print(f"❌ Could not import VideoProctor: {e}")
    VIDEOPROCTOR_LOADED = False


def setup_cheating_logger():
    """Setup logger for cheating detection events"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
    log_file = f"logs/cheating_detection_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger('cheating_detection')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger, log_file


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Video Proctoring System')
    parser.add_argument('--process-videos', action='store_true', help='Process video files from Inputs folder')
    parser.add_argument('--no-display', action='store_true', help='Disable video display for faster processing')
    return parser.parse_args()


def process_input_videos():
    """Process video files using VideoProctor class"""
    print("\n🎬 Processing Input Videos with VideoProctor")
    print("=" * 60)
    
    if not VIDEOPROCTOR_LOADED:
        print("❌ VideoProctor not available - cannot process videos")
        return False
    
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
        # Initialize VideoProctor with target image
        print("\n🤖 Initializing VideoProctor...")
        if not VIDEOPROCTOR_LOADED:
            raise RuntimeError("VideoProctor module not available")
        
        # Import here to ensure it's available
        from video_proctor import create_video_proctor
        
        video_proctor = create_video_proctor(
            target_image_path=str(target_image),
            debug_mode=True
        )
        print("✅ VideoProctor initialized successfully")
        
        # Process video files using VideoProctor
        print(f"\n🎬 Processing videos with VideoProctor...")
        print(f"  - Face camera: {face_video}")
        print(f"  - Hand camera: {hand_video}")
        
        # Use VideoProctor's video file processing method
        results = video_proctor.process_video_files(
            face_video_path=str(face_video),
            hand_video_path=str(hand_video)
        )
        
        # Process results and log cheating detections
        total_cheat_count = 0
        cheating_detections = []
        
        cheat_logger.info(f"Starting analysis of {len(results)} frames")
        
        for frame_idx, result in enumerate(results):
            if 'error' in result:
                print(f"  ⚠️ Frame {frame_idx} processing error: {result['error']}")
                continue
            
            final_score = result.get('Final Score', 0.0)
            
            # Use configurable threshold
            try:
                from config import Config
                cheat_threshold = Config.CHEATING_THRESHOLD if CONFIG_LOADED else 0.5
            except:
                cheat_threshold = 0.5
            
            if final_score > cheat_threshold:  # Cheating detected!
                total_cheat_count += 1
                timestamp_str = f"frame_{frame_idx}"
                
                # Real-time cheating alert
                print(f"  🚨 CHEAT DETECTED #{total_cheat_count} at {timestamp_str} (Score: {final_score:.2f})")
                
                # Log to file with detailed information
                cheat_details = []
                if result.get('H-Hand Detected'):
                    cheat_details.append("Hand in hand camera")
                if result.get('F-Hand Detected'):
                    cheat_details.append("Hand in face camera")
                if result.get('H-Prohibited Item'):
                    cheat_details.append("Prohibited item in hand camera")
                if result.get('F-Prohibited Item'):
                    cheat_details.append("Prohibited item in face camera")
                if not result.get('verification_result', True):
                    cheat_details.append("Identity verification failed")
                
                details_str = ", ".join(cheat_details) if cheat_details else "General suspicious behavior"
                
                cheat_logger.warning(f"CHEAT #{total_cheat_count} - Frame {frame_idx} ({timestamp_str}) - Score: {final_score:.2f} - Details: {details_str}")
                
                cheating_detections.append({
                    'frame': frame_idx,
                    'type': 'combined',
                    'details': result,
                    'cheat_number': total_cheat_count,
                    'final_score': final_score
                })
        
        # Results summary
        print(f"\n✅ VideoProctor processing completed!")
        print(f"📈 Analysis Results:")
        print(f"  - Total frames processed: {len(results)}")
        print(f"  - 🚨 TOTAL CHEATING INSTANCES: {total_cheat_count}")
        print(f"  - Cheating detections: {len(cheating_detections)}")
        
        # Get session statistics from VideoProctor
        stats = video_proctor.get_session_statistics()
        if stats:
            print(f"  - Session statistics:")
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    print(f"    - {key}: {value}")
        
        # Log final summary
        cheat_logger.info(f"=== ANALYSIS COMPLETED ===")
        cheat_logger.info(f"Total frames processed: {len(results)}")
        cheat_logger.info(f"TOTAL CHEATING INSTANCES: {total_cheat_count}")
        cheat_logger.info(f"Cheating detection rate: {total_cheat_count/len(results)*100:.1f}%" if len(results) > 0 else "No frames processed")
        
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
                print(f"    ⚠️ CHEAT #{detection.get('cheat_number', i+1)}: Frame {detection['frame']} - {info_str}")
                
            if len(cheating_detections) > 10:
                print(f"    ... and {len(cheating_detections) - 10} more detections")
                cheat_logger.info(f"Plus {len(cheating_detections) - 10} additional cheating incidents")
        else:
            print(f"\n✅ No cheating detected in processed frames")
            cheat_logger.info("No cheating detected during analysis")
        
        # Final log entry
        cheat_logger.info("=== CHEATING DETECTION SESSION ENDED ===")
        print(f"📝 Complete log saved to: {log_file}")
        print(f"✅ VideoProctor analysis completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ VideoProctor processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    args = parse_arguments()
    
    print("🎬 Video Proctoring System - VideoProctor Integration")
    print("=" * 60)
    
    if args.process_videos:
        print("🎥 Video processing mode enabled")
        if args.no_display:
            print("📺 Display disabled for faster processing")
        success = process_input_videos()
        if success:
            print("\n✅ Video processing completed successfully!")
        else:
            print("\n❌ Video processing failed!")
            return 1
    else:
        print("📋 Available options:")
        print("  --process-videos : Process videos from Inputs folder")
        print("  --no-display     : Disable video display for faster processing")
        print("\nExample: python main.py --process-videos")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
