import os
import cv2
import pandas as pd
import re
from tqdm import tqdm
import logging
import warnings
import sys
import contextlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse project components instead of direct YOLO/MediaPipe
from config import Config
from core.model_manager import ModelManager
from Proctor.static_proctor import StaticProctor

# Match VideoProctor behavior: ensure CWD is the repo root so relative Config paths work
def _set_cwd_to_repo_root():
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(root_dir)
    except Exception:
        pass

_set_cwd_to_repo_root()

# Suppress warnings and logging output
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Context manager to suppress stdout/stderr
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def extract_timestamp(filename):
    """
    Extract timestamp from filenames in formats:
    - frame_number_x:xx:xx.xxxxxx.jpg
    - frame_number_x-xx-xx.xxxxxx.jpg
    - frame_number_x-xx-xx-xxxxxx.jpg
    """
    # Try the format with colon separator and period for microseconds
    match = re.search(r'_(\d+:\d+:\d+\.\d+)\.jpg$', filename)
    if match:
        return match.group(1)
    
    # Try the format with period separator
    match = re.search(r'_(\d+-\d+-\d+\.\d+)\.jpg$', filename)
    if match:
        return match.group(1)
    
    # Try the format with hyphen separator
    match = re.search(r'_(\d+-\d+-\d+-\d+)\.jpg$', filename)
    if match:
        return match.group(1)
        
    return None

def find_frame_paths(video_path, timestamp):
    """
    Find face and hand frames with matching timestamp within a specific video directory.
    Return paths and their respective cheating labels (1 for cheating, 0 for not cheating).
    """
    face_path, face_label = None, None
    hand_path, hand_label = None, None
    
    for is_cheating, cheating_label in [(True, 1), (False, 0)]:
        cheating_str = "cheating_frames" if is_cheating else "not_cheating_frames"
        
        # Check face frame
        face_dir = os.path.join(video_path, "front", cheating_str)
        if os.path.exists(face_dir):
            for file in os.listdir(face_dir):
                if file.endswith('.jpg') and extract_timestamp(file) == timestamp:
                    face_path = os.path.join(face_dir, file)
                    face_label = cheating_label
                    break
                    
        # Check hand frame
        hand_dir = os.path.join(video_path, "side", cheating_str)
        if os.path.exists(hand_dir):
            for file in os.listdir(hand_dir):
                if file.endswith('.jpg') and extract_timestamp(file) == timestamp:
                    hand_path = os.path.join(hand_dir, file)
                    hand_label = cheating_label
                    break
    
    return face_path, face_label, hand_path, hand_label

def get_all_timestamps(video_path):
    """Get all unique timestamps within a specific video directory."""
    all_timestamps = set()
    
    for folder_type in ["front", "side"]:
        for label_type in ["cheating_frames", "not_cheating_frames"]:
            directory = os.path.join(video_path, folder_type, label_type)
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith('.jpg'):
                        timestamp = extract_timestamp(file)
                        if timestamp:
                            all_timestamps.add(timestamp)
    
    return all_timestamps

def process_video(video_path, video_name, split_name, target_frame, proctor, output_dir):
    """Process a single video and return results."""
    results = []
    
    # Get all unique timestamps for this video
    all_timestamps = get_all_timestamps(video_path)
    
    if not all_timestamps:
        print(f"No timestamps found for {split_name}/{video_name}")
        return []
    
    video_progress = tqdm(total=len(all_timestamps), 
                         desc=f"Processing {split_name}/{video_name}", 
                         unit="pair", 
                         leave=False)
    
    total_processed = 0
    mismatched_labels = 0
    
    # Process each timestamp
    for timestamp in all_timestamps:
        face_path, face_label, hand_path, hand_label = find_frame_paths(video_path, timestamp)
        
        # Skip if either frame is missing
        if face_path is None or hand_path is None:
            video_progress.write(f"Warning: Missing frame for timestamp {timestamp} in {split_name}/{video_name}")
            video_progress.update(1)
            continue
        
        # Check for label mismatch
        if face_label != hand_label:
            video_progress.write(f"Note: Label mismatch for timestamp {timestamp} in {split_name}/{video_name} - Face: {face_label}, Hand: {hand_label}")
            mismatched_labels += 1
        
        # Load frames
        face_frame = cv2.imread(face_path)
        hand_frame = cv2.imread(hand_path)
        
        if face_frame is None or hand_frame is None:
            video_progress.write(f"Warning: Could not load frames for timestamp {timestamp} in {split_name}/{video_name}")
            video_progress.update(1)
            continue
        
        # Process frames
        try:
            with suppress_output():
                output = proctor.process_frames(target_frame, face_frame, hand_frame)
            
            # Add metadata with separate labels for face and hand
            output['timestamp'] = timestamp
            output['face_is_cheating'] = face_label
            output['hand_is_cheating'] = hand_label
            output['split'] = split_name
            output['video'] = video_name
            
            # Overall cheating label (1 if either is cheating)
            output['is_cheating'] = 1 if (face_label == 1 or hand_label == 1) else 0
            
            # Append to results
            results.append(output)
            total_processed += 1
            
        except Exception as e:
            video_progress.write(f"Error processing timestamp {timestamp} in {split_name}/{video_name}: {str(e)}")
        
        video_progress.update(1)
    
    video_progress.close()
    
    # Save results for this video
    if results:
        df = pd.DataFrame(results)
        csv_filename = f"{split_name}_{video_name}_proctor_results.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(results)} results to {csv_filename}")
        print(f"  Processed {total_processed} out of {len(all_timestamps)} frame pairs")
        if mismatched_labels > 0:
            print(f"  Found {mismatched_labels} pairs with mismatched labels")
    
    return results

def process_dataset(dataset_path, target_frame_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup detection models via ModelManager and override Hands to static_image_mode=True
    with suppress_output():
        model_manager = ModelManager()
        yolo_model, media_pipe_dict = model_manager.load_detection_models()
        # Ensure static image mode for offline frame processing (preserve original behavior)
        try:
            if 'hands' in media_pipe_dict and hasattr(media_pipe_dict['hands'], 'close'):
                try:
                    media_pipe_dict['hands'].close()
                except Exception:
                    pass
            mpHands = media_pipe_dict.get('mpHands')
            if mpHands is not None and hasattr(mpHands, 'Hands'):
                media_pipe_dict['hands'] = mpHands.Hands(
                    static_image_mode=True,
                    max_num_hands=Config.MAX_NUM_HANDS,
                    min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
                )
        except Exception:
            pass
    
    # Load target frame
    target_frame = cv2.imread(target_frame_path)
    if target_frame is None:
        raise FileNotFoundError(f"Target frame not found at {target_frame_path}")
    
    # Initialize proctor using Config-managed paths
    with suppress_output():
        # Validate mediapipe model path
        try:
            Config.validate_mediapipe_model()
        except Exception:
            pass
        proctor = StaticProctor(yolo_model, media_pipe_dict, Config.DEFAULT_MEDIAPIPE_MODEL)
    
    print(f"Processing dataset at {dataset_path}")
    # return
    all_results = []
    
    # Process Train and Test directories
    for split_name in ["Train", "Test"]:
        split_path = os.path.join(dataset_path, split_name)
        if not os.path.exists(split_path):
            print(f"Warning: {split_name} directory not found at {split_path}")
            continue
            
        print(f"\nProcessing {split_name} split...")
        
        # Get all video directories
        video_dirs = [d for d in os.listdir(split_path) 
                     if os.path.isdir(os.path.join(split_path, d))]
        
        if not video_dirs:
            print(f"No video directories found in {split_name}")
            continue
            
        for video_name in sorted(video_dirs):
            video_path = os.path.join(split_path, video_name)
            print(f"\nProcessing {split_name}/{video_name}...")
            print(f"  Video path: {video_path}")
            video_results = process_video(video_path, video_name, split_name, 
                                        target_frame, proctor, output_dir)
            all_results.extend(video_results)
    
    # return
    # Create a combined CSV with all results
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_csv_path = os.path.join(output_dir, "combined_proctor_results.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\nCombined results saved to combined_proctor_results.csv")
        print(f"Total processed entries: {len(all_results)}")
    
    return all_results

if __name__ == "__main__":
    # Configuration
    dataset_path = r"C:\Users\singl\Desktop\Bhuvanesh\NITK\SEM4\IT255_AI\Project Files\FinalRepo\CheatusDeletus\Cleaned\Inputs\OEPFrame_Dataset\OEPFrame_Dataset"
    target_frame_path = r"C:\Users\singl\Desktop\Bhuvanesh\NITK\SEM4\IT255_AI\Project Files\FinalRepo\CheatusDeletus\Cleaned\Inputs\OEPFrame_Dataset\OEPFrame_Dataset\ID.png"
    output_dir = r"C:\Users\singl\Desktop\Bhuvanesh\NITK\SEM4\IT255_AI\Project Files\FinalRepo\CheatusDeletus\Cleaned\Inputs\csvs"

    # Process the dataset
    results = process_dataset(dataset_path, target_frame_path, output_dir)
    if results:
        sample_df = pd.DataFrame(results[:1])  # Just show first entry structure
        print(f"\nFields captured: {', '.join(sample_df.columns)}")