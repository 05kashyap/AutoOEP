import os
import cv2
import pandas as pd
import torch
import mediapipe as mp
from datetime import datetime
from pathlib import Path
import re
from ultralytics import YOLO
from Proctor.proctor import StaticProctor
from tqdm import tqdm
import logging
import warnings
import sys
import contextlib

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
    - frame_number_x-xx-xx.xxxxxx.jpg
    - frame_number_x-xx-xx-xxxxxx.jpg
    """
    # Try the format with period separator
    match = re.search(r'_(\d+-\d+-\d+\.\d+)\.jpg$', filename)
    if match:
        return match.group(1)
    
    # Try the format with hyphen separator
    match = re.search(r'_(\d+-\d+-\d+-\d+)\.jpg$', filename)
    if match:
        return match.group(1)
        
    return None

def find_frame_paths(dataset_path, timestamp):
    """
    Find face and hand frames with matching timestamp across all directories.
    Return paths and their respective cheating labels (1 for cheating, 0 for not cheating).
    """
    face_path, face_label = None, None
    hand_path, hand_label = None, None
    
    for is_cheating, cheating_label in [(True, 1), (False, 0)]:
        cheating_str = "cheating_frames" if is_cheating else "not_cheating_frames"
        
        # Check face frame
        face_dir = os.path.join(dataset_path, "face_frames", cheating_str)
        if os.path.exists(face_dir):
            for file in os.listdir(face_dir):
                if file.endswith('.jpg') and extract_timestamp(file) == timestamp:
                    face_path = os.path.join(face_dir, file)
                    face_label = cheating_label
                    break
                    
        # Check hand frame
        hand_dir = os.path.join(dataset_path, "hand_frames", cheating_str)
        if os.path.exists(hand_dir):
            for file in os.listdir(hand_dir):
                if file.endswith('.jpg') and extract_timestamp(file) == timestamp:
                    hand_path = os.path.join(hand_dir, file)
                    hand_label = cheating_label
                    break
    
    return face_path, face_label, hand_path, hand_label

def get_all_timestamps(dataset_path):
    """Get all unique timestamps across all directories."""
    all_timestamps = set()
    
    for folder_type in ["face_frames", "hand_frames"]:
        for label_type in ["cheating_frames", "not_cheating_frames"]:
            directory = os.path.join(dataset_path, folder_type, label_type)
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith('.jpg'):
                        timestamp = extract_timestamp(file)
                        if timestamp:
                            all_timestamps.add(timestamp)
    
    return all_timestamps

def process_dataset(dataset_path, target_frame_path, output_csv_path):
    # Setup YOLO model and MediaPipe
    with suppress_output():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO('Models/OEP_YOLOv11n.pt')
        
        mpHands = mp.solutions.hands
        media_pipe_dict = {
            'mpHands': mpHands,
            'hands': mpHands.Hands(static_image_mode=True,
                                 max_num_hands=2,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5),
            'mpdraw': mp.solutions.drawing_utils
        }
    
    # Load target frame
    target_frame = cv2.imread(target_frame_path)
    if target_frame is None:
        raise FileNotFoundError(f"Target frame not found at {target_frame_path}")
    
    # Initialize proctor
    with suppress_output():
        proctor = StaticProctor(model, media_pipe_dict)
    
    # Initialize results list
    results = []
    print(f"Processing dataset at {dataset_path}")
    # Get all unique timestamps
    all_timestamps = get_all_timestamps(dataset_path)
    main_progress = tqdm(total=len(all_timestamps), desc="Processing dataset", unit="pair")
    main_progress.write(f"Found {len(all_timestamps)} unique timestamps across all directories")
    
    total_processed = 0
    mismatched_labels = 0
    
    # Process each timestamp
    for timestamp in all_timestamps:
        face_path, face_label, hand_path, hand_label = find_frame_paths(dataset_path, timestamp)
        
        # Skip if either frame is missing
        if face_path is None or hand_path is None:
            main_progress.write(f"Warning: Missing frame for timestamp {timestamp}")
            main_progress.update(1)
            continue
        
        # Check for label mismatch
        if face_label != hand_label:
            main_progress.write(f"Note: Label mismatch for timestamp {timestamp} - Face: {face_label}, Hand: {hand_label}")
            mismatched_labels += 1
        
        # Load frames
        face_frame = cv2.imread(face_path)
        hand_frame = cv2.imread(hand_path)
        
        if face_frame is None or hand_frame is None:
            main_progress.write(f"Warning: Could not load frames for timestamp {timestamp}")
            main_progress.update(1)
            continue
        
        # Process frames
        try:
            with suppress_output():
                output = proctor.process_frames(target_frame, face_frame, hand_frame)
            
            # Add metadata with separate labels for face and hand
            output['timestamp'] = timestamp
            output['face_is_cheating'] = face_label
            output['hand_is_cheating'] = hand_label
            
            # Overall cheating label (1 if either is cheating)
            output['is_cheating'] = 1 if (face_label == 1 or hand_label == 1) else 0
            
            # Append to results
            results.append(output)
            total_processed += 1
            
        except Exception as e:
            main_progress.write(f"Error processing timestamp {timestamp}: {str(e)}")
        
        main_progress.update(1)
    
    main_progress.close()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\nResults saved to {output_csv_path}")
    print(f"Successfully processed {total_processed} out of {len(all_timestamps)} frame pairs")
    print(f"Found {mismatched_labels} pairs with mismatched labels between face and hand frames")
    
    return df

if __name__ == "__main__":
    # Configuration
    dataset_path = "/home/kashyap/Documents/Projects/PROCTOR/Rjn_Dataset/"
    target_frame_path = "/home/kashyap/Documents/Projects/PROCTOR/Rjn_Dataset/face_frames/ID.png"
    output_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2proctor_results.csv")
    
    # Process the dataset
    results_df = process_dataset(dataset_path, target_frame_path, output_csv_path)
    print(f"Fields captured: {', '.join(results_df.columns)}")