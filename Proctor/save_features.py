import os
import cv2
import pandas as pd
import torch
import mediapipe as mp
from datetime import datetime
from pathlib import Path
import re
from ultralytics import YOLO
from proctor import StaticProctor
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
    """Extract timestamp from filename format frame_number_x-xx-xx.xxxxxx.jpg"""
    match = re.search(r'_(\d+-\d+-\d+\.\d+)\.jpg$', filename)
    if match:
        return match.group(1)
    return None

def process_dataset(dataset_path, target_frame_path, output_csv_path):
    # Setup YOLO model and MediaPipe
    with suppress_output():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO('OEP_YOLOv11n.pt')
        
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
    total_pairs_processed = 0
    
    # First, count total pairs for progress bar
    total_pairs = 0
    for is_cheating in [True, False]:
        cheating_str = "cheating_frames" if is_cheating else "not_cheating_frames"
        face_dir = os.path.join(dataset_path, "face_frames", cheating_str)
        hand_dir = os.path.join(dataset_path, "hand_frames", cheating_str)
        
        if not os.path.exists(face_dir) or not os.path.exists(hand_dir):
            print(f"Warning: Directory not found: {face_dir} or {hand_dir}")
            continue
            
        face_files = {extract_timestamp(f): f for f in os.listdir(face_dir) if f.endswith('.jpg')}
        hand_files = {extract_timestamp(f): f for f in os.listdir(hand_dir) if f.endswith('.jpg')}
        common_timestamps = set(face_files.keys()) & set(hand_files.keys())
        total_pairs += len(common_timestamps)
    
    # Setup main progress bar
    main_progress = tqdm(total=total_pairs, desc="Processing dataset", unit="pair")
    
    # Process directory structure
    for is_cheating, cheating_label in [(True, 1), (False, 0)]:
        # Define paths
        cheating_str = "cheating_frames" if is_cheating else "not_cheating_frames"
        face_dir = os.path.join(dataset_path, "face_frames", cheating_str)
        hand_dir = os.path.join(dataset_path, "hand_frames", cheating_str)
        
        if not os.path.exists(face_dir) or not os.path.exists(hand_dir):
            continue
        
        # Get all face and hand frame files
        face_files = {extract_timestamp(f): f for f in os.listdir(face_dir) if f.endswith('.jpg')}
        hand_files = {extract_timestamp(f): f for f in os.listdir(hand_dir) if f.endswith('.jpg')}
        
        # Process matching pairs
        common_timestamps = set(face_files.keys()) & set(hand_files.keys())
        main_progress.write(f"Found {len(common_timestamps)} matching pairs in {cheating_str}")
        
        for timestamp in common_timestamps:
            face_path = os.path.join(face_dir, face_files[timestamp])
            hand_path = os.path.join(hand_dir, hand_files[timestamp])
            
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
                
                # Add metadata
                output['timestamp'] = timestamp
                output['is_cheating'] = cheating_label
                
                # Append to results
                results.append(output)
                total_pairs_processed += 1
                
            except Exception as e:
                main_progress.write(f"Error processing timestamp {timestamp}: {str(e)}")
            
            main_progress.update(1)
    
    main_progress.close()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\nResults saved to {output_csv_path}")
    print(f"Successfully processed {total_pairs_processed} out of {total_pairs} frame pairs")
    
    return df

if __name__ == "__main__":
    # Configuration
    dataset_path = "/home/kashyap/Documents/Projects/PROCTOR/Bhuv_Dataset/"
    target_frame_path = "/home/kashyap/Documents/Projects/PROCTOR/Bhuv_Dataset/face_frames/ID.png"
    output_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proctor_results.csv")
    
    # Process the dataset
    results_df = process_dataset(dataset_path, target_frame_path, output_csv_path)
    print(f"Fields captured: {', '.join(results_df.columns)}")