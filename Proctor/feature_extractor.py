import os
import cv2
import pandas as pd
import numpy as np
import torch
import mediapipe as mp
import re
from ultralytics import YOLO
from tqdm import tqdm
import warnings
import sys
import contextlib
from typing import Dict, Any, Set, Tuple, Optional, List

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Proctor.proctor import StaticProctor

# Suppress warnings and logging output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Context manager to suppress stdout/stderr from libraries
@contextlib.contextmanager
def suppress_output():
    """A context manager to suppress standard output and error."""
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

class FeatureExtractor:
    """
    A class to extract and process proctoring features from face and hand video frames.
    
    This class encapsulates the models and logic required for feature extraction,
    and can be used for both batch processing of datasets and real-time, single-frame-pair analysis.
    """
    
    def __init__(self, target_frame_path: str, face_landmarker_path: str = 'Models/face_landmarker.task', suppress_runtime_output: bool = True):
        """
        Initializes the FeatureExtractor.

        Args:
            target_frame_path (str): The path to the target image for identity verification.
            face_landmarker_path (str): The path to the face landmarker model file.
        """
        print("Initializing Feature Extractor...")
        self.target_frame = self._load_target_frame(target_frame_path)
        # Control whether to suppress per-frame library logs/prints (e.g., DeepFace, face_inference)
        self.suppress_runtime_output = suppress_runtime_output
        
        with suppress_output():
            # Initialize models (YOLO, MediaPipe)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = YOLO('Models/OEP_YOLOv11n.pt')
            
            # Build MediaPipe dict defensively to avoid import-time issues with type checkers
            mp_solutions = getattr(mp, 'solutions', None)
            mp_hands_module = getattr(mp_solutions, 'hands', None) if mp_solutions else None
            mp_drawing_utils = getattr(mp_solutions, 'drawing_utils', None) if mp_solutions else None
            if mp_hands_module is None or mp_drawing_utils is None:
                raise ImportError("Failed to import mediapipe hands/drawing_utils modules")

            media_pipe_dict = {
                'mpHands': mp_hands_module,
                'hands': mp_hands_module.Hands(static_image_mode=True, max_num_hands=2,
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5),
                'mpdraw': mp_drawing_utils
            }
            
            # Initialize the StaticProctor
            self.proctor = StaticProctor(model, media_pipe_dict, face_landmarker_path)
        
        # Define mappings and column orders for processing
        self._define_processing_parameters()
        print("Initialization complete.")

    def _load_target_frame(self, path: str) -> np.ndarray:
        """Loads the target frame from the given path."""
        target_frame = cv2.imread(path)
        if target_frame is None:
            raise FileNotFoundError(f"Target frame not found at {path}")
        return target_frame

    def _define_processing_parameters(self):
        """Defines parameters used for processing the raw features."""
        self.all_objects = {'cell phone', 'chits', 'closedbook', 'earpiece', 'headphone', 'openbook', 'sheet', 'watch'}
        
        self.mappings = {
            'iris_pos': {'center': 0, 'left': 1, 'right': 2},
            'mouth_zone': {'GREEN': 0, 'YELLOW': 1, 'ORANGE': 2, 'RED': 3},
            'gaze_direction': {'forward': 0, 'left': 1, 'right': 2, 'up': 3, 'down': 4},
            'gaze_zone': {'white': 0, 'yellow': 1, 'red': 2}
        }
        
        self.nan_mappings = {
            'iris_pos': -1, 'mouth_zone': -1, 'gaze_direction': -1, 'gaze_zone': -1,
            "H-Distance": 10000, "F-Distance": 10000,
        }
        
        self.desired_columns = [
            'timestamp', 'verification_result', 'num_faces', 'iris_pos', 'iris_ratio', 
            'mouth_zone', 'mouth_area', 'x_rotation', 'y_rotation', 'z_rotation', 
            'radial_distance', 'gaze_direction', 'gaze_zone', 'watch', 'headphone', 
            'closedbook', 'earpiece', 'cell phone', 'openbook', 'chits', 'sheet',
            'H-Distance', 'F-Distance', 'split', 'video', 'is_cheating'
        ]

    @staticmethod
    def _to_float(val: Any, default: float = 0.0) -> float:
        if val is None:
            return default
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
        if isinstance(val, str):
            v = val.strip()
            if v == '' or v.lower() == 'nan':
                return default
            try:
                return float(v)
            except Exception:
                return default
        return default

    @staticmethod
    def _to_int(val: Any, default: int = 0) -> int:
        if val is None:
            return default
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, (float, np.floating)):
            return int(val)
        if isinstance(val, str):
            v = val.strip()
            if v == '' or v.lower() == 'nan':
                return default
            try:
                return int(float(v))
            except Exception:
                return default
        return default

    def _process_raw_features(self, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single row of raw features into the final format.
        This function is equivalent to the logic in the old `process_csv.py`.
        """
        # 1. Handle verification result if present
        if 'verification_result' in raw_features:
            raw_features['verification_result'] = self._to_int(raw_features['verification_result'], 0)

        # 2. One-hot encode prohibited items
        def parse_items(val) -> Set[str]:
            items: Set[str] = set()
            if val is None:
                return items
            # If already a collection of strings
            if isinstance(val, (list, tuple, set)):
                for x in val:
                    if isinstance(x, str):
                        s = x.strip().lower()
                        if s:
                            items.add(s)
                return items
            # If a string, split by common delimiters
            if isinstance(val, str):
                s = val.strip()
                if not s:
                    return items
                # Remove surrounding brackets if any
                s = s.strip('[]')
                for token in re.split(r"[,;|]", s):
                    t = token.strip().strip("'\"").lower()
                    if t:
                        items.add(t)
                return items
            return items

        observed: Set[str] = set()
        if 'F-Prohibited Item' in raw_features and pd.notna(raw_features['F-Prohibited Item']):
            observed.update(parse_items(raw_features['F-Prohibited Item']))
        if 'H-Prohibited Item' in raw_features and pd.notna(raw_features['H-Prohibited Item']):
            observed.update(parse_items(raw_features['H-Prohibited Item']))
        
        for obj in self.all_objects:
            raw_features[obj] = 1 if obj in observed else 0

        # 3. Apply categorical mappings with robust casing
        # iris_pos and gaze_* expect lowercase; mouth_zone expects uppercase
        if 'iris_pos' in raw_features:
            val = raw_features['iris_pos']
            key = val.lower() if isinstance(val, str) else val
            raw_features['iris_pos'] = self.mappings['iris_pos'].get(key, -1)
        if 'mouth_zone' in raw_features:
            val = raw_features['mouth_zone']
            key = val.upper() if isinstance(val, str) else val
            raw_features['mouth_zone'] = self.mappings['mouth_zone'].get(key, -1)
        if 'gaze_direction' in raw_features:
            val = raw_features['gaze_direction']
            key = val.lower() if isinstance(val, str) else val
            raw_features['gaze_direction'] = self.mappings['gaze_direction'].get(key, -1)
        if 'gaze_zone' in raw_features:
            val = raw_features['gaze_zone']
            key = val.lower() if isinstance(val, str) else val
            raw_features['gaze_zone'] = self.mappings['gaze_zone'].get(key, -1)

        # 4. Coerce numeric fields and fill defaults for missing/empty
        numeric_defaults = {
            'verification_result': 0,
            'num_faces': 0,
            'iris_ratio': 1.0,
            'mouth_area': 0.0,
            'x_rotation': 0.0,
            'y_rotation': 0.0,
            'z_rotation': 0.0,
            'radial_distance': 0.0,
            'H-Distance': 10000.0,
            'F-Distance': 10000.0,
        }
        for k, default in numeric_defaults.items():
            if k in raw_features:
                if k == 'verification_result':
                    raw_features[k] = self._to_int(raw_features[k], int(default))
                else:
                    raw_features[k] = self._to_float(raw_features[k], float(default))
        
        # 5. Select and order columns, ensuring defaults exist for missing keys
        feature_series = pd.Series(raw_features)
        processed_dict = {}
        for col in self.desired_columns:
            if col in feature_series:
                processed_dict[col] = feature_series[col]
            else:
                # Default fallbacks for missing columns
                if col in ['iris_pos', 'mouth_zone', 'gaze_direction', 'gaze_zone']:
                    processed_dict[col] = -1
                elif col in ['watch', 'headphone', 'closedbook', 'earpiece', 'cell phone', 'openbook', 'chits', 'sheet']:
                    processed_dict[col] = 0
                elif col in ['H-Distance', 'F-Distance']:
                    processed_dict[col] = 10000.0
                elif col in ['verification_result', 'num_faces']:
                    processed_dict[col] = 0
                else:
                    processed_dict[col] = 0
        
        return processed_dict

    def process_frame_pair(self, face_frame: np.ndarray, hand_frame: np.ndarray) -> Dict[str, Any]:
        """
        Processes a pair of face and hand frames and returns a dictionary of processed features.

        This is the core function to be used for single-pair processing (e.g., in a real-time application).

        Args:
            face_frame (np.ndarray): The image frame containing the user's face.
            hand_frame (np.ndarray): The image frame containing the user's hands/desk area.

        Returns:
            Dict[str, Any]: A dictionary containing the final, processed features for the frame pair.
        """
        if face_frame is None or hand_frame is None:
            raise ValueError("Input frames cannot be None.")
            
        # Get raw features from StaticProctor
        ctx = suppress_output() if self.suppress_runtime_output else contextlib.nullcontext()
        with ctx:
            raw_features = self.proctor.process_frames(self.target_frame, face_frame, hand_frame)
        
        # Process the raw features into the final format
        processed_features = self._process_raw_features(raw_features)
        
        return processed_features

def _extract_timestamp(filename: str) -> Optional[str]:
    """Extract timestamp from frame filenames."""
    # This regex covers formats like x:xx:xx.xxxxxx, x-xx-xx.xxxxxx, and x-xx-xx-xxxxxx
    match = re.search(r'_(\d+[:\-]\d+[:\-]\d+[\.\-]\d+)\.jpg$', filename)
    return match.group(1) if match else None

def _get_all_timestamps(video_path: str) -> Set[str]:
    """Get all unique timestamps for a given video directory."""
    all_timestamps = set()
    for folder_type in ["face_frames", "hand_frames"]:
        for label_type in ["cheating_frames", "not_cheating_frames"]:
            directory = os.path.join(video_path, folder_type, label_type)
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith('.jpg'):
                        timestamp = _extract_timestamp(file)
                        if timestamp:
                            all_timestamps.add(timestamp)
    return all_timestamps

def _find_frame_paths(video_path: str, timestamp: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[int]]:
    """Find face and hand frame paths and their labels for a given timestamp."""
    face_path, face_label, hand_path, hand_label = None, None, None, None
    for is_cheating, cheating_label in [(True, 1), (False, 0)]:
        cheating_str = "cheating_frames" if is_cheating else "not_cheating_frames"
        
        # Find face frame if not already found
        if not face_path:
            face_dir = os.path.join(video_path, "face_frames", cheating_str)
            if os.path.exists(face_dir):
                for file in os.listdir(face_dir):
                    if file.endswith('.jpg') and _extract_timestamp(file) == timestamp:
                        face_path, face_label = os.path.join(face_dir, file), cheating_label
                        break
        
        # Find hand frame if not already found
        if not hand_path:
            hand_dir = os.path.join(video_path, "hand_frames", cheating_str)
            if os.path.exists(hand_dir):
                for file in os.listdir(hand_dir):
                    if file.endswith('.jpg') and _extract_timestamp(file) == timestamp:
                        hand_path, hand_label = os.path.join(hand_dir, file), cheating_label
                        break
    return face_path, face_label, hand_path, hand_label

def process_dataset_and_save_csv(dataset_path: str, target_frame_path: str, output_dir: str):
    """
    Processes an entire dataset of frames, extracts features, and saves them to CSV files.
    - A separate CSV is saved for each video directory.
    - A final combined CSV with all data is also created.

    Args:
        dataset_path (str): Path to the root of the dataset.
        target_frame_path (str): Path to the target image for identity verification.
        output_dir (str): Directory to save the final processed CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the feature extractor
    feature_extractor = FeatureExtractor(target_frame_path)
    
    all_results: List[Dict[str, Any]] = []
    
    print(f"\nProcessing dataset at {dataset_path}")
    
    for split_name in ["Train", "Test"]:
        split_path = os.path.join(dataset_path, split_name)
        if not os.path.exists(split_path):
            print(f"Warning: {split_name} directory not found at {split_path}, skipping.")
            continue
            
        print(f"\nProcessing {split_name} split...")
        video_dirs = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        
        for video_name in video_dirs:
            video_path = os.path.join(split_path, video_name)
            all_timestamps = _get_all_timestamps(video_path)
            
            if not all_timestamps:
                print(f"No frames found for {split_name}/{video_name}, skipping.")
                continue

            video_results: List[Dict[str, Any]] = []
            progress_bar = tqdm(total=len(all_timestamps), desc=f"  -> {video_name}", unit="frame_pair", leave=False)
            
            for timestamp in all_timestamps:
                face_path, face_label, hand_path, hand_label = _find_frame_paths(video_path, timestamp)
                
                if not face_path or not hand_path:
                    progress_bar.write(f"Warning: Missing frame for timestamp {timestamp} in {video_name}")
                    progress_bar.update(1)
                    continue
                
                face_frame = cv2.imread(face_path)
                hand_frame = cv2.imread(hand_path)
                
                if face_frame is None or hand_frame is None:
                    progress_bar.write(f"Warning: Could not load frames for timestamp {timestamp} in {video_name}")
                    progress_bar.update(1)
                    continue
                
                try:
                    processed_features = feature_extractor.process_frame_pair(face_frame, hand_frame)
                    
                    # Add metadata
                    processed_features['timestamp'] = timestamp
                    processed_features['split'] = split_name
                    processed_features['video'] = video_name
                    processed_features['is_cheating'] = 1 if (face_label == 1 or hand_label == 1) else 0
                    
                    video_results.append(processed_features)
                except Exception as e:
                    progress_bar.write(f"Error processing {timestamp} in {video_name}: {e}")
                
                progress_bar.update(1)
            progress_bar.close()

            # Save CSV for the individual video
            if video_results:
                video_df = pd.DataFrame(video_results)
                video_csv_name = f"{split_name}_{video_name}_processed.csv"
                video_output_path = os.path.join(output_dir, video_csv_name)
                video_df.to_csv(video_output_path, index=False)
                print(f"Saved per-video results to {video_csv_name} ({len(video_df)} entries)")
                all_results.extend(video_results)
    
    if not all_results:
        print("\nNo features were extracted. No combined CSV file will be generated.")
        return

    # Create and save the final combined DataFrame
    final_df = pd.DataFrame(all_results)
    
    # Convert timestamp string to a sortable format for ordering
    def convert_timestamp_to_seconds(ts_str: str) -> float:
        # Handles formats like '0:01:06.066617' or '0-01-06-066617'
        clean_ts = ts_str.replace('-', ':', 2).replace('-', '.')
        try:
            parts = clean_ts.split(':')
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
        except (ValueError, IndexError):
            pass # Return 0.0 if conversion fails
        return 0.0

    final_df['timestamp_sec'] = final_df['timestamp'].apply(convert_timestamp_to_seconds)
    final_df = final_df.sort_values(by=['split', 'video', 'timestamp_sec']).drop(columns=['timestamp_sec'])

    # Reorder columns to ensure consistency
    final_columns = [col for col in feature_extractor.desired_columns if col in final_df.columns]
    final_df = final_df[final_columns]

    output_path = os.path.join(output_dir, "final_features.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"\nProcessing complete.")
    print(f"Saved final combined data to {output_path}")
    print(f"  Total entries: {len(final_df)}")
    if 'is_cheating' in final_df.columns:
        print(f"  Overall cheating distribution:\n{final_df['is_cheating'].value_counts()}")

if __name__ == "__main__":
    # --- Configuration ---
    # NOTE: Update these paths to match your local environment.
    # The dataset path should point to the directory containing 'Train' and 'Test' folders.
    # The target frame is the ID image used for face verification.
    
    # Using relative paths for better portability
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Default dataset path assumes it's in a 'Dataset_Parser/OEPFrame_Dataset' directory
    # relative to the project root.
    DEFAULT_DATASET_PATH = os.path.join(project_root, "Dataset_Parser", "OEPFrame_Dataset")
    
    # Check if the default path exists, otherwise, prompt the user.
    if os.path.exists(DEFAULT_DATASET_PATH):
        dataset_path = DEFAULT_DATASET_PATH
    else:
        # A placeholder path. USER MUST UPDATE THIS.
        print("="*50)
        print("WARNING: Default dataset path not found.")
        print(f"Tried to find it at: {DEFAULT_DATASET_PATH}")
        dataset_path = "PLEASE_UPDATE_THIS_PATH/CheatusDeletus/Dataset_Parser/OEPFrame_Dataset"
        print(f"Please update the 'dataset_path' variable in the script to your correct path.")
        print("="*50)

    target_frame_path = os.path.join(dataset_path, "ID.png")
    output_dir = os.path.join(project_root, "processed_results")
    
    # --- Run Processing ---
    if "PLEASE_UPDATE_THIS_PATH" in dataset_path:
        print("\nExiting: Please configure the dataset path in the script before running.")
    else:
        try:
            process_dataset_and_save_csv(dataset_path, target_frame_path, output_dir)
        except FileNotFoundError as e:
            print(f"\nERROR: A file was not found. Please check your paths.")
            print(f"Details: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
