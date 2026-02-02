import os
import csv
import cv2
import numpy as np
import torch
import mediapipe as mp
import re
from ultralytics import YOLO
from tqdm import tqdm
import warnings
import sys
import contextlib
from typing import Dict, Any, Set, Optional, List
import argparse

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Proctor.proctor import StaticProctor

# Suppress warnings and logging output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

try:
    cv2.setUseOptimized(True)
    _cpus = os.cpu_count() if hasattr(os, 'cpu_count') else None
    if isinstance(_cpus, int) and _cpus and _cpus > 1:
        cv2.setNumThreads(max(1, _cpus - 1))
except Exception:
    pass


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
    Extracts and processes proctoring features from face and hand video frames.
    """
    
    # Class-level constants
    ALL_OBJECTS = {'cell phone', 'chits', 'closedbook', 'earpiece', 'headphone', 'openbook', 'sheet', 'watch'}
    
    CATEGORICAL_MAPPINGS = {
        'iris_pos': {'center': 0, 'left': 1, 'right': 2},
        'mouth_zone': {'GREEN': 0, 'YELLOW': 1, 'ORANGE': 2, 'RED': 3},
        'gaze_direction': {'forward': 0, 'left': 1, 'right': 2, 'up': 3, 'down': 4},
        'gaze_zone': {'white': 0, 'yellow': 1, 'red': 2}
    }
    
    # Column order for output CSV
    OUTPUT_COLUMNS = [
        'video_id', 'frame_id', 'verification_result', 'num_faces', 'iris_pos', 'iris_ratio',
        'mouth_zone', 'mouth_area', 'x_rotation', 'y_rotation', 'z_rotation',
        'radial_distance', 'gaze_direction', 'gaze_zone', 'watch', 'headphone',
        'closedbook', 'earpiece', 'cell phone', 'openbook', 'chits', 'sheet',
        'H-Distance', 'F-Distance', 'is_cheating'
    ]
    
    def __init__(
        self,
        target_frame_path: str,
        face_landmarker_path: str = 'best_models/face_landmarker.task',
        yolo_model_path: str = 'best_models/OEP_YOLOv11n.pt',
        device: Optional[str] = None,
        use_half: Optional[bool] = None,
        suppress_runtime_output: bool = True,
    ):
        """
        Initialize the FeatureExtractor.

        Args:
            target_frame_path: Path to the target image for identity verification.
            face_landmarker_path: Path to the MediaPipe face landmarker model.
            yolo_model_path: Path to the YOLO model weights.
            device: Inference device ('auto', 'cpu', 'cuda').
            use_half: Enable FP16 inference on CUDA.
            suppress_runtime_output: Suppress per-frame library logs.
        """
        print("Initializing Feature Extractor...")
        self.target_frame = self._load_image(target_frame_path, "Target frame")
        self.suppress_runtime_output = suppress_runtime_output
        
        with suppress_output():
            resolved_device = (
                torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                if device in (None, 'auto') else torch.device(device)
            )
            
            if resolved_device.type == 'cuda':
                torch.backends.cudnn.benchmark = True

            model = YOLO(yolo_model_path)
            try:
                model.to(resolved_device)
            except Exception:
                pass
            try:
                model.fuse()
            except Exception:
                pass
            
            self._yolo_device = resolved_device
            self._yolo_half = (resolved_device.type == 'cuda') if use_half is None else (use_half and resolved_device.type == 'cuda')
            
            mp_solutions = getattr(mp, 'solutions', None)
            mp_hands_module = getattr(mp_solutions, 'hands', None) if mp_solutions else None
            mp_drawing_utils = getattr(mp_solutions, 'drawing_utils', None) if mp_solutions else None
            if mp_hands_module is None or mp_drawing_utils is None:
                raise ImportError("Failed to import mediapipe hands/drawing_utils modules")

            media_pipe_dict = {
                'mpHands': mp_hands_module,
                'hands': mp_hands_module.Hands(
                    static_image_mode=True, max_num_hands=2,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5
                ),
                'mpdraw': mp_drawing_utils
            }
            
            self.proctor = StaticProctor(model, media_pipe_dict, face_landmarker_path)
        
        print("Initialization complete.")

    @staticmethod
    def _load_image(path: str, name: str = "Image") -> np.ndarray:
        """Load an image from disk."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"{name} not found at {path}")
        return img

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

    def _parse_prohibited_items(self, val: Any) -> Set[str]:
        """Parse prohibited items from raw feature value."""
        items: Set[str] = set()
        if val is None:
            return items
        if isinstance(val, (list, tuple, set, np.ndarray)):
            val_iter = val.tolist() if isinstance(val, np.ndarray) else val
            for x in val_iter:
                if isinstance(x, str):
                    s = x.strip().lower()
                    if s:
                        items.add(s)
            return items
        if isinstance(val, str):
            s = val.strip().strip('[]')
            if not s:
                return items
            for token in re.split(r"[,;|]", s):
                t = token.strip().strip("'\"").lower()
                if t:
                    items.add(t)
        return items

    def _process_raw_features(self, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw features into the final format."""
        processed = {}
        
        # One-hot encode prohibited items
        observed: Set[str] = set()
        for key in ['F-Prohibited Item', 'H-Prohibited Item']:
            if key in raw_features:
                observed.update(self._parse_prohibited_items(raw_features[key]))
        
        for obj in self.ALL_OBJECTS:
            processed[obj] = 1 if obj in observed else 0

        # Apply categorical mappings
        for field, mapping in self.CATEGORICAL_MAPPINGS.items():
            if field in raw_features:
                val = raw_features[field]
                key = val.upper() if field == 'mouth_zone' else (val.lower() if isinstance(val, str) else val)
                processed[field] = mapping.get(key, -1)
            else:
                processed[field] = -1

        # Numeric fields with defaults
        numeric_fields = {
            'verification_result': (self._to_int, 0),
            'num_faces': (self._to_int, 0),
            'iris_ratio': (self._to_float, 1.0),
            'mouth_area': (self._to_float, 0.0),
            'x_rotation': (self._to_float, 0.0),
            'y_rotation': (self._to_float, 0.0),
            'z_rotation': (self._to_float, 0.0),
            'radial_distance': (self._to_float, 0.0),
            'H-Distance': (self._to_float, 10000.0),
            'F-Distance': (self._to_float, 10000.0),
        }
        
        for field, (converter, default) in numeric_fields.items():
            processed[field] = converter(raw_features.get(field), default)
        
        return processed

    def process_frame_pair(self, face_frame: np.ndarray, hand_frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a pair of face and hand frames.

        Args:
            face_frame: Image frame containing the user's face (front camera).
            hand_frame: Image frame containing the user's hands/desk (side camera).

        Returns:
            Dictionary of processed features.
        """
        if face_frame is None or hand_frame is None:
            raise ValueError("Input frames cannot be None.")
            
        ctx = suppress_output() if self.suppress_runtime_output else contextlib.nullcontext()
        with ctx, torch.inference_mode():
            raw_features = self.proctor.process_frames(self.target_frame, face_frame, hand_frame)
        
        return self._process_raw_features(raw_features)

def timestamp_to_seconds(timestamp: str) -> float:
    """Convert a timestamp string to seconds for sorting."""
    parts = timestamp.replace('.jpg', '').split('-')
    if len(parts) >= 4:
        hours = int(parts[0])
        minutes = int(parts[1])
        secs = int(parts[2])
        microseconds = int(parts[3])
        return hours * 3600 + minutes * 60 + secs + microseconds / 1_000_000
    return 0.0


def load_labels(video_folder: str) -> Dict[str, int]:
    """Load labels from labels.csv in the video folder."""
    labels_path = os.path.join(video_folder, "labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.csv not found at {labels_path}")
    
    labels = {}
    with open(labels_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = row['frame_id'].strip()
            label = int(row['label'])
            labels[frame_id] = label
    return labels


def get_frame_ids(video_folder: str) -> List[str]:
    """Get list of frame IDs (timestamps) that exist in both front and side folders."""
    front_folder = os.path.join(video_folder, "front")
    side_folder = os.path.join(video_folder, "side")
    
    if not os.path.exists(front_folder):
        raise FileNotFoundError(f"Front folder not found: {front_folder}")
    if not os.path.exists(side_folder):
        raise FileNotFoundError(f"Side folder not found: {side_folder}")
    
    # Get frame IDs from both folders (remove .jpg extension)
    front_frames = {f.replace('.jpg', '') for f in os.listdir(front_folder) if f.endswith('.jpg')}
    side_frames = {f.replace('.jpg', '') for f in os.listdir(side_folder) if f.endswith('.jpg')}
    
    # Only keep frames that exist in both
    common_frames = front_frames & side_frames
    
    # Sort by timestamp
    return sorted(common_frames, key=timestamp_to_seconds)


def process_video_folder(
    video_folder: str,
    target_frame_path: str,
    output_csv: str,
    video_id: str,
    *,
    face_landmarker_path: str = 'best_models/face_landmarker.task',
    yolo_model_path: str = 'best_models/OEP_YOLOv11n.pt',
    device: Optional[str] = 'auto',
    use_half: Optional[bool] = None,
    suppress_runtime_output: bool = True,
):
    """
    Process all frames in a video folder and append features to output CSV.

    Args:
        video_folder: Path to the video folder (e.g., Dataset/frames_vid1).
        target_frame_path: Path to the target identity image.
        output_csv: Path to the output CSV file (will be created or appended to).
        video_id: Identifier for this video (e.g., "vid1").
        face_landmarker_path: Path to MediaPipe face landmarker model.
        yolo_model_path: Path to YOLO model weights.
        device: Inference device ('auto', 'cpu', 'cuda').
        use_half: Enable FP16 inference on CUDA.
        suppress_runtime_output: Suppress per-frame library logs.
    """
    # Load labels
    labels = load_labels(video_folder)
    
    # Get frame IDs that exist in both cameras
    frame_ids = get_frame_ids(video_folder)
    
    if not frame_ids:
        print(f"No matching frame pairs found in {video_folder}")
        return
    
    print(f"\nProcessing {len(frame_ids)} frames from {video_folder}")
    
    # Initialize feature extractor
    extractor = FeatureExtractor(
        target_frame_path,
        face_landmarker_path=face_landmarker_path,
        yolo_model_path=yolo_model_path,
        device=device,
        use_half=use_half,
        suppress_runtime_output=suppress_runtime_output,
    )
    
    # Check if CSV exists to determine if we need to write headers
    write_header = not os.path.exists(output_csv)
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    front_folder = os.path.join(video_folder, "front")
    side_folder = os.path.join(video_folder, "side")
    
    processed_count = 0
    skipped_count = 0
    
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FeatureExtractor.OUTPUT_COLUMNS)
        
        if write_header:
            writer.writeheader()
        
        for frame_id in tqdm(frame_ids, desc=f"Processing {video_id}", unit="frame"):
            # Load frames
            face_frame = cv2.imread(os.path.join(front_folder, f"{frame_id}.jpg"))
            hand_frame = cv2.imread(os.path.join(side_folder, f"{frame_id}.jpg"))
            
            if face_frame is None or hand_frame is None:
                skipped_count += 1
                continue
            
            # Get label (default to 0 if not in labels.csv)
            label = labels.get(frame_id, 0)
            
            try:
                features = extractor.process_frame_pair(face_frame, hand_frame)
                
                # Add metadata
                features['video_id'] = video_id
                features['frame_id'] = frame_id
                features['is_cheating'] = label
                
                # Ensure column order
                row = {col: features.get(col, 0) for col in FeatureExtractor.OUTPUT_COLUMNS}
                writer.writerow(row)
                processed_count += 1
                
            except Exception as e:
                tqdm.write(f"Error processing {frame_id}: {e}")
                skipped_count += 1
    
    print(f"Completed: {processed_count} frames processed, {skipped_count} skipped")
    print(f"Results appended to: {output_csv}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    default_output_csv = os.path.join(project_root, "processed_results", "features.csv")
    default_mediapipe_task = os.path.join(project_root, "best_models", "face_landmarker.task")
    default_yolo_model = os.path.join(project_root, "best_models", "OEP_YOLOv11n.pt")

    parser = argparse.ArgumentParser(
        description="Extract proctoring features from video frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python feature_extractor.py --video Dataset/frames_vid1 --target target.jpg --video-id vid1
  python feature_extractor.py --video Dataset/frames_vid2 --target target2.jpg --video-id vid2 --output features.csv
        """
    )
    parser.add_argument("--video", required=True, help="Path to video folder containing front/, side/, and labels.csv")
    parser.add_argument("--target", required=True, help="Path to the target identity image for verification")
    parser.add_argument("--video-id", required=True, help="Identifier for this video (e.g., vid1)")
    parser.add_argument("--output", default=default_output_csv, help=f"Output CSV path (default: {default_output_csv})")
    parser.add_argument("--mediapipe-task", default=default_mediapipe_task, help="Path to MediaPipe face landmarker .task file")
    parser.add_argument("--yolo-model", default=default_yolo_model, help="Path to YOLO model weights")
    parser.add_argument("--device", default='auto', choices=['auto', 'cpu', 'cuda'], help="Inference device (default: auto)")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 inference on CUDA")
    parser.add_argument("--no-suppress-logs", action="store_true", help="Show per-frame library logs")

    args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(args.video):
        print(f"ERROR: Video folder not found: {args.video}")
        sys.exit(1)
    if not os.path.isfile(args.target):
        print(f"ERROR: Target image not found: {args.target}")
        sys.exit(1)
    if not os.path.isfile(args.mediapipe_task):
        print(f"WARNING: MediaPipe task file not found at {args.mediapipe_task}")
    if not os.path.isfile(args.yolo_model):
        print(f"WARNING: YOLO model file not found at {args.yolo_model}")

    try:
        process_video_folder(
            video_folder=args.video,
            target_frame_path=args.target,
            output_csv=args.output,
            video_id=args.video_id,
            face_landmarker_path=args.mediapipe_task,
            yolo_model_path=args.yolo_model,
            device=args.device,
            use_half=args.fp16 if args.fp16 else None,
            suppress_runtime_output=not args.no_suppress_logs,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
