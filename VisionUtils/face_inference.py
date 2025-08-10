from collections import defaultdict, deque
import time
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from VisionUtils.FaceDetailsCalculator import FaceDetails
from deepface import DeepFace

face_attributes = [
    "num_faces",
    "iris_pos",
    "iris_ratio",
    "mouth_zone",
    "mouth_area",
    "x_rotation",
    "y_rotation",
    "z_rotation",
    "radial_distance",
    "gaze_direction",
    "gaze_zone"
]

# model_path = r"C:\Users\singl\Desktop\Bhuvanesh\NITK\SEM4\IT255_AI\Project Files\Face New\face_landmarker.task"

# def verify_id(frame, referenceframe):
#     try:
#         frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # referenceframe = cv2.imread(r"C:\Users\singl\Desktop\Bhuvanesh\NITK\SEM4\IT255_AI\Project Files\DATASET\bhuv_fac\ID.png")
#         referenceframe = cv2.cvtColor(referenceframe, cv2.COLOR_BGR2RGB)
#         result = DeepFace.verify(img1_path=frame_new, img2_path=referenceframe, detector_backend='mediapipe', model_name='ArcFace')
#         print(result)
#         return result['verified']
#     except Exception as e:
#         print("Error in DeepFace verification:", e)
#         return False
all_backends = ["retinaface", "mediapipe", "opencv"]
primary_backend = all_backends[0]  # Start with retinaface

# Track recent history & performance
success_count = defaultdict(int)
attempt_count = defaultdict(int)
avg_time = defaultdict(float)
history = {b: deque(maxlen=50) for b in all_backends}

# Preload models once (important for CPU performance)

def try_verification(frame, referenceframe, backend):
    """Try verification with given backend; returns (success_flag, result_dict_or_None)."""
    attempt_count[backend] += 1
    start = time.time()
    try:
        result = DeepFace.verify(
            img1_path=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            img2_path=cv2.cvtColor(referenceframe, cv2.COLOR_BGR2RGB),
            detector_backend=backend,
            model_name='ArcFace'
        )
        elapsed = (time.time() - start) * 1000
        avg_time[backend] = 0.9 * avg_time[backend] + 0.1 * elapsed
        if result.get('verified', False):
            success_count[backend] += 1
            history[backend].append(1)
            return True, result
    except Exception:
        pass
    history[backend].append(0)
    return False, None

def choose_primary():
    """Pick the best backend based on last N frames' success rate & time."""
    rates = {}
    for b in all_backends:
        if history[b]:
            rates[b] = (sum(history[b]) / len(history[b]), avg_time[b] or 9999)
    if not rates:
        return primary_backend
    # Sort: highest success rate, then lowest avg time
    return sorted(rates.items(), key=lambda x: (-x[1][0], x[1][1]))[0][0]

def verify_id(frame, referenceframe, frame_num=None):
    global primary_backend
    try:
        # Validate frames
        if (frame is None or referenceframe is None or
            not hasattr(frame, 'shape') or not hasattr(referenceframe, 'shape') or
            len(frame.shape) != 3 or len(referenceframe.shape) != 3):
            return False
        
        # Try primary backend
        success, result = try_verification(frame, referenceframe, primary_backend)

        # If primary fails → try the other backend
        if not success:
            for fb in [b for b in all_backends if b != primary_backend]:
                success, result = try_verification(frame, referenceframe, fb)
                if success:
                    primary_backend = fb  # switch immediately for next call
                    print(f"[INFO] Switched primary to {fb}")
                    break
        
        # Periodic backend reassessment (optional if called many times)
        if frame_num is not None and frame_num % 50 == 0:
            new_primary = choose_primary()
            if new_primary != primary_backend:
                print(f"[INFO] Changing primary from {primary_backend} → {new_primary}")
                primary_backend = new_primary

        if result:
            print(f"Verification result: {result}")
        return success
    except Exception as e:
        print(f"Error during verification: {e}")
        return False
def get_landmark_details(result, input_image: mp.Image, timestamp_ms: int):
    face_details = FaceDetails(result, input_image.numpy_view())
    details = {attr: getattr(face_details, attr) for attr in face_attributes}
    return details

# BaseOptions = mp.tasks.BaseOptions
# FaceLandmarker = mp.tasks.vision.FaceLandmarker
# FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
# FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
# FaceDetector = mp.tasks.vision.FaceDetector
# FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
# FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
# VisionRunningMode = mp.tasks.vision.RunningMode
# model_path = 'face_landmarker.task' 

# options = FaceLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.IMAGE,
#     num_faces = 3)

# landmarker =  FaceLandmarker.create_from_options(options)

# Parse command-line arguments

# # Directory containing the images
# image_dirs = [r"C:\Users\singl\Desktop\Bhuvanesh\NITK\SEM4\IT255_AI\Project Files\DATASET\bhuv_fac\cheating_frames",
#               r"C:\Users\singl\Desktop\Bhuvanesh\NITK\SEM4\IT255_AI\Project Files\DATASET\bhuv_fac\not_cheating_frames"]

# List to store results
# results = []

def get_face_inference(frame, target_frame, landmarker):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    face_landmarker_result = landmarker.detect(mp_image)
    output_details = get_landmark_details(face_landmarker_result, mp_image, 0)
    verification_result = verify_id(mp_image.numpy_view().copy(), target_frame)
    output_details['verification_result'] = verification_result
    return output_details

# Process each image in the directory
# for image_dir in image_dirs:
#     for image_file in os.listdir(image_dir):
#         image_path = os.path.join(image_dir, image_file)
#         frame = cv2.imread(image_path)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
#         face_landmarker_result = landmarker.detect(mp_image)
#         output_details = print_landmark_details(face_landmarker_result, mp_image, 0)
#         verification_result = verify(mp_image.numpy_view().copy())
#         output_details['verification_result'] = verification_result
#         results.append(output_details)
#         print(f"Processed {image_file}")

#     # Convert results to a DataFrame
#     df = pd.DataFrame(results)
#     print(df)
#     # Extract subfolder name and save DataFrame as CSV
#     subfolder_name = os.path.basename(image_dir)
#     csv_path = os.path.join(os.path.dirname(image_dir), f"{subfolder_name}.csv")
#     df.to_csv(csv_path, index=False)
#     print(f"Results saved to {csv_path}")
