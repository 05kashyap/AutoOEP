import cv2
from deepface import DeepFace
import numpy as np
import mediapipe as mp
from cheat_prob import calculate_cheat_score
from FaceDetailsCalculator import FaceDetails
import torch
from handpose import inference
from ultralytics import YOLO  

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Path to the face landmark model
model_path = 'face_landmarker.task'

def enhance_frame(image):
    alpha = 1.3
    beta = 30
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    kernel = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
    new = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return new

def preprocess_for_object_detection(image):
    """
    Enhance image to improve object detection by:
    1. Adjusting contrast and brightness
    2. Reducing noise
    3. Enhancing edges
    """
    # Convert to RGB if needed (YOLO typically works with RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image.copy()
    
    # Increase contrast and brightness
    alpha = 1.4  # Contrast control (1.0-3.0)
    beta = 15    # Brightness control (0-100)
    contrast_bright = cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)
    
    # Reduce noise while preserving edges
    denoised = cv2.fastNlMeansDenoisingColored(contrast_bright, None, 10, 10, 7, 21)
    
    # Enhance edges
    kernel_sharpen = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
    
    # Convert back to BGR for OpenCV operations
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR)
    return sharpened


class StaticProctor:
    def __init__(self, yolo_model, media_pipe):
        self.yolo_model = yolo_model
        self.media_pipe = media_pipe
        
        # Initialize MediaPipe with IMAGE mode instead of LIVE_STREAM
        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE)
            
        self.landmarker = FaceLandmarker.create_from_options(self.options)
        
    def process_frames(self, target_frame, input_frame):
        output = {}
        
        preprocessed_frame = preprocess_for_object_detection(input_frame)

        # Hand detection
        hand_dict = inference(preprocessed_frame, self.yolo_model, self.media_pipe)
        if hand_dict:
            hand_keys = {
                'hand_detected': 'Hand Detected',
                'prohibited_item_use': 'Prohibited Item Use', 
                'distance': 'Distance',
                'illegal_objects': 'Illegal Objects',
                'prohibited_item': 'Prohibited Item'
            }
            
            for src_key, dst_key in hand_keys.items():
                if src_key in hand_dict:
                    output[dst_key] = hand_dict[src_key]

        # Face verification
        processed_frame = enhance_frame(cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB))
        processed_target = enhance_frame(cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)) if target_frame.dtype != np.uint8 else target_frame
        
        output['Identity'] = DeepFace.verify(
            img1_path=processed_target, 
            img2_path=processed_frame,
            model_name='ArcFace',
            detector_backend='mediapipe',
            normalization='ArcFace',
            align=False,
            enforce_detection=False)['verified']

        # Face landmarks - for static images
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_frame)
        landmark_result = self.landmarker.detect(mp_image)
        
        if landmark_result and landmark_result.face_landmarks:
            face_details = FaceDetails(landmark_result, input_frame)
            output['Face Direction'] = face_details.gaze_direction
            output['Face Zone'] = face_details.gaze_zone
            output['Eye Direction'] = face_details.iris_pos
            output['Mouth'] = face_details.mouth_zone
            output['Number of people'] = face_details.num_faces

        # Calculate cheat score
        output['Cheat Score'] = calculate_cheat_score(output)
        return output, preprocessed_frame

    def __del__(self):
        # Cleanup
        if hasattr(self, 'landmarker'):
            self.landmarker.close()


def example_usage():
    # Example of how to use the StaticProctor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('/home/kashyap/Documents/Projects/PROCTOR/CheatusDeletus/Proctor/OEP_YOLOv11n.pt')

    
    mpHands = mp.solutions.hands
    media_pipe_dict = {
        'mpHands': mpHands,
        'hands': mpHands.Hands(static_image_mode=True,  # Set to True for static images
                             max_num_hands=2,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5),
        'mpdraw': mp.solutions.drawing_utils
    }

    # Load target and input frames
    target_frame = cv2.imread('Images/facecam1.png')
    input_frame = cv2.imread('Images/handcam2.png')
    
    # Process frames
    proctor = StaticProctor(model, media_pipe_dict)
    result, preprocessed = proctor.process_frames(target_frame, input_frame)
    print(result)
    # cv2.imshow("Original", input_frame)
    # cv2.imshow("Preprocessed", preprocessed)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    example_usage()