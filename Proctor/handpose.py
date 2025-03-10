import cv2
import mediapipe as mp
import math
import torch
import os
import warnings
import logging
from ultralytics import YOLO 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('mediapipe').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
mpHands = None
hands = None
mpdraw = None
device = None
model = None
mylmList = []

def initialize(yolo_model, media_pipe):
    """
    Initializes the necessary components for the handpose module.

    This function initializes the hand detection model, drawing utilities, and other variables required for handpose operations.

    Args:
        yolo_model: The YOLO model for object detection
        media_pipe: Dictionary containing MediaPipe components

    Returns:
        None
    """
    global mpHands, hands, mpdraw, device, model, mylmList

    mpHands = media_pipe['mpHands']
    hands = media_pipe['hands']
    mpdraw = media_pipe['mpdraw']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = yolo_model
    mylmList = []

def initialize_camera():
    """
    Initialize the camera and return the VideoCapture object.
    """
    cap = cv2.VideoCapture(0)
    return cap

def process_frame(cap):
    """
    Process a single frame from the camera and return the image.
    """
    success, image = cap.read()
    return success, image

def calculate_distance(x1, y1, x2, y2):
    """
    Calculates the distance between two points in a 2D plane.

    Parameters:
    x1 (float): The x-coordinate of the first point.
    y1 (float): The y-coordinate of the first point.
    x2 (float): The x-coordinate of the second point.
    y2 (float): The y-coordinate of the second point.

    Returns:
    float: The distance between the two points.
    """
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def extract_hand_landmarks(image):
    """
    Extract hand landmarks using MediaPipe Hands.
    """
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    return results.multi_hand_landmarks

def handYOLO(frame, model):
    """
    Detects hands and prohibited items in a given frame using YOLOv11 object detection and Mediapipe Hands.

    Args:
        frame: The input frame to be processed.
        model: The YOLOv11 model

    Returns:
        A dictionary containing the following information:
        - 'prohibited_item': The name of the prohibited item detected.
        - 'hand_detected': A boolean indicating whether a hand was detected.
        - 'distance': The distance between the center of the prohibited item and the center of the hand.
        - 'Prohibited Item Use': A boolean indicating whether the prohibited item was used.
        - 'illegal_objects': The number of prohibited items detected in the frame.
    """
    output = {}
    output['prohibited_item'] = None
    output['hand_detected'] = False
    output['distance'] = None
    output['Prohibited Item Use'] = False
    output['illegal_objects'] = 0

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB for Mediapipe processing

    # Run YOLOv11 inference
    detections = model(frame)
    yolo_bboxes = []

    # Process YOLOv11 detections
    for detection in detections[0]:
        boxes = detection.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            item_name = model.names[class_id]
            # print(f"Detected: {item_name} with confidence: {confidence:.2f}")

            if(confidence < 0.4):
                continue
            output['prohibited_item'] = item_name
            
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            yolo_bboxes.append(bbox) #gets added to list to process hands
            output['illegal_objects'] += 1
            
            # Draw rectangle and label for illegal object
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            
            # Add text with item name and confidence
            label = f"{item_name}: {confidence:.2f}"
            cv2.putText(frame, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    allHands = []
    h, w, c = frame.shape  # Get the height, width, and number of channels of the frame

    results = extract_hand_landmarks(img)
    if not results:
        return output
    
    # Process each detected hand in the frame
    for handLms in results:
        output['hand_detected'] = True
        myHand = {}
        mylmList = []
        xList = []
        yList = []

        # Extract landmark points and store them in lists
        for id, lm in enumerate(handLms.landmark):
            px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
            mylmList.append([id, px, py])
            xList.append(px)
            yList.append(py)

        # Calculate bounding box around the hand landmarks
        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        boxW, boxH = xmax - xmin, ymax - ymin
        bbox = xmin, ymin, boxW, boxH
        cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

        # Store hand information in a dictionary
        myHand["lmList"] = mylmList
        myHand["bbox"] = bbox
        myHand["center"] = (cx, cy)

        allHands.append(myHand)

        # Draw landmarks and bounding box on the frame
        mpdraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
        cv2.rectangle(frame, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 0, 255), 2)

    for hand in allHands:
        for yolo_bbox in yolo_bboxes:
            x1, y1, x2, y2 = yolo_bbox
            yolo_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            hand_center = hand["center"]
            distance = calculate_distance(yolo_center[0], yolo_center[1], hand_center[0], hand_center[1])
            output['distance'] = distance

            if distance < 200:
                output['Prohibited Item Use'] = True
                # Draw a line connecting hand and object to visualize proximity
                cv2.line(frame, hand_center, yolo_center, (255, 0, 0), 2)
                cv2.putText(frame, f"Alert! Distance: {distance:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Add overall status to the frame
    if output['Prohibited Item Use']:
        status = "ALERT: Prohibited Item in Use!"
        color = (0, 0, 255)  # Red for alert
    elif output['prohibited_item']:
        status = f"Detected: {output['prohibited_item']}"
        color = (0, 255, 255)  # Yellow for detection
    else:
        status = "No prohibited items"
        color = (0, 255, 0)  # Green for all clear
        
    cv2.putText(frame, status, (10, h-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return output

def inference(frame, yolo_model, media_pipe):
    """
    Perform hand pose inference on the given frame.

    Args:
        frame: The input frame on which to perform hand pose inference.
        yolo_model: The YOLOv11 model
        media_pipe: Dictionary containing MediaPipe components

    Returns:
        The output of the hand pose inference.
    """
    initialize(yolo_model, media_pipe)
    output = handYOLO(frame, model)
    return output

if __name__ == '__main__':
    cap = initialize_camera()
    
    # Initialize YOLOv11 model
    model = YOLO('/home/kashyap/Documents/Projects/PROCTOR/CheatusDeletus/Proctor/OEP_YOLOv11n.pt')
    
    mpHands = mp.solutions.hands
    media_pipe_dict = {
        'mpHands': mpHands,
        'hands': mpHands.Hands(static_image_mode=False, 
                              max_num_hands=2,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5),
        'mpdraw': mp.solutions.drawing_utils
    }

    while True:
        ret, frame = process_frame(cap)
        if ret:
            output = inference(frame, model, media_pipe_dict)
            print(output)
            cv2.imshow('Hand Pose Estimation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()