import cv2
import numpy as np
from ultralytics import YOLO
import argparse

def detect_cell_phone(image_path, confidence_threshold=0.25):
    """
    Detect cell phones in an image using YOLOv8
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence score to consider a detection valid
        
    Returns:
        Annotated image and boolean indicating if cell phone was detected
    """
    # Load the YOLOv8 model
    model = YOLO('/home/kashyap/Documents/Projects/PROCTOR/CheatusDeletus/Proctor/OEP_YOLOv11n.pt')  # Using the nano model, can use 's', 'm', 'l', or 'x' for larger models
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Perform detection
    results = model(image)
    
    # Process results
    cell_phone_detected = False
    annotated_image = image.copy()
    
    # COCO dataset class 67 is 'cell phone'
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class ID
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id]
            conf = box.conf.item()
            
            # Check if the detected object is a cell phone and confidence is above threshold
            if cls_name == 'cell phone' and conf > confidence_threshold:
                cell_phone_detected = True
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{cls_name}: {conf:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_image, cell_phone_detected

def main():
    parser = argparse.ArgumentParser(description="Detect cell phones in images using YOLOv8")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save", action="store_true", help="Save the annotated image")
    args = parser.parse_args()
    
    try:
        # Perform detection
        annotated_image, cell_phone_detected = detect_cell_phone(args.image_path, args.conf)
        
        # Display results
        if cell_phone_detected:
            print("Cell phone detected in the image!")
        else:
            print("No cell phone detected in the image.")
        
        # Display the image
        cv2.imshow("Detection Result", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the annotated image if requested
        if args.save:
            output_path = args.image_path.rsplit(".", 1)[0] + "_detected.jpg"
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to {output_path}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()