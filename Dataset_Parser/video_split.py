import cv2
import os
import sys

def split_video(input_path, train_ratio=0.8):
    """
    Split a video into training and testing portions.
    
    Args:
        input_path: Path to the input video file
        train_ratio: Ratio for training split (default 0.8 for 80%)
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} does not exist")
        return
    
    # Get video properties
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate split point
    train_frames = int(total_frames * train_ratio)
    
    # Get output filenames
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.dirname(input_path)
    train_output = os.path.join(output_dir, f"{base_name}_train.avi")
    test_output = os.path.join(output_dir, f"{base_name}_test.avi")
    
    # Define codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Create video writers
    train_writer = cv2.VideoWriter(train_output, fourcc, fps, (width, height))
    test_writer = cv2.VideoWriter(test_output, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count < train_frames:
            train_writer.write(frame)
        else:
            test_writer.write(frame)
            
        frame_count += 1
    
    # Release everything
    cap.release()
    train_writer.release()
    test_writer.release()
    
    print(f"Video split complete:")
    print(f"Training video: {train_output} ({train_frames} frames)")
    print(f"Testing video: {test_output} ({total_frames - train_frames} frames)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python video_split.py <input_video_path>")
        sys.exit(1)
    
    input_video = sys.argv[1]
    split_video(input_video)