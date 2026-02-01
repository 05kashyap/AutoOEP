import cv2
import os
import csv
from datetime import timedelta
from typing import List, Tuple


def format_timestamp(seconds: float) -> str:
    """Convert seconds to a timestamp string with colons replaced by dashes.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string like "0-00-05-123456" (H-MM-SS-microseconds)
    """
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    microseconds = td.microseconds
    return f"{hours}-{minutes:02d}-{secs:02d}-{microseconds:06d}"


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert a timestamp string back to seconds.
    
    Args:
        timestamp: Formatted timestamp string like "0-00-05-123456"
        
    Returns:
        Time in seconds as float
    """
    parts = timestamp.replace('.jpg', '').split('-')
    if len(parts) >= 4:
        hours = int(parts[0])
        minutes = int(parts[1])
        secs = int(parts[2])
        microseconds = int(parts[3])
        return hours * 3600 + minutes * 60 + secs + microseconds / 1_000_000
    return 0.0


def parser(video_dir: str, video_id: int, output_base: str = "Dataset") -> Tuple[str, float, int]:
    """Parse front and side videos for a given video ID and extract frames.
    
    Creates folder structure:
        output_base/frames_vid{ID}/
            front/
                timestamp1.jpg
                timestamp2.jpg
                ...
            side/
                timestamp1.jpg
                timestamp2.jpg
                ...
            labels.csv
    
    Args:
        video_dir: Directory containing the video files
        video_id: The ID of the video pair (e.g., 1 for vid1_front.avi and vid1_side.avi)
        output_base: Base directory for output folders
        
    Returns:
        Tuple of (output_folder_path, fps, total_frames)
    """
    front_video_path = os.path.join(video_dir, f"vid{video_id}_front.avi")
    side_video_path = os.path.join(video_dir, f"vid{video_id}_side.avi")
    
    # Validate video files exist
    if not os.path.exists(front_video_path):
        raise FileNotFoundError(f"Front video not found: {front_video_path}")
    if not os.path.exists(side_video_path):
        raise FileNotFoundError(f"Side video not found: {side_video_path}")
    
    # Create output folder structure
    output_folder = os.path.join(output_base, f"frames_vid{video_id}")
    front_folder = os.path.join(output_folder, "front")
    side_folder = os.path.join(output_folder, "side")
    csv_path = os.path.join(output_folder, "labels.csv")
    
    os.makedirs(front_folder, exist_ok=True)
    os.makedirs(side_folder, exist_ok=True)
    
    # Open both videos with FFMPEG backend for better codec support
    # Try FFMPEG first, fall back to default if not available
    front_video = cv2.VideoCapture(front_video_path, cv2.CAP_FFMPEG)
    side_video = cv2.VideoCapture(side_video_path, cv2.CAP_FFMPEG)
    
    if not front_video.isOpened():
        front_video = cv2.VideoCapture(front_video_path)
    if not side_video.isOpened():
        side_video = cv2.VideoCapture(side_video_path)
    
    front_fps = front_video.get(cv2.CAP_PROP_FPS)
    side_fps = side_video.get(cv2.CAP_PROP_FPS)
    
    # Use front video's fps as reference (they should be the same)
    fps = front_fps
    if abs(front_fps - side_fps) > 0.1:
        print(f"Warning: FPS mismatch - front: {front_fps}, side: {side_fps}. Using front FPS.")
    
    frame_index = 0  # Logical frame index for timestamp calculation
    saved_count = 0  # Successfully saved frame pairs
    skipped_count = 0  # Skipped due to corruption
    frame_ids = []
    
    print(f"\nParsing vid{video_id} (front + side)...")
    print(f"  Front video: {front_video_path}")
    print(f"  Side video: {side_video_path}")
    print(f"  Output folder: {output_folder}")
    
    while True:
        ret_front, frame_front = front_video.read()
        ret_side, frame_side = side_video.read()
        
        # Stop if either video ends
        if not ret_front or not ret_side:
            break
        
        # Check for corrupted/empty frames and skip them
        if frame_front is None or frame_side is None:
            skipped_count += 1
            frame_index += 1
            continue
        
        # Additional check: skip if frame is mostly black/corrupted (optional heuristic)
        # A corrupted frame often has very low mean pixel values or is uniform
        try:
            front_mean = frame_front.mean()
            side_mean = frame_side.mean()
            
            # Skip if frame appears corrupted (all black or invalid)
            if front_mean < 1.0 or side_mean < 1.0:
                skipped_count += 1
                frame_index += 1
                continue
        except Exception:
            skipped_count += 1
            frame_index += 1
            continue
        
        # Generate timestamp-based filename
        timestamp = format_timestamp(frame_index / fps)
        frame_filename = f"{timestamp}.jpg"
        
        # Save frames to respective folders
        front_output = os.path.join(front_folder, frame_filename)
        side_output = os.path.join(side_folder, frame_filename)
        
        try:
            cv2.imwrite(front_output, frame_front)
            cv2.imwrite(side_output, frame_side)
            frame_ids.append(timestamp)
            saved_count += 1
        except Exception as e:
            print(f"  Warning: Failed to save frame {frame_index}: {e}")
            skipped_count += 1
        
        frame_index += 1
        
        # Print progress every 500 frames
        if frame_index % 500 == 0:
            print(f"  Processed {frame_index} frames (saved: {saved_count}, skipped: {skipped_count})...")
    
    front_video.release()
    side_video.release()
    
    # Create CSV with frame_id and label (default 0)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['frame_id', 'label'])
        writer.writeheader()
        for frame_id in frame_ids:
            writer.writerow({'frame_id': frame_id, 'label': 0})
    
    print(f"  Completed: {saved_count} frame pairs saved, {skipped_count} skipped")
    print(f"  CSV saved: {csv_path}")
    
    return output_folder, fps, saved_count


def get_annotation_intervals() -> List[Tuple[float, float]]:
    """Prompt user for cheating time intervals.
    
    Returns:
        List of (start_seconds, end_seconds) tuples representing cheating intervals
    """
    intervals = []
    print("\nEnter cheating time intervals (start and end in seconds).")
    print("Type 'q' when done entering all intervals.\n")
    
    while True:
        start = input("Enter start time (seconds) or 'q' to finish: ").strip()
        if start.lower() == 'q':
            break
        
        try:
            start_sec = float(start)
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
            
        end = input("Enter end time (seconds): ").strip()
        try:
            end_sec = float(end)
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        if end_sec <= start_sec:
            print("End time must be greater than start time. Try again.")
            continue
            
        intervals.append((start_sec, end_sec))
        print(f"  Added interval: {start_sec}s - {end_sec}s")
    
    return intervals


def annotator(video_folder: str, intervals: List[Tuple[float, float]] = None) -> int:
    """Annotate frames within specified time intervals as cheating (label=1).
    
    Args:
        video_folder: Path to the video folder (e.g., "Dataset/frames_vid1")
        intervals: List of (start_seconds, end_seconds) tuples. 
                   If None, will prompt user for input.
    
    Returns:
        Number of frames annotated as cheating
    """
    csv_path = os.path.join(video_folder, "labels.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")
    
    # Get intervals from user if not provided
    if intervals is None:
        intervals = get_annotation_intervals()
    
    if not intervals:
        print("No intervals provided. No annotations made.")
        return 0
    
    print(f"\nAnnotating frames in: {video_folder}")
    print(f"Intervals to annotate: {intervals}")
    
    # Read existing CSV
    rows = []
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    
    # Annotate frames within intervals
    annotated_count = 0
    for row in rows:
        frame_id = row['frame_id']
        frame_seconds = timestamp_to_seconds(frame_id)
        
        # Check if frame falls within any cheating interval
        is_cheating = any(start <= frame_seconds < end for start, end in intervals)
        
        if is_cheating:
            row['label'] = 1
            annotated_count += 1
    
    # Write updated CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['frame_id', 'label'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Annotated {annotated_count} frames as cheating (label=1)")
    print(f"Updated CSV: {csv_path}")
    
    return annotated_count


def main():
    """Main entry point with interactive menu."""
    print("=" * 50)
    print("Video Parser and Annotator for Dual Camera Setup")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Parse videos (extract frames)")
        print("2. Annotate frames (label cheating intervals)")
        print("3. Parse and annotate")
        print("q. Quit")
        
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == 'q':
            print("Goodbye!")
            break
        
        if choice == '1' or choice == '3':
            video_dir = input("Enter path to video directory: ").strip()
            if not os.path.isdir(video_dir):
                print(f"Error: Directory not found: {video_dir}")
                continue
                
            video_id_input = input("Enter video ID (e.g., 1 for vid1_front.avi): ").strip()
            try:
                video_id = int(video_id_input)
            except ValueError:
                print("Invalid video ID. Please enter a number.")
                continue
            
            output_base = input("Enter output base directory [Dataset]: ").strip() or "Dataset"
            
            try:
                output_folder, fps, total_frames = parser(video_dir, video_id, output_base)
                print(f"\nParsing complete! Extracted {total_frames} frames at {fps:.2f} FPS")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                continue
            
            if choice == '3':
                # Continue to annotation
                annotator(output_folder)
        
        elif choice == '2':
            video_folder = input("Enter path to video frames folder (e.g., Dataset/frames_vid1): ").strip()
            if not os.path.isdir(video_folder):
                print(f"Error: Folder not found: {video_folder}")
                continue
            
            try:
                annotator(video_folder)
            except FileNotFoundError as e:
                print(f"Error: {e}")
        
        else:
            print("Invalid choice. Please enter 1, 2, 3, or q.")


if __name__ == "__main__":
    main()

