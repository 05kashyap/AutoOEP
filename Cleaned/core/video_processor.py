"""
Video processing utilities for video proctor
"""
import cv2
import time
import numpy as np
from collections import deque


class VideoProcessor:
    """Handles video capture, processing, and output"""
    
    def __init__(self, fps=30):
        self.fps = fps
    
    def process_videos(self, face_video_path, hand_video_path, target_frame_path, 
                      frame_processor, visualizer, output_path=None, display=True, 
                      test_duration=None):
        """
        Process video streams from face and hand cameras
        
        Args:
            face_video_path: Path to face camera video
            hand_video_path: Path to hand/desk camera video
            target_frame_path: Path to reference face image
            frame_processor: Function to process frame pairs
            visualizer: Visualizer instance for display
            output_path: Path to save output video (optional)
            display: Whether to display the processed video
            test_duration: Duration (in seconds) to process for testing (optional)
            
        Returns:
            List of analysis results
        """
        # Load target frame
        target_frame = cv2.imread(target_frame_path)
        if target_frame is None:
            raise FileNotFoundError(f"Target frame not found at {target_frame_path}")
        
        # Open video captures
        face_cap = cv2.VideoCapture(face_video_path)
        hand_cap = cv2.VideoCapture(hand_video_path)
        
        if not face_cap.isOpened() or not hand_cap.isOpened():
            raise ValueError("Error opening video streams")
        
        # Initialize video writer and visualization
        out = None
        if output_path or display:
            visualizer.initialize_plot()

        results = []
        frame_count = 0
        start_time = time.time()
        
        # Calculate maximum frames for testing
        max_frames = None
        if test_duration:
            max_frames = int(test_duration * self.fps)
        
        try:
            while True:
                face_ret, face_frame = face_cap.read()
                hand_ret, hand_frame = hand_cap.read()
                
                if not face_ret or not hand_ret:
                    break
                
                # Process frame pair
                result = frame_processor(target_frame, face_frame, hand_frame)
                results.append(result)
                
                # Create display frame
                display_frame = visualizer.create_display_frame(face_frame, hand_frame, result)
                
                # Initialize video writer after first frame
                if out is None and output_path:
                    out = self._initialize_video_writer(output_path, display_frame)
                
                # Update visualization
                visualizer.update_plot()
                    
                # Save to output video
                if output_path and out is not None:
                    combined_frame = self._create_combined_frame(display_frame, visualizer)
                    out.write(combined_frame)
                
                # Display video
                if display:
                    cv2.imshow('Video Proctor', display_frame)
                    if cv2.waitKey(1) == 27:  # ESC key
                        break
                
                frame_count += 1
                
                if max_frames and frame_count >= max_frames:
                    break
        
        finally:
            # Cleanup
            self._cleanup_resources(face_cap, hand_cap, out, visualizer, display)
            
            # Print statistics
            elapsed_time = time.time() - start_time
            processed_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({processed_fps:.2f} FPS)")
            
            if output_path:
                print(f"Video output saved to: {output_path}")
        
        return results
    
    def _initialize_video_writer(self, output_path, display_frame):
        """Initialize video writer with proper dimensions"""
        out_width = display_frame.shape[1]
        out_height = display_frame.shape[0] + 200  # Extra space for visualization
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (out_width, out_height))
        print(f"Initialized video writer with dimensions: {out_width}x{out_height}")
        return out
    
    def _create_combined_frame(self, display_frame, visualizer):
        """Create combined frame with plot for video output"""
        plot_frame = visualizer.get_plot_frame()
        plot_frame = cv2.resize(plot_frame, (display_frame.shape[1], 200))
        
        combined_frame = np.zeros((display_frame.shape[0] + plot_frame.shape[0], 
                                 display_frame.shape[1], 3), dtype=np.uint8)
        combined_frame[:display_frame.shape[0], :] = display_frame
        combined_frame[display_frame.shape[0]:, :] = plot_frame
        
        return combined_frame
    
    def _cleanup_resources(self, face_cap, hand_cap, out, visualizer, display):
        """Clean up video capture and display resources"""
        face_cap.release()
        hand_cap.release()
        
        if out:
            out.release()
        
        if display:
            cv2.destroyAllWindows()
        
        visualizer.close_plot()
