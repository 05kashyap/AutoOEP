"""
Visualization utilities for video proctor
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class Visualizer:
    """Handles visualization of frames and prediction plots"""
    
    def __init__(self):
        self.plot_initialized = False
        self.fig = None
        self.ax = None
        self.line1 = None
        self.line2 = None
        self.line3 = None
        
        # Data for visualization
        self.timestamps = deque(maxlen=100)
        self.predictions = deque(maxlen=100)
        self.static_scores = deque(maxlen=100)
        self.xgboost_scores = deque(maxlen=100)
    
    def create_display_frame(self, face_frame, hand_frame, result):
        """Create a display frame with annotations for visualization"""
        # Resize frames to the same height if necessary
        max_height = max(face_frame.shape[0], hand_frame.shape[0])
        
        face_frame = self._resize_frame_to_height(face_frame, max_height)
        hand_frame = self._resize_frame_to_height(hand_frame, max_height)
        
        # Create a combined frame
        combined_width = face_frame.shape[1] + hand_frame.shape[1]
        combined_frame = np.zeros((max_height, combined_width, 3), dtype=np.uint8)
        
        # Place frames side by side
        combined_frame[:, :face_frame.shape[1]] = face_frame
        combined_frame[:, face_frame.shape[1]:] = hand_frame
        
        # Add annotations
        self._add_annotations(combined_frame, result)
        
        return combined_frame
    
    def _resize_frame_to_height(self, frame, target_height):
        """Resize frame to target height while maintaining aspect ratio"""
        if frame.shape[0] != target_height:
            scale = target_height / frame.shape[0]
            new_width = int(frame.shape[1] * scale)
            frame = cv2.resize(frame, (new_width, target_height))
        return frame
    
    def _add_annotations(self, frame, result):
        """Add text annotations to the frame"""
        static_results = result['static_results']
        temporal_pred = result.get('temporal_prediction')
        xgboost_pred = result.get('xgboost_prediction')
        static_model_pred = result.get('static_model_prediction')
        
        y_offset = 30
        
        # Draw static cheat score
        static_score = static_results.get('Cheat Score', 0)
        cv2.putText(frame, f"Static Score: {static_score:.2f}", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
        
        # Draw temporal prediction
        if temporal_pred is not None:
            color = (0, 0, 255) if temporal_pred > 0.5 else (0, 255, 0)
            cv2.putText(frame, f"Temporal Prediction: {temporal_pred:.2f}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        # Draw XGBoost prediction
        if xgboost_pred is not None:
            color = (0, 0, 255) if xgboost_pred > 0.5 else (0, 255, 0)
            cv2.putText(frame, f"XGBoost Prediction: {xgboost_pred:.2f}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        # Draw static model prediction
        if static_model_pred is not None:
            color = (0, 0, 255) if static_model_pred > 0.5 else (0, 255, 0)
            cv2.putText(frame, f"Static Model: {static_model_pred:.2f}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        # Add warning for high probability
        warning_threshold = 0.7
        high_risk_predictions = [p for p in [temporal_pred, xgboost_pred, static_model_pred] 
                               if p is not None and p > warning_threshold]
        
        if high_risk_predictions:
            cv2.putText(frame, "WARNING: Likely Cheating Detected", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def initialize_plot(self):
        """Initialize the matplotlib plot for displaying predictions over time"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.line1, = self.ax.plot([], [], 'r-', label='Temporal Prediction')
        self.line2, = self.ax.plot([], [], 'b-', label='Static Score')
        self.line3, = self.ax.plot([], [], 'g-', label='XGBoost Prediction')
        
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Cheat Probability')
        self.ax.set_title('Cheating Detection')
        self.ax.legend()
        self.ax.grid(True)
        
        self.plot_initialized = True
        plt.show(block=False)
    
    def update_plot(self):
        """Update the plot with new data"""
        if not self.plot_initialized or len(self.timestamps) < 2:
            return
        
        if len(self.timestamps) > 0:
            rel_times = [t - self.timestamps[0] for t in self.timestamps]
            
            self.line1.set_data(rel_times, self.predictions)
            self.line2.set_data(rel_times, self.static_scores)
            self.line3.set_data(rel_times, self.xgboost_scores)
            
            self.ax.set_xlim(0, rel_times[-1] + 0.5)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
    
    def get_plot_frame(self):
        """Render the matplotlib plot as an image for video output"""
        self.fig.canvas.draw()
        plot_image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)
    
    def add_prediction_data(self, timestamp, temporal_pred, static_score, xgboost_pred):
        """Add new prediction data for visualization"""
        self.timestamps.append(timestamp)
        self.predictions.append(temporal_pred if temporal_pred is not None else 0)
        self.static_scores.append(static_score)
        self.xgboost_scores.append(xgboost_pred if xgboost_pred is not None else 0)
    
    def close_plot(self):
        """Close the matplotlib plot"""
        if self.fig:
            plt.close(self.fig)
