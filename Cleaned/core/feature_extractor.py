"""
Feature extraction utilities for video proctor
"""
import time
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class FeatureExtractor:
    """Handles extraction of features from static proctor results"""
    
    def __init__(self):
        self.start_time = None
    
    def extract_features_from_results(self, results):
        """
        Extract numerical features from static proctor results for temporal analysis
        
        Args:
            results: Dictionary of results from static proctor
            
        Returns:
            List of features matching the format expected by the LSTM model
        """
        features = []
        
        # Add timestamp - using relative timestamp for feature extraction
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
        features.append(current_time - self.start_time)
        
        # Add verification result (Identity)
        features.append(int(results.get('verification_result', False)))
        
        # Add number of faces
        features.append(float(results.get('num_faces', 0)))
        
        # Eye/iris position
        features.append(self._encode_iris_position(results.get('iris_pos', 'center')))
        
        # Add iris ratio
        iris_ratio = results.get('iris_ratio', 1.0)
        features.append(float(iris_ratio) if iris_ratio is not None else 1.0)
        
        # Mouth zone
        features.append(self._encode_mouth_zone(results.get('mouth_zone', 'GREEN')))
        
        # Mouth area
        mouth_area = results.get('mouth_area', 0.0)
        features.append(float(mouth_area) if mouth_area is not None else 0.0)
        
        # Face rotation (x, y, z)
        for rotation_key in ['x_rotation', 'y_rotation', 'z_rotation']:
            rotation = results.get(rotation_key, 0.0)
            features.append(float(rotation) if rotation is not None else 0.0)
        
        # Radial distance
        radial_distance = results.get('radial_distance', 0.0)
        features.append(float(radial_distance) if radial_distance is not None else 0.0)
        
        # Gaze direction and zone
        features.append(self._encode_gaze_direction(results.get('gaze_direction', 'forward')))
        features.append(self._encode_gaze_zone(results.get('gaze_zone', 'white')))
        
        # Process prohibited items
        features.extend(self._extract_prohibited_items(results))
        
        # Add distance values
        h_distance = results.get('H-Distance')
        features.append(float(h_distance) if h_distance is not None else 1000)
        
        f_distance = results.get('F-Distance')
        features.append(float(f_distance) if f_distance is not None else 1000)
        
        return features
    
    def _encode_iris_position(self, eye_dir):
        """Encode iris position to numerical value"""
        if isinstance(eye_dir, str):
            return Config.IRIS_POSITION_MAPPING.get(eye_dir.lower(), -1)
        return -1
    
    def _encode_mouth_zone(self, mouth_zone):
        """Encode mouth zone to numerical value"""
        if isinstance(mouth_zone, str):
            return Config.MOUTH_ZONE_MAPPING.get(mouth_zone, -1)
        return -1
    
    def _encode_gaze_direction(self, gaze_dir):
        """Encode gaze direction to numerical value"""
        if isinstance(gaze_dir, str):
            return Config.GAZE_DIRECTION_MAPPING.get(gaze_dir.lower(), -1)
        return -1
    
    def _encode_gaze_zone(self, gaze_zone):
        """Encode gaze zone to numerical value"""
        if isinstance(gaze_zone, str):
            return Config.GAZE_ZONE_MAPPING.get(gaze_zone.lower(), -1)
        return -1
    
    def _extract_prohibited_items(self, results):
        """Extract prohibited items as one-hot encoded features"""
        prohibited_items = {item: 0 for item in Config.PROHIBITED_ITEMS}
        
        # Check for prohibited items in both face and hand frames
        face_prohibited = results.get('F-Prohibited Item', [])
        hand_prohibited = results.get('H-Prohibited Item', [])
        
        for prohibited_list in [face_prohibited, hand_prohibited]:
            if isinstance(prohibited_list, list):
                for item in prohibited_list:
                    if item in prohibited_items:
                        prohibited_items[item] = 1
        
        # Return values in consistent order
        return [prohibited_items[item] for item in Config.PROHIBITED_ITEMS]
    
    @staticmethod
    def get_feature_names():
        """Return the names of all extracted features"""
        return Config.FEATURE_NAMES
