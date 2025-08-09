"""
Debug and validation utilities for video proctor
"""
import csv
import os
import pandas as pd


class DebugValidator:
    """Handles debugging and validation of features and predictions"""
    
    def __init__(self, debug_features=False):
        self.debug_features = debug_features
        self.log_features = debug_features
    
    def validate_frame_pair(self, face_frame, hand_frame):
        """
        Validate that the frame pair is suitable for processing
        
        Args:
            face_frame: Frame from face camera
            hand_frame: Frame from hand/desk camera
            
        Returns:
            bool: True if frames are valid, False otherwise
        """
        try:
            # Check if frames are not None
            if face_frame is None or hand_frame is None:
                if self.debug_features:
                    print("DEBUG: One or both frames are None")
                return False
            
            # Check if frames have proper shape (height, width, channels)
            if len(face_frame.shape) != 3 or len(hand_frame.shape) != 3:
                if self.debug_features:
                    print(f"DEBUG: Invalid frame shapes - Face: {face_frame.shape}, Hand: {hand_frame.shape}")
                return False
            
            # Check minimum dimensions (reasonable for processing)
            min_height, min_width = 100, 100
            if (face_frame.shape[0] < min_height or face_frame.shape[1] < min_width or
                hand_frame.shape[0] < min_height or hand_frame.shape[1] < min_width):
                if self.debug_features:
                    print(f"DEBUG: Frames too small - Face: {face_frame.shape[:2]}, Hand: {hand_frame.shape[:2]}")
                return False
            
            # Frames are valid
            if self.debug_features:
                print(f"DEBUG: Frame validation passed - Face: {face_frame.shape}, Hand: {hand_frame.shape}")
            
            return True
            
        except Exception as e:
            if self.debug_features:
                print(f"DEBUG: Frame validation error: {e}")
            return False
    
    def validate_feature_ranges(self, results_dict):
        """
        Validate feature ranges from a results dictionary
        
        Args:
            results_dict: Dictionary containing analysis results
        """
        if not self.debug_features:
            return
        
        # Extract key features for validation
        key_features = {}
        
        # Get common features that should be validated
        if 'verification_result' in results_dict:
            key_features['verification_result'] = results_dict['verification_result']
        if 'num_faces' in results_dict:
            key_features['num_faces'] = results_dict['num_faces']
        if 'Final Score' in results_dict:
            key_features['Final Score'] = results_dict['Final Score']
        if 'Cheat Score' in results_dict:
            key_features['Cheat Score'] = results_dict['Cheat Score']
        if 'Temporal Score' in results_dict:
            key_features['Temporal Score'] = results_dict['Temporal Score']
        
        # Define expected ranges
        expected_ranges = {
            'verification_result': (0, 1),
            'num_faces': (0, 10),
            'Final Score': (0, 1),
            'Cheat Score': (0, 1),
            'Temporal Score': (0, 1),
            'iris_pos': (-1, 2),
            'mouth_zone': (-1, 3),
            'gaze_direction': (-1, 4),
            'gaze_zone': (-1, 2),
            'H-Distance': (0, 10000),
            'F-Distance': (0, 10000)
        }
        
        warnings = []
        for name, value in key_features.items():
            if name in expected_ranges:
                min_val, max_val = expected_ranges[name]
                if not (min_val <= value <= max_val):
                    warnings.append(f"WARNING: {name} = {value} outside range [{min_val}, {max_val}]")
        
        if warnings:
            print("\n⚠️  FEATURE RANGE WARNINGS:")
            for warning in warnings:
                print(f"   {warning}")
        elif self.debug_features:
            print("✅ All validated features within expected ranges")
    
    def verify_feature_ranges(self, features, feature_names):
        """Verify that features are within expected ranges"""
        expected_ranges = {
            'verification_result': (0, 1),
            'num_faces': (0, 10),
            'iris_pos': (-1, 2),
            'mouth_zone': (-1, 3),
            'gaze_direction': (-1, 4),
            'gaze_zone': (-1, 2),
            'H-Distance': (0, 10000),
            'F-Distance': (0, 10000)
        }
        
        warnings = []
        for i, (name, value) in enumerate(zip(feature_names, features)):
            if name in expected_ranges:
                min_val, max_val = expected_ranges[name]
                if not (min_val <= value <= max_val):
                    warnings.append(f"WARNING: {name} = {value} outside range [{min_val}, {max_val}]")
        
        if warnings:
            print("\n⚠️  FEATURE RANGE WARNINGS:")
            for warning in warnings:
                print(f"   {warning}")
        else:
            print("✅ All features within expected ranges")
    
    def show_expected_features(self, reference_csv_path=None):
        """Show expected features from training data during initialization"""
        if reference_csv_path is None:
            reference_csv_path = 'Datasets/training_proctor_results.csv'
        
        try:
            ref_df = pd.read_csv(reference_csv_path)
            ref_features = ref_df.drop(['timestamp', 'is_cheating'], axis=1, errors='ignore')
            expected_feature_names = ref_features.columns.tolist()
            
            print("\n=== EXPECTED FEATURES FROM TRAINING DATA ===")
            print(f"Total expected features: {len(expected_feature_names)}")
            for i, feat in enumerate(expected_feature_names):
                print(f"  {i:2d}. {feat}")
            
            print(f"\nTraining data shape: {ref_df.shape}")
            print(f"Features will be extracted in this order during processing.")
            
        except Exception as e:
            print(f"Warning: Could not load training data for feature verification: {e}")
    
    def verify_feature_consistency(self, current_features=None, reference_csv_path=None):
        """Verify that current features match the format of training data"""
        if reference_csv_path is None:
            reference_csv_path = 'Datasets/training_proctor_results.csv'
        
        try:
            ref_df = pd.read_csv(reference_csv_path)
            ref_features = ref_df.drop(['timestamp', 'is_cheating'], axis=1, errors='ignore')
            expected_feature_names = ref_features.columns.tolist()
            
            current_feature_names = [
                'verification_result', 'num_faces', 'iris_pos', 'iris_ratio', 
                'mouth_zone', 'mouth_area', 'x_rotation', 'y_rotation', 'z_rotation',
                'radial_distance', 'gaze_direction', 'gaze_zone', 'watch', 'headphone',
                'closedbook', 'earpiece', 'cell phone', 'openbook', 'chits', 'sheet',
                'H-Distance', 'F-Distance'
            ]
            
            print("\n=== FEATURE CONSISTENCY CHECK ===")
            print(f"Expected features (from training): {len(expected_feature_names)}")
            print(f"Current feature names: {len(current_feature_names)}")
            
            if current_features is not None:
                print(f"Current feature values: {len(current_features) - 1}")  # -1 for timestamp
            
            # Check for missing features
            missing = set(expected_feature_names) - set(current_feature_names)
            extra = set(current_feature_names) - set(expected_feature_names)
            
            if missing:
                print(f"❌ Missing features: {missing}")
            if extra:
                print(f"⚠️  Extra features: {extra}")
            if not missing and not extra:
                print("✅ Feature names match perfectly")
            
            # Check feature order
            if current_feature_names == expected_feature_names:
                print("✅ Feature order matches")
            else:
                print("⚠️  Feature order differs:")
                for i, (curr, exp) in enumerate(zip(current_feature_names, expected_feature_names)):
                    if curr != exp:
                        print(f"   Position {i}: got '{curr}', expected '{exp}'")
            
            return missing, extra
            
        except Exception as e:
            print(f"Error in feature consistency check: {e}")
            return None, None
    
    def log_features_to_csv(self, features, feature_names, csv_path):
        """Write features to CSV for logging"""
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=feature_names)

            if not file_exists:
                writer.writeheader()

            row = dict(zip(feature_names, features))
            writer.writerow(row)
    
    def validate_features(self, features, feature_names):
        """Perform comprehensive feature validation"""
        if not self.debug_features:
            return
        
        print(f"\nDEBUG: Extracted {len(features)} features")
        
        expected_feature_count = 23
        if len(features) != expected_feature_count:
            print(f"WARNING: Extracted {len(features)} features, expected {expected_feature_count}")
        
        if self.log_features:
            print("\n=== FEATURE VERIFICATION ===")
            print(f"Expected features: {len(feature_names)}")
            print(f"Extracted features: {len(features)}")
            
            for i, (name, value) in enumerate(zip(feature_names, features)):
                print(f"{i:2d}. {name:15s}: {value}")
            
            self.verify_feature_ranges(features, feature_names)
            self.verify_feature_consistency(features)
            self.log_features_to_csv(features, feature_names, "realtime_features.csv")
