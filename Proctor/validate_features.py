import pandas as pd
import numpy as np
from video_proctor import VideoProctor
import torch
from ultralytics import YOLO
import mediapipe as mp
import cv2

def validate_feature_pipeline():
    """
    Comprehensive feature validation
    """
    print("="*60)
    print("FEATURE PIPELINE VALIDATION")
    print("="*60)
    
    # 1. Load training data and check expected features
    training_csv = 'Proctor/Datasets/training_proctor_results.csv'
    df = pd.read_csv(training_csv)
    
    expected_features = df.drop(['timestamp', 'is_cheating'], axis=1).columns.tolist()
    print(f"Expected features from training data ({len(expected_features)}):")
    for i, feat in enumerate(expected_features):
        print(f"  {i:2d}. {feat}")
    
    # 2. Initialize VideoProctor with debug mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_model = YOLO('Proctor/Models/OEP_YOLOv11n.pt')
    
    mpHands = mp.solutions.hands
    media_pipe_dict = {
        'mpHands': mpHands,
        'hands': mpHands.Hands(static_image_mode=False,
                             max_num_hands=2,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5),
        'mpdraw': mp.solutions.drawing_utils
    }
    
    proctor = VideoProctor(
        lstm_model_path='Proctor/Models/temporal_proctor_model.pt',
        xgboost_model_path='Proctor/Models/xgboost_cheating_model_20250625_143239.pkl',
        xgboost_scaler_path='Proctor/Models/scaler_20250625_143239.pkl',
        yolo_model_path='Proctor/Models/OEP_YOLOv11n.pt',
        mediapipe_model_path='Proctor/Models/face_landmarker.task',
        window_size=15,
        input_size=23,
        debug_features=True  # Enable debug mode
    )
    
    # 3. Process a test frame and check features
    target_frame = cv2.imread('Proctor/Images/identity.jpeg')
    face_frame = cv2.imread('Proctor/Images/facecam1.png')
    hand_frame = cv2.imread('Proctor/Images/test.jpg')
    
    if all(frame is not None for frame in [target_frame, face_frame, hand_frame]):
        print("\n" + "="*40)
        print("PROCESSING TEST FRAMES")
        print("="*40)
        
        # Process multiple frames to fill the buffer for LSTM prediction
        for i in range(20):  # Process 20 frames to ensure we have enough for LSTM
            result = proctor.process_frame_pair(target_frame, face_frame, hand_frame)
            if i == 0:  # Only show details for the first frame
                # Extract features manually for verification
                static_results = result['static_results']
                features = proctor.extract_features_from_results(static_results)
                
                print(f"\nExtracted {len(features)} features")
                print(f"Expected {len(expected_features) + 1} features (including timestamp)")
        
        # 4. Check if XGBoost model can handle the features
        
        if proctor.xgboost_model is not None:
            try:
                xgb_features = np.array(features[1:]).reshape(1, -1)  # Skip timestamp
                xgb_features_scaled = proctor.xgboost_scaler.transform(xgb_features)
                prediction = proctor.xgboost_model.predict_proba(xgb_features_scaled)[0][1]
                print(f"✅ XGBoost prediction successful: {prediction:.3f}")
            except Exception as e:
                print(f"❌ XGBoost prediction failed: {e}")
        else:
            print("⚠️  XGBoost model not loaded")
        
        # 5. Check if LSTM model can handle the features
        try:
            # Use temporal_proctor's window_size instead of proctor.window_size
            if len(proctor.feature_buffer) >= proctor.temporal_proctor.window_size:
                lstm_prediction = proctor.temporal_proctor.make_realtime_prediction(list(proctor.feature_buffer))
                print(f"✅ LSTM prediction successful: {lstm_prediction:.3f}")
            else:
                print(f"⚠️  Not enough frames for LSTM prediction yet. Have {len(proctor.feature_buffer)}, need {proctor.temporal_proctor.window_size}")
        except Exception as e:
            print(f"❌ LSTM prediction failed: {e}")
    
    else:
        print("❌ Could not load test images")
        missing_files = []
        if target_frame is None:
            missing_files.append("Proctor/Images/identity.jpeg")
        if face_frame is None:
            missing_files.append("Proctor/Images/facecam1.png")
        if hand_frame is None:
            missing_files.append("Proctor/Images/test.jpg")
        print(f"Missing files: {missing_files}")

def compare_with_csv_sample():
    """
    Compare extracted features with a sample from CSV
    """
    print("\n" + "="*40)
    print("CSV SAMPLE COMPARISON")
    print("="*40)
    
    try:
        # Load a sample from training data
        df = pd.read_csv('Proctor/Datasets/training_proctor_results.csv')
        sample = df.iloc[0]
        
        print("Sample from training CSV:")
        for col in df.columns:
            if col not in ['timestamp', 'is_cheating']:
                print(f"  {col}: {sample[col]}")
        
        print("\nFeature value ranges in training data:")
        feature_cols = df.drop(['timestamp', 'is_cheating'], axis=1).columns
        for col in feature_cols:
            print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")
    except FileNotFoundError:
        print("❌ Training CSV file not found: Proctor/Datasets/training_proctor_results.csv")
    except Exception as e:
        print(f"❌ Error reading training data: {e}")

if __name__ == "__main__":
    validate_feature_pipeline()
    compare_with_csv_sample()