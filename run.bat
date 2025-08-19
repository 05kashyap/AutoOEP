@echo off
python video_proctor.py ^
    --face Datasets\nocheat\front_test.avi ^
    --hand Datasets\nocheat\side_test.avi ^
    --target Datasets\ID.png ^
    --lstm-model best_models\temporal_proctor_trained_on_processed.pt ^
    --static-model best_models\lightgbm_cheating_model_20250818_132555.pkl ^
    --static-scaler best_models\scaler_20250818_132555.pkl ^
    --static-metadata best_models\model_metadata_20250818_132555.pkl ^
    --mediapipe-task best_models\face_landmarker.task ^
    --yolo-model best_models\OEP_YOLOv11n.pt ^
    --process-fps 1 ^
    --debug