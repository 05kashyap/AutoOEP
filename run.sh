#!/bin/bash
python video_proctor.py \
  --face Datasets/Videos/output_front_view.avi \
  --hand Datasets/Videos/output_side_view.avi \
  --target Datasets/Videos/Bhuvanesh_img.jpg \
  --output Outputs \
  --lstm-model Models/temporal_proctor_model.pt \
  --xgboost-model Models/xgboost_cheating_model_20250625_143239.pkl \
  --xgboost-scaler Models/scaler_20250625_143239.pkl \
  --test-duration 5 \
  --yolo-model Models/OEP_YOLOv11n.pt \
  --mediapipe-task Models/face_landmarker.task

