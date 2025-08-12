#!/bin/bash
python video_proctor.py \
  --face Datasets/front_test.avi \
  --hand Datasets/side_test.avi \
  --target Datasets/ID.png \
  --lstm-model temporal_proctor_fixed_params.pt \
  --static-model archive/Models/lightgbm_cheating_model_20250801_200619.pkl \
  --static-scaler archive/Models/scaler_20250801_200619.pkl \
  --static-metadata archive/Models/model_metadata_20250801_200619.pkl \
  --test-duration 5 \
  --yolo-model archive/Models/OEP_YOLOv11n.pt \
  --mediapipe-task archive/Models/face_landmarker.task \