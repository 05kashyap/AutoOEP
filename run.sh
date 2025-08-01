#!/bin/bash
python video_proctor.py \
  --face Dataset_Parser/Media/NoCheat_Videos/Test/front_test.avi \
  --hand Dataset_Parser/Media/NoCheat_Videos/Test/side_test.avi \
  --target Dataset_Parser/OEPFrame_Dataset/ID.png \
  --output Outputs \
  --lstm-model Models/temporal_proctor_trained_on_processed.pt \
  --static-model Models/lightgbm_cheating_model_20250801_200619.pkl \
  --static-scaler Models/scaler_20250801_200619.pkl \
  --static-metadata Models/model_metadata_20250801_200619.pkl \
  --test-duration 5 \
  --yolo-model Models/OEP_YOLOv11n.pt \
  --mediapipe-task Models/face_landmarker.task \
  --display \

