!python /kaggle/working/CheatusDeletus/Proctor/video_proctor.py \
--face /kaggle/input/oep-lab-data/output_front_view.avi\
--hand /kaggle/input/oep-lab-data/output_side_view.avi\
--target /kaggle/input/oep-lab-data/Bhuvanesh_img.jpg\
--output /kaggle/working\
--lstm-model /kaggle/working/CheatusDeletus/Proctor/temporal_proctor_model.pt\
--xgboost-model /kaggle/working/CheatusDeletus/Proctor/best_xgboost_model.pkl\
--xgboost-scaler /kaggle/working/CheatusDeletus/Proctor/scaler.joblib\
--yolo-model /kaggle/working/CheatusDeletus/Proctor/OEP_YOLOv11n.pt\
--mediapipe-task /kaggle/working/CheatusDeletus/Proctor/face_landmarker.task