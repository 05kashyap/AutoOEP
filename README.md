# CheatusDeletus

## Overview

CheatusDeletus is a video proctoring pipeline for detecting cheating behavior in exam settings using computer vision and machine learning. The system processes video data, extracts features, trains static and temporal models, and provides real-time cheat score predictions.

---

## Pipeline Steps

### 1. **Extract Frames from Videos**

Use `video_parse.py` to extract frames from your raw video files and sort them into `cheating_frames` and `not_cheating_frames` folders.

**Usage:**
```bash
python Dataset_Parser/video_parse.py
```
- Follow the prompts to specify cheating intervals or classify all frames.
- Output: `cheating_frames` and `not_cheating_frames` directories for each video.

---

### 2. **Extract Features from Frames**

Use `save_features.py` to process the sorted frames and extract features for each frame pair (face and hand). This generates raw CSV files for each video.

**Usage:**
```bash
python Proctor/save_features.py
```
- Configure `dataset_path` and `target_frame_path` as needed.
- Output: CSV files in `Proctor/results/` (one per video and a combined file).

---

### 3. **Process Raw CSVs for Training/Testing**

Use `Static/process_csv.py` to clean and standardize the raw CSVs, ensuring consistent feature order and handling missing values.

**Usage:**
```bash
python Static/process_csv.py
```
- Input: CSVs from `Proctor/results/`
- Output: Cleaned CSVs in `processed_results/` (ready for model training)

---

### 4. **Train Static Model**

Use `Static/static_trainer.py` to train static models (LightGBM, XGBoost, RandomForest) on the processed data. The best model and scaler are saved for later use.

**Usage:**
```bash
python Static/static_trainer.py
```
- Input: Processed CSVs from `processed_results/`
- Output: Best static model (`Models/*_cheating_model_*.pkl`), scaler, and metadata.

---

### 5. **Train Temporal Model**

Use `Temporal/temporal_trainer.py` to train an LSTM/GRU-based temporal model on the processed data.

**Usage:**
```bash
python Temporal/temporal_trainer.py
```
- Input: Processed CSVs from `processed_results/`
- Output: Trained temporal model (`Models/temporal_proctor_trained_on_processed.pt`)

---

### 6. **Run Real-Time Cheating Detection**

Use `video_proctor.py` to process test videos and get real-time cheat score predictions using both static and temporal models.

**Usage:**
```bash
bash run.sh
```
or
```bash
python video_proctor.py \
  --face <face_video_path> \
  --hand <hand_video_path> \
  --target <target_image_path> \
  --output <output_dir> \
  --lstm-model <temporal_model_path> \
  --static-model <static_model_path> \
  --static-scaler <scaler_path> \
  --static-metadata <metadata_path> \
  --yolo-model <yolo_model_path> \
  --mediapipe-task <mediapipe_task_path> \
  --display
```
- Output: Real-time predictions and annotated video (if `--output` is specified).

---

## Notes

- Ensure feature order and column consistency between all steps.
- Always use processed CSVs from `processed_results/` for training/testing.
- For best results, verify feature extraction and model input/output shapes using debug mode in `video_proctor.py`.

---

## Directory Structure

```
CheatusDeletus/
├── Dataset_Parser/
│   └── video_parse.py
├── Proctor/
│   └── save_features.py
├── Static/
│   ├── process_csv.py
│   └── static_trainer.py
├── Temporal/
│   └── temporal_trainer.py
├── processed_results/
├── Models/
├── Outputs/
├── run.sh
└── video_proctor.py
```

