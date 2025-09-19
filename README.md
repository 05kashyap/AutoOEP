# AutoOEP
## Automated Online Exam Proctoring System
AutoOEP is an advanced online exam proctoring system designed to ensure the integrity and security of online examinations. It leverages cutting-edge technologies such as facial recognition, behavior analysis, and real-time monitoring to prevent cheating and maintain a fair testing environment. It utilizes a combination of computer vision, machine learning, and deep learning techniques to monitor test-takers effectively. Using two webcams, it captures both frontal and side views of the test-taker, allowing for comprehensive monitoring of their actions and surroundings.

![Exam Proctor flow](Assets/OEP_Final_Flow.png) 

## Libraries and Frameworks Used
- **OpenCV**: For real-time video processing 
- **DeepFace**: For facial recognition and verification
- **Mediapipe**: For pose estimation and behavior analysis
- **YOLOv11**: For object detection
- **LSTM**: For monitoring the sequence of actions and detecting suspicious behavior
- **LightGBM**: For providing static support in behavior analysis and benchmarking
- **PyTorch**: For building and training deep learning models

## Features
### Face Module
![Face Module](Assets/OEP_Front_flow.jpeg)
- **Face Detection**: Utilizes Mediapipe to detect and count number of faces in the frame.
- **Face Recognition**: Uses DeepFace to verify the identity of the test-taker against a pre-registered image.
- **Pose Estimation**: Employs Mediapipe to monitor head movements, gaze, and track eyes and mouth to detect if the test-taker is looking away or talking.

### Hand Module
![Hand Module](Assets/OEP_Side_flow.jpeg)
- **Hand Tracking**: Uses Mediapipe to track hand movements and positions.
- **Object Detection**: Implements YOLOv11 to identify unauthorized objects (e.g., phones, notes) in the test-taker's vicinity.
- **Behavior Analysis**: Analyzes hand movements to detect suspicious behaviors such as reaching out for unauthorized materials.

### Behavior Module
- **Action Sequence Monitoring**: Utilizes LSTM to analyze the sequence of actions and detect patterns indicative of cheating.
- **Static Behavior Analysis**: Uses LightGBM to provide additional support in identifying unusual behaviors.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/05kashyap/AutoOEP.git
   cd AutoOEP
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Extract the pretrained models from the `final_models.zip` file and place them in the `Models` directory.
4. Make sure to have two webcams connected to your system, one for frontal view and another for side view.
5. Make sure to have front and side videos of the test-taker for testing.
6. Upload the registered image of the test taker in same folder.
7. Correct the paths in `run.bat` or `run.sh` file as per your setup.
8. Run the `run.bat` or `run.sh` file as per your OS.
