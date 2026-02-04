# Vision-Based Electronic Component Identification System

## 🔍 Overview

This project presents a Vision-Based Electronic Component Identification System that automatically detects and classifies electronic components using computer vision and deep learning.

The system is built using a YOLO-based object detection model and is capable of identifying multiple electronic components in real time from images or live camera input. It is designed to reduce manual effort in electronics labs, manufacturing, inspection, and educational environments.


## 🎯 Problem Statement

Identification of electronic components such as resistors, capacitors, ICs, and modules is traditionally done manually. This process is time-consuming, error-prone, and requires prior expertise, especially when dealing with similar-looking or miniaturized components.

With increasing automation and the need for real-time inspection, there is a requirement for an intelligent system that can automatically detect and classify electronic components accurately and efficiently without human intervention.


## 🧠 Solution Approach

The proposed solution uses a deep learning–based object detection approach to identify electronic components from images.

A YOLO (You Only Look Once) model is trained on a custom-labeled dataset of electronic components. The model learns visual features such as shape, size, and layout to accurately classify components. Once trained, the model can detect multiple components in a single image and display bounding boxes along with class labels and confidence scores.

The system supports both image-based and real-time camera-based detection, making it suitable for practical and industrial use cases.

## 🏗️ System Architecture

The system follows a modular pipeline:

1. Image Acquisition: Input images are captured using a camera or loaded from a dataset.
2. Preprocessing: Images are resized and normalized as required by the YOLO model.
3. Model Inference: The trained YOLO model performs object detection on the input image.
4. Post-processing: Non-Maximum Suppression is applied to remove duplicate detections.
5. Output Display: Detected components are shown with bounding boxes, labels, and confidence scores.

The architecture is designed for real-time performance and easy integration with inspection or automation systems.

docs/architecture.png


## 🧪 Model & Training Details

The system is built using a YOLO-based object detection model trained on a custom dataset of electronic components.

- Model: YOLO (Ultralytics)
- Input Image Size: 640 × 640
- Training Type: Supervised learning with labeled bounding boxes
- Classes: Multiple electronic components (resistors, capacitors, ICs, modules, etc.)

Data augmentation techniques such as rotation and brightness variation were applied to improve generalization. The model was trained for multiple epochs and the best-performing weights were selected based on validation performance.


## 📊 Results & Performance

The trained model demonstrates accurate detection and classification of electronic components in real-world images.

- Successfully detects multiple components in a single frame
- Performs well on unseen validation images
- Provides confidence scores for each detected component
- Suitable for real-time inference

Sample detection outputs, confusion matrix, precision–recall curves, and other evaluation results are included in the `docs/results/` directory.

## Video Demo of Project



https://github.com/user-attachments/assets/da9817c1-d6bc-4885-ae37-ffcf1cf6665d



## 🚀 How to Run the Project
### 1. Clone the repository
```bash
git clone https://github.com/your-username/electronic_component_detection.git
cd electronic_component_detection

2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Run detection on an image
python run_predict.py --source path/to/image.jpg

5. Run live detection (camera)
streamlit run app.py


## 🗂️ Project Structure

electronic_component_detection/
│
├── app.py
├── detect_components.py
├── run_predict.py
├── train_model.py
├── data.yaml
├── requirements.txt
├── README.md
│
├── weights/
│   └── best.pt
│
├── sample_outputs/
│
└── docs/
    └── results/


## 🛠️ Tech Stack

- Programming Language: Python
- Deep Learning Framework: YOLO (Ultralytics)
- Computer Vision: OpenCV
- Model Training & Inference: PyTorch
- Web Interface: Streamlit


## 🔮 Future Improvements

- Support for additional electronic components
- Improved accuracy with larger and more diverse datasets
- Real-time video stream optimization
- Integration with robotic or industrial inspection systems
- Automatic datasheet retrieval for detected components


## 📄 Research & Documentation

This project is supported by a detailed project report and a research paper describing the methodology, experiments, and results.


## 👤 Author

Bhaktesh Chandajkar  
B.Tech – Instrumentation & Control Engineering  

