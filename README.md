# YOLO Object Detection with PyTorch and ONNX Conversion

This project demonstrates object detection using a YOLO model with both PyTorch and ONNX formats. It includes two main tasks:

1. Running inference using the original PyTorch YOLO model
2. Converting the model to ONNX format and running inference with the converted model

## Project Structure
```
├── task1.py          # PyTorch YOLO inference
├── task2.py          # ONNX conversion and inference
├── requirements.txt  # Python dependencies
├── yolo11n (1).pt   # Original YOLO model
└── image-2.png      # Test image
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Task 1: PyTorch YOLO Inference
```bash
python task1.py
```
This will run inference on the test image using the PyTorch model and save the results as `output.png`.

### Task 2: ONNX Conversion and Inference
```bash
python task2.py
```
This will:
1. Convert the PyTorch model to ONNX format
2. Run inference using the ONNX model
3. Save the results as `output_onnx.png`

## Requirements
- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenCV
- ONNX Runtime
