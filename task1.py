import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

def load_model(model_path):
    # Load the YOLO model using Ultralytics
    model = YOLO(model_path)
    return model

def process_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    return image

def visualize_results(image, results, output_path='output.png'):
    # Get the plotted image with boxes from results
    result_image = results[0].plot()
    
    # Save the output image
    cv2.imwrite(output_path, result_image)
    
    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def main():
    model_path = 'yolo11n (1).pt'
    image_path = 'image-2.png'
    
    # Load model
    print("Loading YOLO model...")
    model = load_model(model_path)
    
    # Run inference
    print("Running inference...")
    results = model.predict(image_path, conf=0.25)  # confidence threshold of 0.25
    
    # Load image for visualization
    image = process_image(image_path)
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(image, results)
    
    # Print detection results
    for r in results:
        print("\nDetections:")
        for box in r.boxes:
            print(f"Class: {box.cls.item()}, Confidence: {box.conf.item():.2f}")
            print(f"Bounding Box: {box.xyxy[0].tolist()}")
    
    print("\nDone! Results have been saved to 'output.png'")

if __name__ == "__main__":
    main()
