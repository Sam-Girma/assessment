from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def main():
    model_path = 'yolo11n (1).pt'
    image_path = 'image-2.png'
    
    print("Loading YOLO model...")
    # Load the YOLO model
    model = YOLO(model_path)
    
    print("Converting model to ONNX format...")
    # Export the model to ONNX format
    model.export(format="onnx")  # creates 'yolo11n.onnx'
    
    print("Loading ONNX model and running inference...")
    # Load the exported ONNX model
    onnx_model = YOLO("yolo11n (1).onnx")
    
    # Run inference with ONNX model
    results = onnx_model(image_path)
    
    # Plot the results
    print("Visualizing results...")
    res_plotted = results[0].plot()
    
    # Save the output image
    cv2.imwrite('output_onnx.png', res_plotted)
    
    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    # Print detection results
    print("\nDetections:")
    for r in results:
        for box in r.boxes:
            print(f"Class: {box.cls.item()}, Confidence: {box.conf.item():.2f}")
            print(f"Bounding Box: {box.xyxy[0].tolist()}")
    
    print("\nDone! Results have been saved to 'output_onnx.png'")

if __name__ == "__main__":
    main()
