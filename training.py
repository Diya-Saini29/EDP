from ultralytics import YOLO

def train_model():
    """
    Model Architecture: YOLOv8 (You Only Look Once, version 8)
    Framework: PyTorch via Ultralytics
    Type: Single-stage Object Detector (CNN-based)
    
    Architecture Details:
    - Backbone: Modified CSPDarknet53 (Extracts feature maps)
    - Neck: PANet (Feature pyramid for multi-scale detection)
    - Head: Anchor-free decoupled head (Predicts classes and bounding boxes independently)
    """
    
    print("[INFO] Initializing YOLOv8 Nano Architecture...")
    # Load the base model (pre-trained on COCO for transfer learning)
    model = YOLO("yolov8n.pt") 

    print("[INFO] Starting Training Pipeline...")
    # Train the model on the custom traffic sign dataset
    # Optimization: Using mixed-precision training and auto-batching natively in YOLOv8
    results = model.train(
        data="data.yaml",   # Path to dataset configuration
        epochs=25,          # Number of training loops
        imgsz=640,          # Input image size for the CNN
        batch=16,           # Batch size for memory optimization
        device=0,           # Set to 0 for GPU, 'cpu' for local machine
        name="traffic_sign_detector" 
    )
    
    print("[INFO] Training Complete. Model saved to runs/detect/traffic_sign_detector/weights/best.pt")

if __name__ == "__main__":
    # Note: This script was executed on a Cloud GPU to generate traffic_model_96.pt
    train_model()