"""
optimize.py - Convert PyTorch model to TensorRT FP16 engine for Jetson Nano
"""
from ultralytics import YOLO
import time
import os

print("="*50)
print("TENSORRT OPTIMIZATION FOR JETSON NANO")
print("="*50)

# Check if original model exists
if not os.path.exists("traffic_model_96.pt"):
    print("❌ Error: traffic_model_96.pt not found!")
    print("   Please ensure your trained model is in this directory.")
    exit(1)

# Load your original model
print("\n📦 Loading original model: traffic_model_96.pt")
model = YOLO("traffic_model_96.pt")

# Get original file size
original_size = os.path.getsize("traffic_model_96.pt") / (1024 * 1024)
print(f"   Original model size: {original_size:.1f} MB")

print("\n🚀 Starting TensorRT Optimization...")
print("   Converting FP32 → FP16 for Jetson Nano GPU")
print("   This takes 10-15 minutes on Jetson Nano\n")
start_time = time.time()

# Export to TensorRT format for Jetson hardware
model.export(
    format="engine",   # TensorRT engine format
    half=True,         # FP16 precision (key optimization)
    workspace=2,       # 2GB workspace for optimization
    imgsz=640,         # Match training input size
    device=0           # Use GPU
)

conversion_time = time.time() - start_time

# Check if export succeeded
if os.path.exists("traffic_model_96.engine"):
    engine_size = os.path.getsize("traffic_model_96.engine") / (1024 * 1024)
    print("\n" + "="*50)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*50)
    print(f"   Time taken: {conversion_time/60:.1f} minutes")
    print(f"   Original size: {original_size:.1f} MB")
    print(f"   Optimized size: {engine_size:.1f} MB")
    print(f"   Size reduction: {(1 - engine_size/original_size)*100:.1f}%")
    print("\n   Saved as: traffic_model_96.engine")
    print("\n🎯 Ready for deployment on Jetson Nano!")
else:
    print("\n❌ Optimization failed! Check TensorRT installation.")