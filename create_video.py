import cv2
import os

image_folder = r"C:\Users\saini\OneDrive\Desktop\EDP\sign_images"
output_video = r"C:\Users\saini\OneDrive\Desktop\EDP\traffic_video.mp4"

images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png", ".jpeg"))]
images.sort()

print(f"Found {len(images)} images: {images}")

# Target size (use first image as reference)
target_size = None
all_frames = []

for img in images:
    img_path = os.path.join(image_folder, img)
    frame = cv2.imread(img_path)
    
    if frame is None:
        print(f"⚠️ Could not read: {img}")
        continue
    
    h, w = frame.shape[:2]
    
    if target_size is None:
        target_size = (w, h)
        print(f"Target size: {w}x{h}")
    else:
        # Resize if different size
        if (w, h) != target_size:
            frame = cv2.resize(frame, target_size)
            print(f"Resized: {img} from {w}x{h} to {target_size}")
    
    # Add 6 frames per image (3 seconds at 2 fps)
    for _ in range(6):
        all_frames.append(frame)

if len(all_frames) == 0:
    print("❌ No frames to write!")
    exit()

# Create video
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MJPG'), 2, target_size)

for frame in all_frames:
    out.write(frame)

out.release()

# Verify
cap = cv2.VideoCapture(output_video)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = frame_count / fps if fps > 0 else 0
cap.release()

print(f"\n✅ Video saved!")
print(f"   Total images: {len(images)}")
print(f"   Total frames: {frame_count}")
print(f"   FPS: {fps}")
print(f"   Duration: {duration:.1f} seconds")