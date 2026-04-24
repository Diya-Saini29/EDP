# import cv2
# import time
# from ultralytics import YOLO
# from config import MODEL_PATH, INPUT_SOURCE, CONFIDENCE_THRESHOLD
# from logger import init_logger, log_damage
# from inference import run_inference
# from utils import draw_metrics

# def main():
#     print("[INFO] Initializing system...")
#     init_logger()
    
#     print("[INFO] Loading AI Model...")
#     model = YOLO(MODEL_PATH)
    
#     print("[INFO] Starting Video Stream...")
#     cap = cv2.VideoCapture(INPUT_SOURCE)
#     print("\n✅ SYSTEM READY.")
#     print("👉 Press 's' to Save a Report. Press 'q' to Quit.\n")
    
#     prev_frame_time = 0

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break
            
#         # Calculate Frames Per Second (FPS)
#         new_frame_time = time.time()
#         fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
#         prev_frame_time = new_frame_time

#         # Call Inference Pipeline
#         annotated_frame, sign_name, condition, inf_time = run_inference(model, frame, CONFIDENCE_THRESHOLD)
        
#         # Overlay the FPS and Inference metrics on the frame
#         annotated_frame = draw_metrics(annotated_frame, fps, inf_time)

#         # Display Output
#         cv2.imshow("Edge AI Damage Assessment Pipeline", annotated_frame)

#         # Handle Keyboard Inputs
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('s'):
#             if sign_name != "None":
#                 log_damage(sign_name, condition)
#             else:
#                 print("⚠️ No sign detected in this frame to save!")

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
import cv2
import time
from ultralytics import YOLO
from config import MODEL_PATH, INPUT_SOURCE, CONFIDENCE_THRESHOLD
from logger import init_logger, log_damage
from inference import run_inference
from utils import draw_metrics

def main():
    print("[INFO] Initializing system...")
    init_logger()
    
    print("[INFO] Loading AI Model...")
    model = YOLO(MODEL_PATH)
    
    print(f"[INFO] Loading Video: {INPUT_SOURCE}")
    cap = cv2.VideoCapture(INPUT_SOURCE)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file {INPUT_SOURCE}")
        return
    
    # Get video properties
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    out = cv2.VideoWriter('output_detected.mp4', 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps_input, (width, height))
    
    print(f"📹 Input video: {fps_input} FPS, {width}x{height}")
    print("🎥 Processing video... Press 'q' to stop\n")
    
    prev_frame_time = 0
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time

        # Run inference
        annotated_frame, sign_name, condition, inf_time = run_inference(model, frame, CONFIDENCE_THRESHOLD)
        annotated_frame = draw_metrics(annotated_frame, fps, inf_time)
        
        # Write to output video
        out.write(annotated_frame)
        
        # Display (optional - can remove for headless Jetson)
        cv2.imshow("Traffic Sign Damage Assessment", annotated_frame)
        
        # Print progress
        print(f"Frame {frame_count} | FPS: {fps:.1f} | Inf: {inf_time:.1f}ms", end='\r')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n\n✅ Processing complete!")
    print(f"   Total frames processed: {frame_count}")
    print(f"   Output video saved: output_detected.mp4")

if __name__ == "__main__":
    main()