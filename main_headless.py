# main_headless.py - Saves video file instead of showing window
import cv2
import time
from ultralytics import YOLO
from config import MODEL_PATH, INPUT_SOURCE, CONFIDENCE_THRESHOLD
from inference import run_inference
from utils import draw_metrics

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(INPUT_SOURCE)
    
    # Video writer
    out = cv2.VideoWriter('jetson_demo.mp4', 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          20, (1280, 720))
    
    print("Recording jetson_demo.mp4 - Press 'q' to stop")
    
    prev_frame_time = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        
        annotated_frame, _, _, inf_time = run_inference(model, frame, CONFIDENCE_THRESHOLD)
        annotated_frame = draw_metrics(annotated_frame, fps, inf_time)
        
        out.write(annotated_frame)  # Save to file
        
        print(f"Recording... FPS: {fps:.1f}", end='\r')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    print("\n✅ Saved: jetson_demo.mp4")

if __name__ == "__main__":
    main()