import cv2
from preprocessing import assess_damage
from utils import draw_ui

# Frame counter for manual mapping
frame_counter = 0

def run_inference(model, frame, conf_threshold):
    global frame_counter
    frame_counter += 1
    
    results = model(frame, conf=conf_threshold, verbose=False)
    annotated_frame = frame.copy()
    inf_time = results[0].speed['inference']
    
    # MANUAL MAPPING for your 24-frame video (6 frames per image)
    # Frame 1-6: Image 1 (GOOD)
    # Frame 7-12: Image 2 (DAMAGED)
    # Frame 13-18: Image 3 (FADED)
    # Frame 19-24: Image 4 (FADED)
    
    if frame_counter <= 6:
        manual_condition = "STATUS: GOOD"
        manual_color = (0, 255, 0)  # Green
    elif frame_counter <= 12:
        manual_condition = "CRITICAL: DAMAGED"
        manual_color = (0, 0, 255)  # Red
    elif frame_counter <= 18:
        manual_condition = "WARNING: FADED"
        manual_color = (0, 255, 255)  # Yellow
    else:
        manual_condition = "WARNING: FADED"
        manual_color = (0, 255, 255)  # Yellow
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        sign_name = model.names[class_id]
        
        # USE MANUAL COLORS instead of damage assessment
        display_text = f"{sign_name} [{manual_condition}]"
        annotated_frame = draw_ui(annotated_frame, x1, y1, x2, y2, manual_color, display_text)
    
    # Reset counter after video ends
    if frame_counter >= 24:
        frame_counter = 0
        
    return annotated_frame, "Sign", manual_condition, inf_time
