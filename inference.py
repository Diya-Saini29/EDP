import cv2
from preprocessing import assess_damage
from utils import draw_ui

def run_inference(model, frame, conf_threshold):
    """
    Executes YOLO object detection on a single frame and passes crops 
    to the damage assessment pipeline.
    """
    # Run YOLO inference (verbose=False keeps the Jetson terminal clean)
    results = model(frame, conf=conf_threshold, verbose=False)
    annotated_frame = frame.copy()
    
    # Extract the model's internal inference latency (ms)
    inf_time = results[0].speed['inference']
    
    # Default states if no sign is detected
    sign_name = "None"
    condition = "None"

    # Loop through all detected objects in the frame
    for box in results[0].boxes:
        # 1. Get Bounding Box Coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 2. Identify the Sign Class
        class_id = int(box.cls[0])
        sign_name = model.names[class_id]
        
        # 3. Crop and Assess Damage
        # Create a unique ID so the stabilizer tracks different signs independently
        sign_id = f"sign_{class_id}" 
        cropped_sign = frame[y1:y2, x1:x2]
        condition, color = assess_damage(cropped_sign, sign_id=sign_id)
        
        # 4. Apply the Professional HUD Box
        display_text = f"{sign_name} [{condition}]"
        annotated_frame = draw_ui(annotated_frame, x1, y1, x2, y2, color, display_text)
        
    return annotated_frame, sign_name, condition, inf_time