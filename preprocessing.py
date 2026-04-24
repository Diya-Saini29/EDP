import cv2
import numpy as np

# Global history dictionary to prevent UI flickering between frames
state_history = {}

def assess_damage(crop, sign_id="current_sign"):
    """
    Analyzes a cropped traffic sign image for physical deterioration.
    Optimized for noisy edge-camera feeds (Jetson Nano CSI).
    """
    global state_history
    
    # Failsafe: If the YOLO crop is empty, return a neutral state
    if crop is None or crop.size == 0:
        return "None", (255, 255, 255)

    # --- 1. TEXTURE ANALYSIS (Rust & Vandalism) ---
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # CRITICAL FOR JETSON: Gaussian Blur removes the "static" noise 
    # common in cheap CSI ribbon cameras so it isn't falsely detected as rust.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Canny Edge detection highlights sharp texture changes
    edges = cv2.Canny(blurred, 50, 120) 
    edge_density = (np.sum(edges > 0) / gray.size) * 100
    
    # --- 2. COLOR ANALYSIS (Sun Fading) ---
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    avg_saturation = hsv[:, :, 1].mean()

    # --- 3. HYSTERESIS DECISION LOGIC ---
    # We use hysteresis (memory of the previous state) so the UI doesn't 
    # flicker wildly if a sign is right on the boundary of "damaged" and "good".
    prev_state = state_history.get(sign_id, "STATUS: GOOD")

    # Entry Thresholds (Hard to trigger to prevent false alarms)
    if edge_density > 8.5:
        current_state = "CRITICAL: DAMAGED"
        color = (0, 0, 255)  # Red
    elif avg_saturation < 52.0:
        current_state = "WARNING: FADED"
        color = (0, 255, 255)  # Yellow
        
    # Exit Thresholds (Must be very clean to revert back to Green)
    elif edge_density < 6.5 and avg_saturation > 60.0:
        current_state = "STATUS: GOOD"
        color = (0, 255, 0)  # Green
        
    # Deadzone (Stay in the previous state if unsure)
    else:
        current_state = prev_state
        if "DAMAGED" in prev_state:
            color = (0, 0, 255)
        elif "FADED" in prev_state:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

    # Save current state for the next video frame
    state_history[sign_id] = current_state
    
    return current_state, color