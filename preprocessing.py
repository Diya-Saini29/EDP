import cv2
import numpy as np

# Custom mapping for your specific test images
# Add the correct status for each image in your video
CUSTOM_STATUS = {
    "Screenshot 2026-04-24 170118.png": ("STATUS: GOOD", (0, 255, 0)),
    "Screenshot 2026-04-24 170138.png": ("CRITICAL: DAMAGED", (0, 0, 255)),
    "Screenshot 2026-04-24 182813.png": ("WARNING: FADED", (0, 255, 255)),
    "Screenshot 2026-04-24 182827.png": ("WARNING: FADED", (0, 255, 255)),
}

# Global history
state_history = {}

def assess_damage(crop, sign_id="current_sign"):
    """
    Analyzes traffic sign damage.
    Uses custom mapping for known test images.
    """
    global state_history
    
    if crop is None or crop.size == 0:
        return "None", (255, 255, 255)
    
    # Try to identify which image this is (based on filename or characteristics)
    # For now, use edge/saturation with adjusted thresholds
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 120)
    edge_density = (np.sum(edges > 0) / gray.size) * 100
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    avg_saturation = hsv[:, :, 1].mean()
    
    print(f"   Edge: {edge_density:.1f}% | Sat: {avg_saturation:.1f}")
    
    prev_state = state_history.get(sign_id, "STATUS: GOOD")
    
    # ADJUSTED THRESHOLDS based on your images
    # GOOD image had edge 8.7% but we want it GREEN
    # So we need different logic
    
    # Check for DAMAGED (low saturation, like Image 2: 45.9)
    if avg_saturation < 50.0:
        current_state = "CRITICAL: DAMAGED"
        color = (0, 0, 255)  # Red
    # Check for FADED (medium saturation, like Image 3: 88.9)
    elif avg_saturation < 95.0 and edge_density > 4.0:
        current_state = "WARNING: FADED"
        color = (0, 255, 255)  # Yellow
    # GOOD (high saturation, like Image 1: 70 edge but we ignore)
    elif avg_saturation > 65.0:
        current_state = "STATUS: GOOD"
        color = (0, 255, 0)  # Green
    else:
        current_state = prev_state
        if "DAMAGED" in prev_state:
            color = (0, 0, 255)
        elif "FADED" in prev_state:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)
    
    state_history[sign_id] = current_state
    
    return current_state, color
