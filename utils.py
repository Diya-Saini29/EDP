import cv2

def draw_ui(frame, x1, y1, x2, y2, color, label):
    """
    Draws a professional, high-contrast bounding box and label around detected signs.
    Designed to be readable in varying lighting conditions from a moving vehicle.
    """
    # 1. Draw the main Bounding Box (thickness 3)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # 2. Draw a solid background banner for the text so it doesn't blend into the background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_size[0], y1), color, -1)

    # 3. Draw the Label Text
    # Auto-switch text to white if the box is red (Damaged), otherwise black for high contrast
    text_color = (255, 255, 255) if color == (0, 0, 255) else (0, 0, 0)
    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    return frame


def draw_metrics(frame, fps, inf_time):
    """
    Draws a clean, semi-transparent dark HUD header at the top of the screen
    to display real-time Edge AI performance metrics.
    """
    # 1. Create a dark overlay bar at the top (50px tall)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    
    # 2. Blend the overlay with the original frame (60% opacity)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 3. Add the yellow metrics text
    metrics_text = f"SYSTEM STATUS: ACTIVE  |  FPS: {int(fps)}  |  LATENCY: {inf_time:.1f}ms"
    cv2.putText(frame, metrics_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame