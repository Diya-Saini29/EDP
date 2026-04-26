import cv2
import numpy as np
import os

image_folder = r"C:\Users\saini\OneDrive\Desktop\EDP\sign_images"

for img_file in os.listdir(image_folder):
    if img_file.endswith(('.jpg', '.png')):
        img = cv2.imread(os.path.join(image_folder, img_file))
        
        # Calculate damage metrics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 120)
        edge_density = (np.sum(edges > 0) / gray.size) * 100
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_saturation = hsv[:, :, 1].mean()
        
        print(f"\n{img_file}:")
        print(f"   Edge Density: {edge_density:.1f}%")
        print(f"   Saturation: {avg_saturation:.1f}")