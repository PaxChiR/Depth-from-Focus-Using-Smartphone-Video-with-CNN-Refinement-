# focus_measures.py
import cv2
import numpy as np

""" def laplacian_variance(gray: np.ndarray) -> np.ndarray:
    
    Compute per-pixel sharpness using the squared Laplacian (LoG energy).
    Input:
        gray: HxW single-channel uint8 or float32 image
    Output:
        sharpness: HxW float32 map
    
    if gray.dtype != np.float32:
        gray_f = gray.astype(np.float32) / 255.0
    else:
        gray_f = gray

    lap = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
    sharp = lap ** 2
    return sharp
 """
def sharpness_measure(gray):
    gray = gray.astype(np.float32) / 255.0
    
    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
    lap_score = lap**2

    # Tenengrad (Sobel magnitude)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    tenengrad_score = gx*gx + gy*gy

    # Combine both
    return lap_score * 0.5 + tenengrad_score * 0.5
