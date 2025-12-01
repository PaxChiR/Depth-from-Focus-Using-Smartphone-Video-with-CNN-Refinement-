# dff_pipeline.py

import os
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from focus_measures import sharpness_measure


def compute_sharpness_stack(frames: List[np.ndarray]) -> np.ndarray:
    """
    Compute sharpness map for each frame using sharpness_measure().
    Output:
        sharp_stack: (N, H, W) float32
    """
    sharp_list = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sharp = sharpness_measure(gray)
        sharp_list.append(sharp)
    sharp_stack = np.stack(sharp_list, axis=0).astype(np.float32)
    return sharp_stack


def normalize_stack(stack: np.ndarray) -> np.ndarray:
    """
    Normalize stack per-pixel across frames to [0,1].
    """
    min_v = stack.min(axis=0, keepdims=True)
    max_v = stack.max(axis=0, keepdims=True)
    denom = max_v - min_v
    denom[denom == 0] = 1.0
    norm = (stack - min_v) / denom
    return norm.astype(np.float32)


def fit_parabola_and_get_peak(sharp_curve: np.ndarray) -> float:
    """
    Given a 1D sharpness curve over frames (length N),
    fit a parabola around the maximum and return a sub-frame
    peak location (float). If fitting fails, return the argmax index.
    """
    sharp_curve = np.asarray(sharp_curve, dtype=np.float32)
    i = int(np.argmax(sharp_curve))

    # can't fit a 3-point parabola at the ends
    if i == 0 or i == len(sharp_curve) - 1:
        return float(i)

    y0, y1, y2 = sharp_curve[i - 1], sharp_curve[i], sharp_curve[i + 1]

    denom = 2.0 * (y0 - 2.0 * y1 + y2)
    if denom == 0:
        return float(i)

    # vertex offset relative to i
    offset = (y0 - y2) / denom
    peak = i + offset

    # clamp to valid range
    if peak < 0:
        peak = 0.0
    if peak > len(sharp_curve) - 1:
        peak = float(len(sharp_curve) - 1)

    return float(peak)


def depth_from_focus(sharp_stack: np.ndarray,
                     smooth_sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Depth-from-focus with per-pixel parabola fitting over the sharpness curve.
    Inputs:
        sharp_stack: (N, H, W) sharpness values for each frame
    Outputs:
        depth_index_map: (H, W) float32 sub-frame indices of best focus
        confidence_map: (H, W) float32 maximum sharpness per pixel
    """
    # optional spatial smoothing of sharpness maps
    if smooth_sigma > 0:
        sharp_stack = gaussian_filter(sharp_stack,
                                      sigma=(0, smooth_sigma, smooth_sigma))

    N, H, W = sharp_stack.shape
    depth_index_map = np.zeros((H, W), dtype=np.float32)
    confidence_map = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            curve = sharp_stack[:, y, x]
            peak_idx = fit_parabola_and_get_peak(curve)
            depth_index_map[y, x] = peak_idx
            confidence_map[y, x] = float(curve.max())

    return depth_index_map, confidence_map


def depth_to_grayscale(depth_index_map: np.ndarray) -> np.ndarray:
    """
    Convert depth indices (float or int) to 0–255 grayscale image.
    """
    d = depth_index_map.astype(np.float32)
    d_min = d.min()
    d_max = d.max()
    if d_max == d_min:
        return np.zeros_like(d, dtype=np.uint8)
    d_norm = (d - d_min) / (d_max - d_min)
    depth_img = (d_norm * 255.0).astype(np.uint8)
    return depth_img


def all_in_focus(frames: List[np.ndarray],
                 depth_index_map: np.ndarray) -> np.ndarray:
    """
    Construct all-in-focus image by selecting, for each pixel, the frame
    where it is sharpest (according to depth_index_map).
    """
    H, W = depth_index_map.shape
    N = len(frames)
    out = np.zeros_like(frames[0])

    for y in range(H):
        for x in range(W):
            idx = int(round(depth_index_map[y, x]))
            idx = max(0, min(N - 1, idx))
            out[y, x, :] = frames[idx][y, x, :]

    return out


def save_visualizations(depth_img: np.ndarray,
                        confidence_map: np.ndarray,
                        out_dir: str):
    """
    Save depth map and confidence map as PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)

    depth_path = os.path.join(out_dir, "depth_map.png")

    conf = confidence_map.copy()
    conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
    conf_img = (conf * 255.0).astype(np.uint8)
    conf_path = os.path.join(out_dir, "confidence_map.png")

    cv2.imwrite(depth_path, depth_img)
    cv2.imwrite(conf_path, conf_img)

    return depth_path, conf_path
