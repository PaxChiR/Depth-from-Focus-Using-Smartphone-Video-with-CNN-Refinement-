# video_utils.py
import cv2
import os
import numpy as np
from typing import List, Tuple

def extract_frames(video_path: str, out_dir: str, max_frames: int = None) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_paths = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and idx >= max_frames:
            break
        frame_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        idx += 1

    cap.release()
    return frame_paths

def load_frames(frame_paths: List[str]) -> List[np.ndarray]:
    frames = []
    for p in frame_paths:
        img = cv2.imread(p)
        if img is None:
            raise RuntimeError(f"Failed to read frame {p}")
        frames.append(img)
    return frames

def stabilize_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Simple global motion stabilization using feature tracking (ECC or keypoints).
    Here we use ECC-based alignment to the first frame (reference).
    """
    if len(frames) == 0:
        return []

    ref = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    h, w = ref.shape
    warp_mode = cv2.MOTION_AFFINE
    number_of_iterations = 100
    termination_eps = 1e-5

    stabilized = [frames[0]]
    for i in range(1, len(frames)):
        cur = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    number_of_iterations, termination_eps)
        try:
            (cc, warp_matrix) = cv2.findTransformECC(ref, cur, warp_matrix, warp_mode, criteria)
            aligned = cv2.warpAffine(frames[i], warp_matrix, (w, h),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except cv2.error:
            # If ECC fails, just keep original frame
            aligned = frames[i]
        stabilized.append(aligned)

    return stabilized

def save_frames(frames: List[np.ndarray], out_dir: str, prefix: str = "stab") -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, f in enumerate(frames):
        path = os.path.join(out_dir, f"{prefix}_{i:04d}.png")
        cv2.imwrite(path, f)
        paths.append(path)
    return paths
