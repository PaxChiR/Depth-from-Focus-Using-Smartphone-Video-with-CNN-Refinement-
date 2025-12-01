# main.py (updated portions)
import numpy as np
import os
import argparse
import cv2
import numpy as np
import torch
from video_utils import extract_frames, load_frames, stabilize_frames, save_frames
from dff_pipeline import (
    compute_sharpness_stack,
    normalize_stack,
    depth_from_focus,
    depth_to_grayscale,
    all_in_focus,
    save_visualizations,
)
from cnn_model import DfFRefineUNet  # NEW

def refine_with_cnn(norm_sharp_stack, depth_index_map, work_dir, device="cpu"):
    """
    Use trained CNN to refine depth map.
    Inputs:
        norm_sharp_stack: N x H x W (normalized)
        depth_index_map: H x W (int)
    Returns:
        refined_depth_img: H x W uint8 image
        refined_depth_norm: H x W float32 normalized [0,1]
    """
    N, H, W = norm_sharp_stack.shape

    # Build feature maps (same as in training)
    max_sharp = norm_sharp_stack.max(axis=0)
    mean_sharp = norm_sharp_stack.mean(axis=0)
    depth_norm = (depth_index_map.astype(np.float32) - depth_index_map.min()) / \
                 (depth_index_map.max() - depth_index_map.min() + 1e-8)

    feature_maps = np.stack([
        max_sharp,
        mean_sharp,
        depth_norm
    ], axis=0).astype(np.float32)  # C x H x W

    # Load model
    model_path = os.path.join(work_dir, "dff_cnn_refine.pth")
    if not os.path.exists(model_path):
        print("[CNN] No trained model found at", model_path)
        return None, None

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = DfFRefineUNet(in_channels=feature_maps.shape[0], base_ch=32)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        feat_tensor = torch.from_numpy(feature_maps[None, ...]).float().to(device)
        pred = model(feat_tensor)  # 1 x 1 x H x W
        pred_depth = pred.squeeze(0).squeeze(0).cpu().numpy()  # H x W, [0,1]

    # Convert to 0–255 depth image
    pred_img = (pred_depth * 255.0).clip(0, 255).astype(np.uint8)
    return pred_img, pred_depth

def main():
    parser = argparse.ArgumentParser(description="Depth from Focus using smartphone video")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video, e.g., IMG_6115.MOV")
    parser.add_argument("--work_dir", type=str, default="results",
                        help="Directory to save intermediate and final outputs.")
    parser.add_argument("--max_frames", type=int, default=50,
                        help="Max number of frames to process (for speed).")
    parser.add_argument("--smooth_sigma", type=float, default=1.0,
                        help="Spatial smoothing sigma for sharpness stack.")
    parser.add_argument("--use_cnn_refine", action="store_true",
                        help="Use trained CNN to refine depth map.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for CNN inference: cuda or cpu")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    frames_dir = os.path.join(args.work_dir, "frames")
    stab_dir = os.path.join(args.work_dir, "stabilized_frames")
    sharp_dir = os.path.join(args.work_dir, "sharpness_maps")
    depth_dir = os.path.join(args.work_dir, "depth_maps")
    aif_dir = os.path.join(args.work_dir, "all_in_focus")
    os.makedirs(sharp_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(aif_dir, exist_ok=True)

    # 1. Extract frames
    print("[1/6] Extracting frames...")
    frame_paths = extract_frames(args.video_path, frames_dir, max_frames=args.max_frames)
    frames = load_frames(frame_paths)

    # 2. Stabilize frames
    print("[2/6] Stabilizing frames...")
    stab_frames = stabilize_frames(frames)
    save_frames(stab_frames, stab_dir, prefix="stab")

    # 3. Compute sharpness stack
    print("[3/6] Computing sharpness stack...")
    sharp_stack = compute_sharpness_stack(stab_frames)   # N x H x W (float32)
    norm_sharp_stack = normalize_stack(sharp_stack)

    # Save for CNN training if needed
    np.save(os.path.join(args.work_dir, "sharpness_stack.npy"), norm_sharp_stack)

    for i in range(min(5, norm_sharp_stack.shape[0])):
        s = norm_sharp_stack[i]
        s_img = (s * 255).astype("uint8")
        cv2.imwrite(os.path.join(sharp_dir, f"sharp_{i:04d}.png"), s_img)

    # 4. Classical depth from focus
    print("[4/6] Estimating depth (classical)...")
    depth_index_map, confidence_map = depth_from_focus(norm_sharp_stack,
                                                       smooth_sigma=args.smooth_sigma)
    depth_img = depth_to_grayscale(depth_index_map)
    depth_path, conf_path = save_visualizations(depth_img, confidence_map, depth_dir)

    # Save depth index map for CNN training
    np.save(os.path.join(args.work_dir, "depth_index_map.npy"), depth_index_map)

    # 5. All-in-focus image (classical)
    print("[5/6] Generating all-in-focus image...")
    aif = all_in_focus(stab_frames, depth_index_map)
    aif_path = os.path.join(aif_dir, "all_in_focus.png")
    cv2.imwrite(aif_path, aif)

    print("[INFO] Classical depth map:", depth_path)
    print("[INFO] All-in-focus:", aif_path)

    # 6. CNN refinement (optional)
    if args.use_cnn_refine:
        print("[6/6] Refining depth with CNN...")
        refined_img, refined_norm = refine_with_cnn(
            norm_sharp_stack, depth_index_map, args.work_dir, device=args.device
        )
        if refined_img is not None:
            refined_path = os.path.join(depth_dir, "depth_map_cnn_refined.png")
            cv2.imwrite(refined_path, refined_img)
            print("[CNN] Refined depth map:", refined_path)
        else:
            print("[CNN] Skipping refinement (no model found).")
    else:
        print("[6/6] CNN refinement disabled.")

if __name__ == "__main__":
    main()
