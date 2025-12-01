# train_cnn_refine.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import argparse

from cnn_model import DfFRefineUNet

class SharpnessDepthDataset(Dataset):
    def __init__(self, feature_maps, target_depth, patch_size=128, num_patches=2000):
        """
        feature_maps: C x H x W numpy array
        target_depth: 1 x H x W numpy array (normalized [0,1])
        """
        self.feature_maps = feature_maps
        self.target_depth = target_depth
        self.C, self.H, self.W = feature_maps.shape
        self.patch_size = patch_size

        # Pre-sample patch top-left coordinates
        self.coords = []
        for _ in range(num_patches):
            y = np.random.randint(0, self.H - patch_size + 1)
            x = np.random.randint(0, self.W - patch_size + 1)
            self.coords.append((y, x))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        ps = self.patch_size
        feat_patch = self.feature_maps[:, y:y+ps, x:x+ps]
        depth_patch = self.target_depth[:, y:y+ps, x:x+ps]
        feat_patch = torch.from_numpy(feat_patch).float()
        depth_patch = torch.from_numpy(depth_patch).float()
        return feat_patch, depth_patch

def main():
    parser = argparse.ArgumentParser(description="Train CNN refinement for depth-from-focus")
    parser.add_argument("--work_dir", type=str, default="results",
                        help="Directory where classical pipeline saved .npy files")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--num_patches", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # === Load classical outputs ===
    # These will be saved by an updated main.py (we'll modify it below).
    sharp_path = os.path.join(args.work_dir, "sharpness_stack.npy")
    depth_idx_path = os.path.join(args.work_dir, "depth_index_map.npy")

    sharp_stack = np.load(sharp_path)  # N x H x W
    depth_index_map = np.load(depth_idx_path)  # H x W

    N, H, W = sharp_stack.shape

    # Build feature maps
    max_sharp = sharp_stack.max(axis=0)           # H x W
    mean_sharp = sharp_stack.mean(axis=0)         # H x W

    depth_norm = (depth_index_map.astype(np.float32) - depth_index_map.min()) / \
                 (depth_index_map.max() - depth_index_map.min() + 1e-8)

    # Optional target smoothing to give nicer supervision
    depth_target = gaussian_filter(depth_norm, sigma=1.0)

    # Stack features into C x H x W
    feature_maps = np.stack([
        max_sharp,
        mean_sharp,
        depth_norm  # the baseline index as an extra cue
    ], axis=0).astype(np.float32)

    # Expand target to 1 x H x W
    target_depth = depth_target[None, :, :].astype(np.float32)

    dataset = SharpnessDepthDataset(feature_maps, target_depth,
                                    patch_size=args.patch_size,
                                    num_patches=args.num_patches)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)

    # === Model ===
    model = DfFRefineUNet(in_channels=feature_maps.shape[0], base_ch=32)
    model = model.to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for feat_batch, depth_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            feat_batch = feat_batch.to(device)
            depth_batch = depth_batch.to(device)

            optimizer.zero_grad()
            pred = model(feat_batch)
            loss = criterion(pred, depth_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * feat_batch.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.6f}")

    # Save model
    os.makedirs(args.work_dir, exist_ok=True)
    model_path = os.path.join(args.work_dir, "dff_cnn_refine.pth")
    torch.save(model.state_dict(), model_path)
    print("Saved model to:", model_path)

if __name__ == "__main__":
    main()
