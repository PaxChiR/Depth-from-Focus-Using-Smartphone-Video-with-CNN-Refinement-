## Depth from Focus Using Smartphone Video (with CNN Refinement)**

## 📌 Overview

This project implements **Depth from Focus (DfF)** using only a **single smartphone video** where the camera performs a **focus sweep** (manually changing focus during capture).
The pipeline:

1. Extracts frames from the video
2. Stabilizes them
3. Computes per-pixel sharpness curves
4. Estimates a **relative depth map** using classical DfF (argmax over focus)
5. Synthesizes an **all-in-focus image**
6. *(Stretch Goal)* Trains a **CNN (U-Net)** to refine the noisy classical depth map using pseudo-labeling

The result demonstrates how **computational photography** can extract meaningful depth information using only a standard smartphone camera — no stereo, no LiDAR, no special sensors.

---

## 📂 Project Structure

```
.
├── cnn_model.py
├── dff_pipeline.py
├── focus_measures.py
├── video_utils.py
├── main.py
├── train_cnn_refine.py
├── requirements.txt
├── IMG_6115.MOV                # Your input video
└── results/
    ├── frames/
    ├── stabilized_frames/
    ├── sharpness_maps/
    ├── depth_maps/
    ├── all_in_focus/
    ├── sharpness_stack.npy
    ├── depth_index_map.npy
    └── dff_cnn_refine.pth
```

---

## 🛠 Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key libraries:

* Python 3.8+
* NumPy
* OpenCV
* SciPy
* Matplotlib
* PyTorch (CPU version OK)
* tqdm

---

## 🎥 Input Video Requirements

Use a smartphone to record a **focus-sweep video**:

* 1080p preferred
* Hold phone as still as possible
* Tap-to-focus on different depths OR manually rack focus
* 5–15 seconds is ideal

Save the video as `.MOV` or `.MP4`.

---

# 🚀 Running the Project

## 1️⃣ Run Classical Depth-from-Focus Pipeline

```bash
python main.py --video_path IMG_6115.MOV --work_dir results
```

This outputs:

```
results/
├── depth_maps/depth_map.png
├── all_in_focus/all_in_focus.png
├── depth_index_map.npy
└── sharpness_stack.npy
```

---

# 🤖 (Optional) Using the CNN Depth Refinement

The CNN refines the classical depth map using pseudo-label supervision.

### Step 1: Train CNN

```bash
python train_cnn_refine.py --work_dir results --device cpu
```

This produces:

```
results/dff_cnn_refine.pth
```

### Step 2: Run pipeline with CNN refinement enabled

```bash
python main.py --video_path IMG_6115.MOV --work_dir results --use_cnn_refine --device cpu
```

Outputs:

```
results/depth_maps/depth_map_cnn_refined.png
```

---

# 📊 Output Files Explained

### 📌 Classical DfF Outputs

| File                  | Description                                     |
| --------------------- | ----------------------------------------------- |
| `depth_map.png`       | Relative depth map from sharpness-argmax        |
| `all_in_focus.png`    | Composited sharpest pixels across video         |
| `depth_index_map.npy` | Raw depth index values (frame number per pixel) |
| `sharpness_stack.npy` | N × H × W sharpness volumes                     |

### 📌 CNN Refinement Outputs

| File                        | Description              |
| --------------------------- | ------------------------ |
| `dff_cnn_refine.pth`        | Trained U-Net model      |
| `depth_map_cnn_refined.png` | Smoothed, denoised depth |

---

# 🧠 Method Summary

## Classical Depth from Focus

* Compute per-frame sharpness using Laplacian variance
* Normalize sharpness across frames
* Stabilize frames to remove motion
* For each pixel:

  * Take the frame index where sharpness is maximized
  * This index corresponds to **relative depth**
* Normalize into grayscale depth image

## CNN Refinement (Stretch Goal)

* Build feature maps:

  * Max sharpness
  * Mean sharpness
  * Normalized classical depth index
* Use handcrafted depth as pseudo labels
* Train a **U-Net** to produce a cleaner depth map
* Output is smoother, less noisy, and has cleaner edges

---

# 🖼 Visual Examples

**Classical Depth vs CNN Refinement**
(Add your images here)

```
results/depth_maps/depth_map.png
results/depth_maps/depth_map_cnn_refined.png
```

---

# 📝 Citation / References

* Nayar & Nakagawa, *Shape from Focus*, IEEE PAMI, 1994
* Hazirbas et al., *Deep Depth from Focus*, ECCV 2018
* Smartphone computational photography research

---

# 🧩 Notes / Limitations

* Produces **relative** depth, not metric depth
* Requires stable video with visible focus changes
* Textureless regions remain challenging
* CNN refinement requires pseudo-labels (no real depth supervision)

---

# 🎯 Summary

This project shows how a single smartphone focus-sweep video can be used to estimate depth by combining:

* Classical computational photography (Depth-from-Focus)
* Modern deep learning (U-Net refinement)

It demonstrates the power of software-only 3D reconstruction on consumer devices and serves as a foundation for AR, refocusing, segmentation, and computational photography techniques.
