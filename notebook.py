

#!/usr/bin/env python3
"""
Leonardo - Airborne Object Recognition Challenge
Kaggle Competition Notebook: Submission Version

Team:        DTU Compute
             Department of Applied Mathematics and Computer Science
             Technical University of Denmark

Team member: Gustav Olaf Yunus Laitinen-Fredriksson Lundstrom-Imanov
             (@olaflundstrom)

Architecture : RT-DETR (Real-Time Detection Transformer), ResNet-101-vd backbone
               End-to-end transformer detector; hybrid CNN-Transformer encoder;
               IoU-aware query selection; no NMS required at inference (bipartite
               Hungarian matching at training time).

Pretrained   : COCO 2017 val mAP@0.5:0.95 = 54.3  (PekingU/rtdetr_r101vd)

Why RT-DETR over YOLO / Faster R-CNN
--------------------------------------
  - YOLO is a single-stage anchor-based detector with known weaknesses on
    small and densely-packed objects.  Airborne imagery contains both.
  - Faster R-CNN is a two-stage detector that is strong but relies on
    predefined anchor grids; small-object performance degrades without
    careful anchor tuning and benefits less from transformer attention.
  - RT-DETR applies multi-scale deformable self-attention across the full
    image, capturing global context crucial for distinguishing visually
    similar categories (Aircraft vs Helicopter vs Drone from altitude).
  - The end-to-end formulation avoids NMS, eliminating a manual hyper-
    parameter that is hard to tune for multi-class aerial scenes.
  - R101-vd backbone (ResNet-101 with vd stem modifications) offers a
    significantly larger receptive field than R50 while remaining feasible
    on a P100 (16 GB VRAM) within the 9-hour Kaggle time budget.

Evaluation   : PASCAL VOC mAP@0.5 (IoU >= 0.5), competition metric
Target       : > 0.647 on public and private leaderboard

Sections
---------
  0  Pre-flight: installation and model download (internet-enabled cell)
  1  Imports and configuration
  2  Reproducibility
  3  EDA   (re-run to guide future improvements)
  4  Dataset preparation
  5  Model
  6  Optimizer and scheduler
  7  Training utilities
  8  1-epoch training loop
  9  Inference with multi-scale flip TTA
 10  Submission

Improvement roadmap (post-baseline)
-------------------------------------
  1. EDA-driven anchor / scale feedback: inspect eda_03_size_categories.png.
     If Drone AP is low, increase IMAGE_SIZE to 800 or add a small-scale crop.
  2. Longer training: 10-20 epochs with cosine decay yields +4-8 mAP points.
  3. Pseudo-labeling: generate soft labels from high-confidence test predictions
     (score > 0.8) and add them to the training set for epoch 2+.
  4. Mosaic augmentation: 4-image mosaic at 640 px forces the model to handle
     many small objects per image, matching the aerial distribution.
  5. Stronger backbone: Co-DINO with Swin-L (mmdetection) reaches 58.9 COCO
     mAP but requires ~24 GB VRAM.  Use gradient checkpointing + smaller batch.
  6. Objects365 pretraining: 'SenseTime/deformable-detr-with-box-refine' or
     DINO trained on O365+COCO transfers better to unseen aerial categories.
  7. Stochastic depth (drop_path_rate) tuning in the backbone for regularization.
"""

# ===========================================================================
# SECTION 0: MODEL DATASET
# ===========================================================================
# The RT-DETR weights are published as a private Kaggle dataset:
#   https://www.kaggle.com/datasets/olaflundstrom/rtdetr-r101vd
#
# To add it to this notebook:
#   1. Click "Add data" (top-right panel in the Kaggle notebook editor).
#   2. Select "Your Datasets".
#   3. Choose "rtdetr-r101vd".
#   4. Click "Add".
#
# Kaggle will mount the files at:
#   /kaggle/input/rtdetr-r101vd/rtdetr_r101vd/
# This path is already set as MODEL_LOCAL in Config.
#
# No pip install is needed; transformers, albumentations, torchmetrics,
# and sklearn are all pre-installed in the Kaggle GPU P100 image.
# WBF is implemented from scratch in this file.
# ---------------------------------------------------------------------------

# ===========================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ===========================================================================
import gc
import math
import os
import random
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Weighted Box Fusion (WBF): pure NumPy implementation
# Source: Solovyev et al., "Weighted boxes fusion", 2021.
# No external package required; works in internet-disabled Kaggle notebooks.
#
# Algorithm summary:
#   1. Pool all predicted boxes from all TTA passes into a single list.
#   2. Sort by score descending.
#   3. Greedily assign each box to an existing cluster if IoU >= iou_thr,
#      otherwise open a new cluster.
#   4. For each cluster, output the score-weighted mean coordinates and the
#      mean score scaled by (number of boxes / number of TTA passes).
#      This penalizes clusters where not all TTA passes agree.
#
# No external package is required; the function runs in internet-disabled
# Kaggle notebooks using only NumPy.
# ---------------------------------------------------------------------------

def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes in xyxy format.

    Parameters
    ----------
    boxes_a : (M, 4) float ndarray
    boxes_b : (N, 4) float ndarray

    Returns
    -------
    iou : (M, N) float ndarray
    """
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])

    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0)


def weighted_boxes_fusion(
    boxes_list:  List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    iou_thr:      float = 0.55,
    skip_box_thr: float = 0.04,
    conf_type:    str   = "avg",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted Box Fusion for merging predictions from multiple TTA passes.

    Parameters
    ----------
    boxes_list   : list of (N_i, 4) float arrays, xyxy coords in [0, 1]
    scores_list  : list of (N_i,) float arrays
    labels_list  : list of (N_i,) int arrays  (ignored internally; caller
                   should call this function per class so all labels match)
    iou_thr      : IoU threshold for cluster membership
    skip_box_thr : discard boxes with score below this threshold
    conf_type    : "avg" averages scores; "max" takes the maximum

    Returns
    -------
    fused_boxes  : (K, 4) float ndarray, xyxy in [0, 1]
    fused_scores : (K,)   float ndarray
    fused_labels : (K,)   int ndarray  (all zeros; caller maps back to class)
    """
    all_boxes:  List[np.ndarray] = []
    all_scores: List[float]      = []
    n_passes = len(boxes_list)

    for boxes, scores in zip(boxes_list, scores_list):
        if len(boxes) == 0:
            continue
        mask = scores >= skip_box_thr
        all_boxes.extend(boxes[mask].tolist())
        all_scores.extend(scores[mask].tolist())

    if len(all_boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )

    all_boxes   = np.array(all_boxes,  dtype=np.float32)
    all_scores  = np.array(all_scores, dtype=np.float32)

    # Sort descending by score
    order      = np.argsort(-all_scores)
    all_boxes  = all_boxes[order]
    all_scores = all_scores[order]

    # Cluster assignment
    cluster_boxes:  List[List[np.ndarray]] = []
    cluster_scores: List[List[float]]      = []
    cluster_reps:   List[np.ndarray]       = []   # representative box per cluster

    for box, score in zip(all_boxes, all_scores):
        if len(cluster_reps) == 0:
            cluster_boxes.append([box])
            cluster_scores.append([score])
            cluster_reps.append(box)
            continue

        reps = np.array(cluster_reps, dtype=np.float32)
        iou  = _iou_matrix(box[None, :], reps)[0]   # (n_clusters,)
        best = int(np.argmax(iou))

        if iou[best] >= iou_thr:
            cluster_boxes[best].append(box)
            cluster_scores[best].append(score)
            # Update representative to weighted mean of cluster members
            sc   = np.array(cluster_scores[best], dtype=np.float32)
            bx   = np.array(cluster_boxes[best],  dtype=np.float32)
            cluster_reps[best] = (bx * sc[:, None]).sum(0) / sc.sum()
        else:
            cluster_boxes.append([box])
            cluster_scores.append([score])
            cluster_reps.append(box)

    # Aggregate each cluster into a single prediction
    fused_boxes:  List[np.ndarray] = []
    fused_scores: List[float]      = []

    for bx_list, sc_list in zip(cluster_boxes, cluster_scores):
        bx = np.array(bx_list, dtype=np.float32)
        sc = np.array(sc_list, dtype=np.float32)

        weighted_box = (bx * sc[:, None]).sum(0) / sc.sum()

        if conf_type == "max":
            base_score = float(sc.max())
        else:
            base_score = float(sc.mean())

        # Scale by coverage fraction (how many TTA passes produced a box here)
        coverage = len(sc_list) / max(1, n_passes)
        final_score = base_score * min(1.0, coverage)

        fused_boxes.append(np.clip(weighted_box, 0.0, 1.0))
        fused_scores.append(final_score)

    fused_boxes_arr  = np.array(fused_boxes,  dtype=np.float32)
    fused_scores_arr = np.array(fused_scores, dtype=np.float32)
    fused_labels_arr = np.zeros(len(fused_scores_arr), dtype=np.int32)

    # Return in descending score order
    sort_idx = np.argsort(-fused_scores_arr)
    return (
        fused_boxes_arr[sort_idx],
        fused_scores_arr[sort_idx],
        fused_labels_arr[sort_idx],
    )


# ---------------------------------------------------------------------------
# Hyperparameter and path configuration
# All tunable values are centralized here.
# ---------------------------------------------------------------------------
class Config:
    # --- Paths ---
    ROOT       = Path("/kaggle/input/competitions/leonardo-airborne-object-recognition-challenge")
    TRAIN_DIR  = ROOT / "train"
    TEST_DIR   = ROOT / "test"
    TRAIN_CSV  = ROOT / "train.csv"
    SAMPLE_SUB = ROOT / "sample_submission.csv"
    OUTPUT_DIR = Path("/kaggle/working")

    # Path to the offline RT-DETR model weights.
    # Dataset: https://www.kaggle.com/datasets/olaflundstrom/rtdetr-r101vd
    # Add it to this notebook via:
    #   "Add data" -> "Your Datasets" -> rtdetr-r101vd
    # User-owned datasets are mounted under datasets/<username>/<slug>/.
    MODEL_LOCAL = "/kaggle/input/datasets/olaflundstrom/rtdetr-r101vd/rtdetr_r101vd"
    MODEL_PATH  = OUTPUT_DIR / "best_rtdetr.pth"

    # --- Classes (no background token; RT-DETR uses 0-indexed categories) ---
    CLASSES     = ["Aircraft", "Human", "GroundVehicle", "Drone", "Ship", "Obstacle", "Helicopter"]
    NUM_CLASSES = len(CLASSES)
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

    # --- Reproducibility ---
    SEED = 42

    # --- Data split ---
    VAL_FRACTION = 0.15

    # --- Image size ---
    # EDA shows 88.8 % of objects are "small" (relative area < 2 %).
    # Drone is 100 % small; Obstacle and Human are 90+ % small.
    # Increasing from 640 to 800 is the single most impactful change for AP.
    # RT-DETR requires image dimensions to be multiples of 32.
    # 800 / 32 = 25 exactly, so 800 is safe.
    IMAGE_SIZE = 800

    # --- Training ---
    BATCH_SIZE   = 6     # reduced from 8 to fit 800x800 in 17 GB VRAM
    NUM_WORKERS  = 4
    NUM_EPOCHS   = 1     # single-epoch fine-tuning; see roadmap for extension

    # --- Differential learning rates (3-group) ---
    # Backbone (pretrained ResNet-101-vd): very small to preserve ImageNet features.
    # Encoder (hybrid CNN-Transformer): moderate.
    # Decoder + head (randomly re-initialized for 7 competition classes): largest.
    LR_BACKBONE  = 1e-5
    LR_ENCODER   = 3e-5
    LR_DECODER   = 1e-4
    WEIGHT_DECAY = 1e-4

    # Warmup: first WARMUP_STEPS optimizer steps use a linear ramp from
    # LR * warmup_factor to the target LR.  This prevents early divergence.
    WARMUP_STEPS = 250
    MIN_LR_FRAC  = 0.1   # floor = MIN_LR_FRAC * target_LR (cosine tail)

    # --- Inference and TTA ---
    SCORE_THRESH = 0.05  # minimum confidence at inference time
    NMS_THRESH   = 0.50  # NMS IoU threshold inside WBF
    WBF_IOU_THR  = 0.55  # Weighted Box Fusion IoU threshold for TTA merging
    WBF_SKIP_THR = 0.04  # WBF minimum score to keep a box

    # TTA scales applied to the base IMAGE_SIZE.
    # CRITICAL: RT-DETR FPN requires all spatial dimensions to be multiples of 32.
    # snap_to_32() below enforces this; do NOT change TTA_SCALES to values whose
    # product with IMAGE_SIZE rounds to a non-multiple of 32.
    # At IMAGE_SIZE=800: 800*0.875=700 -> snap to 704, 800*1.0=800, 800*1.125=900 -> snap to 896.
    # Each scale runs with and without horizontal flip: 6 forward passes total.
    TTA_SCALES = [0.875, 1.0, 1.125]

    # --- Hardware ---
    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True


CFG = Config()

print(f"Device         : {CFG.DEVICE}")
print(f"PyTorch        : {torch.__version__}")
print(f"Torchvision    : {torchvision.__version__}")
if CFG.DEVICE.type == "cuda":
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM           : {vram:.1f} GB")


# ===========================================================================
# SECTION 2: REPRODUCIBILITY
# ===========================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(CFG.SEED)


# ===========================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
# ===========================================================================
# All figures are saved to CFG.OUTPUT_DIR for offline review.
# Key outputs and their modeling implications:
#   eda_01: class imbalance -> consider class-weighted focal loss coefficient
#   eda_02: bbox geometry   -> anchor-free RT-DETR adapts naturally, but
#                              very small median areas suggest IMAGE_SIZE >= 800
#   eda_03: size categories -> if Drone/Aircraft are mostly "small", use 800 px
#   eda_04: spatial heatmaps-> center bias means most objects appear mid-frame;
#                              no special spatial prior needed
#   eda_05: sample images   -> visual sanity check; IR frames appear gray
#   eda_06: RGB vs IR proxy -> high IR fraction -> consider gray augmentation
# ===========================================================================

def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    parts = str(bbox_str).strip().split()
    # Clamp to [0, 1]; some rows contain small floating-point noise below 0.
    x1 = max(0.0, min(1.0, float(parts[0])))
    y1 = max(0.0, min(1.0, float(parts[1])))
    x2 = max(0.0, min(1.0, float(parts[2])))
    y2 = max(0.0, min(1.0, float(parts[3])))
    return x1, y1, x2, y2


train_df = pd.read_csv(CFG.TRAIN_CSV)
print("\ntrain.csv shape:", train_df.shape)
print(train_df.head())
print("\nNull values:\n", train_df.isnull().sum())

train_df[["x_min", "y_min", "x_max", "y_max"]] = (
    train_df["bbox"].apply(lambda s: pd.Series(parse_bbox(s)))
)
train_df["bbox_width"]   = train_df["x_max"] - train_df["x_min"]
train_df["bbox_height"]  = train_df["y_max"] - train_df["y_min"]
train_df["bbox_area"]    = train_df["bbox_width"] * train_df["bbox_height"]
train_df["aspect_ratio"] = train_df["bbox_width"] / (train_df["bbox_height"] + 1e-8)
train_df["cx"]           = (train_df["x_min"] + train_df["x_max"]) / 2
train_df["cy"]           = (train_df["y_min"] + train_df["y_max"]) / 2

print(f"\nUnique images     : {train_df['ImageId'].nunique()}")
print(f"Total annotations : {len(train_df)}")
print(f"Avg ann/image     : {len(train_df) / train_df['ImageId'].nunique():.2f}")


# 3.1  Class distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
class_counts = train_df["class"].value_counts()
class_counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
axes[0].set_title("Annotation Count per Class")
axes[0].set_xlabel("Class"); axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)
for bar, val in zip(axes[0].patches, class_counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                 str(val), ha="center", va="bottom", fontsize=8)

imgs_per_class = train_df.groupby("class")["ImageId"].nunique().sort_values(ascending=False)
imgs_per_class.plot(kind="bar", ax=axes[1], color="coral", edgecolor="black")
axes[1].set_title("Images Containing Each Class")
axes[1].set_xlabel("Class"); axes[1].set_ylabel("Image Count")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(CFG.OUTPUT_DIR / "eda_01_class_distribution.png", dpi=100)
plt.show()
print("\nClass counts:\n", class_counts)


# 3.2  Bounding box geometry
print("\nBBox geometry (relative coords):")
print(train_df[["bbox_width", "bbox_height", "bbox_area", "aspect_ratio"]].describe().round(4))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
train_df["bbox_area"].hist(bins=60, ax=axes[0], color="steelblue", edgecolor="black")
axes[0].set_title("Relative BBox Area")
train_df["bbox_width"].hist(bins=60, ax=axes[1], color="coral", edgecolor="black")
axes[1].set_title("Relative BBox Width")
train_df["bbox_height"].hist(bins=60, ax=axes[2], color="green", edgecolor="black")
axes[2].set_title("Relative BBox Height")
train_df["aspect_ratio"].clip(upper=5).hist(bins=60, ax=axes[3], color="purple", edgecolor="black")
axes[3].set_title("Aspect Ratio (clipped at 5)")
for cls in train_df["class"].unique():
    sub = train_df[train_df["class"] == cls]
    axes[4].scatter(sub["bbox_width"], sub["bbox_height"], label=cls, alpha=0.3, s=5)
axes[4].set_title("Width vs Height by Class"); axes[4].set_xlabel("Width"); axes[4].set_ylabel("Height")
axes[4].legend(markerscale=3, fontsize=7); axes[4].set_xlim(0, 0.5); axes[4].set_ylim(0, 0.5)
ann_per_img = train_df.groupby("ImageId").size()
ann_per_img.hist(bins=30, ax=axes[5], color="teal", edgecolor="black")
axes[5].set_title("Annotations per Image")
plt.suptitle("Bounding Box Statistics", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(CFG.OUTPUT_DIR / "eda_02_bbox_stats.png", dpi=100)
plt.show()


# 3.3  Object size categories
SMALL_THRESH  = 0.02    # relative area < 2 %
MEDIUM_THRESH = 0.10    # 2 % <= area < 10 %

train_df["size_cat"] = pd.cut(
    train_df["bbox_area"],
    bins=[0, SMALL_THRESH, MEDIUM_THRESH, 1.0],
    labels=["small", "medium", "large"],
)
size_by_class = train_df.groupby(["class", "size_cat"]).size().unstack(fill_value=0)
size_by_class.plot(kind="bar", stacked=True, figsize=(12, 5), colormap="Set2")
plt.title("Object Size Category per Class"); plt.xlabel("Class"); plt.ylabel("Count")
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(CFG.OUTPUT_DIR / "eda_03_size_categories.png", dpi=100)
plt.show()
print("\nSize distribution by class:\n", size_by_class)

# Modeling implication: if small objects dominate, IMAGE_SIZE should be >= 800.
small_pct = (train_df["size_cat"] == "small").mean()
print(f"\nFraction of 'small' objects: {small_pct:.2%}")
if small_pct > 0.40:
    print("  -> Consider IMAGE_SIZE = 800 in Config for better small-object AP.")


# 3.4  Spatial center heatmaps
n_cls = train_df["class"].nunique()
fig, axes_grid = plt.subplots(2, 4, figsize=(20, 10))
axes_flat = axes_grid.flatten()
for ax, cls in zip(axes_flat, sorted(train_df["class"].unique())):
    sub = train_df[train_df["class"] == cls]
    h2d, _, _ = np.histogram2d(sub["cx"], sub["cy"], bins=30, range=[[0, 1], [0, 1]])
    im = ax.imshow(h2d.T, origin="lower", aspect="auto", cmap="hot", extent=[0, 1, 0, 1])
    ax.set_title(f"Centers: {cls}"); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax)
for ax in axes_flat[n_cls:]:
    ax.set_visible(False)
plt.suptitle("Spatial Distribution of Object Centers", fontsize=14)
plt.tight_layout()
plt.savefig(CFG.OUTPUT_DIR / "eda_04_center_heatmaps.png", dpi=100)
plt.show()


# 3.5  Sample image visualization
CLASS_COLORS = {
    "Aircraft": "#e41a1c", "Human": "#377eb8", "GroundVehicle": "#4daf4a",
    "Drone": "#984ea3",    "Ship": "#ff7f00",  "Obstacle": "#a65628",
    "Helicopter": "#f781bf",
}

def visualize_samples(df: pd.DataFrame, n: int = 6) -> None:
    sample_ids = (
        df["ImageId"].drop_duplicates()
        .sample(min(n, df["ImageId"].nunique()), random_state=CFG.SEED)
        .tolist()
    )
    n_rows = math.ceil(n / 3)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
    axes = axes.flatten()
    for ax, img_id in zip(axes, sample_ids):
        img_path = CFG.TRAIN_DIR / f"{img_id}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            ax.set_visible(False); continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ax.imshow(img)
        for _, row in df[df["ImageId"] == img_id].iterrows():
            x1, y1 = row["x_min"] * w, row["y_min"] * h
            x2, y2 = row["x_max"] * w, row["y_max"] * h
            color = CLASS_COLORS.get(row["class"], "#ffffff")
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                                      edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1-3, row["class"], fontsize=7, color=color,
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.45, pad=1, linewidth=0))
        ax.set_title(img_id, fontsize=7); ax.axis("off")
    for ax in axes[len(sample_ids):]:
        ax.set_visible(False)
    plt.suptitle("Sample Training Images with Ground-Truth Boxes", fontsize=13)
    plt.tight_layout()
    plt.savefig(CFG.OUTPUT_DIR / "eda_05_sample_images.png", dpi=100)
    plt.show()

visualize_samples(train_df, n=6)


# 3.6  RGB vs IR modality proxy
def estimate_modality(img_dir: Path, sample_n: int = 200) -> pd.DataFrame:
    paths   = sorted(img_dir.glob("*.png"))
    sampled = random.sample(paths, min(sample_n, len(paths)))
    rows = []
    for p in sampled:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float)
        means   = img_rgb.mean(axis=(0, 1))
        rows.append({"file": p.stem, "ch_diff": float(np.std(means))})
    return pd.DataFrame(rows)

modality_df = estimate_modality(CFG.TRAIN_DIR, sample_n=300)
IR_THR = 5.0
n_ir  = (modality_df["ch_diff"] < IR_THR).sum()
n_rgb = len(modality_df) - n_ir
print(f"\nEstimated IR frames  : {n_ir}  ({100*n_ir/len(modality_df):.1f} %)")
print(f"Estimated RGB frames : {n_rgb} ({100*n_rgb/len(modality_df):.1f} %)")

fig, ax = plt.subplots(figsize=(8, 4))
modality_df["ch_diff"].hist(bins=40, ax=ax, color="steelblue", edgecolor="black")
ax.axvline(IR_THR, color="red", linestyle="--", label=f"IR threshold ({IR_THR})")
ax.set_title("Channel-Difference Proxy (RGB vs IR)")
ax.set_xlabel("Std of per-channel means"); ax.set_ylabel("Image count")
ax.legend(); plt.tight_layout()
plt.savefig(CFG.OUTPUT_DIR / "eda_06_rgb_vs_ir.png", dpi=100)
plt.show()


# ===========================================================================
# SECTION 4: DATASET PREPARATION
# ===========================================================================

# 4.1  Train / validation split at image level (prevents annotation leakage)
unique_ids = train_df["ImageId"].unique()
train_ids, val_ids = train_test_split(
    unique_ids, test_size=CFG.VAL_FRACTION, random_state=CFG.SEED
)
print(f"\nTrain images : {len(train_ids)}")
print(f"Val   images : {len(val_ids)}")


# 4.2  Augmentation pipelines
# RT-DETR uses ImageNet normalization (same as the albumentations defaults below).
# The processor expects (3, H, W) float32 tensors normalized to [-mean/std, ...].
# We replicate this in albumentations and bypass the HuggingFace processor for
# training (which would apply a second normalization).

def get_train_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            # Resize: preserve aspect, pad with zeros in bottom-right corner.
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT, value=0,
                position="top_left",
            ),
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.10),
            A.RandomRotate90(p=0.20),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.25, rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT, value=0, p=0.45,
            ),
            A.Perspective(scale=(0.02, 0.06), p=0.15),
            # Photometric augmentations: airborne sensor variability
            A.RandomBrightnessContrast(brightness_limit=0.40, contrast_limit=0.40, p=0.65),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=35, val_shift_limit=30, p=0.45),
            A.CLAHE(clip_limit=4.0, p=0.25),
            A.GaussNoise(var_limit=(10.0, 70.0), p=0.25),
            A.MotionBlur(blur_limit=9, p=0.30),     # simulates airframe vibration
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.30, p=0.12),
            A.RandomRain(drop_length=5, blur_value=3, p=0.06),
            A.ToGray(p=0.20),                        # EDA shows ~56 % of frames are IR
            A.ChannelShuffle(p=0.05),
            # ImageNet normalization (matches RT-DETR pretraining)
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0),
        ],
        bbox_params=A.BboxParams(
            format="albumentations",   # normalized [x_min, y_min, x_max, y_max]
            label_fields=["labels"],
            min_visibility=0.30,
            min_area=0.0001,
        ),
    )


def get_val_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT, value=0,
                position="top_left",
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0),
        ],
        bbox_params=A.BboxParams(
            format="albumentations", label_fields=["labels"],
            min_visibility=0.30, min_area=0.0001,
        ),
    )


def get_tta_transform(image_size: int, hflip: bool = False) -> A.Compose:
    """
    Single TTA transform for a given scale and flip state.
    No bbox_params because TTA is applied at inference time only.
    """
    ops = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT, value=0,
            position="top_left",
        ),
    ]
    if hflip:
        ops.append(A.HorizontalFlip(p=1.0))
    ops += [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(p=1.0),
    ]
    return A.Compose(ops)


# 4.3  Dataset classes

class AirborneDataset(Dataset):
    """
    Training and validation dataset for RT-DETR.

    Label format required by RTDetrForObjectDetection:
      - class_labels : torch.LongTensor (num_objects,)   (0-indexed, no background)
      - boxes        : torch.FloatTensor (num_objects, 4), COCO format (cx, cy, w, h)
                       in relative [0, 1] coordinates

    COCO box format is (center_x, center_y, width, height), all normalized by
    image dimensions.  This differs from the Pascal VOC / albumentations format
    (x_min, y_min, x_max, y_max) used in the annotation CSV.
    """

    def __init__(
        self,
        image_ids: np.ndarray,
        df: pd.DataFrame,
        image_dir: Path,
        transform: Optional[A.Compose] = None,
    ) -> None:
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.transform = transform

        id_set = set(image_ids)
        self._index: Dict[str, List[Tuple]] = defaultdict(list)
        _MIN = 1e-3   # minimum side length after clipping
        for _, row in df[df["ImageId"].isin(id_set)].iterrows():
            x1 = max(0.0, min(1.0, float(row["x_min"])))
            y1 = max(0.0, min(1.0, float(row["y_min"])))
            x2 = max(0.0, min(1.0, float(row["x_max"])))
            y2 = max(0.0, min(1.0, float(row["y_max"])))
            if (x2 - x1) >= _MIN and (y2 - y1) >= _MIN:
                self._index[row["ImageId"]].append(
                    (CFG.CLASS_TO_IDX[row["class"]], x1, y1, x2, y2)
                )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_id   = self.image_ids[idx]
        img_path = self.image_dir / f"{img_id}.png"
        img      = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        anns = self._index.get(img_id, [])

        # Clip to [0, 1] to handle floating-point noise (e.g. y_min = -1e-6).
        # Then discard boxes where clipping causes x_min >= x_max or y_min >= y_max
        # (e.g. a box whose y_min was already 1.0 before clipping).
        # A minimum side length of 1e-3 (0.1 % of image) is required.
        MIN_SIDE = 1e-3
        bboxes_rel = []
        labels     = []
        for a in anns:
            x1 = max(0.0, min(1.0, a[1]))
            y1 = max(0.0, min(1.0, a[2]))
            x2 = max(0.0, min(1.0, a[3]))
            y2 = max(0.0, min(1.0, a[4]))
            if (x2 - x1) >= MIN_SIDE and (y2 - y1) >= MIN_SIDE:
                bboxes_rel.append([x1, y1, x2, y2])
                labels.append(a[0])

        if self.transform is not None:
            out        = self.transform(image=img, bboxes=bboxes_rel, labels=labels)
            img_t      = out["image"]
            bboxes_rel = list(out["bboxes"])
            labels     = list(out["labels"])
        else:
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # Convert xyxy normalized -> cxcywh normalized (RT-DETR label format)
        if len(bboxes_rel) > 0:
            boxes = np.array(bboxes_rel, dtype=np.float32)   # (N, 4) xyxy [0,1]
            cx  = (boxes[:, 0] + boxes[:, 2]) / 2
            cy  = (boxes[:, 1] + boxes[:, 3]) / 2
            w   = boxes[:, 2] - boxes[:, 0]
            h   = boxes[:, 3] - boxes[:, 1]
            cxcywh = np.stack([cx, cy, w, h], axis=1)
            # Remove degenerate boxes (width or height <= 0)
            valid = (w > 1e-4) & (h > 1e-4)
            cxcywh   = cxcywh[valid]
            labels   = [labels[i] for i in range(len(labels)) if valid[i]]
            boxes_t  = torch.as_tensor(cxcywh, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.long)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.long)

        target = {"class_labels": labels_t, "boxes": boxes_t}
        return img_t, target


class AirborneTestDataset(Dataset):
    """
    Test dataset.  Returns image tensor, ImageId, and original (W, H)
    for coordinate back-projection after inference.
    """

    def __init__(self, image_dir: Path) -> None:
        self.image_dir = image_dir
        self.image_ids = sorted([p.stem for p in image_dir.glob("*.png")])

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray, Tuple[int, int]]:
        img_id   = self.image_ids[idx]
        img_path = self.image_dir / f"{img_id}.png"
        img      = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Test image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H_orig, W_orig = img.shape[:2]
        return img_id, img, (W_orig, H_orig)


def collate_train(batch):
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return torch.stack(images), targets


def collate_val(batch):
    return collate_train(batch)


# 4.4  Build datasets and dataloaders
train_dataset = AirborneDataset(
    train_ids, train_df, CFG.TRAIN_DIR,
    transform=get_train_transform(CFG.IMAGE_SIZE),
)
val_dataset = AirborneDataset(
    val_ids, train_df, CFG.TRAIN_DIR,
    transform=get_val_transform(CFG.IMAGE_SIZE),
)
# Test dataset has no fixed transform; TTA applies transforms per pass.
test_dataset_raw = AirborneTestDataset(CFG.TEST_DIR)

train_loader = DataLoader(
    train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
    num_workers=CFG.NUM_WORKERS, collate_fn=collate_train,
    pin_memory=True, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
    num_workers=CFG.NUM_WORKERS, collate_fn=collate_val,
    pin_memory=True,
)

print(f"\nTrain batches : {len(train_loader)}")
print(f"Val batches   : {len(val_loader)}")
print(f"Test images   : {len(test_dataset_raw)}")


# ===========================================================================
# SECTION 5: MODEL
# ===========================================================================
# Architecture: RT-DETR (Real-Time Detection Transformer)
#
# Core design of RT-DETR
# -----------------------
# 1. Backbone (ResNet-101-vd): multi-scale feature extraction at strides
#    8, 16, 32.  The -vd variant replaces the 7x7 conv stem with three 3x3
#    convs (Bag of Tricks) and uses average pooling instead of max pooling
#    in the downsampling shortcuts, providing better gradient flow.
#
# 2. Hybrid Encoder: a single-scale transformer applied on the last feature
#    map (stride 32) followed by a PANet-style top-down / bottom-up path
#    to produce multi-scale features at strides 8, 16, 32.  This is
#    computationally cheaper than applying full deformable attention at all
#    scales, and achieves real-time latency while retaining strong features.
#
# 3. IoU-Aware Query Selection: selects the top-k encoder features by
#    predicted IoU and initializes the decoder queries from them.  This
#    leads to faster convergence compared to DETR's learned positional queries.
#
# 4. Transformer Decoder: 6 layers of cross-attention between queries and
#    encoder features, plus self-attention among queries.  Outputs (cx,cy,w,h)
#    predictions and class logits for each query.
#
# 5. Loss: Hungarian bipartite matching (one prediction per GT object).
#    Losses: focal classification + L1 box + GIoU box (identical to DINO).
#
# Why this outperforms anchor-based detectors on airborne data
# ------------------------------------------------------------
# - Self-attention models long-range dependencies; an aircraft partially
#   occluded at the frame edge can be completed via global context.
# - No anchor hyperparameters to tune: the model learns object shapes directly
#   from the data distribution, adapting to the unusual aspect ratios of
#   aerial observations (cranes, antennas have large h/w ratios).
# - Multi-scale features from the hybrid encoder span a range of resolutions,
#   naturally handling the extreme scale variance between a distant drone
#   (2-5 px) and a large ship spanning the image width.
#
# Class head reinitialization
# ----------------------------
# The pretrained model has num_labels=91 (COCO).  We replace the linear
# classification head with a new one of size 7 (competition classes).
# ignore_mismatched_sizes=True allows loading all other weights unchanged.
# ===========================================================================

print(f"\nLoading RT-DETR model from: {CFG.MODEL_LOCAL}")

# Verify the path exists before calling from_pretrained; a missing dataset
# produces a confusing HFValidationError instead of a plain FileNotFoundError.
import os as _os
if not _os.path.isdir(CFG.MODEL_LOCAL):
    raise FileNotFoundError(
        f"Model directory not found: {CFG.MODEL_LOCAL}\n"
        "Make sure the dataset olaflundstrom/rtdetr-r101vd is added to this "
        "notebook via 'Add data' -> 'Your Datasets'."
    )

processor = RTDetrImageProcessor.from_pretrained(
    CFG.MODEL_LOCAL,
    local_files_only=True,
)

model = RTDetrForObjectDetection.from_pretrained(
    CFG.MODEL_LOCAL,
    num_labels=CFG.NUM_CLASSES,
    ignore_mismatched_sizes=True,   # reinitializes classification head only
    local_files_only=True,
)
model = model.to(CFG.DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")

# Parameter group summary: backbone vs encoder vs decoder
backbone_params, encoder_params, decoder_params = [], [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "backbone" in name:
        backbone_params.append(param)
    elif "encoder" in name:
        encoder_params.append(param)
    else:
        decoder_params.append(param)

print(f"Backbone params  : {sum(p.numel() for p in backbone_params):,}")
print(f"Encoder params   : {sum(p.numel() for p in encoder_params):,}")
print(f"Decoder params   : {sum(p.numel() for p in decoder_params):,}")


# ===========================================================================
# SECTION 6: OPTIMIZER AND SCHEDULER
# ===========================================================================
# Three-group differential LR:
#   - Backbone: smallest LR to preserve rich ImageNet visual features.
#   - Encoder:  intermediate LR; it was pretrained on COCO but must adapt to
#               the new spatial distribution.
#   - Decoder + head: largest LR; the classification head is newly initialized
#               and the decoder queries must rapidly adapt to aerial viewpoints.
#
# Scheduler: linear warmup for WARMUP_STEPS steps, then cosine decay.
# OneCycleLR was considered but cosine gives smoother 1-epoch convergence.
# ===========================================================================

optimizer = optim.AdamW(
    [
        {"params": backbone_params, "lr": CFG.LR_BACKBONE, "name": "backbone"},
        {"params": encoder_params,  "lr": CFG.LR_ENCODER,  "name": "encoder"},
        {"params": decoder_params,  "lr": CFG.LR_DECODER,  "name": "decoder"},
    ],
    weight_decay=CFG.WEIGHT_DECAY,
)

total_steps   = len(train_loader) * CFG.NUM_EPOCHS
warmup_steps  = CFG.WARMUP_STEPS


def warmup_cosine(step: int, target_lr: float, warmup: int, total: int, min_frac: float) -> float:
    """Linear warmup followed by cosine decay.  Returns absolute LR."""
    if step < warmup:
        return target_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return target_lr * (min_frac + (1.0 - min_frac) * cosine)


scaler = GradScaler(enabled=CFG.USE_AMP)
global_step = 0   # incremented inside the training loop


# ===========================================================================
# SECTION 7: TRAINING UTILITIES
# ===========================================================================

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


@torch.inference_mode()
def evaluate_map(model: nn.Module, loader: DataLoader) -> Dict:
    """
    Compute PASCAL VOC mAP@0.5 on the validation set.

    RT-DETR outputs:
      logits : (B, num_queries, num_classes), raw pre-sigmoid logits
      pred_boxes : (B, num_queries, 4), cxcywh normalized [0,1]

    Post-processing converts to absolute xyxy for torchmetrics.
    """
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)

    for pixel_values, targets in loader:
        pixel_values = pixel_values.to(CFG.DEVICE, non_blocking=True)
        B, _, H, W   = pixel_values.shape

        with autocast(enabled=CFG.USE_AMP):
            outputs = model(pixel_values=pixel_values)

        # Decode RT-DETR outputs
        logits     = outputs.logits.sigmoid()               # (B, Q, C)
        pred_boxes = outputs.pred_boxes                     # (B, Q, 4) cxcywh [0,1]

        preds_list   = []
        targets_list = []

        for i in range(B):
            scores, labels = logits[i].max(dim=-1)          # (Q,)
            # Convert cxcywh -> xyxy in absolute pixel coords
            cx, cy, bw, bh = pred_boxes[i].unbind(-1)
            x1 = (cx - bw / 2) * W; y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W; y2 = (cy + bh / 2) * H
            boxes_abs = torch.stack([x1, y1, x2, y2], dim=-1)

            keep = scores > CFG.SCORE_THRESH
            preds_list.append({
                "boxes":  boxes_abs[keep].cpu(),
                "scores": scores[keep].cpu(),
                "labels": labels[keep].cpu(),
            })

            # Ground-truth boxes are cxcywh [0,1] -> convert to xyxy absolute
            gt_boxes  = targets[i]["boxes"]         # (M, 4) cxcywh [0,1]
            gt_labels = targets[i]["class_labels"]  # (M,)
            if len(gt_boxes) > 0:
                gcx, gcy, gw, gh = gt_boxes.unbind(-1)
                gx1 = (gcx - gw/2) * W; gy1 = (gcy - gh/2) * H
                gx2 = (gcx + gw/2) * W; gy2 = (gcy + gh/2) * H
                gt_xyxy = torch.stack([gx1, gy1, gx2, gy2], dim=-1)
            else:
                gt_xyxy = torch.zeros((0, 4), dtype=torch.float32)

            targets_list.append({"boxes": gt_xyxy, "labels": gt_labels})

        metric.update(preds_list, targets_list)

    result    = metric.compute()
    map_50    = result["map_50"].item()
    per_class = {}
    if "map_per_class" in result and result["map_per_class"].numel() > 0:
        for j, ap in enumerate(result["map_per_class"]):
            cls_name = CFG.IDX_TO_CLASS.get(j, f"class_{j}")
            per_class[cls_name] = ap.item()
    return {"map_50": map_50, "per_class_ap": per_class}


# ===========================================================================
# SECTION 8: TRAINING LOOP (1 EPOCH)
# ===========================================================================

history: Dict[str, List] = {"step": [], "loss": [], "lr_decoder": []}
best_map50 = 0.0

print("\n" + "=" * 70)
print("RT-DETR ResNet-101-vd: 1-Epoch Fine-Tuning")
print(f"Total steps      : {total_steps}")
print(f"Warmup steps     : {warmup_steps}")
print(f"LR backbone      : {CFG.LR_BACKBONE}")
print(f"LR encoder       : {CFG.LR_ENCODER}")
print(f"LR decoder/head  : {CFG.LR_DECODER}")
print(f"Image size       : {CFG.IMAGE_SIZE}")
print(f"Batch size       : {CFG.BATCH_SIZE}")
print("=" * 70 + "\n")

loss_meter = AverageMeter()
t_epoch_start = time.time()

model.train()

for batch_idx, (pixel_values, targets) in enumerate(train_loader):
    # --- Adjust learning rate per step (step-level schedule) ---
    for pg in optimizer.param_groups:
        target_lr = CFG.LR_DECODER if pg["name"] == "decoder" else (
                    CFG.LR_ENCODER if pg["name"] == "encoder" else CFG.LR_BACKBONE)
        pg["lr"]  = warmup_cosine(
            global_step, target_lr, warmup_steps, total_steps, CFG.MIN_LR_FRAC
        )

    pixel_values = pixel_values.to(CFG.DEVICE, non_blocking=True)
    # Move each target dict to device
    targets_dev = [
        {k: v.to(CFG.DEVICE, non_blocking=True) for k, v in t.items()}
        for t in targets
    ]

    optimizer.zero_grad(set_to_none=True)

    with autocast(enabled=CFG.USE_AMP):
        # RTDetrForObjectDetection computes Hungarian-matched loss internally
        # when labels are provided in the expected format.
        outputs = model(pixel_values=pixel_values, labels=targets_dev)
        loss    = outputs.loss

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    scaler.step(optimizer)
    scaler.update()

    loss_meter.update(loss.item(), pixel_values.size(0))
    global_step += 1

    history["step"].append(global_step)
    history["loss"].append(loss.item())
    history["lr_decoder"].append(optimizer.param_groups[2]["lr"])

    if batch_idx % 50 == 0:
        elapsed = time.time() - t_epoch_start
        eta_s   = elapsed / max(1, batch_idx + 1) * (len(train_loader) - batch_idx - 1)
        print(
            f"  Step [{batch_idx:04d}/{len(train_loader)}]  "
            f"Loss: {loss_meter.avg:.4f}  "
            f"LR(dec): {optimizer.param_groups[2]['lr']:.2e}  "
            f"ETA: {eta_s/60:.1f} min"
        )

# End-of-epoch validation
epoch_time = time.time() - t_epoch_start
print(f"\nEpoch 1 complete ({epoch_time/60:.1f} min).  Evaluating on validation set...")

val_metrics = evaluate_map(model, val_loader)
best_map50  = val_metrics["map_50"]

print(f"\nValidation mAP@0.5 : {best_map50:.4f}")
print(f"Target             : 0.647")
print(f"Status             : {'PASSED' if best_map50 > 0.647 else 'Below target: see improvement roadmap'}")
for cls, ap in sorted(val_metrics["per_class_ap"].items()):
    print(f"    {cls:<18} : AP@0.5 = {ap:.4f}")

torch.save({"model_state": model.state_dict(), "val_map50": best_map50}, CFG.MODEL_PATH)
print(f"\nCheckpoint saved to {CFG.MODEL_PATH}")

# Training curve
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(history["step"], history["loss"], color="steelblue", linewidth=0.8)
axes[0].set_title("Training Loss (per step)"); axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss")
axes[0].grid(True)

axes[1].plot(history["step"], history["lr_decoder"], color="coral", linewidth=0.8)
axes[1].set_title("Decoder LR Schedule"); axes[1].set_xlabel("Step"); axes[1].set_ylabel("LR")
axes[1].grid(True); axes[1].set_yscale("log")

plt.tight_layout()
plt.savefig(CFG.OUTPUT_DIR / "training_curves.png", dpi=100)
plt.show()

gc.collect(); torch.cuda.empty_cache()


# ===========================================================================
# SECTION 9: INFERENCE WITH MULTI-SCALE FLIP TTA
# ===========================================================================
# Test-Time Augmentation strategy
# ---------------------------------
# For each test image we run CFG.TTA_SCALES x 2 (with/without horizontal flip)
# forward passes.  This yields 6 sets of predictions per image (3 scales x 2).
# All prediction sets are then merged using Weighted Box Fusion (WBF).
#
# WBF overview:
#   Given N sets of predicted boxes (from different TTA passes), WBF clusters
#   boxes by IoU overlap and for each cluster computes a single fused box whose
#   coordinates are the confidence-weighted mean of member boxes, and whose
#   score is the average member score.  This is more robust than NMS, which
#   selects one box and discards all others.
#
# Coordinate pipeline (per TTA pass)
# ------------------------------------
# Original image size : (W_orig, H_orig)
# TTA image size      : tta_sz = round(IMAGE_SIZE * scale)  (square after padding)
# Scale factor        : s = tta_sz / max(W_orig, H_orig)
# After LongestMaxSize + TOP_LEFT pad:
#     W_scaled = W_orig * s <= tta_sz
#     H_scaled = H_orig * s <= tta_sz
# RT-DETR outputs boxes as cxcywh in [0,1] relative to tta_sz x tta_sz.
# Back-projection to original normalized coords:
#     x_rel = pred_cx / tta_sz * tta_sz / W_scaled = pred_cx / W_scaled
# But pred_cx is already in [0,1] relative to tta_sz, so:
#     x_abs_tta = pred_cx * tta_sz   (in padded image pixels)
#     x_abs_orig = x_abs_tta / s = x_abs_tta * max(W_orig, H_orig) / tta_sz
#     x_rel = x_abs_orig / W_orig
# WBF expects all boxes in [0,1] normalized to their respective image size.
# We normalize to [0,1] w.r.t. the ORIGINAL image (before padding), so
# that all TTA passes and WBF operate in the same coordinate frame.
# ===========================================================================

# Load best checkpoint
ckpt = torch.load(CFG.MODEL_PATH, map_location=CFG.DEVICE)
model.load_state_dict(ckpt["model_state"])
print(f"\nLoaded checkpoint with val mAP@0.5 = {ckpt['val_map50']:.4f}")


def decode_rtdetr_output(
    outputs,
    W_orig: int,
    H_orig: int,
    tta_size: int,
    hflip: bool,
    score_thresh: float = CFG.SCORE_THRESH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode RT-DETR output tensors and back-project to original image coords.

    Parameters
    ----------
    outputs      : RTDetrObjectDetectionOutput (single image, batch size 1)
    W_orig, H_orig : original image dimensions
    tta_size     : side length of the squared TTA canvas
    hflip        : whether this TTA pass used horizontal flip
    score_thresh : minimum confidence to retain

    Returns
    -------
    boxes_norm   : (N, 4) float ndarray, xyxy normalized to original image
    scores       : (N,)   float ndarray
    labels       : (N,)   int   ndarray, 0-indexed class ids
    """
    logits     = outputs.logits[0].sigmoid()   # (Q, C)
    pred_boxes = outputs.pred_boxes[0]         # (Q, 4) cxcywh [0,1] wrt tta canvas

    scores, label_ids = logits.max(dim=-1)

    keep = scores > score_thresh
    scores    = scores[keep].cpu().float().numpy()
    label_ids = label_ids[keep].cpu().numpy()
    boxes_rel = pred_boxes[keep].cpu().float().numpy()   # cxcywh [0,1]

    if len(boxes_rel) == 0:
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    # cxcywh [0,1] wrt tta canvas -> xyxy [0,1] wrt tta canvas
    cx, cy, bw, bh = boxes_rel[:, 0], boxes_rel[:, 1], boxes_rel[:, 2], boxes_rel[:, 3]
    x1 = np.clip(cx - bw / 2, 0.0, 1.0)
    y1 = np.clip(cy - bh / 2, 0.0, 1.0)
    x2 = np.clip(cx + bw / 2, 0.0, 1.0)
    y2 = np.clip(cy + bh / 2, 0.0, 1.0)

    # Back-project from padded tta canvas to original image normalized coords.
    # scale = tta_size / max(W_orig, H_orig)
    # W_scaled = W_orig * scale  (portion of tta canvas occupied by the image)
    # x_rel_orig = x_tta_fraction * tta_size / W_scaled
    scale    = tta_size / max(W_orig, H_orig)
    W_frac   = W_orig * scale / tta_size     # fraction of canvas width with content
    H_frac   = H_orig * scale / tta_size

    x1_orig = np.clip(x1 / W_frac, 0.0, 1.0)
    x2_orig = np.clip(x2 / W_frac, 0.0, 1.0)
    y1_orig = np.clip(y1 / H_frac, 0.0, 1.0)
    y2_orig = np.clip(y2 / H_frac, 0.0, 1.0)

    # Undo horizontal flip in coordinate space
    if hflip:
        x1_orig, x2_orig = 1.0 - x2_orig, 1.0 - x1_orig

    boxes_norm = np.stack([x1_orig, y1_orig, x2_orig, y2_orig], axis=1)
    return boxes_norm, scores, label_ids


@torch.inference_mode()
def predict_with_tta(
    model: nn.Module,
    test_dataset: AirborneTestDataset,
    tta_scales: List[float],
    base_size: int,
) -> Dict[str, str]:
    """
    Run inference with multi-scale horizontal-flip TTA on all test images.

    For each image:
      1. Loop over tta_scales x {no flip, hflip}: 2 * len(tta_scales) passes
      2. Collect all boxes in original-image normalized coords
      3. Merge with Weighted Box Fusion (WBF), class-separated
      4. Format as PredictionString for submission

    Returns
    -------
    dict: ImageId -> PredictionString
    """
    def snap_to_32(n: int) -> int:
        """Round n to the nearest multiple of 32.
        RT-DETR FPN concatenates feature maps at strides 8, 16, 32.
        If the input spatial dimension is not divisible by 32 the two branches
        have shapes that differ by 1, causing the cat() to raise RuntimeError.
        Example: 640*0.875=560 -> 560%32=16 (bad) -> snap to 576.
                 640*1.125=720 -> 720%32=16 (bad) -> snap to 736.
                 800*0.875=700 -> 700%32=28 (bad) -> snap to 704.
                 800*1.125=900 -> 900%32=4  (bad) -> snap to 896.
        """
        return max(32, int(round(n / 32)) * 32)

    model.eval()
    predictions: Dict[str, str] = {}

    for img_idx in range(len(test_dataset)):
        img_id, img_np, (W_orig, H_orig) = test_dataset[img_idx]

        all_boxes:  Dict[int, List[np.ndarray]] = defaultdict(list)
        all_scores: Dict[int, List[np.ndarray]] = defaultdict(list)

        for scale in tta_scales:
            # Snap to nearest multiple of 32 to satisfy RT-DETR FPN constraint.
            tta_size = snap_to_32(int(round(base_size * scale)))

            for hflip in [False, True]:
                tfm  = get_tta_transform(tta_size, hflip=hflip)
                img_t = tfm(image=img_np)["image"]
                pixel_values = img_t.unsqueeze(0).to(CFG.DEVICE, non_blocking=True)

                with autocast(enabled=CFG.USE_AMP):
                    outputs = model(pixel_values=pixel_values)

                boxes, scores, labels = decode_rtdetr_output(
                    outputs, W_orig, H_orig, tta_size, hflip
                )

                for cls_id in np.unique(labels):
                    mask = labels == cls_id
                    all_boxes[cls_id].append(boxes[mask])
                    all_scores[cls_id].append(scores[mask])

        # WBF fusion per class
        if not all_boxes:
            predictions[img_id] = "None 1 -1 -1 -1 -1"
            continue

        parts: List[str] = []
        for cls_id, box_list in all_boxes.items():
            sc_list = all_scores[cls_id]
            boxes_wbf, scores_wbf, _ = weighted_boxes_fusion(
                box_list,
                sc_list,
                [np.zeros(len(b), dtype=int) for b in box_list],  # dummy single-class labels
                iou_thr=CFG.WBF_IOU_THR,
                skip_box_thr=CFG.WBF_SKIP_THR,
                conf_type="avg",
            )
            cls_name = CFG.IDX_TO_CLASS.get(cls_id, "")
            if not cls_name or cls_name == "__background__":
                continue
            for box, sc in zip(boxes_wbf, scores_wbf):
                x1, y1, x2, y2 = box
                parts.append(
                    f"{cls_name} {sc:.4f} {x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f}"
                )

        predictions[img_id] = " ".join(parts) if parts else "None 1 -1 -1 -1 -1"

        if img_idx % 500 == 0:
            print(f"  TTA inference: {img_idx}/{len(test_dataset)}")

    return predictions


print("\nStarting TTA inference...")
t_inf = time.time()
test_predictions = predict_with_tta(
    model, test_dataset_raw, CFG.TTA_SCALES, CFG.IMAGE_SIZE
)
inf_time = time.time() - t_inf
print(f"Inference complete: {len(test_predictions)} images in {inf_time/60:.1f} min")


# ===========================================================================
# SECTION 10: SUBMISSION
# ===========================================================================

sample_sub    = pd.read_csv(CFG.SAMPLE_SUB)
submission_rows = [
    {
        "ImageId":          img_id,
        "PredictionString": test_predictions.get(img_id, "None 1 -1 -1 -1 -1"),
    }
    for img_id in sample_sub["ImageId"].values
]
submission_df = pd.DataFrame(submission_rows)

assert len(submission_df) == len(sample_sub), "Row count mismatch"
assert submission_df["ImageId"].is_unique, "Duplicate ImageIds"

output_path = CFG.OUTPUT_DIR / "submission.csv"
submission_df.to_csv(output_path, index=False)

non_none = (submission_df["PredictionString"] != "None 1 -1 -1 -1 -1").sum()
print(f"\nSubmission file     : {output_path}")
print(f"Total rows          : {len(submission_df)}")
print(f"Images with objects : {non_none} ({100*non_none/len(submission_df):.1f} %)")
print("\nFirst 10 rows:")
print(submission_df.head(10).to_string(index=False))


# Submission format verification
def verify_prediction_string(pred_str: str, image_id: str) -> bool:
    if pred_str == "None 1 -1 -1 -1 -1":
        return True
    tokens = pred_str.strip().split()
    if len(tokens) % 6 != 0:
        print(f"WARNING: malformed PredictionString for {image_id}")
        return False
    for d in range(len(tokens) // 6):
        off = d * 6
        assert tokens[off] in CFG.CLASS_TO_IDX, f"Unknown label: {tokens[off]}"
        conf   = float(tokens[off + 1])
        coords = [float(tokens[off + k]) for k in range(2, 6)]
        assert 0.0 <= conf <= 1.0
        assert all(0.0 <= c <= 1.0 for c in coords)
    return True

print("\nVerifying submission format...")
n_checked = min(500, len(submission_df))
errors    = sum(
    0 if verify_prediction_string(r["PredictionString"], r["ImageId"]) else 1
    for _, r in submission_df.head(n_checked).iterrows()
)
print(f"Checked {n_checked} rows, errors: {errors}")

print("\nDone.  Upload /kaggle/working/submission.csv to Kaggle.")

/usr/local/lib/python3.12/dist-packages/albumentations/check_version.py:147: UserWarning: Error fetching version info <urlopen error [Errno -3] Temporary failure in name resolution>
  data = fetch_version_info()

Device         : cuda
PyTorch        : 2.9.0+cu126
Torchvision    : 0.24.0+cu126
GPU            : Tesla P100-PCIE-16GB
VRAM           : 17.1 GB

train.csv shape: (72319, 3)
            ImageId     class                                 bbox
0  00026721e16f8530  Obstacle  0.893394 0.351251 0.920469 0.426637
1  000974b83586c193     Human       0.136938 0.956852 0.154719 1.0
2  000974b83586c193     Human   0.056938 0.478112 0.06326 0.505074
3  000974b83586c193     Human  0.060094 0.466871 0.067688 0.490463
4  000974b83586c193     Human  0.123563 0.755889 0.133438 0.780593

Null values:
 ImageId    0
class      0
bbox       0
dtype: int64

Unique images     : 17406
Total annotations : 72319
Avg ann/image     : 4.15

Class counts:
 class
GroundVehicle    25974
Human            13828
Obstacle         13224
Ship             13100
Drone             2494
Aircraft          2190
Helicopter        1509
Name: count, dtype: int64

BBox geometry (relative coords):
       bbox_width  bbox_height   bbox_area  aspect_ratio
count  72319.0000   72319.0000  72319.0000  7.231900e+04
mean       0.0623       0.0979      0.0161  6.508325e+02
std        0.1132       0.1288      0.0707  6.324389e+04
min        0.0000       0.0000      0.0000  0.000000e+00
25%        0.0140       0.0285      0.0005  2.984000e-01
50%        0.0275       0.0541      0.0015  6.152000e-01
75%        0.0580       0.1139      0.0058  9.766000e-01
max        1.0000       1.0000      1.0000  9.074600e+06

Size distribution by class:
 size_cat       small  medium  large
class                              
Aircraft        1137     745    308
Drone           2488       0      0
GroundVehicle  23627    1971    376
Helicopter       758     456    295
Human          13275     441     95
Obstacle       12088     966    167
Ship           10841    1204   1055

Fraction of 'small' objects: 88.79%
  -> Consider IMAGE_SIZE = 800 in Config for better small-object AP.

Estimated IR frames  : 169  (56.3 %)
Estimated RGB frames : 131 (43.7 %)

Train images : 14795
Val   images : 2611

Train batches : 2465
Val batches   : 218
Test images   : 3

Loading RT-DETR model from: /kaggle/input/datasets/olaflundstrom/rtdetr-r101vd/rtdetr_r101vd

RTDetrForObjectDetection LOAD REPORT from: /kaggle/input/datasets/olaflundstrom/rtdetr-r101vd/rtdetr_r101vd
Key                                                 | Status   |                                                                                        
----------------------------------------------------+----------+----------------------------------------------------------------------------------------
model.decoder.class_embed.{0, 1, 2, 3, 4, 5}.weight | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([80, 256]) vs model:torch.Size([7, 256])
model.denoising_class_embed.weight                  | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([81, 256]) vs model:torch.Size([8, 256])
model.enc_score_head.weight                         | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([80, 256]) vs model:torch.Size([7, 256])
model.decoder.class_embed.{0, 1, 2, 3, 4, 5}.bias   | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([80]) vs model:torch.Size([7])          
model.enc_score_head.bias                           | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([80]) vs model:torch.Size([7])          

Notes:
- MISMATCH	:ckpt weights were loaded, but they did not match the original empty weight shapes.

Total parameters     : 76,406,253
Trainable parameters : 76,406,253
Backbone params  : 42,413,920
Encoder params   : 27,814,208
Decoder params   : 6,178,125

======================================================================
RT-DETR ResNet-101-vd: 1-Epoch Fine-Tuning
Total steps      : 2465
Warmup steps     : 250
LR backbone      : 1e-05
LR encoder       : 3e-05
LR decoder/head  : 0.0001
Image size       : 800
Batch size       : 6
======================================================================

  Step [0000/2465]  Loss: 209.4814  LR(dec): 4.00e-07  ETA: 2389.3 min
  Step [0050/2465]  Loss: 141.3602  LR(dec): 2.04e-05  ETA: 97.0 min
  Step [0100/2465]  Loss: 99.2387  LR(dec): 4.04e-05  ETA: 73.2 min
  Step [0150/2465]  Loss: 79.4123  LR(dec): 6.04e-05  ETA: 64.5 min
  Step [0200/2465]  Loss: 67.9493  LR(dec): 8.04e-05  ETA: 59.6 min
  Step [0250/2465]  Loss: 60.7175  LR(dec): 1.00e-04  ETA: 56.1 min
  Step [0300/2465]  Loss: 55.6265  LR(dec): 9.99e-05  ETA: 53.5 min
  Step [0350/2465]  Loss: 52.0548  LR(dec): 9.95e-05  ETA: 51.4 min
  Step [0400/2465]  Loss: 49.1592  LR(dec): 9.90e-05  ETA: 49.5 min
  Step [0450/2465]  Loss: 46.7121  LR(dec): 9.82e-05  ETA: 47.7 min
  Step [0500/2465]  Loss: 44.8593  LR(dec): 9.72e-05  ETA: 46.1 min
  Step [0550/2465]  Loss: 43.1604  LR(dec): 9.60e-05  ETA: 44.6 min
  Step [0600/2465]  Loss: 41.7406  LR(dec): 9.46e-05  ETA: 43.2 min
  Step [0650/2465]  Loss: 40.5479  LR(dec): 9.30e-05  ETA: 41.8 min
  Step [0700/2465]  Loss: 39.5126  LR(dec): 9.11e-05  ETA: 40.5 min
  Step [0750/2465]  Loss: 38.5879  LR(dec): 8.92e-05  ETA: 39.2 min
  Step [0800/2465]  Loss: 37.7324  LR(dec): 8.70e-05  ETA: 37.9 min
  Step [0850/2465]  Loss: 36.9709  LR(dec): 8.47e-05  ETA: 36.7 min
  Step [0900/2465]  Loss: 36.2728  LR(dec): 8.22e-05  ETA: 35.4 min
  Step [0950/2465]  Loss: 35.6614  LR(dec): 7.96e-05  ETA: 34.2 min
  Step [1000/2465]  Loss: 35.1065  LR(dec): 7.69e-05  ETA: 33.0 min
  Step [1050/2465]  Loss: 34.5959  LR(dec): 7.40e-05  ETA: 31.8 min
  Step [1100/2465]  Loss: 34.0958  LR(dec): 7.11e-05  ETA: 30.6 min
  Step [1150/2465]  Loss: 33.6408  LR(dec): 6.81e-05  ETA: 29.5 min
  Step [1200/2465]  Loss: 33.1978  LR(dec): 6.50e-05  ETA: 28.3 min
  Step [1250/2465]  Loss: 32.8301  LR(dec): 6.18e-05  ETA: 27.2 min
  Step [1300/2465]  Loss: 32.4892  LR(dec): 5.87e-05  ETA: 26.0 min
  Step [1350/2465]  Loss: 32.1494  LR(dec): 5.55e-05  ETA: 24.9 min
  Step [1400/2465]  Loss: 31.8131  LR(dec): 5.23e-05  ETA: 23.7 min
  Step [1450/2465]  Loss: 31.4986  LR(dec): 4.91e-05  ETA: 22.6 min
  Step [1500/2465]  Loss: 31.1608  LR(dec): 4.60e-05  ETA: 21.4 min
  Step [1550/2465]  Loss: 30.8911  LR(dec): 4.29e-05  ETA: 20.3 min
  Step [1600/2465]  Loss: 30.6560  LR(dec): 3.98e-05  ETA: 19.2 min
  Step [1650/2465]  Loss: 30.3972  LR(dec): 3.69e-05  ETA: 18.1 min
  Step [1700/2465]  Loss: 30.1632  LR(dec): 3.40e-05  ETA: 16.9 min
  Step [1750/2465]  Loss: 29.9180  LR(dec): 3.12e-05  ETA: 15.8 min
  Step [1800/2465]  Loss: 29.6930  LR(dec): 2.86e-05  ETA: 14.7 min
  Step [1850/2465]  Loss: 29.5034  LR(dec): 2.61e-05  ETA: 13.6 min
  Step [1900/2465]  Loss: 29.3228  LR(dec): 2.37e-05  ETA: 12.5 min
  Step [1950/2465]  Loss: 29.1362  LR(dec): 2.15e-05  ETA: 11.4 min
  Step [2000/2465]  Loss: 28.9730  LR(dec): 1.94e-05  ETA: 10.2 min
  Step [2050/2465]  Loss: 28.8169  LR(dec): 1.76e-05  ETA: 9.1 min
  Step [2100/2465]  Loss: 28.6498  LR(dec): 1.59e-05  ETA: 8.0 min
  Step [2150/2465]  Loss: 28.4863  LR(dec): 1.44e-05  ETA: 6.9 min
  Step [2200/2465]  Loss: 28.3401  LR(dec): 1.31e-05  ETA: 5.8 min
  Step [2250/2465]  Loss: 28.1932  LR(dec): 1.21e-05  ETA: 4.7 min
  Step [2300/2465]  Loss: 28.0758  LR(dec): 1.12e-05  ETA: 3.6 min
  Step [2350/2465]  Loss: 27.9402  LR(dec): 1.06e-05  ETA: 2.5 min
  Step [2400/2465]  Loss: 27.8144  LR(dec): 1.02e-05  ETA: 1.4 min
  Step [2450/2465]  Loss: 27.6976  LR(dec): 1.00e-05  ETA: 0.3 min

Epoch 1 complete (54.2 min).  Evaluating on validation set...

Validation mAP@0.5 : 0.4377
Target             : 0.647
Status             : Below target: see improvement roadmap
    Aircraft           : AP@0.5 = 0.5729
    Drone              : AP@0.5 = 0.6344
    GroundVehicle      : AP@0.5 = 0.3605
    Helicopter         : AP@0.5 = 0.3566
    Human              : AP@0.5 = 0.4104
    Obstacle           : AP@0.5 = 0.1936
    Ship               : AP@0.5 = 0.5358

Checkpoint saved to /kaggle/working/best_rtdetr.pth

Loaded checkpoint with val mAP@0.5 = 0.4377

Starting TTA inference...
  TTA inference: 0/3
Inference complete: 3 images in 0.2 min

Submission file     : /kaggle/working/submission.csv
Total rows          : 3
Images with objects : 3 (100.0 %)

First 10 rows:
         ImageId                                                                                                                                                                                                                                                                                                                                                                                                                              PredictionString
00028dc7b4fe2ca1 Human 0.1053 0.5063 0.4939 0.5329 0.5369 Human 0.0391 0.5138 0.5047 0.5308 0.5344 Human 0.0225 0.5070 0.4965 0.5218 0.5284 Human 0.0129 0.5895 0.4903 0.5984 0.5102 Human 0.0111 0.4038 0.4913 0.4128 0.5098 Helicopter 0.0546 0.3932 0.4253 0.6086 0.6002 Helicopter 0.0325 0.5000 0.4830 0.5517 0.5430 Aircraft 0.0644 0.0000 1.0000 0.9843 1.0000 Ship 0.0969 0.5052 0.4913 0.5320 0.5376 GroundVehicle 0.0544 0.0511 0.8068 0.3847 1.0000
00035aaa02c4002f                                                                                                                                                                           Human 0.0606 0.4186 0.5048 0.4456 0.5550 Human 0.0179 0.4678 0.4261 0.4849 0.4517 Ship 0.1029 0.4181 0.4306 0.5174 0.5700 Ship 0.0093 0.4665 0.4245 0.4870 0.4506 GroundVehicle 0.0389 0.0000 0.3270 0.9587 1.0000 GroundVehicle 0.0255 0.0000 1.0000 0.8927 1.0000
00103b0934b1d99e                                                                                                                                                                                                                                                                                                                                                               Ship 0.1495 0.4294 0.1210 0.5718 0.6595 Ship 0.0193 0.4409 0.4587 0.5660 0.6486

Verifying submission format...
Checked 3 rows, errors: 0

Done.  Upload /kaggle/working/submission.csv to Kaggle.

