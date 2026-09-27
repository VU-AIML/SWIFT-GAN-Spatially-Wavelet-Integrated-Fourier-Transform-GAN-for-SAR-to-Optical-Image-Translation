#!/usr/bin/env python3
"""
Cityscapes Label->Photo semantic evaluation (FCN-score style).

Runs AFTER training completes. Using the best checkpoint (netG_AB_best.pth),
it generates fake photos from testA (label) images, passes them through a
pretrained segmentation network (SegFormer, Cityscapes), and compares the
prediction against the trainId map of the input label.

Produced metrics (Cityscapes table columns):
  - pixAcc   (per-pixel accuracy)
  - classAcc (mean per-class accuracy)  -- also reported as the "mAP" column
  - mIoU     (mean IoU)

NOTE: This uses a pretrained DRN network. It differs from the classic FCN-8s
(Caffe) protocol; the paper MUST state "semantic scores computed with a
pretrained DRN network". Absolute values are not directly comparable to
FCN-8s tables.

metrics.py is NOT touched. This is a fully separate evaluation tool.

Usage:
  python eval_cityscapes_semantic.py \
      --ckpt   checkpoints_bench_cityscapes/netG_AB_best.pth \
      --testA  benchmark_data/cityscapes/testA \
      --out    outputs_bench_cityscapes
"""
import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from model import SwiftGenerator

# ----------------------------------------------------------------------
# Cityscapes 19 train-class color palette (RGB). gtFine_color.png uses these.
# Order = trainId 0..18.
# ----------------------------------------------------------------------
CITYSCAPES_PALETTE = [
    (128, 64,128),  # 0 road
    (244, 35,232),  # 1 sidewalk
    ( 70, 70, 70),  # 2 building
    (102,102,156),  # 3 wall
    (190,153,153),  # 4 fence
    (153,153,153),  # 5 pole
    (250,170, 30),  # 6 traffic light
    (220,220,  0),  # 7 traffic sign
    (107,142, 35),  # 8 vegetation
    (152,251,152),  # 9 terrain
    ( 70,130,180),  # 10 sky
    (220, 20, 60),  # 11 person
    (255,  0,  0),  # 12 rider
    (  0,  0,142),  # 13 car
    (  0,  0, 70),  # 14 truck
    (  0, 60,100),  # 15 bus
    (  0, 80,100),  # 16 train
    (  0,  0,230),  # 17 motorcycle
    (119, 11, 32),  # 18 bicycle
]
NUM_CLASSES = 19
PALETTE = np.array(CITYSCAPES_PALETTE, dtype=np.int64)  # [19,3]


def color_to_trainid(label_rgb):
    """Color label (H,W,3) -> trainId map (H,W). Assign each pixel to nearest palette color."""
    h, w, _ = label_rgb.shape
    flat = label_rgb.reshape(-1, 3).astype(np.int64)      # [HW,3]
    # nearest palette color per pixel (L2)
    d = ((flat[:, None, :] - PALETTE[None, :, :]) ** 2).sum(-1)  # [HW,19]
    ids = d.argmin(1).reshape(h, w)
    return ids


def load_segmenter(device):
    """
    Load a SegFormer segmentation network pretrained on Cityscapes (19 classes).
    Uses HuggingFace transformers (already available in the environment) instead of
    the outdated fyu/drn torch.hub repo, which is not torch.hub compatible.
    Model: nvidia/segformer-b0-finetuned-cityscapes-1024-1024
    The model's label ids follow the standard 19-class Cityscapes trainId order,
    matching CITYSCAPES_PALETTE above.
    """
    from transformers import SegformerForSemanticSegmentation
    name = os.environ.get("SEG_MODEL", "nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained(name)
    model.eval().to(device)
    print(f"[INFO] Segmenter loaded: {name}")
    return model


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Generator (label -> photo)
    G = SwiftGenerator(input_nc=3, output_nc=3).to(device)
    G.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    G.eval()
    print(f"[INFO] Generator loaded: {args.ckpt}")

    # 2. Segmentation network
    seg = load_segmenter(device)

    # ImageNet normalization (expected by DRN)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    label_files = sorted(glob.glob(os.path.join(args.testA, "*.png")) +
                         glob.glob(os.path.join(args.testA, "*.jpg")))
    print(f"[INFO] {len(label_files)} test label images.")

    # confusion matrix [19,19]
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for i, lab_path in enumerate(label_files):
        # --- read input label, feed to generator ---
        lab_img = Image.open(lab_path).convert('RGB').resize((256, 256))
        lab_np = np.asarray(lab_img, dtype=np.int64)             # for ground-truth trainId
        gt_ids = color_to_trainid(lab_np)                        # [256,256]

        # generator input: [-1,1]
        x = torch.from_numpy(np.asarray(lab_img, np.float32)/255.0).permute(2,0,1)
        x = ((x - 0.5)/0.5).unsqueeze(0).to(device)
        fake = G(x)                                              # [-1,1] fake photo
        fake01 = (fake * 0.5 + 0.5).clamp(0, 1)                  # [0,1]

        # --- pass fake photo through segmentation (SegFormer) ---
        seg_in = (fake01 - mean) / std
        out = seg(pixel_values=seg_in)
        logits = out.logits                                      # [1,19,H/4,W/4]
        logits = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False)
        pred = logits.argmax(1)[0].cpu().numpy()                 # [256,256]

        # --- update confusion matrix ---
        mask = (gt_ids >= 0) & (gt_ids < NUM_CLASSES)
        conf += np.bincount(
            NUM_CLASSES * gt_ids[mask].astype(int) + pred[mask].astype(int),
            minlength=NUM_CLASSES**2
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(label_files)} processed")

    # --- metrics ---
    tp = np.diag(conf)
    gt_per_class = conf.sum(1)
    pred_per_class = conf.sum(0)

    pixAcc = tp.sum() / max(conf.sum(), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        cls_acc = tp / np.maximum(gt_per_class, 1)
        iou = tp / np.maximum(gt_per_class + pred_per_class - tp, 1)
    valid = gt_per_class > 0
    classAcc = np.nanmean(cls_acc[valid])
    mIoU = np.nanmean(iou[valid])

    print("\n" + "="*60)
    print(f"[CITYSCAPES SEMANTIC - SegFormer protocol]")
    print(f"  pixAcc   : {pixAcc*100:.2f}")
    print(f"  classAcc : {classAcc*100:.2f}   (table: mAP/classAcc)")
    print(f"  mIoU     : {mIoU*100:.2f}")
    print("="*60)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "cityscapes_semantic_scores.txt"), "w") as f:
        f.write(f"pixAcc {pixAcc*100:.2f}\nclassAcc {classAcc*100:.2f}\nmIoU {mIoU*100:.2f}\n")
    print(f"[INFO] Saved: {args.out}/cityscapes_semantic_scores.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",  required=True, help="netG_AB_best.pth (label->photo)")
    ap.add_argument("--testA", required=True, help="cityscapes/testA (color labels)")
    ap.add_argument("--out",   default="./outputs_bench_cityscapes")
    args = ap.parse_args()
    run(args)