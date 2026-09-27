#!/usr/bin/env python3
"""
Cityscapes -> trainA/trainB (Label->Photo) organizer.

Cityscapes raw structure:
  leftImg8bit/{train,val}/{city}/{city}_{seq}_{frame}_leftImg8bit.png   (real photo)
  gtFine/{train,val}/{city}/{city}_{seq}_{frame}_gtFine_color.png        (color label)

Output (FolderDataset-compatible, flat folders):
  cityscapes/trainA/*.png   <- color semantic labels (Label, input)
  cityscapes/trainB/*.png   <- real photos (Photo, target)
  cityscapes/testA, testB   <- from val (Cityscapes test has no labels, we use val as test)

Task direction: Label -> Photo  => A=label, B=photo (no swap needed, A->B is correct).

Usage:
  python prepare_cityscapes.py \
      --root benchmark_data \
      --out  benchmark_data/cityscapes
"""
import os, glob, shutil, argparse


def collect(gt_dir, img_dir, outA, outB):
    os.makedirs(outA, exist_ok=True)
    os.makedirs(outB, exist_ok=True)

    # color labels
    label_files = sorted(glob.glob(os.path.join(gt_dir, "*", "*_gtFine_color.png")))
    n = 0
    for lab in label_files:
        base = os.path.basename(lab).replace("_gtFine_color.png", "")
        city = os.path.basename(os.path.dirname(lab))
        photo = os.path.join(img_dir, city, base + "_leftImg8bit.png")
        if not os.path.isfile(photo):
            print(f"  [skip] no matching photo: {base}")
            continue
        # copy with same name so A/B stay aligned (paired ground-truth preserved)
        shutil.copy(lab,   os.path.join(outA, base + ".png"))
        shutil.copy(photo, os.path.join(outB, base + ".png"))
        n += 1
    print(f"  -> {n} pairs copied: {outA} , {outB}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory containing leftImg8bit and gtFine")
    ap.add_argument("--out",  required=True, help="output cityscapes directory")
    args = ap.parse_args()

    gt   = os.path.join(args.root, "gtFine")
    img  = os.path.join(args.root, "leftImg8bit")

    print("== TRAIN ==")
    collect(os.path.join(gt, "train"), os.path.join(img, "train"),
            os.path.join(args.out, "trainA"), os.path.join(args.out, "trainB"))

    print("== VAL (used as test) ==")
    collect(os.path.join(gt, "val"), os.path.join(img, "val"),
            os.path.join(args.out, "testA"), os.path.join(args.out, "testB"))

    print("\nDone. Structure:")
    for d in ["trainA", "trainB", "testA", "testB"]:
        p = os.path.join(args.out, d)
        cnt = len(glob.glob(os.path.join(p, "*.png"))) if os.path.isdir(p) else 0
        print(f"  {p}: {cnt} images")


if __name__ == "__main__":
    main()