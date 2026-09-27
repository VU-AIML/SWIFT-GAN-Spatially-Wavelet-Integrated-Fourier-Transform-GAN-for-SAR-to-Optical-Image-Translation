#!/usr/bin/env python3
"""
Benchmark qualitative figure for SWIFT-GAN (Table 2 companion).

For every benchmark dataset, shows N sample pairs:
    [ Input | SWIFT-GAN output ]

Datasets are grouped as in the paper:
    geometry fixed / appearance-level : Cityscapes, Summer->Winter,
                                       Map->Satellite, Horse->Zebra
    anatomical re-shaping             : Cat->Dog

Each benchmark has its own checkpoint under swift_gan_src/checkpoints_bench_*/.
Test images are read from benchmark_data/<name>/testA (or trainA as fallback).

Usage (run from .../sar_to_optical/swift_gan_src):
    python make_benchmark_figure.py --n-per-set 3 --out benchmark_qualitative.png

    # dump every test image so you can hand-pick the best ones:
    python make_benchmark_figure.py --dump-all --dump-dir bench_dumps
"""
import os, sys, glob, argparse, importlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

# (display name, data dir under benchmark_data/, checkpoint dir, swap_ab)
# swap_ab=True means the training used A=B-domain (see run_maps.sh), so the
# input we want to show is trainB/testB instead of testA.
BENCHMARKS = [
    # --- structure-preserving ---
    ("Cityscapes (Label$\\to$Photo)", "cityscapes",              "checkpoints_bench_cityscapes",              False),
    ("Summer $\\to$ Winter",          "summer2winter_yosemite",  "checkpoints_bench_summer2winter_yosemite",  False),
    ("Map $\\to$ Satellite",          "maps",                    "checkpoints_bench_maps",                    True),
    # --- shape-altering ---
    ("Cat $\\to$ Dog",                "cat2dog",                 "checkpoints_bench_cat2dog",                 False),
    ("Horse $\\to$ Zebra",            "horse2zebra",             "checkpoints_bench_horse2zebra",             False),
]


# ---------------- model ----------------
def load_swift_generator(ckpt_path):
    """Import the SWIFT-GAN generator (swift_gan_src/model.py) and load weights."""
    sys.path.insert(0, ROOT)
    try:
        mod = importlib.import_module("model")
        net = mod.SwiftGenerator(input_nc=3, output_nc=3).to(DEVICE)
        ck = torch.load(ckpt_path, map_location=DEVICE)
        if isinstance(ck, dict) and "model_state_dict" in ck:
            ck = ck["model_state_dict"]
        net.load_state_dict(ck)
        net.eval()
        return net
    finally:
        if ROOT in sys.path:
            sys.path.remove(ROOT)


def find_checkpoint(ckpt_dir):
    d = os.path.join(ROOT, ckpt_dir)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"missing checkpoint dir: {d}")
    for pat in ["netG_AB_best.pth", "*AB_best.pth", "netG_AB_ep*.pth"]:
        hits = sorted(glob.glob(os.path.join(d, pat)))
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"no generator checkpoint in {d}")


# ---------------- data ----------------
def load_image(path):
    """RGB image -> [1,3,256,256] tensor in [-1,1], plus the display array."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # [H,W,3] in [0,1]
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    t = (t - 0.5) / 0.5
    return t, arr


def list_test_images(data_dir, swap_ab):
    """Return the input-domain image paths for a benchmark."""
    base = os.path.join(ROOT, "benchmark_data", data_dir)
    dom = "B" if swap_ab else "A"
    for split in [f"test{dom}", f"train{dom}"]:
        d = os.path.join(base, split)
        if os.path.isdir(d):
            files = sorted(glob.glob(os.path.join(d, "*.*")))
            files = [f for f in files
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if files:
                return files, split
    raise FileNotFoundError(f"no input images for {data_dir}")


def denorm(t):
    img = t.detach().cpu().numpy() * 0.5 + 0.5
    return np.transpose(np.clip(img, 0, 1), (1, 2, 0))


@torch.no_grad()
def translate(net, tensor):
    return denorm(net(tensor.to(DEVICE))[0])


# ---------------- selection ----------------
def pick_by_names(net, files, names):
    """
    Select specific files by name. `names` come from the dump directory and look
    like "023_frankfurt_000000_011007.png": a NNN_ prefix added by --dump-all
    followed by the original filename. We strip the prefix and match on the rest.
    """
    def strip_prefix(n):
        base = os.path.basename(n.strip())
        parts = base.split("_", 1)
        return parts[1] if len(parts) == 2 and parts[0].isdigit() else base

    wanted = [strip_prefix(n) for n in names]
    by_base = {os.path.basename(f): f for f in files}

    out = []
    for w in wanted:
        f = by_base.get(w)
        if f is None:                       # tolerate minor name drift
            hits = [p for b, p in by_base.items() if b.endswith(w) or w.endswith(b)]
            f = hits[0] if hits else None
        if f is None:
            print(f"[WARN]   pick not found: {w}")
            continue
        t, arr = load_image(f)
        out.append((arr, translate(net, t), os.path.basename(f)))
    return out


def pick_samples(net, files, n, mode="spread"):
    """
    Choose n input files.
      spread : evenly spaced across the (sorted) test set - unbiased default
      change : largest input/output difference, i.e. most actual translation
    Returns list of (input_array, output_array, filename).
    """
    if mode == "spread":
        idx = np.linspace(0, len(files) - 1, num=min(n, len(files)), dtype=int)
        chosen = [files[i] for i in idx]
        out = []
        for f in chosen:
            t, arr = load_image(f)
            out.append((arr, translate(net, t), os.path.basename(f)))
        return out

    # mode == "change": score every image, keep the top-n
    scored = []
    for f in files[:200]:                     # cap for speed
        t, arr = load_image(f)
        pred = translate(net, t)
        scored.append((float(np.mean(np.abs(pred - arr))), f, arr, pred))
    scored.sort(reverse=True)
    return [(a, p, os.path.basename(f)) for (_, f, a, p) in scored[:n]]


# ---------------- figure ----------------
def build_figure(rows, out_path, n_per_set):
    """rows: list of (bench_name, [(inp, out, fname), ...])"""
    n_bench = len(rows)
    n_cols = 2 * n_per_set                    # (input,output) pairs side by side
    fig, axes = plt.subplots(n_bench, n_cols,
                             figsize=(1.5 * n_cols, 1.6 * n_bench))
    if n_bench == 1:
        axes = np.expand_dims(axes, 0)

    for r, (bench, samples) in enumerate(rows):
        for i in range(n_per_set):
            ci, co = 2 * i, 2 * i + 1
            if i < len(samples):
                inp, out, _ = samples[i]
                axes[r, ci].imshow(inp)
                axes[r, co].imshow(out)
            for c in (ci, co):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
                for s in axes[r, c].spines.values():
                    s.set_linewidth(0.4)
            if r == 0:
                axes[r, ci].set_title("Input", fontsize=6, pad=3)
                axes[r, co].set_title("SWIFT-GAN", fontsize=6, pad=3)
        axes[r, 0].set_ylabel(bench, fontsize=7, rotation=0,
                              ha="right", va="center", labelpad=6)

    plt.subplots_adjust(wspace=0.03, hspace=0.05,
                        left=0.14, right=0.99, top=0.95, bottom=0.01)
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"[INFO] saved: {out_path} (+ .pdf)")


def dump_all(net, files, data_dir, dump_dir, limit):
    """Write every input/output pair to disk so the best ones can be picked by eye."""
    d = os.path.join(ROOT, dump_dir, data_dir)
    os.makedirs(d, exist_ok=True)
    for i, f in enumerate(files[:limit]):
        t, arr = load_image(f)
        pred = translate(net, t)
        side = np.concatenate([arr, pred], axis=1)
        Image.fromarray((side * 255).astype(np.uint8)).save(
            os.path.join(d, f"{i:03d}_{os.path.basename(f)}"))
    print(f"[INFO]   dumped {min(limit, len(files))} pairs -> {d}")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-set", type=int, default=3)
    ap.add_argument("--out", type=str, default="benchmark_qualitative.png")
    ap.add_argument("--mode", choices=["spread", "change"], default="spread",
                    help="spread = evenly sampled (unbiased); "
                         "change = most-translated (cherry-picked)")
    ap.add_argument("--dump-all", action="store_true",
                    help="write all input/output pairs instead of a figure")
    ap.add_argument("--dump-dir", type=str, default="bench_dumps")
    ap.add_argument("--dump-limit", type=int, default=40)
    ap.add_argument("--picks", type=str, default=None,
                    help='Hand-picked files per dataset, groups separated by ";", '
                         'files by ",". Example: '
                         '"cityscapes:023_a.png,025_b.png;maps:006_c.jpg". '
                         'Filenames may contain spaces and may keep the NNN_ '
                         'prefix added by --dump-all.')
    args = ap.parse_args()

    print(f"[INFO] device: {DEVICE}")

    picks = {}
    if args.picks:
        # groups are separated by ";" (filenames may contain spaces)
        for group in args.picks.split(";"):
            if ":" not in group:
                continue
            key, files_str = group.split(":", 1)
            picks[key.strip()] = [f for f in files_str.split(",") if f.strip()]
        print(f"[INFO] hand-picked sets: {list(picks.keys())}")

    rows = []
    for (name, data_dir, ckpt_dir, swap) in BENCHMARKS:
        print(f"[INFO] === {name} ===")
        try:
            ckpt = find_checkpoint(ckpt_dir)
            print(f"[INFO]   ckpt: {os.path.basename(ckpt)}")
            files, split = list_test_images(data_dir, swap)
            print(f"[INFO]   inputs: {len(files)} from {split}")
            net = load_swift_generator(ckpt)

            if args.dump_all:
                dump_all(net, files, data_dir, args.dump_dir, args.dump_limit)
            elif data_dir in picks:
                rows.append((name, pick_by_names(net, files, picks[data_dir])))
            else:
                rows.append((name, pick_samples(net, files, args.n_per_set, args.mode)))
            del net
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[WARN] {name} skipped: {e}")

    if args.dump_all:
        print("[INFO] dump complete - inspect the pairs, then rerun without --dump-all")
        return
    if not rows:
        raise RuntimeError("no benchmark produced output")
    build_figure(rows, args.out, args.n_per_set)


if __name__ == "__main__":
    main()