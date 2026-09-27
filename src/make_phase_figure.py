#!/usr/bin/env python3
"""
Direct evidence that phase-locking works.

Two outputs:

  (1) A figure, per sample row:
        input | SWIFT-GAN output | input phase | output phase | phase error | amplitude change

  (2) Two aggregate numbers whose CONTRAST is the whole argument:
        mean |phase error|          -> should be SMALL  (geometry held)
        mean |log-amplitude change| -> should be LARGE  (texture rewritten)

      An unconstrained baseline (CycleGAN) is measured on the same patches so the
      phase-error number has a reference point.

Preprocessing is copied verbatim from make_qualitative_figure.py / inference.py
(2/98-percentile stretch per polarisation, ratio = clip(vh_n/vv_n, 0, 1), stacked
as (VV, VH, ratio) and mapped to [-1,1]) and it reads the same scene, so this
figure is directly comparable to the qualitative comparison figure.

Run from .../sar_to_optical/swift_gan_src :
    python make_phase_figure.py --n-samples 3 --n-eval 60 --out phase_analysis.png
"""
import os, sys, glob, argparse, importlib
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SAR_ROOT = os.path.dirname(ROOT)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same scene as the qualitative comparison figure, so the two stay consistent.
VV_PATH = "/scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/data/Sentinel-1/22588247ff6d53170dfec01c75255b58/S1A_IW_GRDH_1SDV_20240104T043534_20240104T043559_051952_0646F6_FEFD_VV.tif"
VH_PATH = VV_PATH.replace("_VV.tif", "_VH.tif")

PATCH_SIZE = 256
STRIDE = 2048


# ---------- preprocessing (verbatim from make_qualitative_figure.py) ----------
def _normalize_s1(data):
    valid = data[~np.isnan(data)]
    if valid.size == 0 or np.max(valid) == 0:
        return np.zeros_like(data)
    p2, p98 = np.percentile(valid, (2, 98))
    stretched = np.clip(data, p2, p98)
    if p98 - p2 == 0:
        return np.zeros_like(data)
    return (stretched - p2) / (p98 - p2)


def preprocess_dual_pol_patch(vv, vh):
    vv_n, vh_n = _normalize_s1(vv), _normalize_s1(vh)
    ratio = np.clip(np.divide(vh_n, vv_n + 1e-6), 0, 1)
    t = torch.from_numpy(np.stack([vv_n, vh_n, ratio], 0)).float()
    return ((t - 0.5) / 0.5).unsqueeze(0)


def denorm(t):
    img = t.cpu().detach().numpy() * 0.5 + 0.5
    return np.transpose(np.clip(img, 0, 1), (1, 2, 0))


# ---------- patches (verbatim logic from make_qualitative_figure.py) ----------
def select_patches(n):
    """Most-textured valid patches, spread across the scene."""
    cands = []
    with rasterio.open(VV_PATH) as s:
        W, H = s.width, s.height
        print(f"[INFO] SAR scene: {W}x{H}")
        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                p = s.read(1, window=Window(x, y, PATCH_SIZE, PATCH_SIZE))
                if np.max(p) < 1e-4:
                    continue
                cands.append((float(np.std(p)), x, y))
    if not cands:
        raise RuntimeError("no valid SAR patches")
    cands.sort(reverse=True)
    step = max(1, len(cands) // (n * 4))
    coords = [(c[1], c[2]) for c in cands[::step]][:n]
    print(f"[INFO] selected {len(coords)} patches (of {len(cands)} valid)")
    return coords


def read_sar(coords):
    out = []
    with rasterio.open(VV_PATH) as sv, rasterio.open(VH_PATH) as sh:
        for (x, y) in coords:
            w = Window(x, y, PATCH_SIZE, PATCH_SIZE)
            out.append(preprocess_dual_pol_patch(sv.read(1, window=w),
                                                 sh.read(1, window=w)))
    return out


# ---------- models ----------
def load_generator(src_dir, module_file, class_name, ckpt_dir):
    """Each competitor ships its own model*.py importing as 'model'; isolate them."""
    src_path = os.path.join(SAR_ROOT, src_dir)
    for name in list(sys.modules):
        if name == "model" or name.startswith("model_"):
            del sys.modules[name]
    sys.path.insert(0, src_path)
    try:
        mod = importlib.import_module(module_file)
        net = getattr(mod, class_name)(input_nc=3, output_nc=3).to(DEVICE)
        d = os.path.join(src_path, ckpt_dir)
        hits = (glob.glob(os.path.join(d, "*AB_best.pth"))
                or glob.glob(os.path.join(d, "*best*.pth"))
                or sorted(glob.glob(os.path.join(d, "netG_AB_ep*.pth"))))
        hits = [h for h in hits if "netD" not in os.path.basename(h)
                and "netF" not in os.path.basename(h)
                and "netG_BA" not in os.path.basename(h)]
        if not hits:
            raise FileNotFoundError(f"no generator checkpoint in {d}")
        ck = hits[-1]
        state = torch.load(ck, map_location=DEVICE)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        net.load_state_dict(state, strict=False)
        net.eval()
        return net, os.path.basename(ck)
    finally:
        if src_path in sys.path:
            sys.path.remove(src_path)


# ---------- spectral analysis ----------
def spectra(t):
    """Grayscale 2D FFT -> (phase, log-amplitude), fftshifted."""
    gray = t[0].mean(0)
    F = torch.fft.fftshift(torch.fft.fft2(gray))
    return torch.angle(F).cpu().numpy(), torch.log1p(torch.abs(F)).cpu().numpy()


def wrap(d):
    """Angular difference wrapped to [-pi, pi]; raw differences are meaningless."""
    return (d + np.pi) % (2 * np.pi) - np.pi


@torch.no_grad()
def measure(net, inputs):
    """Mean |phase error| (rad) and mean |log-amplitude change| over inputs."""
    pe, ae = [], []
    for x in inputs:
        x = x.to(DEVICE)
        y = net(x)
        px, ax_ = spectra(x)
        py, ay = spectra(y)
        pe.append(np.abs(wrap(py - px)).mean())
        ae.append(np.abs(ay - ax_).mean())
    return float(np.mean(pe)), float(np.mean(ae)), len(pe)


# ---------- figure ----------
@torch.no_grad()
def build_figure(net, inputs, out_path):
    cols = ["Input SAR", "SWIFT-GAN", r"Phase $\Phi(x)$", r"Phase $\Phi(G(x))$",
            "Phase error", "Amplitude change"]
    n = len(inputs)
    fig, axes = plt.subplots(n, 6, figsize=(15, 2.6 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for r, x in enumerate(inputs):
        x = x.to(DEVICE)
        y = net(x)
        px, ax_ = spectra(x)
        py, ay = spectra(y)
        perr = np.abs(wrap(py - px))
        adiff = ay - ax_
        vmax = float(np.abs(adiff).max())

        panels = [
            (denorm(x[0]), None, None),
            (denorm(y[0]), None, None),
            (px, "twilight", "phase"),
            (py, "twilight", "phase"),
            (perr, "inferno", "err"),
            (adiff, "coolwarm", "amp"),
        ]
        for c, (data, cmap, kind) in enumerate(panels):
            a = axes[r, c]
            if cmap is None:
                a.imshow(data)
            elif kind == "phase":
                a.imshow(data, cmap=cmap, vmin=-np.pi, vmax=np.pi)
            elif kind == "err":
                a.imshow(data, cmap=cmap, vmin=0, vmax=np.pi)
            else:
                a.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(cols[c], fontsize=9)

        axes[r, 4].set_xlabel(f"mean $|\\Delta\\Phi|$ = {perr.mean():.3f} rad", fontsize=7)
        axes[r, 5].set_xlabel(f"mean $|\\Delta \\log A|$ = {np.abs(adiff).mean():.3f}", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"[INFO] saved: {out_path} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=3, help="rows in the figure")
    ap.add_argument("--n-eval", type=int, default=60,
                    help="patches used for the aggregate statistics")
    ap.add_argument("--out", type=str, default="phase_analysis.png")
    args = ap.parse_args()

    print(f"[INFO] device: {DEVICE}")

    # One scan; the figure uses the first n_samples of the same patch pool.
    coords = select_patches(max(args.n_samples, args.n_eval))
    inputs = read_sar(coords)
    print(f"[INFO] loaded {len(inputs)} patches")

    net, ck = load_generator("swift_gan_src", "model", "SwiftGenerator",
                             "checkpoints_swift_gan_v2")
    print(f"[INFO] SWIFT-GAN checkpoint: {ck}\n")

    build_figure(net, inputs[:args.n_samples], args.out)

    print(f"\n[INFO] measuring spectra over {len(inputs)} patches ...")
    pe, ae, n = measure(net, inputs)
    print(f"\n=== SWIFT-GAN (n={n}) ===")
    print(f"  mean |phase error|          : {pe:.4f} rad  (uniform-random would be ~{np.pi/2:.3f})")
    print(f"  mean |log-amplitude change| : {ae:.4f}")
    print(f"  amplitude/phase ratio       : {ae/pe:.2f}")

    del net
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    try:
        base, bck = load_generator("cyclegan_baseline_src", "model",
                                   "ResnetGenerator", "checkpoints_cyclegan")
        bpe, bae, bn = measure(base, inputs)
        print(f"\n=== CycleGAN (n={bn}, ckpt {bck}) ===")
        print(f"  mean |phase error|          : {bpe:.4f} rad")
        print(f"  mean |log-amplitude change| : {bae:.4f}")
        print(f"\n[RESULT] SWIFT-GAN phase error is {bpe/pe:.2f}x lower than CycleGAN")
    except Exception as e:
        print(f"\n[WARN] baseline comparison skipped: {e}")


if __name__ == "__main__":
    main()