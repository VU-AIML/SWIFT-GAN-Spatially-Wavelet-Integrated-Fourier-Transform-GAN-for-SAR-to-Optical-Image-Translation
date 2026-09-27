#!/usr/bin/env python3
"""
Inference-speed benchmark: single-pass SWIFT-GAN vs iterative baselines.

Reports mean +/- std wall-clock time per 256x256 image, plus parameter counts.
Every model is loaded in isolation (each competitor ships its own model*.py that
imports as "model"), timed with proper CUDA synchronisation and warm-up.

Run from .../sar_to_optical/swift_gan_src :
    python measure_inference_speed.py --runs 100 --out inference_speed.tex

The table this prints is meant to back the "single forward pass" claim against
diffusion / Schrodinger-bridge methods that need iterative sampling.
"""
import os, sys, glob, time, argparse, importlib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
SAR_ROOT = os.path.dirname(ROOT)          # .../sar_to_optical
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

# (display name, source dir, module file, class name, checkpoint dir, iterative?)
MODELS = [
    ("CycleGAN",    "cyclegan_baseline_src", "model",      "Generator",      "checkpoints_cyclegan",   True),
    ("Cycle-CBAM",  "cyclegan_cbam_src",     "model",      "Generator",      "checkpoints_cbam",       True),
    ("CUT",         "cut_src",               "model",      "ResnetGenerator", "checkpoints_cut",       True),
    ("SwinCUT",     "swincut_src",           "model",      "SwinGenerator",  "checkpoints_swincut",    True),
    ("QS-Attn",     "qs_attn_src",           "model",      "QSGenerator",    "checkpoints_qs_attn",    True),
    ("ITTR",        "ittr_unsb_src",         "model_ittr", "ITTRGenerator",  "checkpoints_ittr",       True),
    ("UNSB",        "ittr_unsb_src",         "model_unsb", "UNSBGenerator",  "checkpoints_unsb",       True),
    ("SWIFT-GAN",   "swift_gan_src",         "model",      "SwiftGenerator", "checkpoints_swift_gan_v2", True),
]

UNSB_NUM_TIMESTEPS = 5      # UNSB samples iteratively over this many steps


def purge_modules():
    """Competitors all expose a module literally named 'model'; drop stale ones."""
    for name in list(sys.modules):
        if name == "model" or name.startswith("model_"):
            del sys.modules[name]


def find_checkpoint(src_dir, ckpt_dir):
    d = os.path.join(SAR_ROOT, src_dir, ckpt_dir)
    if not os.path.isdir(d):
        return None
    for pat in ["*AB_best.pth", "netG_AB_ep*.pth", "netG_A*.pth", "*.pth"]:
        hits = [h for h in sorted(glob.glob(os.path.join(d, pat)))
                if "netD" not in os.path.basename(h)
                and "netF" not in os.path.basename(h)
                and "netG_BA" not in os.path.basename(h)]
        if hits:
            def epoch_of(p):
                b = os.path.basename(p)
                if "best" in b:
                    return -1
                digits = "".join(c for c in b if c.isdigit())
                return int(digits) if digits else 0
            hits.sort(key=epoch_of)
            return hits[-1] if "best" not in os.path.basename(hits[0]) else hits[0]
    return None


def load_model(name, src_dir, module_file, class_name, ckpt_dir):
    src_path = os.path.join(SAR_ROOT, src_dir)
    purge_modules()
    sys.path.insert(0, src_path)
    try:
        mod = importlib.import_module(module_file)
        cls = getattr(mod, class_name)
        net = cls(input_nc=3, output_nc=3).to(DEVICE)
        ckpt = find_checkpoint(src_dir, ckpt_dir)
        if ckpt is not None:
            state = torch.load(ckpt, map_location=DEVICE)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            net.load_state_dict(state, strict=False)
        net.eval()
        return net, ckpt
    finally:
        if src_path in sys.path:
            sys.path.remove(src_path)


@torch.no_grad()
def time_model(net, name, iterative, runs, warmup=10):
    """Wall-clock ms per image, averaged over `runs` timed forward passes."""
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    def forward():
        if name == "UNSB":
            # UNSB denoises over several timesteps; time the full sampling chain
            out = x
            for t in reversed(range(UNSB_NUM_TIMESTEPS)):
                ts = torch.full((1,), t, device=DEVICE, dtype=torch.long)
                out = net(out, ts)
            return out
        return net(x)

    for _ in range(warmup):
        forward()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        forward()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)), float(np.std(times))


def count_params(net):
    return sum(p.numel() for p in net.parameters()) / 1e6


def emit_latex(rows, path):
    """Write a LaTeX table ready to paste into the paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Inference cost for a single $256\times256$ image on one NVIDIA "
        r"V100, averaged over 100 runs after warm-up. SWIFT-GAN enforces "
        r"phase-locking in a single forward pass, whereas methods that place the "
        r"structural constraint inside a stochastic process require iterative "
        r"sampling. Steps denotes the number of network evaluations per image.}",
        r"\label{tab:inference_speed}",
        r"\small",
        r"\begin{tabular}{l|c|c|c}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Steps} & \textbf{Params (M)} & \textbf{Time (ms)} \\",
        r"\midrule",
    ]
    for (name, steps, params, mean, std, _has_ck) in rows:
        tag = r"\textbf{" + name + "}" if name == "SWIFT-GAN" else name
        lines.append(f"{tag} & {steps} & {params:.1f} & {mean:.1f} $\\pm$ {std:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] LaTeX table -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--out", type=str, default="inference_speed.tex")
    args = ap.parse_args()

    if DEVICE.type != "cuda":
        print("[WARN] no GPU visible - timings on CPU are NOT reportable. "
              "Run this inside a GPU job (srun/sbatch).")

    print(f"[INFO] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[INFO] gpu: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] {args.runs} timed runs per model, input 1x3x{IMG_SIZE}x{IMG_SIZE}\n")

    rows = []
    for (name, src, mfile, cls, ckpt_dir, iterative) in MODELS:
        try:
            net, ckpt = load_model(name, src, mfile, cls, ckpt_dir)
            steps = UNSB_NUM_TIMESTEPS if name == "UNSB" else 1
            mean, std = time_model(net, name, iterative, args.runs)
            params = count_params(net)
            rows.append((name, steps, params, mean, std, ckpt is not None))
            ck = os.path.basename(ckpt) if ckpt else "RANDOM INIT - timing valid, weights are not"
            print(f"  {name:12s} steps={steps}  {params:6.1f}M  "
                  f"{mean:7.2f} +/- {std:5.2f} ms   [{ck}]")
            del net
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [WARN] {name} skipped: {e}")

    if not rows:
        raise RuntimeError("no model could be timed")

    print()
    emit_latex(rows, args.out)

    swift = [r for r in rows if r[0] == "SWIFT-GAN"]
    if swift:
        s = swift[0][3]
        print("[INFO] SWIFT-GAN relative cost:")
        for (name, steps, params, mean, std, _has_ck) in rows:
            if name != "SWIFT-GAN":
                print(f"         vs {name:12s}: {mean/s:5.2f}x")


if __name__ == "__main__":
    main()