#!/usr/bin/env python3
"""Multi-seed stress evaluation for P02 ResNet-18 HAR.

Evaluates 3 trained checkpoints (seeds 42, 43, 44) on the SAME stress test
sets, computes mean +/- std, saves JSON results, and regenerates sweep figures
with error bars.

EVAL-ONLY: no retraining, no data regeneration.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
RESULTS_ROOT = _HERE.parents[1]
_BASE = _HERE.parents[2]
REPO_ROOT = next((_BASE / _c for _c in ("_repo", "repo") if (_BASE / _c).is_dir()), _BASE / "_repo")
STRESS_DIR = RESULTS_ROOT / "stress"
FIGURES_DIR = RESULTS_ROOT / "figures"

CHECKPOINTS = {
    "s42": RESULTS_ROOT / "artifacts" / "resnet18_s42" / "best_model.pt",
    "s43": RESULTS_ROOT / "artifacts" / "resnet18_s43" / "best_model.pt",
    "s44": RESULTS_ROOT / "artifacts" / "resnet18_s44" / "best_model.pt",
}

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "projects" / "p02_resnet18_har"))

from model import make_har_model
from shared.micro_doppler import ACTIVITY_LABELS, N_CLASSES
from common.hdf5_io import load_hdf5

CLASS_NAMES = list(ACTIVITY_LABELS)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: str) -> nn.Module:
    model = make_har_model("resnet18", n_classes=N_CLASSES).to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_test_dataset(h5_path: Path) -> TensorDataset:
    data = load_hdf5(str(h5_path), ["x", "y"])
    x = torch.as_tensor(data["x"], dtype=torch.float32)
    y = torch.as_tensor(data["y"], dtype=torch.long)
    return TensorDataset(x, y)


@torch.no_grad()
def evaluate(model: nn.Module, dataset: TensorDataset, device: str):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_preds, all_labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    acc = float(np.mean(preds == labels))
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        if mask.sum() > 0:
            per_class[name] = float(np.mean(preds[mask] == labels[mask]))
    return acc, per_class


# ---------------------------------------------------------------------------
# SNR sweep evaluation
# ---------------------------------------------------------------------------

def run_snr_multiseed(device: str):
    print("=" * 60)
    print("SNR SWEEP - Multi-seed evaluation")
    print("=" * 60)

    snr_values = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
    results = []

    for i, snr in enumerate(snr_values):
        h5_path = STRESS_DIR / f"snr_{snr:+d}dB.h5"
        if not h5_path.exists():
            print(f"  WARNING: {h5_path} not found, skipping")
            continue

        ds = load_test_dataset(h5_path)
        seed_accs = {}
        seed_per_class = {}

        for seed_name, ckpt_path in CHECKPOINTS.items():
            model = load_model(ckpt_path, device)
            acc, pc = evaluate(model, ds, device)
            seed_accs[seed_name] = acc
            seed_per_class[seed_name] = pc
            del model
            torch.cuda.empty_cache()

        accs = list(seed_accs.values())
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=0))

        # Per-class mean/std
        per_class_mean = {}
        per_class_std = {}
        for cls in CLASS_NAMES:
            vals = [seed_per_class[s].get(cls, 0.0) for s in CHECKPOINTS]
            per_class_mean[cls] = float(np.mean(vals))
            per_class_std[cls] = float(np.std(vals, ddof=0))

        entry = {
            "snr_db": snr,
            "data_seed": 7000 + i,
            "per_model_seed": seed_accs,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "per_class_mean": per_class_mean,
            "per_class_std": per_class_std,
            "per_model_per_class": {s: seed_per_class[s] for s in CHECKPOINTS},
        }
        results.append(entry)

        print(f"  SNR={snr:+3d} dB | "
              f"s42={seed_accs['s42']:.4f}  s43={seed_accs['s43']:.4f}  s44={seed_accs['s44']:.4f} | "
              f"mean={mean_acc:.4f} +/- {std_acc:.4f}")

    out_path = STRESS_DIR / "snr_sweep_multiseed.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


# ---------------------------------------------------------------------------
# Aspect sweep evaluation
# ---------------------------------------------------------------------------

def run_aspect_multiseed(device: str):
    print("\n" + "=" * 60)
    print("ASPECT SWEEP - Multi-seed evaluation")
    print("=" * 60)

    aspect_bands = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90)]
    results = []

    for i, (alo, ahi) in enumerate(aspect_bands):
        h5_path = STRESS_DIR / f"aspect_{alo}-{ahi}deg.h5"
        if not h5_path.exists():
            print(f"  WARNING: {h5_path} not found, skipping")
            continue

        ds = load_test_dataset(h5_path)
        seed_accs = {}
        seed_per_class = {}

        for seed_name, ckpt_path in CHECKPOINTS.items():
            model = load_model(ckpt_path, device)
            acc, pc = evaluate(model, ds, device)
            seed_accs[seed_name] = acc
            seed_per_class[seed_name] = pc
            del model
            torch.cuda.empty_cache()

        accs = list(seed_accs.values())
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=0))

        per_class_mean = {}
        per_class_std = {}
        for cls in CLASS_NAMES:
            vals = [seed_per_class[s].get(cls, 0.0) for s in CHECKPOINTS]
            per_class_mean[cls] = float(np.mean(vals))
            per_class_std[cls] = float(np.std(vals, ddof=0))

        entry = {
            "aspect_band": f"{alo}-{ahi}",
            "aspect_lo": alo,
            "aspect_hi": ahi,
            "data_seed": 8000 + i,
            "per_model_seed": seed_accs,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "per_class_mean": per_class_mean,
            "per_class_std": per_class_std,
            "per_model_per_class": {s: seed_per_class[s] for s in CHECKPOINTS},
        }
        results.append(entry)

        print(f"  Aspect {alo:2d}-{ahi:2d} deg | "
              f"s42={seed_accs['s42']:.4f}  s43={seed_accs['s43']:.4f}  s44={seed_accs['s44']:.4f} | "
              f"mean={mean_acc:.4f} +/- {std_acc:.4f}")

    out_path = STRESS_DIR / "aspect_sweep_multiseed.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(snr_results, aspect_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: SNR sweep with error bars ---
    print("\n--- Figure: SNR sweep (multi-seed) ---")
    snrs = [r["snr_db"] for r in snr_results]
    means = [r["mean_accuracy"] * 100 for r in snr_results]
    stds = [r["std_accuracy"] * 100 for r in snr_results]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Shade training band
    ax.axvspan(5, 25, alpha=0.12, color="green", label="Training SNR band (5-25 dB)")

    # Mean line with error bars
    ax.errorbar(snrs, means, yerr=stds, fmt="o-", color="#1f77b4", linewidth=2,
                markersize=7, capsize=4, capthick=1.5, ecolor="#1f77b4", elinewidth=1.5,
                label="Mean accuracy (3 seeds)")

    # Shaded band for +/- std
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(snrs, means_arr - stds_arr, np.minimum(means_arr + stds_arr, 105),
                     alpha=0.15, color="#1f77b4")

    # Chance line
    chance = 100 / N_CLASSES
    ax.axhline(chance, color="red", linestyle="--", linewidth=1,
               label=f"Chance ({chance:.1f}%)")

    # Transition zone annotation (sub-0-dB region is seed-variable)
    ax.annotate(
        "Transition zone:\nhigh seed variance",
        xy=(-5, means[snrs.index(-5)]),
        xytext=(-16, 65),
        fontsize=9, fontweight="bold", color="#333",
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
    )

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("ResNet-18 HAR: Accuracy vs. SNR (3-seed mean +/- std)", fontsize=13)
    ax.set_ylim(-2, 108)
    ax.set_xlim(min(snrs) - 2, max(snrs) + 2)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # Value annotations
    for s, m, sd in zip(snrs, means, stds):
        if sd > 0.05:
            label = f"{m:.1f}+/-{sd:.1f}"
        else:
            label = f"{m:.1f}"
        ax.annotate(label, (s, m), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=7.5)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(str(FIGURES_DIR / f"snr_sweep.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved: snr_sweep.png/pdf -> {FIGURES_DIR}")

    # --- Figure 2: Aspect sweep with error bars ---
    print("--- Figure: Aspect sweep (multi-seed) ---")
    band_labels = [f"{r['aspect_lo']}-{r['aspect_hi']}" for r in aspect_results]
    aspect_means = [r["mean_accuracy"] * 100 for r in aspect_results]
    aspect_stds = [r["std_accuracy"] * 100 for r in aspect_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(band_labels))

    # Colors: green for in-training (<=60), red for OOD (>60)
    colors = ["#2ca02c" if r["aspect_hi"] <= 60 else "#d62728" for r in aspect_results]

    bars = ax.bar(x_pos, aspect_means, color=colors, edgecolor="black", linewidth=0.5,
                  yerr=aspect_stds, capsize=5, error_kw=dict(lw=1.5, capthick=1.5, color="black"))

    # Chance line
    ax.axhline(chance, color="red", linestyle="--", linewidth=1,
               label=f"Chance ({chance:.1f}%)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{b} deg" for b in band_labels], fontsize=10)
    ax.set_xlabel("Aspect Angle Band", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("ResNet-18 HAR: Accuracy vs. Aspect Angle (3-seed mean +/- std, SNR=15 dB)",
                 fontsize=12)
    ax.set_ylim(0, 115)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", edgecolor="black", label="In-distribution (<=60 deg)"),
        Patch(facecolor="#d62728", edgecolor="black", label="Out-of-distribution (>60 deg)"),
        plt.Line2D([0], [0], color="red", linestyle="--", label=f"Chance ({chance:.1f}%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.3)

    for i, (m, sd) in enumerate(zip(aspect_means, aspect_stds)):
        if sd > 0.05:
            label = f"{m:.1f}+/-{sd:.1f}"
        else:
            label = f"{m:.1f}"
        ax.annotate(label, (i, m + sd + 1.5), ha="center", fontsize=8.5)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(str(FIGURES_DIR / f"aspect_sweep.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved: aspect_sweep.png/pdf -> {FIGURES_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoints: {list(CHECKPOINTS.keys())}")

    snr_results = run_snr_multiseed(device)
    aspect_results = run_aspect_multiseed(device)
    make_figures(snr_results, aspect_results)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nSNR Sweep (mean +/- std across 3 seeds):")
    print(f"  {'SNR':>6}  {'s42':>8}  {'s43':>8}  {'s44':>8}  {'mean':>8}  {'std':>8}")
    for r in snr_results:
        ps = r["per_model_seed"]
        print(f"  {r['snr_db']:>+4d} dB  "
              f"{ps['s42']*100:>7.2f}%  {ps['s43']*100:>7.2f}%  {ps['s44']*100:>7.2f}%  "
              f"{r['mean_accuracy']*100:>7.2f}%  {r['std_accuracy']*100:>7.3f}%")

    print("\nAspect Sweep (mean +/- std across 3 seeds):")
    print(f"  {'Band':>8}  {'s42':>8}  {'s43':>8}  {'s44':>8}  {'mean':>8}  {'std':>8}")
    for r in aspect_results:
        ps = r["per_model_seed"]
        print(f"  {r['aspect_band']:>6} deg  "
              f"{ps['s42']*100:>7.2f}%  {ps['s43']*100:>7.2f}%  {ps['s44']*100:>7.2f}%  "
              f"{r['mean_accuracy']*100:>7.2f}%  {r['std_accuracy']*100:>7.3f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
