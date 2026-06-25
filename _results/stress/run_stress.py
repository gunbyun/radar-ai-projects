#!/usr/bin/env python3
"""Stress / robustness evaluation for P02 ResNet-18 HAR model.

Eval-only: loads an existing checkpoint and evaluates on synthetic test sets
generated at controlled SNR and aspect-angle conditions.

Usage (from repo root or anywhere -- paths are absolute):
    python run_stress.py                    # full sweep
    python run_stress.py --snr_only         # SNR sweep only
    python run_stress.py --aspect_only      # aspect sweep only
    python run_stress.py --figures_only      # regenerate figures from saved JSON
"""

import argparse
import json
import sys
import time
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
CHECKPOINT = RESULTS_ROOT / "artifacts" / "resnet18_s42" / "best_model.pt"
STRESS_DIR = RESULTS_ROOT / "stress"
FIGURES_DIR = RESULTS_ROOT / "figures"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "projects" / "p02_resnet18_har"))

from model import make_har_model
from shared.micro_doppler import ACTIVITY_LABELS, N_CLASSES
from common.hdf5_io import load_hdf5
from common.seed import seed_everything

# Re-use the generation function directly from generate_data.py
P02_DIR = REPO_ROOT / "projects" / "p02_resnet18_har"
sys.path.insert(0, str(P02_DIR))
from generate_data import _generate_split, RADAR

CLASS_NAMES = list(ACTIVITY_LABELS)  # ['walk','run','sit_down','fall','wave','idle']
N_PER_CLASS = 200
N_SAMPLES = N_PER_CLASS * N_CLASSES  # 1200

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: str) -> nn.Module:
    model = make_har_model("resnet18", n_classes=N_CLASSES).to(device)
    state = torch.load(str(CHECKPOINT), map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")
    print(f"Device: {device}")
    return model


# ---------------------------------------------------------------------------
# Dataset from HDF5
# ---------------------------------------------------------------------------

def load_test_dataset(h5_path: Path) -> TensorDataset:
    data = load_hdf5(str(h5_path), ["x", "y"])
    x = torch.as_tensor(data["x"], dtype=torch.float32)
    y = torch.as_tensor(data["y"], dtype=torch.long)
    return TensorDataset(x, y)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, dataset: TensorDataset, device: str):
    """Return accuracy, per-class accuracy, confusion matrix, and raw preds/labels."""
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
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(labels, preds):
        confusion[int(t), int(p)] += 1

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        if mask.sum() > 0:
            per_class[name] = float(np.mean(preds[mask] == labels[mask]))

    return {
        "accuracy": acc,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "class_names": CLASS_NAMES,
    }, preds, labels


# ---------------------------------------------------------------------------
# Data generation helper
# ---------------------------------------------------------------------------

def generate_test_set(out_path: Path, snr_lo: float, snr_hi: float,
                      aspect_lo: float, aspect_hi: float, seed: int) -> None:
    """Generate a balanced test HDF5 using the repo's _generate_split."""
    seed_everything(seed)
    _generate_split(
        out_path,
        n_samples=N_SAMPLES,
        seed=seed,
        snr_lo=snr_lo,
        snr_hi=snr_hi,
        aspect_angle_range_deg=(aspect_lo, aspect_hi),
    )


# ---------------------------------------------------------------------------
# Step 1: SNR sweep
# ---------------------------------------------------------------------------

def run_snr_sweep(model, device):
    print("\n" + "=" * 60)
    print("STEP 1: SNR Sweep (aspect 0-60 deg, fixed per training)")
    print("=" * 60)

    snr_values = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
    results = []

    for i, snr in enumerate(snr_values):
        seed = 7000 + i
        h5_path = STRESS_DIR / f"snr_{snr:+d}dB.h5"
        print(f"\n--- SNR = {snr:+d} dB (seed={seed}) ---")

        generate_test_set(h5_path, snr_lo=float(snr), snr_hi=float(snr),
                          aspect_lo=0.0, aspect_hi=60.0, seed=seed)
        ds = load_test_dataset(h5_path)
        res, _, _ = evaluate(model, ds, device)

        entry = {
            "snr_db": snr,
            "seed": seed,
            "accuracy": res["accuracy"],
            "per_class": res["per_class"],
            "confusion_matrix": res["confusion_matrix"],
        }
        results.append(entry)
        print(f"  Accuracy: {res['accuracy']:.4f}")
        for cls, a in res["per_class"].items():
            print(f"    {cls:<14s} {a:.4f}")

    out_path = STRESS_DIR / "snr_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


# ---------------------------------------------------------------------------
# Step 2: Aspect sweep
# ---------------------------------------------------------------------------

def run_aspect_sweep(model, device):
    print("\n" + "=" * 60)
    print("STEP 2: Aspect Sweep (SNR fixed 15 dB)")
    print("=" * 60)

    aspect_bands = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90)]
    results = []

    for i, (alo, ahi) in enumerate(aspect_bands):
        seed = 8000 + i
        h5_path = STRESS_DIR / f"aspect_{alo}-{ahi}deg.h5"
        print(f"\n--- Aspect {alo}-{ahi} deg (seed={seed}) ---")

        generate_test_set(h5_path, snr_lo=15.0, snr_hi=15.0,
                          aspect_lo=float(alo), aspect_hi=float(ahi), seed=seed)
        ds = load_test_dataset(h5_path)
        res, _, _ = evaluate(model, ds, device)

        entry = {
            "aspect_band": f"{alo}-{ahi}",
            "aspect_lo": alo,
            "aspect_hi": ahi,
            "seed": seed,
            "accuracy": res["accuracy"],
            "per_class": res["per_class"],
            "confusion_matrix": res["confusion_matrix"],
        }
        results.append(entry)
        print(f"  Accuracy: {res['accuracy']:.4f}")
        for cls, a in res["per_class"].items():
            print(f"    {cls:<14s} {a:.4f}")

    out_path = STRESS_DIR / "aspect_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


# ---------------------------------------------------------------------------
# Step 3: Failure cases at knee condition
# ---------------------------------------------------------------------------

def find_knee_and_collect_failures(model, device, snr_results, aspect_results):
    """Pick a knee condition and collect misclassified samples."""
    print("\n" + "=" * 60)
    print("STEP 3: Failure Case Collection")
    print("=" * 60)

    # Find SNR knee: accuracy closest to 70-90%
    snr_candidates = [r for r in snr_results if 0.40 <= r["accuracy"] <= 0.92]
    if not snr_candidates:
        snr_candidates = sorted(snr_results, key=lambda r: abs(r["accuracy"] - 0.80))
    snr_knee = min(snr_candidates, key=lambda r: abs(r["accuracy"] - 0.80))

    # Find aspect knee: accuracy closest to 70-90%
    aspect_candidates = [r for r in aspect_results if 0.40 <= r["accuracy"] <= 0.92]
    if not aspect_candidates:
        aspect_candidates = sorted(aspect_results, key=lambda r: abs(r["accuracy"] - 0.80))
    aspect_knee = min(aspect_candidates, key=lambda r: abs(r["accuracy"] - 0.80))

    # Pick the one with accuracy closer to 80%
    if abs(snr_knee["accuracy"] - 0.80) <= abs(aspect_knee["accuracy"] - 0.80):
        knee = snr_knee
        knee_type = "snr"
        knee_label = f"SNR={knee['snr_db']:+d} dB"
        h5_path = STRESS_DIR / f"snr_{knee['snr_db']:+d}dB.h5"
    else:
        knee = aspect_knee
        knee_type = "aspect"
        knee_label = f"Aspect {knee['aspect_band']} deg"
        h5_path = STRESS_DIR / f"aspect_{knee['aspect_band']}deg.h5"

    print(f"  Knee condition: {knee_label} (accuracy={knee['accuracy']:.4f})")

    # Re-evaluate to get per-sample predictions + spectrograms
    data = load_hdf5(str(h5_path), ["x", "y"])
    x_np = data["x"]  # (N, 1, H, W)
    y_np = data["y"]
    ds = TensorDataset(
        torch.as_tensor(x_np, dtype=torch.float32),
        torch.as_tensor(y_np, dtype=torch.long),
    )
    _, preds, labels = evaluate(model, ds, device)

    # Collect misclassified
    misclassified_idx = np.where(preds != labels)[0]
    correct_idx = np.where(preds == labels)[0]
    print(f"  Total misclassified: {len(misclassified_idx)} / {len(labels)}")

    # Pick up to 10 diverse misclassified (spread across class pairs)
    n_fail = min(10, len(misclassified_idx))
    rng = np.random.default_rng(999)
    if len(misclassified_idx) > n_fail:
        chosen_fail = rng.choice(misclassified_idx, size=n_fail, replace=False)
    else:
        chosen_fail = misclassified_idx

    # Pick 4 correct for contrast
    n_correct = min(4, len(correct_idx))
    chosen_correct = rng.choice(correct_idx, size=n_correct, replace=False)

    failure_data = {
        "knee_type": knee_type,
        "knee_label": knee_label,
        "knee_accuracy": knee["accuracy"],
        "knee_confusion_matrix": knee["confusion_matrix"],
        "knee_per_class": knee["per_class"],
        "misclassified_samples": [],
        "correct_samples": [],
    }

    for idx in chosen_fail:
        failure_data["misclassified_samples"].append({
            "index": int(idx),
            "true_label": CLASS_NAMES[int(labels[idx])],
            "pred_label": CLASS_NAMES[int(preds[idx])],
            "true_idx": int(labels[idx]),
            "pred_idx": int(preds[idx]),
        })
    for idx in chosen_correct:
        failure_data["correct_samples"].append({
            "index": int(idx),
            "true_label": CLASS_NAMES[int(labels[idx])],
            "pred_label": CLASS_NAMES[int(preds[idx])],
            "true_idx": int(labels[idx]),
            "pred_idx": int(preds[idx]),
        })

    # Save spectrograms for figure generation
    spec_fail = x_np[chosen_fail, 0, :, :]  # (n_fail, H, W)
    spec_correct = x_np[chosen_correct, 0, :, :]
    np.savez(
        str(STRESS_DIR / "failure_spectrograms.npz"),
        spec_fail=spec_fail,
        spec_correct=spec_correct,
        fail_info=json.dumps(failure_data["misclassified_samples"]),
        correct_info=json.dumps(failure_data["correct_samples"]),
    )

    out_path = STRESS_DIR / "failure_cases.json"
    with open(out_path, "w") as f:
        json.dump(failure_data, f, indent=2)
    print(f"  Saved: {out_path}")
    return failure_data


# ---------------------------------------------------------------------------
# Step 4: Figures
# ---------------------------------------------------------------------------

def make_figures(snr_results=None, aspect_results=None, failure_data=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load from JSON if not provided
    if snr_results is None:
        with open(STRESS_DIR / "snr_sweep.json") as f:
            snr_results = json.load(f)
    if aspect_results is None:
        with open(STRESS_DIR / "aspect_sweep.json") as f:
            aspect_results = json.load(f)
    if failure_data is None:
        with open(STRESS_DIR / "failure_cases.json") as f:
            failure_data = json.load(f)

    # --- Figure 1: SNR sweep ---
    print("\n--- Figure: SNR sweep ---")
    snrs = [r["snr_db"] for r in snr_results]
    accs = [r["accuracy"] * 100 for r in snr_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snrs, accs, "o-", color="#1f77b4", linewidth=2, markersize=7,
            label="Overall accuracy")
    ax.axhspan(0, 100, xmin=0, xmax=1, alpha=0.0)  # placeholder
    # Shade in-training band
    ax.axvspan(5, 25, alpha=0.15, color="green", label="Training SNR band (5-25 dB)")
    ax.axhline(100 / N_CLASSES, color="red", linestyle="--", linewidth=1,
               label=f"Chance ({100/N_CLASSES:.1f}%)")
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("ResNet-18 HAR: Accuracy vs. SNR", fontsize=14)
    ax.set_ylim(-2, 105)
    ax.set_xlim(min(snrs) - 2, max(snrs) + 2)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    for s, a in zip(snrs, accs):
        ax.annotate(f"{a:.1f}", (s, a), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(str(FIGURES_DIR / f"snr_sweep.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved: snr_sweep.png/pdf")

    # --- Figure 2: Aspect sweep ---
    print("--- Figure: Aspect sweep ---")
    bands = [r["aspect_band"] for r in aspect_results]
    band_labels = [f"{r['aspect_lo']}-{r['aspect_hi']}" for r in aspect_results]
    aspect_accs = [r["accuracy"] * 100 for r in aspect_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(bands))
    colors = ["#2ca02c" if r["aspect_hi"] <= 60 else "#d62728" for r in aspect_results]
    bars = ax.bar(x_pos, aspect_accs, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(100 / N_CLASSES, color="red", linestyle="--", linewidth=1,
               label=f"Chance ({100/N_CLASSES:.1f}%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{b}°" for b in band_labels], fontsize=10)
    ax.set_xlabel("Aspect Angle Band", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("ResNet-18 HAR: Accuracy vs. Aspect Angle (SNR = 15 dB)", fontsize=14)
    ax.set_ylim(0, 110)

    # Add custom legend entries
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", edgecolor="black", label="In-training (≤60°)"),
        Patch(facecolor="#d62728", edgecolor="black", label="Out-of-distribution (>60°)"),
        plt.Line2D([0], [0], color="red", linestyle="--", label=f"Chance ({100/N_CLASSES:.1f}%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    for i, a in enumerate(aspect_accs):
        ax.annotate(f"{a:.1f}", (i, a), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(str(FIGURES_DIR / f"aspect_sweep.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved: aspect_sweep.png/pdf")

    # --- Figure 3: Stress confusion matrix ---
    print("--- Figure: Stress confusion matrix ---")
    cm = np.array(failure_data["knee_confusion_matrix"])
    # Normalize by row (true label)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "Proportion"})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(
        f"Confusion Matrix @ {failure_data['knee_label']}\n"
        f"(Overall accuracy: {failure_data['knee_accuracy']:.1%})",
        fontsize=13,
    )
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(str(FIGURES_DIR / f"stress_confusion.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved: stress_confusion.png/pdf")

    # --- Figure 4: Failure examples ---
    print("--- Figure: Failure examples ---")
    npz = np.load(str(STRESS_DIR / "failure_spectrograms.npz"), allow_pickle=True)
    spec_fail = npz["spec_fail"]
    spec_correct = npz["spec_correct"]
    fail_info = json.loads(str(npz["fail_info"]))
    correct_info = json.loads(str(npz["correct_info"]))

    n_fail = len(fail_info)
    n_correct = len(correct_info)
    n_total = n_fail + n_correct
    n_cols = min(5, n_total)
    n_rows = (n_total + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    # Plot misclassified first
    for i, info in enumerate(fail_info):
        ax = axes[i]
        im = ax.imshow(spec_fail[i], origin="lower", aspect="auto", cmap="magma")
        ax.set_title(
            f"TRUE: {info['true_label']}\nPRED: {info['pred_label']}",
            fontsize=9, color="red", fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot correct ones
    for j, info in enumerate(correct_info):
        idx = n_fail + j
        if idx >= len(axes):
            break
        ax = axes[idx]
        im = ax.imshow(spec_correct[j], origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(
            f"CORRECT: {info['true_label']}",
            fontsize=9, color="green", fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for k in range(n_total, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        f"Failure Examples @ {failure_data['knee_label']} "
        f"(red = misclassified, green = correct)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(str(FIGURES_DIR / f"failure_examples_real.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: failure_examples_real.png/pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="P02 HAR Stress Test")
    parser.add_argument("--snr_only", action="store_true")
    parser.add_argument("--aspect_only", action="store_true")
    parser.add_argument("--figures_only", action="store_true")
    parser.add_argument("--no_figures", action="store_true")
    args = parser.parse_args()

    STRESS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.figures_only:
        make_figures()
        print("\nDone (figures only).")
        return

    model = load_model(device)

    snr_results = None
    aspect_results = None
    failure_data = None

    if not args.aspect_only:
        snr_results = run_snr_sweep(model, device)

    if not args.snr_only:
        aspect_results = run_aspect_sweep(model, device)

    # Need both for failure case selection
    if snr_results is None:
        with open(STRESS_DIR / "snr_sweep.json") as f:
            snr_results = json.load(f)
    if aspect_results is None:
        with open(STRESS_DIR / "aspect_sweep.json") as f:
            aspect_results = json.load(f)

    failure_data = find_knee_and_collect_failures(
        model, device, snr_results, aspect_results
    )

    if not args.no_figures:
        make_figures(snr_results, aspect_results, failure_data)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nSNR Sweep:")
    for r in snr_results:
        snr_val = r.get("snr_db", "?")
        print(f"  SNR={snr_val:>4} dB  -> Acc={r['accuracy']:.4f}")
    print("\nAspect Sweep:")
    for r in aspect_results:
        print(f"  {r['aspect_band']:>8} deg  -> Acc={r['accuracy']:.4f}")
    print(f"\nKnee: {failure_data['knee_label']} "
          f"(accuracy={failure_data['knee_accuracy']:.4f})")
    print("\nOutputs:")
    print(f"  JSON: {STRESS_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
