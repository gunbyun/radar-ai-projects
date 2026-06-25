#!/usr/bin/env python3
"""Collect failure cases and regenerate figures at the real knee conditions.

The automatic knee finder picked SNR=0 dB (99.75%) which is too clean.
The actual stress knees are:
  - Aspect 75-90 deg @ 15 dB: 40.6% accuracy (primary knee)
  - SNR -5 dB @ aspect 0-60: 24.4% accuracy (collapse point)
  - Aspect 60-75 deg @ 15 dB: 97.25% (onset of degradation)

This script:
  1. Re-collects failure cases at aspect 75-90 (the best knee)
  2. Regenerates the stress confusion matrix at aspect 75-90
  3. Regenerates the failure examples figure
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

CLASS_NAMES = list(ACTIVITY_LABELS)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_har_model("resnet18", n_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(str(CHECKPOINT), map_location=device))
    model.eval()
    print(f"Loaded checkpoint, device={device}")

    # Use aspect 75-90 as primary knee
    h5_path = STRESS_DIR / "aspect_75-90deg.h5"
    data = load_hdf5(str(h5_path), ["x", "y"])
    x_np = data["x"]  # (N, 1, H, W)
    y_np = data["y"]

    ds = TensorDataset(
        torch.as_tensor(x_np, dtype=torch.float32),
        torch.as_tensor(y_np, dtype=torch.long),
    )

    # Evaluate
    model.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
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

    print(f"Aspect 75-90 deg: accuracy={acc:.4f}")
    for cls, a in per_class.items():
        print(f"  {cls:<14s} {a:.4f}")

    # Collect misclassified samples
    misclassified_idx = np.where(preds != labels)[0]
    correct_idx = np.where(preds == labels)[0]
    print(f"Total misclassified: {len(misclassified_idx)} / {len(labels)}")

    # Pick 10 diverse misclassified (spread across confusion pairs)
    rng = np.random.default_rng(999)
    # Group by (true, pred) pair
    pair_map = {}
    for idx in misclassified_idx:
        pair = (int(labels[idx]), int(preds[idx]))
        if pair not in pair_map:
            pair_map[pair] = []
        pair_map[pair].append(idx)

    # Sort pairs by count (most confused first) and pick from each
    sorted_pairs = sorted(pair_map.items(), key=lambda x: -len(x[1]))
    chosen_fail = []
    target = 10
    # Round-robin from top pairs
    while len(chosen_fail) < target:
        added = False
        for pair, indices in sorted_pairs:
            remaining = [i for i in indices if i not in chosen_fail]
            if remaining and len(chosen_fail) < target:
                chosen_fail.append(rng.choice(remaining))
                added = True
        if not added:
            break

    chosen_fail = np.array(chosen_fail)

    # Pick 4 correct from different classes
    chosen_correct = []
    for cls_idx in range(min(4, N_CLASSES)):
        cls_correct = correct_idx[labels[correct_idx] == cls_idx]
        if len(cls_correct) > 0:
            chosen_correct.append(rng.choice(cls_correct))
    chosen_correct = np.array(chosen_correct)

    failure_data = {
        "knee_type": "aspect",
        "knee_label": "Aspect 75-90 deg (SNR=15 dB)",
        "knee_accuracy": acc,
        "knee_confusion_matrix": confusion.tolist(),
        "knee_per_class": per_class,
        "n_misclassified": int(len(misclassified_idx)),
        "n_total": int(len(labels)),
        "most_confused_pairs": [
            {
                "true": CLASS_NAMES[pair[0]],
                "pred": CLASS_NAMES[pair[1]],
                "count": len(indices),
            }
            for pair, indices in sorted_pairs[:10]
        ],
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

    # Save spectrograms
    spec_fail = x_np[chosen_fail, 0, :, :]
    spec_correct = x_np[chosen_correct, 0, :, :]
    np.savez(
        str(STRESS_DIR / "failure_spectrograms.npz"),
        spec_fail=spec_fail,
        spec_correct=spec_correct,
        fail_info=json.dumps(failure_data["misclassified_samples"]),
        correct_info=json.dumps(failure_data["correct_samples"]),
    )

    with open(STRESS_DIR / "failure_cases.json", "w") as f:
        json.dump(failure_data, f, indent=2)
    print(f"Saved: failure_cases.json")

    # ---- Regenerate figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns

    # Load sweep results
    with open(STRESS_DIR / "snr_sweep.json") as f:
        snr_results = json.load(f)
    with open(STRESS_DIR / "aspect_sweep.json") as f:
        aspect_results = json.load(f)

    # --- Figure 1: SNR sweep ---
    print("\n--- Figure: SNR sweep ---")
    snrs = [r["snr_db"] for r in snr_results]
    accs = [r["accuracy"] * 100 for r in snr_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snrs, accs, "o-", color="#1f77b4", linewidth=2, markersize=7,
            label="Overall accuracy")
    ax.axvspan(5, 25, alpha=0.15, color="green", label="Training SNR band (5–25 dB)")
    ax.axhline(100 / N_CLASSES, color="red", linestyle="--", linewidth=1,
               label=f"Chance ({100/N_CLASSES:.1f}%)")
    # Mark the cliff
    ax.annotate("Cliff:\n−5 → 0 dB", xy=(-2.5, 62), fontsize=10,
                ha="center", color="#d62728", fontweight="bold")
    ax.annotate("", xy=(0, 99.75), xytext=(-5, 24.4),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=2))
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("ResNet-18 HAR: Accuracy vs. SNR", fontsize=14)
    ax.set_ylim(-2, 108)
    ax.set_xlim(min(snrs) - 2, max(snrs) + 2)
    ax.legend(loc="center right", fontsize=10)
    ax.grid(True, alpha=0.3)
    for s, a in zip(snrs, accs):
        offset_y = -14 if a < 50 else 8
        ax.annotate(f"{a:.1f}%", (s, a), textcoords="offset points",
                    xytext=(0, offset_y), ha="center", fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(str(FIGURES_DIR / f"snr_sweep.{ext}"), dpi=300)
    plt.close(fig)
    print("  Saved: snr_sweep.png/pdf")

    # --- Figure 2: Aspect sweep ---
    print("--- Figure: Aspect sweep ---")
    bands = [r["aspect_band"] for r in aspect_results]
    band_labels = [f"{r['aspect_lo']}–{r['aspect_hi']}" for r in aspect_results]
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
    ax.set_ylim(0, 115)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", edgecolor="black", label="In-training (≤60°)"),
        Patch(facecolor="#d62728", edgecolor="black", label="Out-of-distribution (>60°)"),
        plt.Line2D([0], [0], color="red", linestyle="--", label=f"Chance ({100/N_CLASSES:.1f}%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    for i, a in enumerate(aspect_accs):
        ax.annotate(f"{a:.1f}%", (i, a), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(str(FIGURES_DIR / f"aspect_sweep.{ext}"), dpi=300)
    plt.close(fig)
    print("  Saved: aspect_sweep.png/pdf")

    # --- Figure 3: Stress confusion matrix at aspect 75-90 ---
    print("--- Figure: Stress confusion matrix (aspect 75-90) ---")
    cm = np.array(failure_data["knee_confusion_matrix"])
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
        f"Confusion Matrix @ Aspect 75–90°, SNR=15 dB\n"
        f"(Overall accuracy: {failure_data['knee_accuracy']:.1%})",
        fontsize=13,
    )
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(str(FIGURES_DIR / f"stress_confusion.{ext}"), dpi=300)
    plt.close(fig)
    print("  Saved: stress_confusion.png/pdf")

    # --- Figure 4: Failure examples ---
    print("--- Figure: Failure examples (aspect 75-90) ---")
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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, info in enumerate(fail_info):
        ax = axes[i]
        ax.imshow(spec_fail[i], origin="lower", aspect="auto", cmap="magma")
        ax.set_title(
            f"TRUE: {info['true_label']}\nPRED: {info['pred_label']}",
            fontsize=9, color="red", fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for j, info in enumerate(correct_info):
        idx = n_fail + j
        if idx >= len(axes):
            break
        ax = axes[idx]
        ax.imshow(spec_correct[j], origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(
            f"CORRECT: {info['true_label']}",
            fontsize=9, color="green", fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for k in range(n_total, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        "Failure Examples @ Aspect 75–90° (red=misclassified, green=correct)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(str(FIGURES_DIR / f"failure_examples_real.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: failure_examples_real.png/pdf")

    # Print most confused pairs
    print("\nMost confused pairs:")
    for p in failure_data["most_confused_pairs"][:8]:
        print(f"  {p['true']:>10s} -> {p['pred']:<10s}  count={p['count']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
