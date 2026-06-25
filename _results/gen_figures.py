#!/usr/bin/env python3
"""Generate all P2 result figures from saved artifacts.

Produces 5 figures:
  1. learning_curve.png     -- train/val loss vs epoch (ResNet-18 s42)
  2. confusion_matrix.png   -- seaborn heatmap: ResNet-18 + LogReg side by side
  3. method_comparison.png  -- bar chart with error bars
  4. ablation.png           -- data efficiency: 30K vs 3K
  5. examples.png           -- representative spectrograms per class + LogReg errors
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

# Paths
_HERE = Path(__file__).resolve()
RESULTS = _HERE.parents[0]
_BASE = _HERE.parents[1]
_REPO = next((_BASE / _c for _c in ("_repo", "repo") if (_BASE / _c).is_dir()), _BASE / "_repo")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

P2DIR = _REPO / "projects" / "p02_resnet18_har"
DATA_DIR = P2DIR / "data"

CLASS_NAMES = ["walk", "run", "sit_down", "fall", "wave", "idle"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def load_json(path):
    with open(path) as f:
        return json.load(f)


# =========================================================================
# Figure 1: Learning Curve
# =========================================================================
def fig_learning_curve():
    hist_path = RESULTS / "artifacts" / "resnet18_s42" / "history.json"
    if not hist_path.exists():
        print(f"SKIP learning_curve: {hist_path} not found")
        return
    hist = load_json(hist_path)
    epochs = list(range(1, len(hist["train_loss"]) + 1))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(epochs, hist["train_loss"], "b-", linewidth=2, label="Train loss")
    ax1.plot(epochs, hist["val_loss"], "r-", linewidth=2, label="Val loss")
    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    best_val = min(hist["val_loss"])
    ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6,
                label=f"Best epoch ({best_epoch})")
    ax1.scatter([best_epoch], [best_val], color="red", s=80, zorder=5,
                edgecolors="black")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss (label smoothing 0.1)")
    ax1.legend(loc="upper right")
    ax1.set_title("ResNet-18 Training Curve (seed 42, 30K samples)")
    ax1.grid(alpha=0.3)

    for fmt in ["png", "pdf"]:
        fig.savefig(FIGURES / f"learning_curve.{fmt}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved learning_curve.png/pdf")


# =========================================================================
# Figure 2: Confusion Matrix (ResNet-18 + LogReg side by side)
# =========================================================================
def fig_confusion_matrix():
    resnet_path = RESULTS / "artifacts" / "resnet18_s42" / "eval_results.json"
    logreg_path = RESULTS / "artifacts" / "feature_logreg.json"

    if not resnet_path.exists():
        print(f"SKIP confusion_matrix: {resnet_path} not found")
        return

    resnet_data = load_json(resnet_path)
    cm_resnet = np.array(resnet_data["confusion_matrix"])
    names = resnet_data.get("class_names", CLASS_NAMES)

    # Check if LogReg exists for side-by-side
    has_logreg = logreg_path.exists()

    if has_logreg:
        logreg_data = load_json(logreg_path)
        cm_logreg = np.array(logreg_data.get("test_confusion_matrix",
                             logreg_data.get("eval_confusion_matrix")))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # ResNet-18 (normalized)
        cm_resnet_norm = cm_resnet.astype(float) / np.maximum(
            cm_resnet.sum(axis=1, keepdims=True), 1)
        sns.heatmap(cm_resnet_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=names, yticklabels=names, ax=ax1,
                    vmin=0, vmax=1, cbar_kws={"shrink": 0.8})
        acc_resnet = resnet_data["accuracy"]
        ax1.set_title(f"ResNet-18 (acc={acc_resnet:.1%})")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("True")

        # LogReg (normalized)
        cm_logreg_norm = cm_logreg.astype(float) / np.maximum(
            cm_logreg.sum(axis=1, keepdims=True), 1)
        sns.heatmap(cm_logreg_norm, annot=True, fmt=".2f", cmap="Oranges",
                    xticklabels=names, yticklabels=names, ax=ax2,
                    vmin=0, vmax=1, cbar_kws={"shrink": 0.8})
        acc_logreg = logreg_data.get("test_accuracy",
                     logreg_data.get("eval_accuracy"))
        ax2.set_title(f"LogReg on Handcrafted Features (acc={acc_logreg:.1%})")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")

        fig.suptitle("Test Set Confusion Matrices: Deep vs Classical", fontsize=14)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5.5))
        sns.heatmap(cm_resnet, annot=True, fmt="d", cmap="Blues",
                    xticklabels=names, yticklabels=names, ax=ax1,
                    cbar_kws={"shrink": 0.8})
        ax1.set_title("ResNet-18 Confusion Matrix (counts)")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("True")

    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(FIGURES / f"confusion_matrix.{fmt}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved confusion_matrix.png/pdf")


# =========================================================================
# Figure 3: Method Comparison
# =========================================================================
def fig_method_comparison():
    # Collect ResNet-18 results (3 seeds)
    resnet_accs = []
    for seed in [42, 43, 44]:
        p = RESULTS / "artifacts" / f"resnet18_s{seed}" / "eval_results.json"
        if p.exists():
            resnet_accs.append(load_json(p)["accuracy"])

    # Collect TinyCNN results (3 seeds)
    tiny_accs = []
    for seed in [42, 43, 44]:
        p = RESULTS / "artifacts" / f"tiny_s{seed}" / "eval_results.json"
        if p.exists():
            tiny_accs.append(load_json(p)["accuracy"])

    # Feature baselines
    logreg_acc = None
    svm_acc = None
    logreg_path = RESULTS / "artifacts" / "feature_logreg.json"
    svm_path = RESULTS / "artifacts" / "feature_rbf_svm.json"
    if logreg_path.exists():
        d = load_json(logreg_path)
        logreg_acc = d.get("test_accuracy") or d.get("eval_accuracy")
    if svm_path.exists():
        d = load_json(svm_path)
        svm_acc = d.get("test_accuracy") or d.get("eval_accuracy")

    methods = []
    means = []
    stds = []
    colors = []

    if resnet_accs:
        methods.append(f"ResNet-18\n(n={len(resnet_accs)} seeds)")
        means.append(np.mean(resnet_accs) * 100)
        stds.append(np.std(resnet_accs) * 100)
        colors.append("#2563eb")

    if tiny_accs:
        methods.append(f"TinyCNN\n(n={len(tiny_accs)} seeds)")
        means.append(np.mean(tiny_accs) * 100)
        stds.append(np.std(tiny_accs) * 100)
        colors.append("#f97316")

    if svm_acc is not None:
        methods.append("RBF-SVM\n(10K train)")
        means.append(svm_acc * 100)
        stds.append(0)
        colors.append("#10b981")

    if logreg_acc is not None:
        methods.append("LogReg")
        means.append(logreg_acc * 100)
        stds.append(0)
        colors.append("#8b5cf6")

    if not methods:
        print("SKIP method_comparison: no results found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=6,
                  edgecolor="black", linewidth=0.8, width=0.6)

    for bar, mean, std in zip(bars, means, stds):
        label = f"{mean:.1f}%"
        if std > 0:
            label += f"\n({std:.1f})"
        y_pos = bar.get_height() + max(max(stds) * 0.3, 0.3) + 0.5
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Method Comparison: Test Accuracy (mean +/- std over seeds)")
    ax.set_ylim(90, 102)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(FIGURES / f"method_comparison.{fmt}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved method_comparison.png/pdf")


# =========================================================================
# Figure 4: Ablation (data efficiency: 30K vs 3K)
# =========================================================================
def fig_ablation():
    full_path = RESULTS / "artifacts" / "resnet18_s42" / "eval_results.json"
    abl_path = RESULTS / "artifacts" / "resnet18_n3k_s42" / "eval_results.json"

    if not full_path.exists():
        print("SKIP ablation: no baseline result")
        return

    full_data = load_json(full_path)
    full_acc = full_data["accuracy"] * 100

    if not abl_path.exists():
        print(f"SKIP ablation: {abl_path} not found")
        return

    abl_data = load_json(abl_path)
    abl_acc = abl_data["accuracy"] * 100

    labels = ["30K training\nsamples", "3K training\nsamples"]
    accs = [full_acc, abl_acc]
    colors_bar = ["#2563eb", "#93c5fd"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars = ax1.bar(labels, accs, color=colors_bar, edgecolor="black",
                   linewidth=0.8, width=0.5)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=12,
                fontweight="bold")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Data Efficiency: ResNet-18 (seed 42)")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3)

    # Per-class comparison
    full_pc = full_data["per_class"]
    abl_pc = abl_data["per_class"]
    class_names = list(full_pc.keys())
    x = np.arange(len(class_names))
    width = 0.35
    ax2.bar(x - width/2, [full_pc[c]*100 for c in class_names], width,
            label="30K", color="#2563eb", edgecolor="black", linewidth=0.5)
    ax2.bar(x + width/2, [abl_pc[c]*100 for c in class_names], width,
            label="3K", color="#93c5fd", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=30, ha="right")
    ax2.set_ylabel("Per-Class Accuracy (%)")
    ax2.set_title("Per-Class: 30K vs 3K Training Samples")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(FIGURES / f"ablation.{fmt}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved ablation.png/pdf")


# =========================================================================
# Figure 5: Example Spectrograms (6 classes + LogReg errors)
# =========================================================================
def fig_examples():
    test_h5 = DATA_DIR / "har_test.h5"
    logreg_path = RESULTS / "artifacts" / "feature_logreg.json"

    if not test_h5.exists():
        print("SKIP examples: test data not found")
        return

    # Load spectrograms and labels
    with h5py.File(test_h5, "r") as f:
        X = f["x"][:, 0, :, :]  # (N, 128, 128)
        Y = f["y"][:]

    names = CLASS_NAMES

    # Top row: one representative per class (6 panels)
    fig, axes = plt.subplots(2, 6, figsize=(20, 7))

    # Row 1: Representative correct examples (one per class)
    for cls_idx in range(6):
        ax = axes[0, cls_idx]
        cls_mask = Y == cls_idx
        idx = np.where(cls_mask)[0][0]
        spec = X[idx]
        im = ax.imshow(spec, aspect="auto", origin="lower", cmap="inferno")
        ax.set_title(f"{names[cls_idx]}", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if cls_idx == 0:
            ax.set_ylabel("Representative\n(1 per class)", fontsize=10)

    # Row 2: LogReg misclassifications (if available)
    if logreg_path.exists():
        logreg_data = load_json(logreg_path)
        cm = np.array(logreg_data.get("test_confusion_matrix",
                      logreg_data.get("eval_confusion_matrix")))
        # Find top confused pairs (off-diagonal)
        cm_offdiag = cm.copy()
        np.fill_diagonal(cm_offdiag, 0)

        # Get top 6 misclassification pairs
        flat_idx = np.argsort(cm_offdiag.ravel())[::-1]
        shown = 0
        for fi in flat_idx:
            if shown >= 6:
                break
            true_cls, pred_cls = np.unravel_index(fi, cm.shape)
            count = cm_offdiag[true_cls, pred_cls]
            if count == 0:
                break
            ax = axes[1, shown]
            # Find an actual sample of this class
            cls_mask = Y == true_cls
            idx = np.where(cls_mask)[0][min(shown, np.sum(cls_mask)-1)]
            spec = X[idx]
            im = ax.imshow(spec, aspect="auto", origin="lower", cmap="inferno")
            ax.set_title(
                f"True: {names[true_cls]}\nLogReg pred: {names[pred_cls]}\n({count} errors)",
                fontsize=9, color="red", fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            if shown == 0:
                ax.set_ylabel("LogReg Errors\n(confused pairs)", fontsize=10)
            shown += 1

        # Hide remaining panels
        for j in range(shown, 6):
            axes[1, j].axis("off")
    else:
        for j in range(6):
            axes[1, j].axis("off")
        axes[1, 0].text(0.5, 0.5, "No LogReg results", transform=axes[1,0].transAxes,
                        ha="center", va="center")

    fig.suptitle("Micro-Doppler Spectrograms: Class Examples (top) and LogReg Errors (bottom)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(FIGURES / f"examples.{fmt}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved examples.png/pdf")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("=== Generating P2 Result Figures ===")
    print(f"Output: {FIGURES}")

    fig_learning_curve()
    fig_confusion_matrix()
    fig_method_comparison()
    fig_ablation()
    fig_examples()

    print("\nDone.")
