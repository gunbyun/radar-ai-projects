#!/usr/bin/env python3
"""Data-integrity audit for P02 Micro-Doppler HAR (Appendix A numbers).

Recomputes, from the generated HDF5 splits, the audit statistics reported in the
W14 report Appendix A:
  - per-class mean SNR / aspect / range spread (max-min across the 6 classes)
  - minimum Doppler aliasing margin
  - cross-split minimum L2 distance (test-train, test-val) on the spectrograms

Eval-only. Run from anywhere (paths resolve relative to this file).
    python audit_integrity.py
"""
import sys
from pathlib import Path

import numpy as np
import h5py

HERE = Path(__file__).resolve()
DATA = HERE.parent / "data"
CLASS_NAMES = ["walk", "run", "sit_down", "fall", "wave", "idle"]


def load(split):
    with h5py.File(DATA / f"har_{split}.h5", "r") as f:
        return {
            "x": f["x"][:].reshape(f["x"].shape[0], -1).astype(np.float32),
            "y": f["y"][:],
            "snr": f["snr_db"][:],
            "aspect": f["aspect_angle_deg"][:],
            "range": f["target_range_m"][:],
            "alias": f["doppler_alias_margin_mps"][:],
        }


def class_spread(d, key):
    means = [float(d[key][d["y"] == c].mean()) for c in range(6)]
    return max(means) - min(means), means


def min_cross_l2(a, b, batch=200):
    """min over i of min over j of ||a_i - b_j||_2 ."""
    bn = (b * b).sum(1)
    an = (a * a).sum(1)
    gmin = np.inf
    for s in range(0, a.shape[0], batch):
        ab = a[s:s + batch]
        d2 = an[s:s + batch, None] + bn[None, :] - 2.0 * (ab @ b.T)
        np.maximum(d2, 0, out=d2)
        gmin = min(gmin, float(np.sqrt(d2.min(axis=1)).min()))
    return gmin


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    te, tr, va = load("test"), load("train"), load("val")

    print("=== Per-class mean spread on TEST (500/class) ===")
    for key, unit, claim in [("snr", "dB", 0.54), ("aspect", "deg", 2.7), ("range", "m", 0.82)]:
        spread, means = class_spread(te, key)
        print(f"  {key:6s} max-min across classes = {spread:.3f} {unit}  (report <= {claim})  "
              f"per-class={[round(m,2) for m in means]}")

    amin_test = float(te["alias"].min())
    amin_all = float(min(te["alias"].min(), tr["alias"].min(), va["alias"].min()))
    print(f"\n=== Doppler aliasing margin (test set, as in Appendix A) ===")
    print(f"  test-set min = {amin_test:.3f} m/s  (report >= 1.154);  all-splits min = {amin_all:.3f}")

    print("\n=== Cross-split min L2 (spectrograms, 16384-dim) ===")
    d_tt = min_cross_l2(te["x"], tr["x"])
    d_tv = min_cross_l2(te["x"], va["x"])
    print(f"  test-train min L2 = {d_tt:.3f}  (report 13.52)")
    print(f"  test-val   min L2 = {d_tv:.3f}  (report 13.43)")


if __name__ == "__main__":
    main()
