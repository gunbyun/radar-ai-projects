# P02 ResNet-18 HAR -- Stress / Robustness Evaluation (3-seed)

**Checkpoints:** `resnet18_s42`, `resnet18_s43`, `resnet18_s44` (`artifacts/resnet18_s{42,43,44}/best_model.pt`)
**In-distribution training regime:** SNR 5-25 dB, aspect 0-60 deg
**Classes (index order):** walk, run, sit_down, fall, wave, idle
**Date:** 2026-06-16

> The W14 report Tables 3-4 and Figures 6-7 use the **3-seed mean +/- std** below
> (`snr_sweep_multiseed.json`, `aspect_sweep_multiseed.json`, produced by `run_multiseed_eval.py`).
> The single-seed files `snr_sweep.json` / `aspect_sweep.json` (produced by `run_stress.py`) are the
> seed-42-only intermediate; the seed-42 column of the multiseed tables reproduces them exactly.
> The knee confusion matrix (Step 3, Table 5 / Figure 8) is reported as the **seed-42 single run**
> (40.58% for that seed; the 3-seed mean is 38.7%).

---

## Reproduction Commands

```bash
# from results/stress (paths resolve relative to this script)
python run_stress.py            # generate stress HDF5 datasets + single-seed (s42) sweep + s42 knee
python run_multiseed_eval.py    # evaluate the 3 checkpoints (s42/s43/s44) on the SAME sets -> mean+/-std + figures
python collect_failures.py      # re-collect aspect 75-90 knee failures (s42) + regenerate figures
```

All datasets: 200 samples/class = 1200 samples per condition. Fixed data seeds (7000+i for SNR, 8000+i for aspect). Model seeds 42/43/44 are the trained checkpoints (eval-only, no retraining).

---

## Step 1 -- SNR Sweep (aspect 0-60 deg held, 3-seed)

| SNR (dB) | Accuracy (mean +/- std) | s42 | s43 | s44 | Data seed |
|----------|-------------------------|-------|-------|-------|------|
| -20      | 16.7 +/- 0.0   | 16.7  | 16.7  | 16.7  | 7000 |
| -15      | 18.0 +/- 1.3   | 16.7  | 17.6  | 19.8  | 7001 |
| -10      | 29.6 +/- 11.1  | 16.7  | 28.4  | 43.7  | 7002 |
| -5       | 62.8 +/- 27.1  | 24.4  | 81.8  | 82.2  | 7003 |
| **0**    | **99.6 +/- 0.2** | 99.8  | 99.3  | 99.8  | 7004 |
| 5        | 100.0 +/- 0.0  | 100.0 | 100.0 | 100.0 | 7005 |
| 10       | 100.0 +/- 0.0  | 100.0 | 100.0 | 100.0 | 7006 |
| 15       | 100.0 +/- 0.0  | 100.0 | 100.0 | 100.0 | 7007 |
| 20       | 100.0 +/- 0.0  | 100.0 | 100.0 | 100.0 | 7008 |
| 25       | 100.0 +/- 0.0  | 100.0 | 100.0 | 100.0 | 7009 |

**Key finding:** The model holds near-perfect accuracy down to 0 dB (5 dB below the training range), then collapses. The sub-0-dB transition (-10 to -5 dB) is **seed-dependent**: at -5 dB the per-seed accuracies are 24.4 / 81.8 / 82.2 (std 27.1). Only the two ends are reproducible: >= 0 dB is robust (99.6 +/- 0.2 at 0 dB), and <= -15 dB converges to chance (16.7-18.0%, single-class collapse, with the collapse class differing by seed: s42 -> run, s43 -> fall, s44 -> walk at the floor).

---

## Step 2 -- Aspect Sweep (SNR fixed 15 dB, 3-seed)

| Aspect Band | Accuracy (mean +/- std) | s42 | s43 | s44 | Data seed | OOD? |
|-------------|-------------------------|-------|-------|-------|------|------|
| 0-15 deg    | 100.0 +/- 0.0   | 100.0 | 100.0 | 100.0 | 8000 | No   |
| 15-30 deg   | 100.0 +/- 0.0   | 100.0 | 100.0 | 100.0 | 8001 | No   |
| 30-45 deg   | 100.0 +/- 0.0   | 100.0 | 100.0 | 100.0 | 8002 | No   |
| 45-60 deg   | 100.0 +/- 0.0   | 100.0 | 100.0 | 100.0 | 8003 | No   |
| **60-75 deg** | **97.7 +/- 0.3** | 97.25 | 98.08 | 97.75 | 8004 | Yes  |
| **75-90 deg** | **38.7 +/- 2.5** | 40.58 | 40.25 | 35.17 | 8005 | Yes  |

Per-class accuracy at 75-90 deg (3-seed mean): walk 0.06, run 0.27, sit_down 0.20, fall 0.51, wave 0.28, idle 1.00.

**Key finding:** In-training bands (0-60 deg) are all 100%. The first OOD band (60-75) degrades to 97.7%; the far-OOD band (75-90) collapses to 38.7 +/- 2.5% (reproducible, low std), with only idle keeping 100% and all motion classes severely degraded.

---

## Step 3 -- Knee Condition: Aspect 75-90 deg, SNR=15 dB (seed 42 single run)

**Accuracy:** 40.58% (713 / 1200 misclassified) for the seed-42 checkpoint (the 3-seed mean is 38.7%; see Step 2).

### Confusion Matrix (row-normalized, seed 42)

|             | walk | run  | sit_down | fall | wave | idle |
|-------------|------|------|----------|------|------|------|
| **walk**    | 0.05 | 0.00 | 0.37     | 0.00 | 0.45 | 0.14 |
| **run**     | 0.01 | 0.29 | 0.35     | 0.00 | 0.28 | 0.07 |
| **sit_down**| 0.00 | 0.00 | 0.23     | 0.00 | 0.00 | 0.77 |
| **fall**    | 0.00 | 0.00 | 0.20     | 0.57 | 0.01 | 0.23 |
| **wave**    | 0.00 | 0.00 | 0.00     | 0.00 | 0.29 | 0.71 |
| **idle**    | 0.00 | 0.00 | 0.00     | 0.00 | 0.00 | 1.00 |

### Most Confused Pairs (seed 42)

| True     | Predicted | Count (/200) |
|----------|-----------|--------------|
| sit_down | idle      | 154          |
| wave     | idle      | 142          |
| walk     | wave      | 89           |
| walk     | sit_down  | 74           |
| run      | sit_down  | 71           |
| run      | wave      | 55           |
| fall     | idle      | 45           |
| fall     | sit_down  | 40           |

---

## Physical Interpretation

### SNR degradation
Low SNR buries the micro-Doppler signatures in thermal noise; the STFT spectrogram becomes noise-dominated and the time-frequency patterns that distinguish activities vanish. The cliff between -5 and 0 dB is the threshold where the micro-Doppler signal-to-noise ratio at the target range bin stops producing recognizable texture. Near the floor the model collapses to a single class (the class differs by seed), which is why the sub-0-dB transition is seed-dependent while the two ends are reproducible.

### Aspect angle degradation
Near 90 deg (broadside) the radial velocity of all scatterers scales as v * cos(theta) -> 0, so all Doppler shifts shrink toward zero regardless of actual motion. Effects: (1) all activities look like idle (dominant confusion toward idle); (2) gait activities (walk, run) break first because they rely on periodic limb micro-Doppler that vanishes under cosine projection; (3) transient activities (fall) partially survive due to a strong radial/vertical velocity component and a distinct transient onset; (4) idle is immune (a near-zero-Doppler spectrogram at broadside matches its training pattern).

---

## Machine-Readable Data

| File | Description |
|------|-------------|
| `stress/snr_sweep_multiseed.json` | SNR sweep, 3-seed: per-model-seed + mean/std + per-class (**reported numbers**) |
| `stress/aspect_sweep_multiseed.json` | Aspect sweep, 3-seed: per-model-seed + mean/std + per-class (**reported numbers**) |
| `stress/snr_sweep.json` | SNR sweep, seed-42 single (intermediate; = s42 column above) |
| `stress/aspect_sweep.json` | Aspect sweep, seed-42 single (intermediate; = s42 column above) |
| `stress/failure_cases.json` | Knee condition (seed 42) details + most confused pairs |
| `stress/failure_spectrograms.npz` | Saved spectrograms for failure/correct examples |

## Figures (regenerated by `run_multiseed_eval.py` with error bars)

| Figure | Path |
|--------|------|
| SNR sweep (3-seed mean +/- std) | `figures/snr_sweep.png` / `.pdf` |
| Aspect sweep (3-seed mean +/- std) | `figures/aspect_sweep.png` / `.pdf` |
| Stress confusion matrix (75-90 deg, seed 42) | `figures/stress_confusion.png` / `.pdf` |
| Failure examples (spectrograms, seed 42) | `figures/failure_examples_real.png` / `.pdf` |

## Scripts

| Script | Purpose |
|--------|---------|
| `stress/run_stress.py` | Generate stress datasets + seed-42 sweep + seed-42 knee + single-seed figures |
| `stress/run_multiseed_eval.py` | Eval-only on s42/s43/s44 -> 3-seed mean/std JSON + error-bar figures (reported) |
| `stress/collect_failures.py` | Re-collect aspect 75-90 knee failures (seed 42) + regenerate knee figures |
