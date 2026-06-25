# P2 Micro-Doppler HAR -- Experiment Results

## Environment

| Item | Value |
|------|-------|
| PyTorch | 2.5.1+cu121 |
| CUDA | Yes |
| GPU | NVIDIA GeForce GTX 1660 Ti (6 GB) |
| Python | 3.12 |
| Dependencies | `_repo/requirements.txt` (pinned minimums) |

## Dataset Configuration

| Parameter | Value |
|-----------|-------|
| Training samples | 30,000 (balanced, 5,000/class) |
| Validation samples | 3,000 (balanced, 500/class) |
| Test samples | 3,000 (balanced, 500/class) |
| Classes | walk, run, sit_down, fall, wave, idle |
| STFT shape | 128 x 128 |
| SNR range | [5.0, 25.0] dB |
| Aspect range | [0, 60] deg |
| Target range | [6.0, 18.0] m |
| Radar fc | 77.0 GHz |
| Schema version | 6 |
| Data seed | 42 (train), 1042 (val), 2042 (test) |

## Results Summary

All numbers below are sourced from the JSON files listed in parentheses.

### Test Accuracy (all on the same test set)

| Method | Test Accuracy | Seeds | Mean +/- Std | Source |
|--------|--------------|-------|--------------|--------|
| ResNet-18 | 1.0, 1.0, 1.0 | 42, 43, 44 | 1.0 +/- 0.0 | `resnet18_s{42,43,44}/eval_results.json` |
| TinyCNN | 1.0, 1.0, 1.0 | 42, 43, 44 | 1.0 +/- 0.0 | `tiny_s{42,43,44}/eval_results.json` |
| RBF-SVM (10K train) | 0.9843 | 42 | -- | `feature_rbf_svm.json` |
| LogReg | 0.9763 | 42 | -- | `feature_logreg.json` |

### Per-Class Accuracy (from eval JSON files)

| Class | LogReg | RBF-SVM | ResNet-18 | TinyCNN |
|-------|--------|---------|-----------|---------|
| walk | 0.924 | 0.940 | 1.0 | 1.0 |
| run | 0.950 | 0.968 | 1.0 | 1.0 |
| sit_down | 0.996 | 1.0 | 1.0 | 1.0 |
| fall | 0.992 | 1.0 | 1.0 | 1.0 |
| wave | 0.996 | 1.0 | 1.0 | 1.0 |
| idle | 1.0 | 0.998 | 1.0 | 1.0 |

Source: `feature_logreg.json`, `feature_rbf_svm.json`, `resnet18_s42/eval_results.json`, `tiny_s42/eval_results.json`

### Most Confused Class Pair

walk vs run (from `feature_logreg.json` confusion matrix: 22 walk->run, 25 run->walk misclassifications; from `feature_rbf_svm.json`: 21 walk->run, 16 run->walk). Neural models have zero confusion (diagonal confusion matrices).

### LogReg Confusion Matrix (from `feature_logreg.json`)

```
            walk  run  sit  fall wave idle
walk         462   22    3    4    9    0
run           25  475    0    0    0    0
sit_down       0    0  498    0    0    2
fall           2    0    1  496    1    0
wave           2    0    0    0  498    0
idle           0    0    0    0    0  500
```

## Ablation: Data Efficiency (Stage 5)

ResNet-18 (seed 42) trained on 3,000 vs 30,000 samples.

| Dataset | Test Accuracy | Errors | Source |
|---------|--------------|--------|--------|
| 30K training | 1.0 (3000/3000 correct) | 0 | `resnet18_s42/eval_results.json` |
| 3K training | 0.9997 (2999/3000 correct) | 1 (run -> walk) | `resnet18_n3k_s42/eval_results.json` |

Finding: ResNet-18 nearly saturates even with 10x less training data on this simulated dataset. The single error is a run sample misclassified as walk, reflecting the physical similarity of these two gait patterns.

## Training Details

### ResNet-18 (seed 42, from `resnet18_s42/history.json`)
- Parameters: 11,173,318
- Epochs: 30
- Batch size: 64
- Optimizer: Adam (lr=1e-3)
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Loss: CrossEntropyLoss (label_smoothing=0.1)
- Best val loss: 0.4210 (epoch 27)
- Total training time: 1,252s (~21 min)
- Average epoch time: 41.7s

### TinyCNN (seed 42, from `tiny_s42/history.json`)
- Parameters: 328,446 (~0.33M)
- Architecture: 4 blocks of (Conv3x3 + BN + ReLU + Conv3x3 + BN + ReLU + MaxPool2d), width=24
- Total training time: 1,566s (~26 min)
- Average epoch time: 52.2s

## Figure Paths

| Figure | PNG | PDF |
|--------|-----|-----|
| Learning curve | `_results/figures/learning_curve.png` | `_results/figures/learning_curve.pdf` |
| Confusion matrix | `_results/figures/confusion_matrix.png` | `_results/figures/confusion_matrix.pdf` |
| Method comparison | `_results/figures/method_comparison.png` | `_results/figures/method_comparison.pdf` |
| Ablation | `_results/figures/ablation.png` | `_results/figures/ablation.pdf` |
| Examples | `_results/figures/examples.png` | `_results/figures/examples.pdf` |

## Reproduce

```bash
PY="C:/Users/remi/AppData/Local/Programs/Python/Python312/python.exe"
cd _repo/projects/p02_resnet18_har

# Generate data
$PY generate_data.py --n_train 30000 --n_val 3000 --n_test 3000 --seed 42

# Train ResNet-18
$PY train.py --epochs 30 --seed 42 --artifact_dir artifacts/resnet18_s42

# Train TinyCNN
$PY train.py --epochs 30 --seed 42 --model tiny_cnn --artifact_dir artifacts/tiny_s42

# Feature baselines
$PY evaluate_feature_baseline.py --data_dir data --model logreg --out artifacts/feature_logreg.json
$PY evaluate_feature_baseline.py --data_dir data --model rbf_svm --max_train 10000 --out artifacts/feature_rbf_svm.json
```

## Stages Completed

- [x] Stage 0: Setup (CUDA PyTorch 2.5.1+cu121)
- [x] Stage 1: Smoke test (passed)
- [x] Stage 2: ResNet-18 baseline, 3 seeds (all 1.0 accuracy)
- [x] Stage 3: TinyCNN ablation, 3 seeds (all 1.0 accuracy)
- [x] Stage 4: Feature baselines (LogReg 0.9763, RBF-SVM 0.9843)
- [x] Stage 5: Data efficiency ablation (30K: 1.0, 3K: 0.9997)
- [x] Stage 6: Figures (9 PNG+PDF pairs: 5 main + 4 stress)
- [x] Stage 7: Results manifest (this file + results.json)

## Deviations

- **Stage 5**: The data-efficiency ablation (30K vs 3K training samples) is the primary single-variable ablation and demonstrates that ResNet-18 nearly saturates even at 3K samples. The SNR and aspect-angle stress sweeps were additionally run as a 3-seed robustness study (see `STRESS_RESULTS.md`).
- **Concurrent GPU training**: TinyCNN seeds 43 and 44 ran concurrently on the GPU, causing ~2x wall-clock slowdown per run due to CUDA context switching. Results are unaffected (same accuracy).
- **No val tuning on test**: All models used validation loss for checkpoint selection; test accuracy was reported once per run from the best-val checkpoint.

## Total Wall Clock

Approximately 3 hours (data generation: ~28 min x2, ResNet-18 training: ~21 min x3, TinyCNN training: ~26 min + ~55 min concurrent, ablation: ~3.5 min, feature baselines: ~7s).
