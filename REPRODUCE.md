# P2 Micro-Doppler HAR — 재현 가이드 (W14 제출용)

**변건 · 2026.06.15**

본 문서는 보고서 `202650765_변건_W14.pdf`의 모든 수치·그림을 처음부터 재현하는 절차이다. 데이터는 외부 다운로드 없이 코드가 합성 생성한다.

## 1. 환경

| 항목 | 값 |
|---|---|
| OS / GPU | Windows 11 / NVIDIA GTX 1660 Ti (6 GB) |
| Python | 3.12.10 |
| 핵심 패키지 | PyTorch 2.5.1+cu121, h5py 3.16, scikit-learn 1.9, seaborn 0.13.2, matplotlib 3.10, tqdm |
| 코드 저장소 | https://github.com/gunbyun/radar-ai-projects (경로 `projects/p02_resnet18_har/`) |
| 의존성 | `requirements.txt` (저장소 루트, 최소 버전 고정) |

## 2. 설치

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install "h5py>=3.8" "scikit-learn>=1.2" tqdm seaborn matplotlib
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"  # True 확인
```

> 실행 명령의 `python`은 채점 환경의 Python 3.12 인터프리터를 가리킨다(`python` 또는 `python3`).

## 3. 한 줄 재현 (기본 학습)

```bash
cd projects/p02_resnet18_har
python train.py --generate --epochs 30        # 데이터 생성 + ResNet-18 30 epoch 학습 + 평가
```

## 4. 보고서 전체 재현 (다중 시드 + 기준선 + 스트레스)

### 4.1 데이터 생성 (1회, 고정 시드)
```bash
cd projects/p02_resnet18_har
python generate_data.py --n_train 30000 --n_val 3000 --n_test 3000 --seed 42
# 분할별 데이터 시드: train 42 / val 1042 / test 2042 (생성기 내부 고정)
```

### 4.2 ResNet-18 기준선 (시드 42/43/44, 동일 데이터)
```bash
python train.py --epochs 30 --seed 42 --artifact_dir artifacts/resnet18_s42
python train.py --epochs 30 --seed 43 --artifact_dir artifacts/resnet18_s43
python train.py --epochs 30 --seed 44 --artifact_dir artifacts/resnet18_s44
```

### 4.3 TinyCNN 용량 절삭 (동일 데이터·시드)
```bash
python train.py --epochs 30 --seed 42 --model tiny_cnn --artifact_dir artifacts/tiny_s42
python train.py --epochs 30 --seed 43 --model tiny_cnn --artifact_dir artifacts/tiny_s43
python train.py --epochs 30 --seed 44 --model tiny_cnn --artifact_dir artifacts/tiny_s44
```

### 4.4 고전 특징 기준선
```bash
python evaluate_feature_baseline.py --data_dir data --model logreg  --out artifacts/feature_logreg.json
python evaluate_feature_baseline.py --data_dir data --model rbf_svm --max_train 10000 --out artifacts/feature_rbf_svm.json
```

### 4.5 데이터 효율 절삭 (3K)
```bash
python train.py --generate --n_train 3000 --epochs 30 --seed 42 --artifact_dir artifacts/resnet18_n3k_s42
```

### 4.6 SNR·경사각 스트레스 평가 (학습된 체크포인트 사용, 재학습 없음)
```bash
cd ../../_results/stress     # 저장소 기준 상대 경로는 본인 배치에 맞게 조정
python run_stress.py            # SNR(-20~25dB) + 경사각(0~90deg) 스윕 데이터 생성 + 단일시드(s42) 평가
python run_multiseed_eval.py    # 3개 시드(42/43/44) 평가 -> 평균±표준편차 + 에러바 그림 (보고서 표3/4·그림6/7)
python collect_failures.py      # 경사각 75-90deg 오분류 사례(s42) 수집 + 그림 재생성
```

## 5. 출력 위치

| 산출물 | 경로 | 저장소 포함 |
|---|---|---|
| 결과 요약 | `_results/RESULTS.md`, `_results/STRESS_RESULTS.md` | 포함 |
| 스트레스 결과(JSON) | `_results/stress/*.json` (`snr_sweep`·`aspect_sweep`·`failure_cases` + `*_multiseed`) | 포함 |
| 그림(PNG+PDF) | `_results/figures/` | 포함 |
| 시드별 eval JSON·학습 체크포인트 | 각 실행의 `--artifact_dir` (예: `artifacts/resnet18_s42/`) | 미포함(재현 시 생성) |
| 합성 데이터(HDF5) | `data/` | 미포함(재현 시 생성) |

## 6. 재현성 정책

- 시드 고정: 모델 학습 시드 42/43/44, 데이터 분할 시드 42/1042/2042.
- 모델 선택은 검증 손실 최소 기준 체크포인트로만 수행하고, 테스트 세트는 최종 1회 평가에만 사용한다(테스트 시간 튜닝 없음).
- 모든 비교는 동일 데이터·동일 분할·동일 에폭에서 단일 변수만 변경한다.

## 7. 제출

- 보고서 PDF(`202650765_변건_W14.pdf`)는 충남대학교 사이버캠퍼스에 별도 제출.
- 코드는 본 저장소 `https://github.com/gunbyun/radar-ai-projects`로 공개 제출하였다. reference 코드(`common/`·`shared/`·`projects/p02_resnet18_har/`) + 본인 추가 스크립트(`_results/stress/`, `_results/gen_figures.py`) + 본 재현 가이드를 포함하며, 코드 출처는 저장소 `README.md`에 명시되어 있다.
