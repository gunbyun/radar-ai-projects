# Radar-AI P2: Micro-Doppler HAR (W14 제출)

변건 · 학번 202650765 · 충남대학교 대학원 레이더+AI · 2026.06

## 코드 출처 (중요)
- `common/`, `shared/`, `projects/p02_resnet18_har/` 의 모델·데이터 합성·학습·평가 코드는 **강의에서 제공한 reference 코드**(REMI Lab, `remilab-cnu/radar-ai-projects`)를 기반으로 한다.
- **본인이 추가·작성한 부분:**
  - `_results/stress/run_stress.py` : SNR·경사각 스트레스 데이터 생성 + 단일시드(s42) 평가
  - `_results/stress/run_multiseed_eval.py` : 3시드(42/43/44) 평가 → 평균±표준편차 + 에러바 그림 (보고서 표3·4·그림6·7)
  - `_results/stress/collect_failures.py` : 경사각 75–90° 오분류 사례 수집
  - `_results/gen_figures.py` : 보고서 그림 생성
- 모든 실험(ResNet/TinyCNN 3시드 재현 · 데이터효율 ablation · SNR/경사각 스트레스)은 본인 GPU(NVIDIA GTX 1660 Ti)에서 직접 실행하였다.

## 한 줄 재현
```
cd projects/p02_resnet18_har
python train.py --generate --epochs 30
```
전체 재현 절차(다중시드·기준선·스트레스)는 [REPRODUCE.md](REPRODUCE.md) 참조.

## 결과 요약
- 분포 내: ResNet-18 100.0±0.0% · TinyCNN 100.0±0.0% · LogReg 97.63% · RBF-SVM 98.43%
- 경사각 75–90°: 38.7±2.5% · SNR −5 dB: 62.8±27.1% (3시드)
- 상세: [`_results/RESULTS.md`](_results/RESULTS.md), [`_results/STRESS_RESULTS.md`](_results/STRESS_RESULTS.md)

## 데이터
외부 다운로드 불필요. 코드가 합성 생성한다(`generate_data.py`). 대용량 데이터(`*.h5`)·체크포인트(`*.pt`)는 저장소에서 제외(재현 시 생성).
