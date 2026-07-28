# 제조 시계열 이상 탐지를 위한 데이터 증강 연구

이 저장소는 심한 클래스 불균형이 있는 제조 시계열 데이터에서 이상 탐지 성능을 개선하기 위한 실험을 정리한 프로젝트입니다. 원본 데이터만 사용한 기준 모델, 생성형 이상 데이터 증강, 임계값 선택 전략, 균형/불균형 학습 조건을 비교합니다.

주요 데이터 단위는 용접 공정 시계열에서 만든 고정 길이 윈도우입니다. 모델 평가는 별도로 분리한 실제 윈도우에서 수행하며, 최종 테스트 라벨에 직접 맞추는 것을 피하기 위해 임계값은 실제 검증 데이터에서만 선택합니다.

## 저장소 구조

```text
data/
  augmentation_split/   학습, 검증, 최종 테스트용 윈도우 분할 데이터
  baseline/             기준 모델 비교 및 튜닝 결과
  raw_data/             원본 데이터와 초기 Research01 노트북
  research01/           RandomForest 기준 모델 출력
  research02/           생성 윈도우, 품질 비교 그림, IsolationForest 결과
  research03/           지도학습 기반 증강 검증 결과와 그림
  research05/           균형/불균형 증강 실험 결과
notebooks/
  Research01_baseline_pipeline.ipynb
  Research02_Generate_Data.ipynb
  Research03_Validate_Data.ipynb
  Research04.ipynb
  Research05.ipynb
tools/
  research01_randomforest_baseline.py
  research02_generate_data.py
  research03_model_augmented_validation.py
  research03_diffusion1000_hyperparameter_tuning.py
  research03_diffusion1000_threshold_optimization.py
  research05_balanced_aug_experiment.py
```

## 데이터 분할

실험용 데이터 분할 결과는 `data/augmentation_split/`에 저장되어 있습니다.

| 분할 | 윈도우 수 | 정상 | 이상 |
| --- | ---: | ---: | ---: |
| normal_train | 135,730 | 135,730 | 0 |
| anomaly_train_seed | 247 | 0 | 247 |
| validation | 1,447 | 1,283 | 164 |
| final_test | 3,081 | 2,785 | 296 |

## 실험 흐름

1. `Research01`은 원본 데이터만 사용한 기준 모델을 만듭니다.
   - IsolationForest 기준 모델 결과는 `data/baseline/`에 저장됩니다.
   - RandomForest 기준 모델 결과는 `data/research01/`에 저장됩니다.

2. `Research02`는 이상 윈도우를 생성하고 생성 품질을 비교합니다.
   - 비교 방법은 GT-GAN, Diffusion, Masking GT-GAN, Masking Diffusion입니다.
   - Masking 기반 방법은 seed 윈도우에서 관측된 값은 유지하고, 구조화된 시간 구간 및 feature group mask 영역만 복원합니다.
   - 생성된 윈도우는 `data/research02/generated/`에 저장됩니다.

3. `Research03`은 생성 이상 데이터를 RandomForest 지도학습에 추가했을 때의 성능을 검증합니다.
   - 생성 이상 데이터는 모델 학습에만 추가합니다.
   - 임계값은 실제 검증 윈도우에서만 선택합니다.
   - 결과는 `data/research03/results/`에 저장됩니다.

4. `Research05`는 생성형 증강과 전통적 증강, 그리고 클래스 비율 균형화 효과를 비교합니다.
   - 전통적 증강 방법은 time warping, magnitude warping, frequency-domain augmentation, noise injection, comprehensive augmentation입니다.
   - 결과는 `data/research05/results/`에 저장됩니다.

## 주요 결과

원본 데이터만 사용한 RandomForest 기준 모델:

| 방법 | 정밀도 | 재현율 | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 실제 이상 데이터만 사용 | 0.855 | 0.618 | 0.718 | 0.959 | 0.800 |

`Research03`에서 확인한 최고 지도학습 증강 결과:

| 방법 | 생성 이상 수 | 정밀도 | 재현율 | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Masking Diffusion | 1,000 | 0.962 | 0.764 | 0.851 | 0.990 | 0.938 |

`Research05`의 조건별 최고 결과:

| 조건 | 최고 계열 | 최고 방법 |
| --- | --- | --- |
| 균형 조건 | 전통적 증강 | Frequency domain |
| 불균형 조건 | 전통적 증강 | Magnitude warping |

이 결과는 구조적 masking 기반 생성이 원본 데이터 기준 모델보다 지도학습 이상 탐지 성능을 개선할 수 있음을 보여줍니다. 동시에 클래스 비율의 영향과 전통적 증강 방법도 중요한 비교 기준임을 확인했습니다.

## 스크립트 실행 방법

스크립트 실행에 필요한 Python 패키지를 설치합니다.

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

저장소 루트에서 아래 순서로 실험 스크립트를 실행할 수 있습니다.

```powershell
python tools/research01_randomforest_baseline.py
python tools/research02_generate_data.py
python tools/research03_model_augmented_validation.py
python tools/research03_diffusion1000_hyperparameter_tuning.py
python tools/research03_diffusion1000_threshold_optimization.py
python tools/research05_balanced_aug_experiment.py
```

각 스크립트는 CSV, JSON, NPZ, 그림 파일을 해당 `data/researchXX/` 디렉터리에 저장합니다.

## 참고 사항

- 적용 가능한 실험 스크립트에는 고정 random seed가 설정되어 있습니다.
- `__pycache__/`와 Python bytecode 파일은 Git에서 제외합니다.
- 이 저장소는 모든 실험을 다시 실행하지 않아도 분석 내용을 확인할 수 있도록 생성된 결과 파일을 함께 저장합니다.
