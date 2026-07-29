# Manufacturing Anomaly Detection Data Augmentation Research

제조 용접 공정의 시계열 센서 데이터에서 불량 윈도우를 탐지하기 위한 데이터 증강 실험 프로젝트입니다. 불량 데이터가 정상 데이터보다 매우 적은 조건에서, 원본 불량 데이터만 사용한 기준 모델과 생성형/전통적 증강을 적용한 모델을 비교했습니다.

핵심 질문은 다음과 같습니다.

- 생성형 시계열 증강이 원본 데이터만 사용한 기준 모델보다 이상 탐지 성능을 개선하는가?
- 성능 변화가 생성 샘플 수 때문인지, 클래스 비율 보정 때문인지 구분할 수 있는가?
- 실제 제출/운영 관점에서 임계값 안정성, seed 수 부족, 반복 실행 안정성까지 유지되는가?

## Project Structure

```text
data/
  augmentation_split/   train/validation/final test 윈도우 분할 결과
  baseline/             초기 기준 모델 결과
  research01/           원본 데이터 기반 RandomForest 기준 모델
  research02/           GT-GAN, Diffusion, Masking 기반 생성 데이터 품질 비교
  research03/           생성형 증강 데이터의 지도학습 성능 검증
  research05/           생성형/전통적 증강 및 클래스 비율 비교
  research07/           임계값 강건성, 반복 안정성, scarce seed 실험
notebooks/
  Research01_baseline_pipeline.ipynb
  Research02_Generate_Data.ipynb
  Research03_Validate_Data.ipynb
  Research04_MaskingDiffusion_RF_Threshold_Optimization.ipynb
  Research05_Balanced_vs_Imbalanced_Augmentation.ipynb
  Research06_filtering.ipynb
  Research07.ipynb
tools/
  research01_randomforest_baseline.py
  research02_generate_data.py
  research03_model_augmented_validation.py
  research03_diffusion1000_hyperparameter_tuning.py
  research03_diffusion1000_threshold_optimization.py
  research05_balanced_aug_experiment.py
```

## Data Split

최종 성능 평가는 학습과 분리된 final test 윈도우에서 수행했습니다. 임계값은 final test에 직접 맞추지 않고 validation split에서만 선택했습니다.

| Split | Windows | Normal | Anomaly |
| --- | ---: | ---: | ---: |
| normal_train | 135,730 | 135,730 | 0 |
| anomaly_train_seed | 247 | 0 | 247 |
| validation | 1,447 | 1,283 | 164 |
| final_test | 3,081 | 2,785 | 296 |

## Experiment Flow

1. **Research01: Baseline**
   - 원본 불량 seed 247개와 정상 윈도우 5,000개만 사용해 RandomForest 기준 모델을 구성했습니다.
   - 기준 모델의 final test F1은 0.718, recall은 0.618입니다.

2. **Research02: Generated Data Quality**
   - GT-GAN, Diffusion, Masking GT-GAN, Masking Diffusion을 비교했습니다.
   - Masking 계열은 실제 불량 seed의 관측 값을 보존하고, 시간 블록과 feature group 단위로 가려진 영역만 복원하도록 설계했습니다.
   - 생성 품질 평가는 t-SNE, 품질 지표, IsolationForest 기반 downstream 성능으로 확인했습니다.

   ![Generated data quality metrics](data/research02/figures/generated_quality_metrics.png)

3. **Research03: Supervised Augmentation Validation**
   - 생성 불량 윈도우를 RandomForest 학습 데이터에 추가하고, validation split에서 선택한 임계값으로 final test를 평가했습니다.
   - 가장 좋은 결과는 Masking Diffusion 1,000개 증강 조건에서 나왔습니다.

   ![Supervised augmentation performance](data/research03/figures/supervised_model_augmented_performance.png)

4. **Research05: Balanced vs Imbalanced Augmentation**
   - 생성형 증강, 품질 필터링 생성형 증강, 전통적 증강, hybrid 증강을 비교했습니다.
   - balanced 조건은 정상 데이터를 불량 수에 맞춰 downsampling했고, imbalanced 조건은 정상 5,000개를 유지했습니다.
   - 최종적으로 imbalanced 조건에서 전통적 Magnitude warping이 가장 높은 F1을 보였고, 생성형 계열에서는 Filtered Masking Diffusion이 강한 대안으로 확인됐습니다.

   ![Balanced and imbalanced augmentation scores](data/research05/figures/balanced_vs_imbalanced_scores.png)

5. **Research07: Portfolio Robustness Analysis**
   - 포트폴리오 제출용으로 임계값 강건성, 반복 실행 안정성, 불량 seed 부족 조건을 추가 분석했습니다.
   - 단일 최고 점수만 제시하지 않고, 실제 적용 시 선택 가능한 임계값 폭과 seed 수 변화에 따른 성능 변화를 함께 정리했습니다.

   ![Threshold robustness comparison](data/research07/figures/research07_threshold_robustness_comparison.png)

   ![Stability recall comparison](data/research07/figures/research07_stability_recall_comparison.png)

   ![Scarce seed efficiency comparison](data/research07/figures/research07_scarce_seed_efficiency_comparison.png)

## Key Results

### Baseline vs Best Augmentation

| Method | Generated anomalies | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real anomaly only | 0 | 0.855 | 0.618 | 0.718 | 0.959 | 0.800 |
| Masking Diffusion | 1,000 | 0.962 | 0.764 | 0.851 | 0.990 | 0.938 |
| Magnitude warping | 1,000 | 0.964 | 0.814 | 0.883 | 0.994 | 0.959 |
| Filtered@750 Masking Diffusion | 750 | 0.953 | 0.818 | 0.880 | 0.991 | 0.944 |

### Research07 Robustness Summary

| Analysis | Best observed method | Main takeaway |
| --- | --- | --- |
| Precision floor threshold sweep | Magnitude warping | Precision 0.90 이상을 유지하면서 최대 recall 0.858까지 확보 |
| 30-run stability | Magnitude warping | 평균 F1 0.860, 평균 recall 0.802로 반복 실행 안정성이 가장 높음 |
| Scarce seed analysis | Magnitude warping | seed가 25, 50, 100, 150, 247개일 때 전반적으로 가장 안정적 |
| Generative alternative | Filtered@750 Masking Diffusion | 생성형 증강 중 가장 실용적인 후보로, 247 seed 조건에서 평균 F1 0.812와 AUPRC 0.907 기록 |

## Interpretation

원본 데이터만 사용한 기준 모델은 precision은 높지만 recall이 낮아 실제 불량을 놓치는 문제가 있었습니다. Masking Diffusion은 원본 seed의 구조를 보존하면서 가려진 구간만 복원하는 방식으로 기준 모델 대비 F1과 AUPRC를 크게 개선했습니다.

다만 최종 robustness 분석에서는 전통적 Magnitude warping이 가장 높은 평균 성능과 안정성을 보였습니다. 이 결과는 생성형 증강 자체가 항상 우월하다기보다, 불량 seed 수가 적은 제조 시계열 문제에서는 증강 방식의 품질 관리, 클래스 비율, 임계값 선택이 함께 성능을 결정한다는 점을 보여줍니다.

## Reproduce

필요 패키지 예시는 다음과 같습니다.

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

저장소 루트에서 아래 순서로 실행할 수 있습니다.

```powershell
python tools/research01_randomforest_baseline.py
python tools/research02_generate_data.py
python tools/research03_model_augmented_validation.py
python tools/research03_diffusion1000_hyperparameter_tuning.py
python tools/research03_diffusion1000_threshold_optimization.py
python tools/research05_balanced_aug_experiment.py
```

각 실험은 결과 CSV/JSON과 그림 파일을 `data/researchXX/` 하위 디렉터리에 저장합니다.

## Notes

- 모든 주요 평가는 final test split에서 수행했습니다.
- 임계값은 validation split에서 선택해 test leakage를 줄였습니다.
- 결과 파일과 그림을 함께 저장해 노트북을 재실행하지 않아도 실험 흐름과 결론을 확인할 수 있도록 구성했습니다.
