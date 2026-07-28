# Data Augmentation for Manufacturing Time-Series Anomaly Detection

This repository contains experiments for manufacturing time-series anomaly detection under severe class imbalance. The project compares original-data baselines, generative anomaly augmentation, threshold-selection strategies, and balanced-vs-imbalanced training conditions.

The main data unit is a fixed-length window from welding process time series. Models are evaluated on held-out real windows, and threshold selection is performed on a separate real validation split to avoid tuning directly on the final test labels.

## Repository Layout

```text
data/
  augmentation_split/   Train, validation, and final-test window splits
  baseline/             Baseline model comparison and tuning results
  raw_data/             Original raw data and early Research01 notebook
  research01/           RandomForest baseline outputs
  research02/           Generated windows, quality plots, and IsolationForest results
  research03/           Supervised augmentation validation results and figures
  research05/           Balanced-vs-imbalanced augmentation experiment results
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

## Data Split

The generated experiment split is stored in `data/augmentation_split/`.

| Split | Windows | Normal | Anomaly |
| --- | ---: | ---: | ---: |
| normal_train | 135,730 | 135,730 | 0 |
| anomaly_train_seed | 247 | 0 | 247 |
| validation | 1,447 | 1,283 | 164 |
| final_test | 3,081 | 2,785 | 296 |

## Experiment Flow

1. `Research01` builds original-data baselines.
   - IsolationForest baseline results are stored in `data/baseline/`.
   - RandomForest baseline results are stored in `data/research01/`.

2. `Research02` generates anomaly windows and compares generation quality.
   - Methods include GT-GAN, Diffusion, Masking GT-GAN, and Masking Diffusion.
   - Masking methods preserve observed seed-window values and restore structured temporal and feature-group masks.
   - Generated windows are stored in `data/research02/generated/`.

3. `Research03` evaluates generated anomalies in supervised RandomForest training.
   - Generated anomalies are added only to model training.
   - Thresholds are selected only on real validation windows.
   - Results are stored in `data/research03/results/`.

4. `Research05` compares generative augmentation against traditional augmentation and class-ratio balancing.
   - Traditional methods include time warping, magnitude warping, frequency-domain augmentation, noise injection, and comprehensive augmentation.
   - Results are stored in `data/research05/results/`.

## Key Results

Original-data RandomForest baseline:

| Method | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real anomaly only | 0.855 | 0.618 | 0.718 | 0.959 | 0.800 |

Best supervised augmentation result from `Research03`:

| Method | Generated anomalies | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Masking Diffusion | 1,000 | 0.962 | 0.764 | 0.851 | 0.990 | 0.938 |

Best condition-level results from `Research05`:

| Condition | Best family | Best method |
| --- | --- | --- |
| Balanced | Traditional | Frequency domain |
| Imbalanced | Traditional | Magnitude warping |

These results suggest that structured masking-based generation can improve supervised anomaly detection over the original-data baseline, while class-ratio effects and traditional augmentation remain important controls.

## Running the Scripts

Install the Python packages used by the scripts:

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

Run experiments from the repository root:

```powershell
python tools/research01_randomforest_baseline.py
python tools/research02_generate_data.py
python tools/research03_model_augmented_validation.py
python tools/research03_diffusion1000_hyperparameter_tuning.py
python tools/research03_diffusion1000_threshold_optimization.py
python tools/research05_balanced_aug_experiment.py
```

The scripts write CSV, JSON, NPZ, and figure outputs under the corresponding `data/researchXX/` directories.

## Notes

- Random seeds are fixed in the experiment scripts where applicable.
- `__pycache__/` and Python bytecode files are ignored by Git.
- The repository stores generated result artifacts so the analysis can be inspected without rerunning every experiment.
