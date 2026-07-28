from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


RANDOM_STATE = 42
NORMAL_TRAIN_SIZE = 5000
GENERATED_COUNT = 1000
TARGET_METHOD = "Masking Diffusion"
TARGET_GENERATED_FILE = "diffusion_masked_windows.npz"
RESULT_PREFIX = "masking_diffusion1000"
REQUIRED_RESEARCH02_KEYS = {
    "masking_strategy",
    "masking_diffusion_denoiser",
    "temporal_block_ratio",
    "feature_group_ratio",
    "adaptive_masking",
}
THRESHOLDS = np.linspace(0.05, 0.95, 91)


def load_npz_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["X"].astype(np.float32), data["y"].astype(np.int8)


def load_npz_x(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return data["X"].astype(np.float32)


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def to_json_value(value):
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return value
    if isinstance(value, (np.floating, np.float32, np.float64)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    return value


def load_research02_generation_summary(root: Path) -> dict:
    summary_path = root / "data" / "research02" / "results" / "research02_generation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "Research02 generation summary was not found. Run tools/research02_generate_data.py first."
        )
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    missing = REQUIRED_RESEARCH02_KEYS - set(summary)
    if missing:
        raise RuntimeError(
            "Research02 generated data appears to come from the old masking pipeline. "
            f"Rerun tools/research02_generate_data.py before running Research04. Missing keys: {sorted(missing)}"
        )
    return summary


def select_threshold_by_validation_f1(
    y_true: np.ndarray,
    anomaly_proba: np.ndarray,
) -> dict:
    best = None

    for threshold in THRESHOLDS:
        pred = (anomaly_proba >= threshold).astype(np.int8)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        candidate = {
            "threshold": float(threshold),
            "validation_precision": float(precision),
            "validation_recall": float(recall),
            "validation_f1": float(f1),
        }
        if best is None or (
            candidate["validation_f1"],
            candidate["validation_recall"],
            candidate["validation_precision"],
        ) > (
            best["validation_f1"],
            best["validation_recall"],
            best["validation_precision"],
        ):
            best = candidate

    if best is None:
        raise RuntimeError("No threshold candidates were evaluated.")
    return best


def evaluate_model(
    params: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_final: np.ndarray,
    y_final: np.ndarray,
) -> dict:
    model = RandomForestClassifier(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(flatten_windows(x_train), y_train)

    val_proba = model.predict_proba(flatten_windows(x_val))[:, 1]
    threshold_result = select_threshold_by_validation_f1(y_val, val_proba)

    final_proba = model.predict_proba(flatten_windows(x_final))[:, 1]
    final_pred = (final_proba >= threshold_result["threshold"]).astype(np.int8)

    result = {
        "method": TARGET_METHOD,
        "generated_anomaly_used": GENERATED_COUNT,
        "threshold_selection": "Best F1 on real validation_windows only.",
        "threshold_selected_on_real_validation": threshold_result["threshold"],
        "validation_precision": threshold_result["validation_precision"],
        "validation_recall": threshold_result["validation_recall"],
        "validation_f1": threshold_result["validation_f1"],
        "precision": precision_score(y_final, final_pred, zero_division=0),
        "recall": recall_score(y_final, final_pred, zero_division=0),
        "f1": f1_score(y_final, final_pred, zero_division=0),
        "auroc": roc_auc_score(y_final, final_proba),
        "auprc": average_precision_score(y_final, final_proba),
        "pred_normal": int((final_pred == 0).sum()),
        "pred_anomaly": int((final_pred == 1).sum()),
        "true_normal": int((y_final == 0).sum()),
        "true_anomaly": int((y_final == 1).sum()),
        "final_pred": final_pred,
    }
    result.update(params)
    return result


def main() -> None:
    root = Path.cwd()
    split_dir = root / "data" / "augmentation_split"
    generated_dir = root / "data" / "research02" / "generated"
    result_dir = root / "data" / "research03" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    generation_summary = load_research02_generation_summary(root)

    x_normal_train, _ = load_npz_xy(split_dir / "normal_train_windows.npz")
    x_anomaly_seed, _ = load_npz_xy(split_dir / "anomaly_train_seed_windows.npz")
    x_val, y_val = load_npz_xy(split_dir / "validation_windows.npz")
    x_final, y_final = load_npz_xy(split_dir / "final_test_windows.npz")
    x_diffusion = load_npz_x(generated_dir / TARGET_GENERATED_FILE)

    rng = np.random.default_rng(RANDOM_STATE)
    normal_indices = rng.choice(
        len(x_normal_train),
        size=min(NORMAL_TRAIN_SIZE, len(x_normal_train)),
        replace=False,
    )
    diffusion_indices = rng.choice(
        len(x_diffusion),
        size=min(GENERATED_COUNT, len(x_diffusion)),
        replace=False,
    )

    x_normal_sample = x_normal_train[normal_indices]
    x_diffusion_sample = x_diffusion[diffusion_indices]
    x_anomaly_train = np.concatenate([x_anomaly_seed, x_diffusion_sample], axis=0)
    x_train = np.concatenate([x_normal_sample, x_anomaly_train], axis=0)
    y_train = np.concatenate(
        [
            np.zeros(len(x_normal_sample), dtype=np.int8),
            np.ones(len(x_anomaly_train), dtype=np.int8),
        ]
    )

    grid = {
        "n_estimators": [300, 500],
        "max_depth": [None, 12, 20],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt"],
        "class_weight": ["balanced_subsample", {0: 1, 1: 2}, {0: 1, 1: 4}],
    }
    param_names = list(grid)
    param_grid = [
        dict(zip(param_names, values))
        for values in itertools.product(*(grid[name] for name in param_names))
    ]

    results = []
    for i, params in enumerate(param_grid, start=1):
        print(f"[{i}/{len(param_grid)}] {params}")
        results.append(
            evaluate_model(
                params=params,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_final=x_final,
                y_final=y_final,
            )
        )

    tuning_df = pd.DataFrame(
        [{k: v for k, v in result.items() if k != "final_pred"} for result in results]
    ).sort_values(
        ["validation_f1", "validation_recall", "validation_precision"],
        ascending=False,
    )
    tuning_df.to_csv(
        result_dir / f"{RESULT_PREFIX}_randomforest_hyperparameter_tuning.csv",
        index=False,
    )

    best_result = max(
        results,
        key=lambda result: (
            result["validation_f1"],
            result["validation_recall"],
            result["validation_precision"],
        ),
    )
    report = classification_report(
        y_final,
        best_result["final_pred"],
        target_names=["normal", "anomaly"],
        zero_division=0,
        output_dict=True,
    )

    summary = {
        "claim_tested": (
            f"Only {TARGET_METHOD} with 1000 generated anomaly windows is tuned. "
            "Threshold and hyperparameters are selected on real validation_windows."
        ),
        "method": TARGET_METHOD,
        "generated_anomaly_used": GENERATED_COUNT,
        "model": "RandomForestClassifier",
        "normal_train_used": int(len(x_normal_sample)),
        "real_anomaly_seed_used": int(len(x_anomaly_seed)),
        "total_train_rows": int(len(x_train)),
        "hyperparameter_selection": "Best validation F1, then validation recall, then validation precision.",
        "threshold_selection": "Best F1 on real validation_windows only.",
        "research02_generation_summary": generation_summary,
        "n_candidates": len(results),
        "best_result": {
            k: to_json_value(v)
            for k, v in best_result.items()
            if k != "final_pred"
        },
        "best_classification_report": report,
    }
    with open(
        result_dir / f"{RESULT_PREFIX}_randomforest_hyperparameter_tuning_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print(tuning_df.head(10))
    print()
    print("Best validation-selected final-test classification report:")
    print(
        classification_report(
            y_final,
            best_result["final_pred"],
            target_names=["normal", "anomaly"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
