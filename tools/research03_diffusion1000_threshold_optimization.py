from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
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
TUNING_SUMMARY_FILE = f"{RESULT_PREFIX}_randomforest_hyperparameter_tuning_summary.json"
THRESHOLDS = np.linspace(0.05, 0.95, 91)
PRECISION_FLOOR = 0.80
REQUIRED_RESEARCH02_KEYS = {
    "masking_strategy",
    "masking_diffusion_denoiser",
    "temporal_block_ratio",
    "feature_group_ratio",
    "adaptive_masking",
}


def load_npz_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["X"].astype(np.float32), data["y"].astype(np.int8)


def load_npz_x(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return data["X"].astype(np.float32)


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


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


def load_tuned_hyperparameters(result_dir: Path) -> dict:
    summary_path = result_dir / TUNING_SUMMARY_FILE
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} was not found. Run tools/research03_diffusion1000_hyperparameter_tuning.py first."
        )
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    best = summary["best_result"]
    return {
        "n_estimators": best["n_estimators"],
        "max_depth": best["max_depth"],
        "min_samples_leaf": best["min_samples_leaf"],
        "max_features": best["max_features"],
        "class_weight": {int(k): v for k, v in best["class_weight"].items()}
        if isinstance(best["class_weight"], dict)
        else best["class_weight"],
    }


def score_thresholds(y_true: np.ndarray, proba: np.ndarray, prefix: str) -> list[dict]:
    rows = []
    for threshold in THRESHOLDS:
        pred = (proba >= threshold).astype(np.int8)
        rows.append(
            {
                "threshold": float(threshold),
                f"{prefix}_precision": precision_score(y_true, pred, zero_division=0),
                f"{prefix}_recall": recall_score(y_true, pred, zero_division=0),
                f"{prefix}_f1": f1_score(y_true, pred, zero_division=0),
                f"{prefix}_pred_anomaly": int((pred == 1).sum()),
            }
        )
    return rows


def evaluate_at_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    prefix: str,
) -> dict:
    pred = (proba >= threshold).astype(np.int8)
    return {
        f"{prefix}_precision": precision_score(y_true, pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, pred, zero_division=0),
        f"{prefix}_pred_normal": int((pred == 0).sum()),
        f"{prefix}_pred_anomaly": int((pred == 1).sum()),
        f"{prefix}_true_normal": int((y_true == 0).sum()),
        f"{prefix}_true_anomaly": int((y_true == 1).sum()),
    }


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
    tuned_params = load_tuned_hyperparameters(result_dir)

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

    x_train = np.concatenate(
        [
            x_normal_train[normal_indices],
            x_anomaly_seed,
            x_diffusion[diffusion_indices],
        ],
        axis=0,
    )
    y_train = np.concatenate(
        [
            np.zeros(len(normal_indices), dtype=np.int8),
            np.ones(len(x_anomaly_seed) + len(diffusion_indices), dtype=np.int8),
        ]
    )

    model = RandomForestClassifier(
        **tuned_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(flatten_windows(x_train), y_train)

    val_proba = model.predict_proba(flatten_windows(x_val))[:, 1]
    final_proba = model.predict_proba(flatten_windows(x_final))[:, 1]

    val_rows = score_thresholds(y_val, val_proba, "validation")
    final_rows = score_thresholds(y_final, final_proba, "final")
    sweep_df = pd.DataFrame(
        [
            {**val_row, **{k: v for k, v in final_row.items() if k != "threshold"}}
            for val_row, final_row in zip(val_rows, final_rows)
        ]
    )
    sweep_df.to_csv(
        result_dir / f"{RESULT_PREFIX}_tuned_threshold_sweep.csv",
        index=False,
    )

    best_validation_f1 = max(
        val_rows,
        key=lambda row: (
            row["validation_f1"],
            row["validation_recall"],
            row["validation_precision"],
        ),
    )
    precision_floor_candidates = [
        row for row in val_rows if row["validation_precision"] >= PRECISION_FLOOR
    ]
    best_precision_floor = max(
        precision_floor_candidates,
        key=lambda row: (
            row["validation_recall"],
            row["validation_f1"],
            row["validation_precision"],
        ),
    )
    best_validation_recall = max(
        val_rows,
        key=lambda row: (
            row["validation_recall"],
            row["validation_precision"],
            row["validation_f1"],
        ),
    )

    strategies = {
        "best_validation_f1": best_validation_f1,
        f"max_recall_with_validation_precision_at_least_{PRECISION_FLOOR:.2f}": best_precision_floor,
        "max_validation_recall": best_validation_recall,
    }
    rows = []
    for strategy, selected in strategies.items():
        threshold = selected["threshold"]
        row = {
            "strategy": strategy,
            "threshold": threshold,
            **evaluate_at_threshold(y_val, val_proba, threshold, "validation"),
            **evaluate_at_threshold(y_final, final_proba, threshold, "final"),
            "final_auroc": roc_auc_score(y_final, final_proba),
            "final_auprc": average_precision_score(y_final, final_proba),
        }
        rows.append(row)

    strategy_df = pd.DataFrame(rows)
    strategy_df.to_csv(
        result_dir / f"{RESULT_PREFIX}_tuned_threshold_strategy_comparison.csv",
        index=False,
    )

    summary = {
        "claim_tested": (
            f"For the tuned {TARGET_METHOD} 1000 RandomForest model, thresholds are selected "
            "on validation_windows and evaluated once on final_test_windows."
        ),
        "model": "RandomForestClassifier",
        "method": TARGET_METHOD,
        "generated_anomaly_used": GENERATED_COUNT,
        "research02_generation_summary": generation_summary,
        "fixed_hyperparameters": tuned_params,
        "precision_floor": PRECISION_FLOOR,
        "strategies": rows,
    }
    with open(
        result_dir / f"{RESULT_PREFIX}_tuned_threshold_optimization_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(strategy_df)


if __name__ == "__main__":
    main()
