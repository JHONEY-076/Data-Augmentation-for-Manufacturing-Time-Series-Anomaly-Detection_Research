from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


RANDOM_STATE = 42
NORMAL_TRAIN_SIZE = 5000
THRESHOLDS = np.linspace(0.05, 0.95, 91)

BASELINE_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
}
HYPERPARAMETER_APPLIED_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": {0: 1, 1: 2},
}


def load_npz_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["X"].astype(np.float32), data["y"].astype(np.int8)


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def select_threshold_by_validation_f1(
    y_true: np.ndarray,
    anomaly_proba: np.ndarray,
) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in THRESHOLDS:
        pred = (anomaly_proba >= threshold).astype(np.int8)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, float(best_f1)


def to_json_value(value):
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.floating, np.float32, np.float64)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    return value


def evaluate_randomforest_baseline(
    method: str,
    params: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_final: np.ndarray,
    y_final: np.ndarray,
    normal_train_used: int,
    real_anomaly_seed_used: int,
) -> tuple[dict, np.ndarray]:
    model = RandomForestClassifier(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(flatten_windows(x_train), y_train)

    val_proba = model.predict_proba(flatten_windows(x_val))[:, 1]
    threshold, validation_f1 = select_threshold_by_validation_f1(y_val, val_proba)

    final_proba = model.predict_proba(flatten_windows(x_final))[:, 1]
    final_pred = (final_proba >= threshold).astype(np.int8)

    result = {
        "method": method,
        "model": "RandomForestClassifier",
        "normal_train_used": int(normal_train_used),
        "real_anomaly_seed_used": int(real_anomaly_seed_used),
        "generated_anomaly_used": 0,
        "total_train_rows": int(len(x_train)),
        "threshold_selected_on_real_validation": threshold,
        "validation_f1": validation_f1,
        "precision": precision_score(y_final, final_pred, zero_division=0),
        "recall": recall_score(y_final, final_pred, zero_division=0),
        "f1": f1_score(y_final, final_pred, zero_division=0),
        "auroc": roc_auc_score(y_final, final_proba),
        "auprc": average_precision_score(y_final, final_proba),
        "pred_normal": int((final_pred == 0).sum()),
        "pred_anomaly": int((final_pred == 1).sum()),
        "true_normal": int((y_final == 0).sum()),
        "true_anomaly": int((y_final == 1).sum()),
        "params": params,
    }
    return result, final_pred


def main() -> None:
    root = Path.cwd()
    split_dir = root / "data" / "augmentation_split"
    result_dir = root / "data" / "research01" / "results"
    figure_dir = root / "data" / "research01" / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    x_normal_train, _ = load_npz_xy(split_dir / "normal_train_windows.npz")
    x_anomaly_seed, _ = load_npz_xy(split_dir / "anomaly_train_seed_windows.npz")
    x_val, y_val = load_npz_xy(split_dir / "validation_windows.npz")
    x_final, y_final = load_npz_xy(split_dir / "final_test_windows.npz")

    rng = np.random.default_rng(RANDOM_STATE)
    normal_indices = rng.choice(
        len(x_normal_train),
        size=min(NORMAL_TRAIN_SIZE, len(x_normal_train)),
        replace=False,
    )

    x_normal_sample = x_normal_train[normal_indices]
    x_train = np.concatenate([x_normal_sample, x_anomaly_seed], axis=0)
    y_train = np.concatenate(
        [
            np.zeros(len(x_normal_sample), dtype=np.int8),
            np.ones(len(x_anomaly_seed), dtype=np.int8),
        ]
    )

    result, final_pred = evaluate_randomforest_baseline(
        method="Real anomaly only",
        params=BASELINE_PARAMS,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_final=x_final,
        y_final=y_final,
        normal_train_used=len(x_normal_sample),
        real_anomaly_seed_used=len(x_anomaly_seed),
    )
    tuned_result, tuned_final_pred = evaluate_randomforest_baseline(
        method="Real anomaly only + tuned RF hyperparameters",
        params=HYPERPARAMETER_APPLIED_PARAMS,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_final=x_final,
        y_final=y_final,
        normal_train_used=len(x_normal_sample),
        real_anomaly_seed_used=len(x_anomaly_seed),
    )

    result_df = pd.DataFrame([{k: v for k, v in result.items() if k != "params"}])
    result_df.to_csv(result_dir / "randomforest_original_baseline.csv", index=False)
    tuned_result_df = pd.DataFrame([{k: v for k, v in tuned_result.items() if k != "params"}])
    tuned_result_df.to_csv(
        result_dir / "randomforest_hyperparameter_applied_baseline.csv",
        index=False,
    )

    cm = confusion_matrix(y_final, final_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["normal", "anomaly"],
        yticklabels=["normal", "anomaly"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("RandomForest Original Baseline")
    plt.tight_layout()
    plt.savefig(figure_dir / "randomforest_original_baseline_confusion_matrix.png", dpi=200)
    plt.close()

    tuned_cm = confusion_matrix(y_final, tuned_final_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        tuned_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["normal", "anomaly"],
        yticklabels=["normal", "anomaly"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("RandomForest Tuned Hyperparameter Baseline")
    plt.tight_layout()
    plt.savefig(
        figure_dir / "randomforest_hyperparameter_applied_baseline_confusion_matrix.png",
        dpi=200,
    )
    plt.close()

    report = classification_report(
        y_final,
        final_pred,
        target_names=["normal", "anomaly"],
        zero_division=0,
        output_dict=True,
    )

    summary = {
        "purpose": "Original-data RandomForestClassifier baseline before generated anomaly augmentation.",
        "result": {
            k: to_json_value(v)
            for k, v in result.items()
            if k != "params"
        },
        "params": BASELINE_PARAMS,
        "classification_report": report,
    }
    with open(result_dir / "research01_randomforest_baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    tuned_report = classification_report(
        y_final,
        tuned_final_pred,
        target_names=["normal", "anomaly"],
        zero_division=0,
        output_dict=True,
    )
    tuned_summary = {
        "purpose": (
            "Original-data RandomForestClassifier baseline with the tuned hyperparameters "
            "selected from Masking Diffusion 1000 Research04 experiments, but without generated data."
        ),
        "result": {
            k: to_json_value(v)
            for k, v in tuned_result.items()
            if k != "params"
        },
        "params": HYPERPARAMETER_APPLIED_PARAMS,
        "classification_report": tuned_report,
    }
    with open(
        result_dir / "research01_hyperparameter_applied_baseline_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(tuned_summary, f, indent=2, ensure_ascii=False)

    print(result_df)
    print()
    print(tuned_result_df)
    print()
    print(
        classification_report(
            y_final,
            final_pred,
            target_names=["normal", "anomaly"],
            zero_division=0,
        )
    )
    print()
    print(
        classification_report(
            y_final,
            tuned_final_pred,
            target_names=["normal", "anomaly"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
