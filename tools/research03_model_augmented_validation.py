from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


RANDOM_STATE = 42
NORMAL_TRAIN_SIZE = 5000
THRESHOLDS = np.linspace(0.05, 0.95, 91)
GENERATED_COUNTS = [50, 100, 200, 500, 1000]
TARGET_RECALL = 0.90
FBETA_BETA = 2.0
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
            f"Rerun tools/research02_generate_data.py before running Research03. Missing keys: {sorted(missing)}"
        )
    return summary


def select_threshold_by_validation_f1(
    y_true: np.ndarray,
    anomaly_proba: np.ndarray,
) -> dict:
    best_threshold = 0.5
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for threshold in THRESHOLDS:
        pred = (anomaly_proba >= threshold).astype(np.int8)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        f2 = fbeta_score(y_true, pred, beta=FBETA_BETA, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_threshold = float(threshold)

    return {
        "threshold": best_threshold,
        "validation_precision": float(best_precision),
        "validation_recall": float(best_recall),
        "validation_f1": float(best_f1),
        "validation_f2": float(fbeta_score(
            y_true,
            (anomaly_proba >= best_threshold).astype(np.int8),
            beta=FBETA_BETA,
            zero_division=0,
        )),
        "validation_recall_target_met": bool(best_recall >= TARGET_RECALL),
    }


def select_threshold_by_validation_f2(
    y_true: np.ndarray,
    anomaly_proba: np.ndarray,
) -> dict:
    best = None

    for threshold in THRESHOLDS:
        pred = (anomaly_proba >= threshold).astype(np.int8)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        f2 = fbeta_score(y_true, pred, beta=FBETA_BETA, zero_division=0)
        candidate = {
            "threshold": float(threshold),
            "validation_precision": float(precision),
            "validation_recall": float(recall),
            "validation_f1": float(f1),
            "validation_f2": float(f2),
            "validation_recall_target_met": bool(recall >= TARGET_RECALL),
        }
        if best is None or (
            candidate["validation_f2"],
            candidate["validation_precision"],
            candidate["threshold"],
        ) > (
            best["validation_f2"],
            best["validation_precision"],
            best["threshold"],
        ):
            best = candidate

    if best is None:
        raise RuntimeError("No threshold candidates were evaluated.")
    return best


def select_threshold_by_target_recall(
    y_true: np.ndarray,
    anomaly_proba: np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> dict:
    candidates = []

    for threshold in THRESHOLDS:
        pred = (anomaly_proba >= threshold).astype(np.int8)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        f2 = fbeta_score(y_true, pred, beta=FBETA_BETA, zero_division=0)
        candidates.append(
            {
                "threshold": float(threshold),
                "validation_precision": float(precision),
                "validation_recall": float(recall),
                "validation_f1": float(f1),
                "validation_f2": float(f2),
                "validation_recall_target_met": bool(recall >= target_recall),
            }
        )

    target_candidates = [
        candidate for candidate in candidates if candidate["validation_recall"] >= target_recall
    ]
    if target_candidates:
        return max(
            target_candidates,
            key=lambda candidate: (
                candidate["validation_precision"],
                candidate["validation_f1"],
                candidate["threshold"],
            ),
        )

    return max(
        candidates,
        key=lambda candidate: (
            candidate["validation_recall"],
            candidate["validation_precision"],
            candidate["validation_f1"],
        ),
    )


def evaluate_supervised_method(
    method: str,
    x_normal_train: np.ndarray,
    x_anomaly_seed: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_final: np.ndarray,
    y_final: np.ndarray,
    normal_indices: np.ndarray,
    generated_x: np.ndarray | None = None,
    threshold_strategy: str = "best_validation_f1",
) -> dict:
    x_normal_sample = x_normal_train[normal_indices]
    x_anomaly_train = x_anomaly_seed if generated_x is None else np.concatenate(
        [x_anomaly_seed, generated_x],
        axis=0,
    )

    x_train = np.concatenate([x_normal_sample, x_anomaly_train], axis=0)
    y_train = np.concatenate(
        [
            np.zeros(len(x_normal_sample), dtype=np.int8),
            np.ones(len(x_anomaly_train), dtype=np.int8),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(flatten_windows(x_train), y_train)

    val_proba = model.predict_proba(flatten_windows(x_val))[:, 1]
    if threshold_strategy == "recall90_precision":
        threshold_result = select_threshold_by_target_recall(y_val, val_proba)
    elif threshold_strategy == "best_validation_f2":
        threshold_result = select_threshold_by_validation_f2(y_val, val_proba)
    elif threshold_strategy == "best_validation_f1":
        threshold_result = select_threshold_by_validation_f1(y_val, val_proba)
    else:
        raise ValueError(f"Unknown threshold strategy: {threshold_strategy}")

    threshold = threshold_result["threshold"]

    final_proba = model.predict_proba(flatten_windows(x_final))[:, 1]
    final_pred = (final_proba >= threshold).astype(np.int8)

    result = {
        "method": method,
        "model": "RandomForestClassifier",
        "normal_train_used": int(len(x_normal_sample)),
        "real_anomaly_seed_used": int(len(x_anomaly_seed)),
        "generated_anomaly_used": 0 if generated_x is None else int(len(generated_x)),
        "total_train_rows": int(len(x_train)),
        "threshold_strategy": threshold_strategy,
        "target_recall": TARGET_RECALL if threshold_strategy == "recall90_precision" else np.nan,
        "fbeta_beta": FBETA_BETA if threshold_strategy == "best_validation_f2" else np.nan,
        "threshold_selected_on_real_validation": threshold,
        "validation_precision": threshold_result["validation_precision"],
        "validation_recall": threshold_result["validation_recall"],
        "validation_f1": threshold_result["validation_f1"],
        "validation_f2": threshold_result["validation_f2"],
        "validation_recall_target_met": threshold_result["validation_recall_target_met"],
        "precision": precision_score(y_final, final_pred, zero_division=0),
        "recall": recall_score(y_final, final_pred, zero_division=0),
        "f1": f1_score(y_final, final_pred, zero_division=0),
        "f2": fbeta_score(y_final, final_pred, beta=FBETA_BETA, zero_division=0),
        "auroc": roc_auc_score(y_final, final_proba),
        "auprc": average_precision_score(y_final, final_proba),
        "pred_normal": int((final_pred == 0).sum()),
        "pred_anomaly": int((final_pred == 1).sum()),
        "true_normal": int((y_final == 0).sum()),
        "true_anomaly": int((y_final == 1).sum()),
        "final_pred": final_pred,
    }
    return result


def sample_generated_windows(
    generated_x: np.ndarray,
    n_sample: int,
    seed: int,
) -> np.ndarray:
    if len(generated_x) <= n_sample:
        return generated_x
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(generated_x), size=n_sample, replace=False)
    return generated_x[indices]


def save_metric_plot(performance_df: pd.DataFrame, figure_dir: Path) -> None:
    plot_df = performance_df[performance_df["generated_anomaly_used"] > 0].melt(
        id_vars=["method", "generated_anomaly_used"],
        value_vars=["f1", "auprc", "recall"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=plot_df,
        x="generated_anomaly_used",
        y="score",
        hue="method",
        style="metric",
        marker="o",
    )
    plt.title("Performance by Number of Generated Anomaly Windows")
    plt.xlabel("Generated anomaly windows used for training")
    plt.ylabel("Final test score")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(figure_dir / "supervised_model_augmented_performance.png", dpi=200)
    plt.close()


def save_focus_plot(performance_df: pd.DataFrame, figure_dir: Path) -> None:
    focus_df = performance_df[performance_df["generated_anomaly_used"] > 0].melt(
        id_vars=["method", "generated_anomaly_used"],
        value_vars=["f1", "auprc"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=focus_df,
        x="generated_anomaly_used",
        y="score",
        hue="method",
        style="metric",
        marker="o",
    )
    plt.title("F1 and AUPRC by Generated Data Amount")
    plt.xlabel("Generated anomaly windows used for training")
    plt.ylabel("Final test score")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(figure_dir / "supervised_model_augmented_f1_recall_auprc.png", dpi=200)
    plt.close()


def save_best_per_count_plot(performance_df: pd.DataFrame, figure_dir: Path) -> None:
    augmented_df = performance_df[performance_df["generated_anomaly_used"] > 0].copy()
    best_by_count = augmented_df.loc[
        augmented_df.groupby("generated_anomaly_used")["f1"].idxmax()
    ].sort_values("generated_anomaly_used")

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=best_by_count,
        x="generated_anomaly_used",
        y="f1",
        marker="o",
        label="Best F1",
    )
    sns.lineplot(
        data=best_by_count,
        x="generated_anomaly_used",
        y="auprc",
        marker="o",
        label="Best AUPRC",
    )
    plt.title("Best Method Performance by Generated Data Amount")
    plt.xlabel("Generated anomaly windows used for training")
    plt.ylabel("Final test score")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(figure_dir / "best_per_generated_count.png", dpi=200)
    plt.close()


def save_best_confusion_matrix(
    y_final: np.ndarray,
    best_result: dict,
    figure_dir: Path,
) -> None:
    cm = confusion_matrix(y_final, best_result["final_pred"])

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
    plt.title(f"Best Supervised Method\n{best_result['method']}")
    plt.tight_layout()
    plt.savefig(figure_dir / "best_supervised_confusion_matrix.png", dpi=200)
    plt.close()


def main() -> None:
    root = Path.cwd()
    split_dir = root / "data" / "augmentation_split"
    generated_dir = root / "data" / "research02" / "generated"
    quality_path = root / "data" / "research02" / "results" / "generated_quality_metrics.csv"
    generation_summary = load_research02_generation_summary(root)

    result_dir = root / "data" / "research03" / "results"
    figure_dir = root / "data" / "research03" / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    x_normal_train, _ = load_npz_xy(split_dir / "normal_train_windows.npz")
    x_anomaly_seed, _ = load_npz_xy(split_dir / "anomaly_train_seed_windows.npz")
    x_val, y_val = load_npz_xy(split_dir / "validation_windows.npz")
    x_final, y_final = load_npz_xy(split_dir / "final_test_windows.npz")

    generated_sets = {
        "GT-GAN": load_npz_x(generated_dir / "gtgan_series_windows.npz"),
        "Masking GT-GAN": load_npz_x(generated_dir / "gtgan_masked_windows.npz"),
        "Diffusion": load_npz_x(generated_dir / "diffusion_series_windows.npz"),
        "Masking Diffusion": load_npz_x(generated_dir / "diffusion_masked_windows.npz"),
    }

    rng = np.random.default_rng(RANDOM_STATE)
    normal_indices = rng.choice(
        len(x_normal_train),
        size=min(NORMAL_TRAIN_SIZE, len(x_normal_train)),
        replace=False,
    )

    def collect_results(threshold_strategy: str) -> list[dict]:
        strategy_results = [
            evaluate_supervised_method(
                method="Real anomaly only",
                x_normal_train=x_normal_train,
                x_anomaly_seed=x_anomaly_seed,
                x_val=x_val,
                y_val=y_val,
                x_final=x_final,
                y_final=y_final,
                normal_indices=normal_indices,
                generated_x=None,
                threshold_strategy=threshold_strategy,
            )
        ]

        for method_index, (method, generated_x) in enumerate(generated_sets.items()):
            for n_generated in GENERATED_COUNTS:
                generated_sample = sample_generated_windows(
                    generated_x,
                    n_generated,
                    seed=RANDOM_STATE + 1000 * (method_index + 1) + n_generated,
                )
                strategy_results.append(
                    evaluate_supervised_method(
                        method=method,
                        x_normal_train=x_normal_train,
                        x_anomaly_seed=x_anomaly_seed,
                        x_val=x_val,
                        y_val=y_val,
                        x_final=x_final,
                        y_final=y_final,
                        normal_indices=normal_indices,
                        generated_x=generated_sample,
                        threshold_strategy=threshold_strategy,
                    )
                )

        return strategy_results

    results = collect_results("best_validation_f1")
    f2_results = collect_results("best_validation_f2")
    recall90_results = collect_results("recall90_precision")

    performance_df = pd.DataFrame(
        [{k: v for k, v in result.items() if k != "final_pred"} for result in results]
    ).sort_values("f1", ascending=False)

    performance_df.to_csv(
        result_dir / "supervised_model_augmented_performance.csv",
        index=False,
    )

    f2_performance_df = pd.DataFrame(
        [{k: v for k, v in result.items() if k != "final_pred"} for result in f2_results]
    ).sort_values("f2", ascending=False)

    f2_performance_df.to_csv(
        result_dir / "supervised_model_augmented_f2_performance.csv",
        index=False,
    )

    recall90_performance_df = pd.DataFrame(
        [{k: v for k, v in result.items() if k != "final_pred"} for result in recall90_results]
    ).sort_values(["recall", "precision", "f1"], ascending=False)

    recall90_performance_df.to_csv(
        result_dir / "supervised_model_augmented_recall90_performance.csv",
        index=False,
    )

    save_metric_plot(performance_df, figure_dir)
    save_focus_plot(performance_df, figure_dir)
    save_best_per_count_plot(performance_df, figure_dir)

    best_result = max(results, key=lambda result: result["f1"])
    best_f2_result = max(f2_results, key=lambda result: result["f2"])
    best_recall90_result = max(
        recall90_results,
        key=lambda result: (
            result["recall"],
            result["precision"],
            result["f1"],
        ),
    )
    save_best_confusion_matrix(y_final, best_result, figure_dir)

    report = classification_report(
        y_final,
        best_result["final_pred"],
        target_names=["normal", "anomaly"],
        zero_division=0,
        output_dict=True,
    )

    quality_ranking = []
    if quality_path.exists():
        quality_df = pd.read_csv(quality_path)
        quality_ranking = quality_df["method"].tolist()

    summary = {
        "claim_tested": "Generated anomaly windows are added to supervised model training, not to threshold tuning.",
        "model": "RandomForestClassifier",
        "normal_train_used_each_method": int(len(normal_indices)),
        "real_anomaly_seed_used_each_method": int(len(x_anomaly_seed)),
        "generated_anomaly_counts": GENERATED_COUNTS,
        "threshold_selection": "Best F1 on real validation_windows only.",
        "research02_generation_summary": generation_summary,
        "quality_ranking_from_research02": quality_ranking,
        "performance_ranking_by_f1": performance_df["method"].tolist(),
        "best_method": best_result["method"],
        "best_result": {
            k: to_json_value(v)
            for k, v in best_result.items()
            if k != "final_pred"
        },
        "best_classification_report": report,
    }

    with open(result_dir / "research03_model_improvement_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    f2_report = classification_report(
        y_final,
        best_f2_result["final_pred"],
        target_names=["normal", "anomaly"],
        zero_division=0,
        output_dict=True,
    )

    f2_summary = {
        "claim_tested": (
            "Threshold is selected on real validation_windows by F2-score, which weights "
            "recall higher than precision without forcing recall to a fixed target."
        ),
        "model": "RandomForestClassifier",
        "normal_train_used_each_method": int(len(normal_indices)),
        "real_anomaly_seed_used_each_method": int(len(x_anomaly_seed)),
        "generated_anomaly_counts": GENERATED_COUNTS,
        "threshold_selection": "Best F2-score on real validation_windows only.",
        "fbeta_beta": FBETA_BETA,
        "research02_generation_summary": generation_summary,
        "quality_ranking_from_research02": quality_ranking,
        "performance_ranking_by_final_f2": f2_performance_df["method"].tolist(),
        "best_method_by_final_f2": best_f2_result["method"],
        "best_result_by_final_f2": {
            k: to_json_value(v)
            for k, v in best_f2_result.items()
            if k != "final_pred"
        },
        "best_classification_report": f2_report,
    }

    with open(result_dir / "research03_f2_threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump(f2_summary, f, indent=2, ensure_ascii=False)

    recall90_report = classification_report(
        y_final,
        best_recall90_result["final_pred"],
        target_names=["normal", "anomaly"],
        zero_division=0,
        output_dict=True,
    )

    recall90_summary = {
        "claim_tested": (
            "Threshold is selected on real validation_windows to target recall >= 0.90, "
            "then final test performance is evaluated without changing the final test labels."
        ),
        "model": "RandomForestClassifier",
        "normal_train_used_each_method": int(len(normal_indices)),
        "real_anomaly_seed_used_each_method": int(len(x_anomaly_seed)),
        "generated_anomaly_counts": GENERATED_COUNTS,
        "threshold_selection": (
            "Among thresholds with validation recall >= 0.90, choose the threshold with "
            "highest validation precision. If none meet target recall, choose highest validation recall."
        ),
        "target_recall": TARGET_RECALL,
        "research02_generation_summary": generation_summary,
        "quality_ranking_from_research02": quality_ranking,
        "performance_ranking_by_final_recall": recall90_performance_df["method"].tolist(),
        "best_method_by_final_recall": best_recall90_result["method"],
        "best_result_by_final_recall": {
            k: to_json_value(v)
            for k, v in best_recall90_result.items()
            if k != "final_pred"
        },
        "best_classification_report": recall90_report,
    }

    with open(result_dir / "research03_recall90_threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump(recall90_summary, f, indent=2, ensure_ascii=False)

    print(performance_df)
    print()
    print("F2 threshold results:")
    print(f2_performance_df)
    print()
    print("Recall-target threshold results:")
    print(recall90_performance_df)
    print()
    print("Best method:", best_result["method"])
    print("Best final-test classification report:")
    print(
        classification_report(
            y_final,
            best_result["final_pred"],
            target_names=["normal", "anomaly"],
            zero_division=0,
        )
    )
    print()
    print("Best F2 method:", best_f2_result["method"])
    print("Best F2 final-test classification report:")
    print(
        classification_report(
            y_final,
            best_f2_result["final_pred"],
            target_names=["normal", "anomaly"],
            zero_division=0,
        )
    )
    print()
    print("Best recall-target method:", best_recall90_result["method"])
    print("Best recall-target final-test classification report:")
    print(
        classification_report(
            y_final,
            best_recall90_result["final_pred"],
            target_names=["normal", "anomaly"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
