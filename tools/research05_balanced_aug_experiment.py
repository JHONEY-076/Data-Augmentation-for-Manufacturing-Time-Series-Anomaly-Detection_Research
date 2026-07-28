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
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


RANDOM_STATE = 42
NORMAL_TRAIN_SIZE = 5000
GENERATED_COUNT = 1000
QUALITY_FILTER_COUNTS = [500, 750, 900, 950]
QUALITY_RANGE_MULTIPLIER = 3.0
THRESHOLDS = np.linspace(0.05, 0.95, 91)
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
    return ensure_window_3d(data["X"].astype(np.float32)), data["y"].astype(np.int8)


def load_npz_x(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return ensure_window_3d(data["X"].astype(np.float32))


def ensure_window_3d(x: np.ndarray) -> np.ndarray:
    """Return windows as (n_windows, window_size, n_features)."""
    x = np.asarray(x)
    if x.ndim == 3:
        return x
    if x.ndim > 3:
        squeezed = np.squeeze(x)
        if squeezed.ndim == 3:
            return squeezed.astype(np.float32)
        if x.ndim == 4 and x.shape[1] == 1:
            return x[:, 0, :, :].astype(np.float32)
        if x.ndim == 4 and x.shape[-1] == 1:
            return x[:, :, :, 0].astype(np.float32)
    raise ValueError(f"Expected 3D window data, got shape {x.shape}")


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
            f"Rerun tools/research02_generate_data.py before running Research05. Missing keys: {sorted(missing)}"
        )
    return summary


def sample_rows(x: np.ndarray, n_sample: int, seed: int) -> np.ndarray:
    if len(x) <= n_sample:
        return x
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=n_sample, replace=False)
    return x[indices]


def quality_filter_generated_anomalies(
    x_generated: np.ndarray,
    x_anomaly_seed: np.ndarray,
    x_normal_train: np.ndarray,
    n_select: int,
) -> np.ndarray:
    """Keep generated anomalies close to real anomalies and away from normal windows."""
    x_generated = ensure_window_3d(x_generated)
    x_anomaly_seed = ensure_window_3d(x_anomaly_seed)
    x_normal_train = ensure_window_3d(x_normal_train)

    generated_flat = flatten_windows(x_generated)
    anomaly_flat = flatten_windows(x_anomaly_seed)
    normal_flat = flatten_windows(x_normal_train)

    reference_flat = np.concatenate([normal_flat, anomaly_flat], axis=0)
    reference_mean = reference_flat.mean(axis=0, keepdims=True)
    reference_std = reference_flat.std(axis=0, keepdims=True)
    reference_std = np.where(reference_std < 1e-6, 1.0, reference_std)

    generated_z = (generated_flat - reference_mean) / reference_std
    anomaly_z = (anomaly_flat - reference_mean) / reference_std
    normal_z = (normal_flat - reference_mean) / reference_std

    anomaly_centroid = anomaly_z.mean(axis=0, keepdims=True)
    normal_centroid = normal_z.mean(axis=0, keepdims=True)
    dist_to_anomaly = np.linalg.norm(generated_z - anomaly_centroid, axis=1)
    dist_to_normal = np.linalg.norm(generated_z - normal_centroid, axis=1)

    anomaly_mean = anomaly_flat.mean(axis=0, keepdims=True)
    anomaly_std = anomaly_flat.std(axis=0, keepdims=True)
    anomaly_std = np.where(anomaly_std < 1e-6, 1.0, anomaly_std)
    within_anomaly_range = (
        np.abs((generated_flat - anomaly_mean) / anomaly_std) <= QUALITY_RANGE_MULTIPLIER
    ).mean(axis=1)

    anomaly_radius = np.percentile(np.linalg.norm(anomaly_z - anomaly_centroid, axis=1), 95)
    normal_radius = np.percentile(np.linalg.norm(normal_z - normal_centroid, axis=1), 25)
    candidate_mask = (
        (dist_to_anomaly <= anomaly_radius * 1.25)
        & (dist_to_normal >= normal_radius * 0.75)
        & (within_anomaly_range >= 0.90)
    )

    quality_score = (
        -dist_to_anomaly
        + 0.5 * dist_to_normal
        + 10.0 * within_anomaly_range
    )
    candidate_indices = np.where(candidate_mask)[0]
    if len(candidate_indices) < n_select:
        candidate_indices = np.arange(len(x_generated))

    ranked_indices = candidate_indices[np.argsort(quality_score[candidate_indices])[::-1]]
    selected_indices = ranked_indices[: min(n_select, len(ranked_indices))]
    return x_generated[selected_indices].astype(np.float32)


def score_filter_generated_anomalies(
    x_generated: np.ndarray,
    x_anomaly_seed: np.ndarray,
    x_normal_train: np.ndarray,
    n_select: int,
    seed: int,
) -> np.ndarray:
    """Keep generated windows that a real-data classifier scores as anomaly-like."""
    x_generated = ensure_window_3d(x_generated)
    x_anomaly_seed = ensure_window_3d(x_anomaly_seed)
    x_normal_train = ensure_window_3d(x_normal_train)

    x_normal_sample = sample_rows(x_normal_train, NORMAL_TRAIN_SIZE, seed=seed + 31)
    x_train = np.concatenate([x_normal_sample, x_anomaly_seed], axis=0)
    y_train = np.concatenate(
        [
            np.zeros(len(x_normal_sample), dtype=np.int8),
            np.ones(len(x_anomaly_seed), dtype=np.int8),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(flatten_windows(x_train), y_train)
    anomaly_scores = model.predict_proba(flatten_windows(x_generated))[:, 1]
    ranked_indices = np.argsort(anomaly_scores)[::-1]
    selected_indices = ranked_indices[: min(n_select, len(ranked_indices))]
    return x_generated[selected_indices].astype(np.float32)


def select_threshold_by_validation_f1(
    y_true: np.ndarray,
    anomaly_proba: np.ndarray,
) -> dict:
    best = None
    for threshold in THRESHOLDS:
        pred = (anomaly_proba >= threshold).astype(np.int8)
        candidate = {
            "threshold": float(threshold),
            "validation_precision": precision_score(y_true, pred, zero_division=0),
            "validation_recall": recall_score(y_true, pred, zero_division=0),
            "validation_f1": f1_score(y_true, pred, zero_division=0),
            "validation_f2": fbeta_score(y_true, pred, beta=FBETA_BETA, zero_division=0),
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


def smooth_random_curve(length: int, n_features: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_knots = 4
    knot_x = np.linspace(0, length - 1, n_knots)
    curves = []
    for _ in range(n_features):
        knot_y = rng.normal(loc=1.0, scale=sigma, size=n_knots)
        curves.append(np.interp(np.arange(length), knot_x, knot_y))
    return np.stack(curves, axis=1).astype(np.float32)


def time_warp(x: np.ndarray, n_generate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    length = x.shape[1]
    for i in range(n_generate):
        source = x[rng.integers(0, len(x))]
        # Monotonic random time map keeps ordering while changing local speed.
        increments = rng.lognormal(mean=0.0, sigma=0.18, size=length)
        warped_time = np.cumsum(increments)
        warped_time = (warped_time - warped_time[0]) / (warped_time[-1] - warped_time[0])
        warped_time = warped_time * (length - 1)
        generated = np.empty_like(source)
        base_time = np.arange(length)
        for feature_idx in range(source.shape[1]):
            generated[:, feature_idx] = np.interp(base_time, warped_time, source[:, feature_idx])
        out.append(generated)
    return np.stack(out).astype(np.float32)


def magnitude_warp(x: np.ndarray, n_generate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_generate):
        source = x[rng.integers(0, len(x))]
        curve = smooth_random_curve(source.shape[0], source.shape[1], sigma=0.12, seed=seed + i)
        out.append(source * curve)
    return np.stack(out).astype(np.float32)


def weak_magnitude_warp_existing(x: np.ndarray, seed: int, sigma: float = 0.04) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for i, source in enumerate(ensure_window_3d(x)):
        curve = smooth_random_curve(source.shape[0], source.shape[1], sigma=sigma, seed=seed + i)
        jitter = rng.normal(loc=1.0, scale=sigma * 0.25, size=source.shape)
        out.append(source * curve * jitter)
    return np.stack(out).astype(np.float32)


def noise_injection(x: np.ndarray, n_generate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    feature_std = x.reshape(-1, x.shape[-1]).std(axis=0).reshape(1, 1, -1)
    out = []
    for _ in range(n_generate):
        source = x[rng.integers(0, len(x))]
        noise = rng.normal(loc=0.0, scale=0.04, size=source.shape) * feature_std
        out.append(source + noise)
    return np.stack(out).astype(np.float32)


def frequency_domain(x: np.ndarray, n_generate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    length = x.shape[1]
    for _ in range(n_generate):
        source = x[rng.integers(0, len(x))]
        spectrum = np.fft.rfft(source, axis=0)
        amplitude_scale = rng.normal(loc=1.0, scale=0.08, size=spectrum.shape)
        phase_shift = rng.normal(loc=0.0, scale=0.08, size=spectrum.shape)
        phase_shift[0, :] = 0.0
        if length % 2 == 0:
            phase_shift[-1, :] = 0.0
        perturbed = spectrum * amplitude_scale * np.exp(1j * phase_shift)
        generated = np.fft.irfft(perturbed, n=length, axis=0)
        out.append(generated)
    return np.stack(out).astype(np.float32)


def comprehensive_augmentation(x: np.ndarray, n_generate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    length = x.shape[1]
    feature_std = x.reshape(-1, x.shape[-1]).std(axis=0).reshape(1, -1)
    base_time = np.arange(length)
    for i in range(n_generate):
        source = x[rng.integers(0, len(x))]

        increments = rng.lognormal(mean=0.0, sigma=0.12, size=length)
        warped_time = np.cumsum(increments)
        warped_time = (warped_time - warped_time[0]) / (warped_time[-1] - warped_time[0])
        warped_time = warped_time * (length - 1)
        generated = np.empty_like(source)
        for feature_idx in range(source.shape[1]):
            generated[:, feature_idx] = np.interp(base_time, warped_time, source[:, feature_idx])

        curve = smooth_random_curve(length, source.shape[1], sigma=0.08, seed=seed + i)
        generated = generated * curve

        spectrum = np.fft.rfft(generated, axis=0)
        amplitude_scale = rng.normal(loc=1.0, scale=0.04, size=spectrum.shape)
        phase_shift = rng.normal(loc=0.0, scale=0.04, size=spectrum.shape)
        phase_shift[0, :] = 0.0
        if length % 2 == 0:
            phase_shift[-1, :] = 0.0
        generated = np.fft.irfft(spectrum * amplitude_scale * np.exp(1j * phase_shift), n=length, axis=0)

        noise = rng.normal(loc=0.0, scale=0.02, size=generated.shape) * feature_std
        out.append(generated + noise)
    return np.stack(out).astype(np.float32)


def evaluate_condition(
    condition: str,
    augmentation_family: str,
    method: str,
    x_normal_train: np.ndarray,
    x_anomaly_seed: np.ndarray,
    x_generated: np.ndarray | None,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_final: np.ndarray,
    y_final: np.ndarray,
    seed: int,
) -> dict:
    x_normal_train = ensure_window_3d(x_normal_train)
    x_anomaly_seed = ensure_window_3d(x_anomaly_seed)
    x_val = ensure_window_3d(x_val)
    x_final = ensure_window_3d(x_final)

    if x_generated is None:
        x_anomaly_train = x_anomaly_seed
        generated_used = 0
    else:
        x_generated = ensure_window_3d(x_generated)
        generated_sample = sample_rows(x_generated, GENERATED_COUNT, seed=seed + 11)
        generated_sample = ensure_window_3d(generated_sample)
        x_anomaly_train = np.concatenate([x_anomaly_seed, generated_sample], axis=0)
        generated_used = len(generated_sample)

    if condition == "imbalanced":
        normal_used = min(NORMAL_TRAIN_SIZE, len(x_normal_train))
    elif condition == "balanced":
        normal_used = min(len(x_anomaly_train), len(x_normal_train))
    else:
        raise ValueError(f"Unknown condition: {condition}")

    x_normal_sample = sample_rows(x_normal_train, normal_used, seed=seed + 23)
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
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(flatten_windows(x_train), y_train)

    val_proba = model.predict_proba(flatten_windows(x_val))[:, 1]
    final_proba = model.predict_proba(flatten_windows(x_final))[:, 1]
    threshold_result = select_threshold_by_validation_f1(y_val, val_proba)
    threshold = threshold_result["threshold"]
    final_pred = (final_proba >= threshold).astype(np.int8)

    return {
        "condition": condition,
        "augmentation_family": augmentation_family,
        "method": method,
        "normal_train_used": int(len(x_normal_sample)),
        "real_anomaly_seed_used": int(len(x_anomaly_seed)),
        "generated_anomaly_used": int(generated_used),
        "total_anomaly_train_used": int(len(x_anomaly_train)),
        "normal_to_anomaly_ratio": float(len(x_normal_sample) / len(x_anomaly_train)),
        "total_train_rows": int(len(x_train)),
        "threshold": threshold,
        "validation_precision": float(threshold_result["validation_precision"]),
        "validation_recall": float(threshold_result["validation_recall"]),
        "validation_f1": float(threshold_result["validation_f1"]),
        "validation_f2": float(threshold_result["validation_f2"]),
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
    }


def save_plots(results_df: pd.DataFrame, figure_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    metric_df = results_df.melt(
        id_vars=["condition", "augmentation_family", "method"],
        value_vars=["f1", "recall", "auprc"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=metric_df, x="method", y="score", hue="condition")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Imbalanced vs Balanced Training Conditions")
    plt.tight_layout()
    plt.savefig(figure_dir / "balanced_vs_imbalanced_scores.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        data=results_df,
        x="normal_to_anomaly_ratio",
        y="f1",
        hue="augmentation_family",
        style="condition",
        s=120,
    )
    plt.xscale("log")
    plt.ylim(0, 1.05)
    plt.title("F1 by Normal-to-Anomaly Training Ratio")
    plt.tight_layout()
    plt.savefig(figure_dir / "f1_by_class_ratio.png", dpi=200)
    plt.close()


def main() -> None:
    root = Path.cwd()
    split_dir = root / "data" / "augmentation_split"
    generated_dir = root / "data" / "research02" / "generated"
    result_dir = root / "data" / "research05" / "results"
    figure_dir = root / "data" / "research05" / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    generation_summary = load_research02_generation_summary(root)

    x_normal_train, _ = load_npz_xy(split_dir / "normal_train_windows.npz")
    x_anomaly_seed, _ = load_npz_xy(split_dir / "anomaly_train_seed_windows.npz")
    x_val, y_val = load_npz_xy(split_dir / "validation_windows.npz")
    x_final, y_final = load_npz_xy(split_dir / "final_test_windows.npz")

    generated_sets = {
        "Time warping": ("traditional", time_warp(x_anomaly_seed, GENERATED_COUNT, RANDOM_STATE + 101)),
        "Magnitude warping": ("traditional", magnitude_warp(x_anomaly_seed, GENERATED_COUNT, RANDOM_STATE + 102)),
        "Noise injection": ("traditional", noise_injection(x_anomaly_seed, GENERATED_COUNT, RANDOM_STATE + 103)),
        "Frequency domain": ("traditional", frequency_domain(x_anomaly_seed, GENERATED_COUNT, RANDOM_STATE + 104)),
        "Comprehensive": (
            "traditional",
            comprehensive_augmentation(x_anomaly_seed, GENERATED_COUNT, RANDOM_STATE + 105),
        ),
        "GT-GAN": ("generative", load_npz_x(generated_dir / "gtgan_series_windows.npz")),
        "Masking GT-GAN": ("generative", load_npz_x(generated_dir / "gtgan_masked_windows.npz")),
        "Diffusion": ("generative", load_npz_x(generated_dir / "diffusion_series_windows.npz")),
        "Masking Diffusion": ("generative", load_npz_x(generated_dir / "diffusion_masked_windows.npz")),
    }

    filtered_generated_sets = {}
    for method, (family, generated_x) in generated_sets.items():
        if family != "generative":
            continue
        for filter_count in QUALITY_FILTER_COUNTS:
            filtered_generated_sets[f"Filtered@{filter_count} {method}"] = (
                "generative_filtered",
                quality_filter_generated_anomalies(
                    x_generated=generated_x,
                    x_anomaly_seed=x_anomaly_seed,
                    x_normal_train=x_normal_train,
                    n_select=filter_count,
                ),
            )
            filtered_generated_sets[f"ScoreFiltered@{filter_count} {method}"] = (
                "generative_score_filtered",
                score_filter_generated_anomalies(
                    x_generated=generated_x,
                    x_anomaly_seed=x_anomaly_seed,
                    x_normal_train=x_normal_train,
                    n_select=filter_count,
                    seed=RANDOM_STATE + filter_count,
                ),
            )
    generated_sets.update(filtered_generated_sets)

    score_filtered_masking_diffusion_750 = score_filter_generated_anomalies(
        x_generated=generated_sets["Masking Diffusion"][1],
        x_anomaly_seed=x_anomaly_seed,
        x_normal_train=x_normal_train,
        n_select=750,
        seed=RANDOM_STATE + 750,
    )
    generated_sets["Hybrid ScoreFiltered@750 Masking Diffusion + Weak Magnitude"] = (
        "generative_hybrid",
        weak_magnitude_warp_existing(
            score_filtered_masking_diffusion_750,
            seed=RANDOM_STATE + 751,
            sigma=0.04,
        ),
    )

    rows = []
    for condition in ["imbalanced", "balanced"]:
        rows.append(
            evaluate_condition(
                condition=condition,
                augmentation_family="none",
                method="Original only",
                x_normal_train=x_normal_train,
                x_anomaly_seed=x_anomaly_seed,
                x_generated=None,
                x_val=x_val,
                y_val=y_val,
                x_final=x_final,
                y_final=y_final,
                seed=RANDOM_STATE,
            )
        )
        for idx, (method, (family, generated_x)) in enumerate(generated_sets.items()):
            rows.append(
                evaluate_condition(
                    condition=condition,
                    augmentation_family=family,
                    method=method,
                    x_normal_train=x_normal_train,
                    x_anomaly_seed=x_anomaly_seed,
                    x_generated=generated_x,
                    x_val=x_val,
                    y_val=y_val,
                    x_final=x_final,
                    y_final=y_final,
                    seed=RANDOM_STATE + 100 * (idx + 1),
                )
            )

    results_df = pd.DataFrame(rows).sort_values(["condition", "f1"], ascending=[True, False])
    results_df.to_csv(result_dir / "research05_balanced_augmentation_comparison.csv", index=False)

    best_by_condition = results_df.loc[results_df.groupby("condition")["f1"].idxmax()]
    best_by_family = results_df.loc[results_df.groupby(["condition", "augmentation_family"])["f1"].idxmax()]
    best_by_condition.to_csv(result_dir / "research05_best_by_condition.csv", index=False)
    best_by_family.to_csv(result_dir / "research05_best_by_family.csv", index=False)

    save_plots(results_df, figure_dir)

    summary = {
        "research_question": [
            "Are generative time-series augmentations better than traditional augmentations?",
            "Is the performance change caused by more generated samples or by class-ratio balancing?",
        ],
        "conditions": {
            "imbalanced": f"Use {NORMAL_TRAIN_SIZE} normal windows and anomaly seed plus generated anomaly windows.",
            "balanced": "Downsample normal training windows to match the number of anomaly training windows.",
        },
        "generated_count_per_method": GENERATED_COUNT,
        "quality_filtered_counts": QUALITY_FILTER_COUNTS,
        "traditional_augmentation_methods": [
            "Time warping",
            "Magnitude warping",
            "Frequency domain",
            "Noise injection",
            "Comprehensive",
        ],
        "research02_generation_summary": generation_summary,
        "masking_method_note": (
            "Masking GT-GAN and Masking Diffusion use the generated windows produced by Research02. "
            "When Research02 is rerun with structured masking, Research05 evaluates temporal block "
            "and feature-group masked generation rather than element-wise random masking."
        ),
        "quality_filtering_note": (
            "Filtered generative methods rank generated windows by closeness to the real anomaly centroid, "
            "separation from the normal centroid, and consistency with real-anomaly feature ranges before "
            "downstream classifier training. ScoreFiltered methods train a real-data-only RandomForest on "
            "the training split and keep generated windows with the highest anomaly probabilities."
        ),
        "hybrid_augmentation_note": (
            "The hybrid method applies weak magnitude warping to ScoreFiltered@750 Masking Diffusion windows "
            "to increase local anomaly variability after quality-aware generative filtering."
        ),
        "metrics_prioritized": ["f1", "recall", "auprc"],
        "best_by_condition": best_by_condition.to_dict(orient="records"),
        "best_by_family": best_by_family.to_dict(orient="records"),
    }
    with open(result_dir / "research05_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(results_df)
    print()
    print("Best by condition:")
    print(best_by_condition)
    print()
    print("Best by condition and augmentation family:")
    print(best_by_family)


if __name__ == "__main__":
    main()
