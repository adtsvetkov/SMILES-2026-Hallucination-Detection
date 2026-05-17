from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from model import MAX_LENGTH, get_model_and_tokenizer


DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"

OUTPUT_DIR = Path("./artifacts/geometric_uncertainty_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUTPUT = OUTPUT_DIR / "features_dataset_geometric_uncertainty_v2.parquet"
TEST_OUTPUT = OUTPUT_DIR / "features_test_geometric_uncertainty_v2.parquet"

BATCH_SIZE = 4

LAYERS = [11, 12, 13, 14, 15, 16]
LAYER_PAIRS = list(zip(LAYERS[:-1], LAYERS[1:]))

EPS = 1e-8

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()

    return np.asarray(x, dtype=np.float32)

import re
from collections import Counter


UNCERTAINTY_WORDS = [
    "maybe",
    "perhaps",
    "possibly",
    "likely",
    "probably",
    "might",
    "could",
    "unclear",
    "unsure",
    "unknown",
    "approximately",
    "appears",
    "seems",
]

REFUSAL_WORDS = [
    "cannot",
    "can't",
    "unable",
    "not able",
    "sorry",
    "i do not",
    "i don't",
    "i cannot",
]


def safe_div(a, b):
    return float(a) / (float(b) + EPS)


def repetition_ratio(words):
    if len(words) == 0:
        return 0.0

    counts = Counter(words)
    repeated = sum(v for v in counts.values() if v > 1)

    return repeated / len(words)


def unmatched_symbol_count(text, left_symbol, right_symbol):
    return abs(text.count(left_symbol) - text.count(right_symbol))


def safe_mean(x, mask):
    if mask.sum().item() == 0:
        return torch.zeros(x.shape[-1], dtype=x.dtype)

    return x[mask].mean(dim=0)


def safe_l2(x):
    x = to_numpy(x)
    return float(np.linalg.norm(x))


def safe_cosine(a, b):
    a = to_numpy(a)
    b = to_numpy(b)

    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b) + EPS)
    )


def clean_value(x):
    if not np.isfinite(x):
        return 0.0

    return float(x)

def make_token_masks(valid_mask):
    valid_mask = valid_mask.bool().cpu()
    valid_positions = torch.where(valid_mask)[0]

    if valid_positions.numel() == 0:
        return {
            "all": valid_mask.clone(),
            "first70": valid_mask.clone(),
            "last40": valid_mask.clone(),
            "last30": valid_mask.clone(),
            "last20": valid_mask.clone(),
            "last5": valid_mask.clone(),
            "last_token": valid_mask.clone(),
        }

    n_tokens = int(valid_positions.numel())

    def last_fraction(frac):
        n_keep = max(1, int(round(n_tokens * frac)))
        positions = valid_positions[-n_keep:]

        mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        mask[positions] = True

        return mask

    def last_n(n):
        n_keep = max(1, min(int(n), n_tokens))
        positions = valid_positions[-n_keep:]

        mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        mask[positions] = True

        return mask

    last30 = last_fraction(0.30)

    first70 = valid_mask & (~last30)
    if first70.sum().item() == 0:
        first70 = valid_mask.clone()

    return {
        "all": valid_mask,
        "first70": first70,
        "last40": last_fraction(0.40),
        "last30": last30,
        "last20": last_fraction(0.20),
        "last5": last_n(5),
        "last_token": last_n(1),
    }

def value_entropy(values):
    values = np.abs(np.asarray(values, dtype=np.float32))
    total = values.sum()

    if total <= EPS:
        return 0.0

    probs = values / total
    return float(-(probs * np.log(probs + EPS)).sum())


def value_gini(values):
    values = np.sort(np.abs(np.asarray(values, dtype=np.float32)))

    if len(values) == 0 or values.sum() <= EPS:
        return 0.0

    n = len(values)
    ranks = np.arange(1, n + 1)

    return float(
        (2 * np.sum(ranks * values) / (n * values.sum()))
        - ((n + 1) / n)
    )

def weighted_token_mean(token_matrix):
    token_matrix = token_matrix.float()

    if token_matrix.shape[0] == 0:
        return torch.zeros(token_matrix.shape[-1], dtype=token_matrix.dtype)

    n_tokens = token_matrix.shape[0]

    weights = torch.linspace(
        0.5,
        1.5,
        n_tokens,
        dtype=token_matrix.dtype,
        device=token_matrix.device,
    )

    weights = weights / (weights.sum() + EPS)

    return (token_matrix * weights.unsqueeze(-1)).sum(dim=0)


def activation_entropy_stats(token_matrix):
    token_matrix = token_matrix.float()

    if token_matrix.shape[0] == 0:
        return {
            "entropy_mean": 0.0,
            "entropy_std": 0.0,
            "entropy_min": 0.0,
            "entropy_max": 0.0,
        }

    probs = torch.softmax(token_matrix, dim=-1)
    entropy = -(probs * torch.log(probs + EPS)).sum(dim=-1)

    entropy_np = entropy.detach().cpu().numpy()

    return {
        "entropy_mean": float(entropy_np.mean()),
        "entropy_std": float(entropy_np.std()),
        "entropy_min": float(entropy_np.min()),
        "entropy_max": float(entropy_np.max()),
    }


def activation_max_stats(token_matrix):
    token_matrix = token_matrix.float()

    if token_matrix.shape[0] == 0:
        return {
            "max_mean": 0.0,
            "max_std": 0.0,
            "max_p90": 0.0,
            "max_p95": 0.0,
        }

    max_values = token_matrix.max(dim=1).values.detach().cpu().numpy()

    return {
        "max_mean": float(max_values.mean()),
        "max_std": float(max_values.std()),
        "max_p90": float(np.percentile(max_values, 90)),
        "max_p95": float(np.percentile(max_values, 95)),
    }


def activation_variance_stats(token_matrix):
    token_matrix = token_matrix.float()

    if token_matrix.shape[0] <= 1:
        return {
            "var_mean": 0.0,
            "var_std": 0.0,
            "var_min": 0.0,
            "var_max": 0.0,
            "var_p90": 0.0,
            "var_p95": 0.0,
            "var_iqr": 0.0,
        }

    var_values = token_matrix.var(
        dim=0,
        unbiased=False,
    ).detach().cpu().numpy()

    p25 = float(np.percentile(var_values, 25))
    p75 = float(np.percentile(var_values, 75))

    return {
        "var_mean": float(var_values.mean()),
        "var_std": float(var_values.std()),
        "var_min": float(var_values.min()),
        "var_max": float(var_values.max()),
        "var_p90": float(np.percentile(var_values, 90)),
        "var_p95": float(np.percentile(var_values, 95)),
        "var_iqr": p75 - p25,
    }

def add_trajectory_features(features, prefix, values):
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if len(values) == 0:
        values = np.array([0.0], dtype=np.float32)

    diff1 = np.diff(values) if len(values) >= 2 else np.array([0.0], dtype=np.float32)
    diff2 = np.diff(values, n=2) if len(values) >= 3 else np.array([0.0], dtype=np.float32)

    mean_value = float(values.mean())
    std_value = float(values.std())
    max_value = float(values.max())
    min_value = float(values.min())
    value_range = max_value - min_value
    roughness = float(np.abs(diff1).sum())

    early = float(values[:2].mean()) if len(values) >= 2 else mean_value
    late = float(values[-2:].mean()) if len(values) >= 2 else mean_value

    signs = np.sign(diff1)
    signs = signs[signs != 0]

    if len(signs) >= 2:
        sign_changes = int(np.sum(signs[1:] != signs[:-1]))
    else:
        sign_changes = 0

    x = np.arange(len(values), dtype=np.float32)

    if len(values) >= 2:
        slope = float(np.polyfit(x, values, 1)[0])
    else:
        slope = 0.0

    features[f"{prefix}_mean"] = mean_value
    features[f"{prefix}_std"] = std_value
    features[f"{prefix}_max"] = max_value
    features[f"{prefix}_min"] = min_value
    features[f"{prefix}_range"] = value_range
    features[f"{prefix}_cv"] = std_value / (abs(mean_value) + EPS)

    features[f"{prefix}_slope"] = slope
    features[f"{prefix}_acceleration"] = float(np.abs(diff2).mean())
    features[f"{prefix}_roughness"] = roughness
    features[f"{prefix}_smoothness"] = 1.0 / (1.0 + roughness)

    features[f"{prefix}_entropy"] = value_entropy(values)
    features[f"{prefix}_gini"] = value_gini(values)
    features[f"{prefix}_peak_position"] = float(np.argmax(values) / max(len(values) - 1, 1))

    features[f"{prefix}_late_minus_early"] = late - early
    features[f"{prefix}_late_div_early"] = late / (abs(early) + EPS)

    features[f"{prefix}_monotonicity"] = float(
        abs((diff1 > 0).sum() - (diff1 < 0).sum()) / max(len(diff1), 1)
    )
    features[f"{prefix}_num_increases"] = int((diff1 > 0).sum())
    features[f"{prefix}_num_decreases"] = int((diff1 < 0).sum())
    features[f"{prefix}_sign_changes"] = sign_changes

    features[f"{prefix}_second_diff_mean"] = float(diff2.mean())
    features[f"{prefix}_second_diff_max"] = float(diff2.max())

def add_outlier_features(features, prefix, token_matrix, zone_mean, last_token_mean=None, last5_mean=None):
    token_matrix = token_matrix.float()

    if token_matrix.shape[0] == 0:
        features[f"{prefix}_norm_outlier_frac_1std"] = 0.0
        features[f"{prefix}_norm_outlier_frac_2std"] = 0.0
        features[f"{prefix}_distance_outlier_frac_p90"] = 0.0
        features[f"{prefix}_max_token_outlier_distance"] = 0.0
        features[f"{prefix}_last_token_outlier_score"] = 0.0
        features[f"{prefix}_last5_outlier_score"] = 0.0
        return

    token_norms = torch.linalg.norm(
        token_matrix,
        dim=1,
    ).detach().cpu().numpy()

    norm_mean = float(token_norms.mean())
    norm_std = float(token_norms.std()) + EPS

    features[f"{prefix}_norm_outlier_frac_1std"] = float(
        (token_norms > norm_mean + norm_std).mean()
    )

    features[f"{prefix}_norm_outlier_frac_2std"] = float(
        (token_norms > norm_mean + 2 * norm_std).mean()
    )

    distances = torch.linalg.norm(
        token_matrix - zone_mean,
        dim=1,
    ).detach().cpu().numpy()

    p90_distance = float(np.percentile(distances, 90))

    features[f"{prefix}_distance_outlier_frac_p90"] = float(
        (distances > p90_distance).mean()
    )

    features[f"{prefix}_max_token_outlier_distance"] = float(
        distances.max()
    )

    if last_token_mean is not None:
        last_token_distance = safe_l2(last_token_mean - zone_mean)

        features[f"{prefix}_last_token_outlier_score"] = float(
            last_token_distance / (p90_distance + EPS)
        )
    else:
        features[f"{prefix}_last_token_outlier_score"] = 0.0

    if last5_mean is not None:
        last5_distance = safe_l2(last5_mean - zone_mean)

        features[f"{prefix}_last5_outlier_score"] = float(
            last5_distance / (p90_distance + EPS)
        )
    else:
        features[f"{prefix}_last5_outlier_score"] = 0.0

def add_spectral_features(features, prefix, values):
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if len(values) == 0:
        values = np.array([0.0], dtype=np.float32)

    centered = values - values.mean()

    fft_power = np.abs(np.fft.rfft(centered)) ** 2
    fft_power = fft_power.astype(np.float32)

    if len(fft_power) == 0:
        fft_power = np.array([0.0], dtype=np.float32)

    total_power = float(fft_power.sum())

    split_idx = max(1, len(fft_power) // 2)

    low_freq_power = float(fft_power[:split_idx].sum())
    high_freq_power = float(fft_power[split_idx:].sum())

    features[f"{prefix}_fft_energy"] = total_power
    features[f"{prefix}_low_frequency_energy"] = low_freq_power
    features[f"{prefix}_high_frequency_energy"] = high_freq_power

    features[f"{prefix}_high_low_ratio"] = (
        high_freq_power / (low_freq_power + EPS)
    )

    features[f"{prefix}_spectral_entropy"] = value_entropy(fft_power)

    features[f"{prefix}_dominant_frequency_index"] = float(
        np.argmax(fft_power) / max(len(fft_power) - 1, 1)
    )

    features[f"{prefix}_total_spectral_power"] = total_power

    normalized_power = fft_power / (total_power + EPS)

    for i, power in enumerate(normalized_power):
        features[f"{prefix}_normalized_spectral_power_{i}"] = float(power)

def pairwise_cosine_stats(token_matrix):
    token_matrix = to_numpy(token_matrix)

    if token_matrix.shape[0] <= 1:
        return {
            "mean": 1.0,
            "std": 0.0,
            "min": 1.0,
            "p10": 1.0,
            "p90": 1.0,
        }

    norms = np.linalg.norm(token_matrix, axis=1, keepdims=True) + EPS
    normalized = token_matrix / norms

    cosine_matrix = normalized @ normalized.T

    upper_triangle = cosine_matrix[
        np.triu_indices_from(cosine_matrix, k=1)
    ]

    return {
        "mean": float(upper_triangle.mean()),
        "std": float(upper_triangle.std()),
        "min": float(upper_triangle.min()),
        "p10": float(np.percentile(upper_triangle, 10)),
        "p90": float(np.percentile(upper_triangle, 90)),
    }

def covariance_pca_stats(token_matrix):
    token_matrix = to_numpy(token_matrix)

    if token_matrix.shape[0] <= 2:
        return {
            "cov_trace": 0.0,
            "cov_top_eigenvalue": 0.0,
            "top_eigenvalue_ratio": 0.0,
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "cov_spectral_entropy": 0.0,
            "pca_energy_top1": 0.0,
            "pca_energy_top3": 0.0,
            "pca_energy_top5": 0.0,
            "collapse_score": 0.0,
            "spread_score": 0.0,
        }

    centered = token_matrix - token_matrix.mean(axis=0, keepdims=True)

    singular_values = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False,
    )

    eigenvalues = singular_values ** 2
    total_energy = float(eigenvalues.sum()) + EPS

    energy_ratio = eigenvalues / total_energy

    spectral_entropy = float(
        -(energy_ratio * np.log(energy_ratio + EPS)).sum()
    )

    effective_rank = float(np.exp(spectral_entropy))

    participation_ratio = float(
        (eigenvalues.sum() ** 2) /
        ((eigenvalues ** 2).sum() + EPS)
    )

    top1_ratio = float(energy_ratio[:1].sum())
    top3_ratio = float(energy_ratio[:3].sum())
    top5_ratio = float(energy_ratio[:5].sum())

    return {
        "cov_trace": float(total_energy),
        "cov_top_eigenvalue": float(eigenvalues[0]),
        "top_eigenvalue_ratio": top1_ratio,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "cov_spectral_entropy": spectral_entropy,
        "pca_energy_top1": top1_ratio,
        "pca_energy_top3": top3_ratio,
        "pca_energy_top5": top5_ratio,
        "collapse_score": top1_ratio,
        "spread_score": effective_rank,
    }

def add_robust_percentiles(features, prefix, values):
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if len(values) == 0:
        values = np.array([0.0], dtype=np.float32)

    p05 = float(np.percentile(values, 5))
    p10 = float(np.percentile(values, 10))
    p25 = float(np.percentile(values, 25))
    p50 = float(np.percentile(values, 50))
    p75 = float(np.percentile(values, 75))
    p90 = float(np.percentile(values, 90))
    p95 = float(np.percentile(values, 95))
    p99 = float(np.percentile(values, 99))

    features[f"{prefix}_p10"] = p10
    features[f"{prefix}_p25"] = p25
    features[f"{prefix}_p50"] = p50
    features[f"{prefix}_p75"] = p75
    features[f"{prefix}_p90"] = p90
    features[f"{prefix}_p95"] = p95
    features[f"{prefix}_p99"] = p99
    features[f"{prefix}_iqr"] = p75 - p25
    features[f"{prefix}_p90_div_p10"] = p90 / (abs(p10) + EPS)
    features[f"{prefix}_p95_minus_p50"] = p95 - p50
    features[f"{prefix}_p50_minus_p05"] = p50 - p05


def extract_features_one_sample(hidden, valid_mask, prompt_text="", response_text=""):
    """
    hidden: Tensor with shape (n_layers, seq_len, hidden_dim)
    valid_mask: Tensor with shape (seq_len,)
    """

    hidden = hidden.float().cpu()
    masks = make_token_masks(valid_mask)

    features = {}

    layer_means = {}

    for zone_name, zone_mask in masks.items():
        layer_means[zone_name] = {}

        for layer in LAYERS:
            layer_means[zone_name][layer] = safe_mean(
                hidden[layer],
                zone_mask,
            )
        
    # BLOCK 1 + 2: Trajectory features over layer jumps + Spectral / frequency features

    for zone_name in ["all", "first70", "last40", "last30", "last20", "last5", "last_token"]:
        jumps = []

        for left_layer, right_layer in LAYER_PAIRS:
            jump = layer_means[zone_name][right_layer] - layer_means[zone_name][left_layer]
            jumps.append(safe_l2(jump))

        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_jump",
            values=jumps,
        )

        add_spectral_features(
            features=features,
            prefix=f"{zone_name}_jump",
            values=jumps,
        )
    
    # BLOCK 3: Norm dynamics over layers

    for zone_name in ["all", "first70", "last40", "last30", "last20", "last5", "last_token"]:
        norms = []

        for layer in LAYERS:
            norms.append(
                safe_l2(layer_means[zone_name][layer])
            )

        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_norm",
            values=norms,
        )

        features[f"{zone_name}_norm_max_layer"] = float(
            LAYERS[int(np.argmax(norms))]
        )

        features[f"{zone_name}_norm_min_layer"] = float(
            LAYERS[int(np.argmin(norms))]
        )

        features[f"{zone_name}_norm_max_layer_pos"] = float(
            np.argmax(norms) / max(len(norms) - 1, 1)
        )

        features[f"{zone_name}_norm_min_layer_pos"] = float(
            np.argmin(norms) / max(len(norms) - 1, 1)
        )

        features[f"{zone_name}_norm_collapse_score"] = float(
            max(0.0, norms[0] - norms[-1])
        )

        features[f"{zone_name}_norm_explosion_score"] = float(
            max(0.0, norms[-1] - norms[0])
        )
    
    # BLOCK 4: Cosine stability between neighboring layers

    for zone_name in ["all", "first70", "last40", "last30", "last20", "last5", "last_token"]:
        cosines = []

        for left_layer, right_layer in LAYER_PAIRS:
            cos_value = safe_cosine(
                layer_means[zone_name][left_layer],
                layer_means[zone_name][right_layer],
            )
            cosines.append(cos_value)

        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_cosine",
            values=cosines,
        )

        diff_cosines = np.diff(np.asarray(cosines, dtype=np.float32))

        if len(diff_cosines) > 0:
            features[f"{zone_name}_cosine_drop_max"] = float(
                max(0.0, -diff_cosines.min())
            )
        else:
            features[f"{zone_name}_cosine_drop_max"] = 0.0
    
    # BLOCK 5: Curvature / zig-zag over drift vectors

    for zone_name in ["all", "first70", "last40", "last30", "last20", "last5", "last_token"]:
        drift_vectors = []

        for left_layer, right_layer in LAYER_PAIRS:
            drift = (
                layer_means[zone_name][right_layer]
                - layer_means[zone_name][left_layer]
            )
            drift_vectors.append(drift)

        curvature_cosines = []

        for i in range(len(drift_vectors) - 1):
            curvature_cosines.append(
                safe_cosine(
                    drift_vectors[i],
                    drift_vectors[i + 1],
                )
            )

        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_curvature",
            values=curvature_cosines,
        )

        curvature_values = np.asarray(curvature_cosines, dtype=np.float32)

        if len(curvature_values) == 0:
            curvature_values = np.array([0.0], dtype=np.float32)

        features[f"{zone_name}_curvature_negative_steps"] = int(
            (curvature_values < 0).sum()
        )

        features[f"{zone_name}_curvature_max_curvature"] = float(
            1.0 - curvature_values.min()
        )

        features[f"{zone_name}_curvature_angle_entropy"] = value_entropy(
            1.0 - curvature_values
        )

        if len(curvature_values) >= 2:
            early_curvature = float(curvature_values[:2].mean())
            late_curvature = float(curvature_values[-2:].mean())
        else:
            early_curvature = float(curvature_values.mean())
            late_curvature = float(curvature_values.mean())

        features[f"{zone_name}_curvature_late_minus_early"] = (
            late_curvature - early_curvature
        )

        features[f"{zone_name}_curvature_late_div_early"] = (
            late_curvature / (abs(early_curvature) + EPS)
        )
    
    # BLOCK 6: Token disagreement

    pairwise_mean_by_zone = {}

    for zone_name in ["all", "first70", "last40", "last30", "last20", "last5"]:
        pairwise_mean_by_zone[zone_name] = []

        for layer in LAYERS:
            zone_tokens = hidden[layer][masks[zone_name]].float()

            if zone_tokens.shape[0] == 0:
                zone_tokens = hidden[layer][masks["all"]].float()

            if zone_tokens.shape[0] == 0:
                token_norms = np.array([0.0], dtype=np.float32)
                token_variance = np.array([0.0], dtype=np.float32)
                pairwise_stats = {
                    "mean": 1.0,
                    "std": 0.0,
                    "min": 1.0,
                    "p10": 1.0,
                    "p90": 1.0,
                }
            else:
                token_norms = torch.linalg.norm(
                    zone_tokens,
                    dim=1,
                ).detach().cpu().numpy()

                token_variance = zone_tokens.var(
                    dim=0,
                    unbiased=False,
                ).detach().cpu().numpy()

                pairwise_stats = pairwise_cosine_stats(zone_tokens)

            features[f"{zone_name}_l{layer}_token_variance_mean"] = float(
                token_variance.mean()
            )
            features[f"{zone_name}_l{layer}_token_variance_max"] = float(
                token_variance.max()
            )

            features[f"{zone_name}_l{layer}_token_norm_std"] = float(
                token_norms.std()
            )
            features[f"{zone_name}_l{layer}_token_norm_range"] = float(
                token_norms.max() - token_norms.min()
            )

            features[f"{zone_name}_l{layer}_pairwise_cosine_mean"] = pairwise_stats["mean"]
            features[f"{zone_name}_l{layer}_pairwise_cosine_std"] = pairwise_stats["std"]
            features[f"{zone_name}_l{layer}_pairwise_cosine_min"] = pairwise_stats["min"]
            features[f"{zone_name}_l{layer}_pairwise_cosine_p10"] = pairwise_stats["p10"]
            features[f"{zone_name}_l{layer}_pairwise_cosine_p90"] = pairwise_stats["p90"]

            features[f"{zone_name}_l{layer}_token_disagreement"] = (
                1.0 - pairwise_stats["mean"]
            )

            pairwise_mean_by_zone[zone_name].append(pairwise_stats["mean"])

    for layer in LAYERS:
        all_key = f"all_l{layer}_pairwise_cosine_mean"
        last5_key = f"last5_l{layer}_pairwise_cosine_mean"

        features[f"l{layer}_token_consensus_collapse_all_minus_last5"] = (
            features.get(all_key, 1.0) - features.get(last5_key, 1.0)
        )

    for zone_name, values in pairwise_mean_by_zone.items():
        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_pairwise_cosine_by_layer",
            values=values,
        )
    
    # BLOCK 7: Covariance / PCA collapse

    for zone_name in ["all", "first70", "last40", "last30", "last20", "last5"]:
        effective_rank_by_layer = []
        collapse_score_by_layer = []

        for layer in LAYERS:
            zone_tokens = hidden[layer][masks[zone_name]].float()

            if zone_tokens.shape[0] == 0:
                zone_tokens = hidden[layer][masks["all"]].float()

            stats = covariance_pca_stats(zone_tokens)

            for stat_name, stat_value in stats.items():
                features[f"{zone_name}_l{layer}_{stat_name}"] = stat_value

            effective_rank_by_layer.append(stats["effective_rank"])
            collapse_score_by_layer.append(stats["collapse_score"])

        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_effective_rank_by_layer",
            values=effective_rank_by_layer,
        )

        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_collapse_score_by_layer",
            values=collapse_score_by_layer,
        )
    
    # BLOCK 8: Late collapse

    for layer in LAYERS:
        last5_mean = layer_means["last5"][layer]
        all_mean = layer_means["all"][layer]
        last30_mean = layer_means["last30"][layer]
        last_token_mean = layer_means["last_token"][layer]

        features[f"l{layer}_last5_vs_all_norm_diff"] = (
            safe_l2(last5_mean) - safe_l2(all_mean)
        )

        features[f"l{layer}_last5_vs_all_cosine"] = safe_cosine(
            last5_mean,
            all_mean,
        )

        features[f"l{layer}_last5_vs_last30_norm_ratio"] = (
            safe_l2(last5_mean) / (safe_l2(last30_mean) + EPS)
        )

        features[f"l{layer}_last_token_vs_last5_cosine"] = safe_cosine(
            last_token_mean,
            last5_mean,
        )

        features[f"l{layer}_last_token_divergence_from_all"] = safe_l2(
            last_token_mean - all_mean
        )

        features[f"l{layer}_last_token_divergence_from_last30"] = safe_l2(
            last_token_mean - last30_mean
        )

    # last5 jump energy vs all jump energy
    last5_jumps = []
    all_jumps = []

    for left_layer, right_layer in LAYER_PAIRS:
        last5_jump = (
            layer_means["last5"][right_layer]
            - layer_means["last5"][left_layer]
        )
        all_jump = (
            layer_means["all"][right_layer]
            - layer_means["all"][left_layer]
        )

        last5_jumps.append(safe_l2(last5_jump))
        all_jumps.append(safe_l2(all_jump))

    last5_jump_mean = float(np.mean(last5_jumps))
    all_jump_mean = float(np.mean(all_jumps))

    features["last5_vs_all_jump_energy_diff"] = (
        last5_jump_mean - all_jump_mean
    )

    features["last5_vs_all_jump_energy_ratio"] = (
        last5_jump_mean / (all_jump_mean + EPS)
    )

    # last5 curvature vs all curvature
    last5_drift_vectors = []
    all_drift_vectors = []

    for left_layer, right_layer in LAYER_PAIRS:
        last5_drift_vectors.append(
            layer_means["last5"][right_layer]
            - layer_means["last5"][left_layer]
        )

        all_drift_vectors.append(
            layer_means["all"][right_layer]
            - layer_means["all"][left_layer]
        )

    last5_curvatures = []
    all_curvatures = []

    for i in range(len(LAYER_PAIRS) - 1):
        last5_curvatures.append(
            safe_cosine(
                last5_drift_vectors[i],
                last5_drift_vectors[i + 1],
            )
        )

        all_curvatures.append(
            safe_cosine(
                all_drift_vectors[i],
                all_drift_vectors[i + 1],
            )
        )

    last5_curvature_mean = float(np.mean(last5_curvatures))
    all_curvature_mean = float(np.mean(all_curvatures))

    features["last5_vs_all_curvature_diff"] = (
        last5_curvature_mean - all_curvature_mean
    )

    features["last5_vs_all_curvature_ratio"] = (
        last5_curvature_mean / (abs(all_curvature_mean) + EPS)
    )

    # last5 token variance vs all token variance
    last5_token_variances = []
    all_token_variances = []

    for layer in LAYERS:
        last5_tokens = hidden[layer][masks["last5"]].float()
        all_tokens = hidden[layer][masks["all"]].float()

        if last5_tokens.shape[0] > 1:
            last5_var = float(
                last5_tokens.var(dim=0, unbiased=False).mean().item()
            )
        else:
            last5_var = 0.0

        if all_tokens.shape[0] > 1:
            all_var = float(
                all_tokens.var(dim=0, unbiased=False).mean().item()
            )
        else:
            all_var = 0.0

        last5_token_variances.append(last5_var)
        all_token_variances.append(all_var)

        features[f"l{layer}_last5_vs_all_token_variance_diff"] = (
            last5_var - all_var
        )

        features[f"l{layer}_last5_vs_all_token_variance_ratio"] = (
            last5_var / (all_var + EPS)
        )

    features["last5_vs_all_token_variance_mean_diff"] = float(
        np.mean(last5_token_variances) - np.mean(all_token_variances)
    )

    features["last5_vs_all_token_variance_mean_ratio"] = float(
        np.mean(last5_token_variances) /
        (np.mean(all_token_variances) + EPS)
    )

    # BLOCK 9: Layer-specific uncertainty

    for layer in LAYERS:
        all_tokens = hidden[layer][masks["all"]].float()
        last5_tokens = hidden[layer][masks["last5"]].float()
        last_token_tokens = hidden[layer][masks["last_token"]].float()

        if all_tokens.shape[0] == 0:
            all_tokens = torch.zeros((1, hidden.shape[-1]), dtype=hidden.dtype)

        all_token_norms = torch.linalg.norm(
            all_tokens,
            dim=1,
        ).detach().cpu().numpy()

        features[f"l{layer}_token_norm_mean"] = float(
            all_token_norms.mean()
        )

        features[f"l{layer}_token_norm_std"] = float(
            all_token_norms.std()
        )

        if all_tokens.shape[0] > 1:
            token_variance = float(
                all_tokens.var(dim=0, unbiased=False).mean().item()
            )
        else:
            token_variance = 0.0

        features[f"l{layer}_token_variance"] = token_variance

        pairwise_stats = pairwise_cosine_stats(all_tokens)

        features[f"l{layer}_pairwise_cosine_mean"] = pairwise_stats["mean"]
        features[f"l{layer}_pairwise_cosine_std"] = pairwise_stats["std"]

        last_token_mean = layer_means["last_token"][layer]
        all_mean = layer_means["all"][layer]
        last5_mean = layer_means["last5"][layer]

        features[f"l{layer}_last_token_to_all_mean_dist"] = safe_l2(
            last_token_mean - all_mean
        )

        features[f"l{layer}_last5_mean_to_all_mean_dist"] = safe_l2(
            last5_mean - all_mean
        )

        # entropy-like spread based on token norms
        features[f"l{layer}_token_norm_entropy"] = value_entropy(
            all_token_norms
        )

        # covariance / PCA stats
        cov_stats = covariance_pca_stats(all_tokens)

        features[f"l{layer}_covariance_trace"] = cov_stats["cov_trace"]
        features[f"l{layer}_effective_rank"] = cov_stats["effective_rank"]
    
    # BLOCK 10: Cross-zone contrastive features

    zone_pairs = [
        ("last5", "all"),
        ("last_token", "all"),
        ("last20", "first70"),
        ("last30", "first70"),
        ("last40", "first70"),
        ("last5", "last30"),
        ("last_token", "last5"),
    ]

    for zone_a, zone_b in zone_pairs:
        for layer in LAYERS:
            mean_a = layer_means[zone_a][layer]
            mean_b = layer_means[zone_b][layer]

            features[f"{zone_a}_vs_{zone_b}_l{layer}_cosine"] = safe_cosine(
                mean_a,
                mean_b,
            )

            features[f"{zone_a}_vs_{zone_b}_l{layer}_l2_distance"] = safe_l2(
                mean_a - mean_b,
            )

            features[f"{zone_a}_vs_{zone_b}_l{layer}_norm_ratio"] = (
                safe_l2(mean_a) / (safe_l2(mean_b) + EPS)
            )

            tokens_a = hidden[layer][masks[zone_a]].float()
            tokens_b = hidden[layer][masks[zone_b]].float()

            if tokens_a.shape[0] > 1:
                variance_a = float(
                    tokens_a.var(dim=0, unbiased=False).mean().item()
                )
            else:
                variance_a = 0.0

            if tokens_b.shape[0] > 1:
                variance_b = float(
                    tokens_b.var(dim=0, unbiased=False).mean().item()
                )
            else:
                variance_b = 0.0

            features[f"{zone_a}_vs_{zone_b}_l{layer}_variance_ratio"] = (
                variance_a / (variance_b + EPS)
            )

            pairwise_a = pairwise_cosine_stats(tokens_a)
            pairwise_b = pairwise_cosine_stats(tokens_b)

            disagreement_a = 1.0 - pairwise_a["mean"]
            disagreement_b = 1.0 - pairwise_b["mean"]

            features[f"{zone_a}_vs_{zone_b}_l{layer}_disagreement_ratio"] = (
                disagreement_a / (disagreement_b + EPS)
            )

    for zone_a, zone_b in zone_pairs:
        drift_a = []
        drift_b = []

        for left_layer, right_layer in LAYER_PAIRS:
            jump_a = (
                layer_means[zone_a][right_layer]
                - layer_means[zone_a][left_layer]
            )
            jump_b = (
                layer_means[zone_b][right_layer]
                - layer_means[zone_b][left_layer]
            )

            drift_a.append(safe_l2(jump_a))
            drift_b.append(safe_l2(jump_b))

        drift_a_mean = float(np.mean(drift_a))
        drift_b_mean = float(np.mean(drift_b))

        features[f"{zone_a}_vs_{zone_b}_drift_ratio"] = (
            drift_a_mean / (drift_b_mean + EPS)
        )

        features[f"{zone_a}_vs_{zone_b}_drift_diff"] = (
            drift_a_mean - drift_b_mean
        )
    
    # BLOCK 11: Directional drift alignment

    for zone_name in ["all", "first70", "last40", "last30", "last20", "last5", "last_token"]:
        drift_vectors = []

        for left_layer, right_layer in LAYER_PAIRS:
            drift = (
                layer_means[zone_name][right_layer]
                - layer_means[zone_name][left_layer]
            )
            drift_vectors.append(drift)

        first_drift = drift_vectors[0]      # 11 -> 12
        last_drift = drift_vectors[-1]      # 15 -> 16

        early_drift = torch.stack(drift_vectors[:2]).mean(dim=0)
        late_drift = torch.stack(drift_vectors[-2:]).mean(dim=0)

        features[f"{zone_name}_drift_11_12_vs_15_16_cosine"] = safe_cosine(
            first_drift,
            last_drift,
        )

        features[f"{zone_name}_early_vs_late_drift_cosine"] = safe_cosine(
            early_drift,
            late_drift,
        )

        features[f"{zone_name}_early_vs_late_drift_l2_distance"] = safe_l2(
            late_drift - early_drift
        )

        early_np = to_numpy(early_drift)
        late_np = to_numpy(late_drift)

        early_norm = np.linalg.norm(early_np) + EPS

        projection = float(
            np.dot(late_np, early_np) / early_norm
        )

        early_unit = early_np / early_norm

        orthogonal_residual = late_np - projection * early_unit

        features[f"{zone_name}_late_drift_projection_on_early"] = projection

        features[f"{zone_name}_late_drift_orthogonal_residual_norm"] = float(
            np.linalg.norm(orthogonal_residual)
        )

        features[f"{zone_name}_late_drift_projection_ratio"] = (
            projection / (np.linalg.norm(late_np) + EPS)
        )

        features[f"{zone_name}_late_drift_orthogonal_ratio"] = (
            np.linalg.norm(orthogonal_residual) /
            (np.linalg.norm(late_np) + EPS)
        )

    # BLOCK 12: Robust percentiles

    for zone_name in ["all", "first70", "last40", "last30", "last20", "last5"]:
        for layer in LAYERS:
            zone_tokens = hidden[layer][masks[zone_name]].float()

            if zone_tokens.shape[0] == 0:
                zone_tokens = hidden[layer][masks["all"]].float()

            token_norms = torch.linalg.norm(
                zone_tokens,
                dim=1,
            ).detach().cpu().numpy()

            add_robust_percentiles(
                features=features,
                prefix=f"{zone_name}_l{layer}_token_norm",
                values=token_norms,
            )

    for zone_name in ["all", "first70", "last40", "last30", "last20", "last5", "last_token"]:
        drift_norms = []

        for left_layer, right_layer in LAYER_PAIRS:
            drift = (
                layer_means[zone_name][right_layer]
                - layer_means[zone_name][left_layer]
            )

            drift_norms.append(safe_l2(drift))

        add_robust_percentiles(
            features=features,
            prefix=f"{zone_name}_drift_norm",
            values=drift_norms,
        )

    for zone_name in ["last40", "last30", "last20", "last5", "last_token"]:
        distances_to_all = []

        for layer in LAYERS:
            distance = safe_l2(
                layer_means[zone_name][layer]
                - layer_means["all"][layer]
            )

            distances_to_all.append(distance)

        add_robust_percentiles(
            features=features,
            prefix=f"{zone_name}_to_all_distance",
            values=distances_to_all,
        )
    
     # BLOCK 13: Outlier features

    for zone_name in ["all", "first70", "last40", "last30", "last20", "last5"]:
        for layer in LAYERS:
            zone_tokens = hidden[layer][masks[zone_name]].float()

            if zone_tokens.shape[0] == 0:
                zone_tokens = hidden[layer][masks["all"]].float()

            add_outlier_features(
                features=features,
                prefix=f"{zone_name}_l{layer}",
                token_matrix=zone_tokens,
                zone_mean=layer_means[zone_name][layer],
                last_token_mean=layer_means["last_token"][layer],
                last5_mean=layer_means["last5"][layer],
            )
    
    # BLOCK 14: Confidence proxy via margin-like geometry

    for layer in LAYERS:
        last_token_mean = layer_means["last_token"][layer]
        all_mean = layer_means["all"][layer]
        last30_mean = layer_means["last30"][layer]

        all_tokens = hidden[layer][masks["all"]].float()

        if all_tokens.shape[0] == 0:
            all_token_norm_mean = 0.0
        else:
            all_token_norms = torch.linalg.norm(
                all_tokens,
                dim=1,
            ).detach().cpu().numpy()

            all_token_norm_mean = float(all_token_norms.mean())

        features[f"l{layer}_confidence_last_token_to_all_dist"] = safe_l2(
            last_token_mean - all_mean
        )

        features[f"l{layer}_confidence_last_token_to_last30_dist"] = safe_l2(
            last_token_mean - last30_mean
        )

        features[f"l{layer}_confidence_last_token_with_all_cosine"] = safe_cosine(
            last_token_mean,
            all_mean,
        )

        features[f"l{layer}_confidence_last_token_with_last30_cosine"] = safe_cosine(
            last_token_mean,
            last30_mean,
        )

        features[f"l{layer}_confidence_last_token_norm_div_all_norm_mean"] = (
            safe_l2(last_token_mean) / (all_token_norm_mean + EPS)
        )

    # last token drift energy across neighboring layers
    last_token_drift_energy = []

    for left_layer, right_layer in LAYER_PAIRS:
        drift = (
            layer_means["last_token"][right_layer]
            - layer_means["last_token"][left_layer]
        )

        last_token_drift_energy.append(safe_l2(drift))

    add_trajectory_features(
        features=features,
        prefix="last_token_drift_energy",
        values=last_token_drift_energy,
    )

    add_spectral_features(
        features=features,
        prefix="last_token_drift_energy",
        values=last_token_drift_energy,
    )

    # final-layer token spread
    final_layer = LAYERS[-1]
    final_all_tokens = hidden[final_layer][masks["all"]].float()
    final_last30_tokens = hidden[final_layer][masks["last30"]].float()
    final_last5_tokens = hidden[final_layer][masks["last5"]].float()

    for zone_name, zone_tokens in [
        ("all", final_all_tokens),
        ("last30", final_last30_tokens),
        ("last5", final_last5_tokens),
    ]:
        if zone_tokens.shape[0] == 0:
            token_variance = 0.0
            token_norm_std = 0.0
            pairwise_mean = 1.0
            pairwise_std = 0.0
        else:
            token_norms = torch.linalg.norm(
                zone_tokens,
                dim=1,
            ).detach().cpu().numpy()

            token_variance = (
                float(zone_tokens.var(dim=0, unbiased=False).mean().item())
                if zone_tokens.shape[0] > 1
                else 0.0
            )

            token_norm_std = float(token_norms.std())

            pairwise_stats = pairwise_cosine_stats(zone_tokens)
            pairwise_mean = pairwise_stats["mean"]
            pairwise_std = pairwise_stats["std"]

        features[f"final_layer_{zone_name}_token_variance"] = token_variance
        features[f"final_layer_{zone_name}_token_norm_std"] = token_norm_std
        features[f"final_layer_{zone_name}_pairwise_cosine_mean"] = pairwise_mean
        features[f"final_layer_{zone_name}_pairwise_cosine_std"] = pairwise_std
        features[f"final_layer_{zone_name}_token_disagreement"] = 1.0 - pairwise_mean

    # BLOCK 15: Cross-layer token consistency
    
    for zone_name in ["all", "first70", "last40", "last30", "last20", "last5", "last_token"]:
        tokenwise_drift_means = []
        tokenwise_drift_stds = []
        tokenwise_drift_maxes = []
        tokenwise_drift_variances = []

        for left_layer, right_layer in LAYER_PAIRS:
            zone_mask = masks[zone_name]

            left_tokens = hidden[left_layer][zone_mask].float()
            right_tokens = hidden[right_layer][zone_mask].float()

            if left_tokens.shape[0] == 0:
                drift_norms = np.array([0.0], dtype=np.float32)
            else:
                token_drifts = right_tokens - left_tokens

                drift_norms = torch.linalg.norm(
                    token_drifts,
                    dim=1,
                ).detach().cpu().numpy()

            drift_mean = float(drift_norms.mean())
            drift_std = float(drift_norms.std())
            drift_max = float(drift_norms.max())
            drift_var = float(drift_norms.var())

            features[f"{zone_name}_tokenwise_drift_l{left_layer}_to_l{right_layer}_mean"] = drift_mean
            features[f"{zone_name}_tokenwise_drift_l{left_layer}_to_l{right_layer}_std"] = drift_std
            features[f"{zone_name}_tokenwise_drift_l{left_layer}_to_l{right_layer}_max"] = drift_max
            features[f"{zone_name}_tokenwise_drift_l{left_layer}_to_l{right_layer}_variance"] = drift_var

            tokenwise_drift_means.append(drift_mean)
            tokenwise_drift_stds.append(drift_std)
            tokenwise_drift_maxes.append(drift_max)
            tokenwise_drift_variances.append(drift_var)

        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_tokenwise_drift_mean_trajectory",
            values=tokenwise_drift_means,
        )

        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_tokenwise_drift_std_trajectory",
            values=tokenwise_drift_stds,
        )

        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_tokenwise_drift_max_trajectory",
            values=tokenwise_drift_maxes,
        )

        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_tokenwise_drift_variance_trajectory",
            values=tokenwise_drift_variances,
        )

    # Explicit last-token layer drift trajectory
    last_token_layer_drifts = []

    for left_layer, right_layer in LAYER_PAIRS:
        drift = (
            layer_means["last_token"][right_layer]
            - layer_means["last_token"][left_layer]
        )

        last_token_layer_drifts.append(safe_l2(drift))

    add_trajectory_features(
        features=features,
        prefix="last_token_layer_drift_trajectory",
        values=last_token_layer_drifts,
    )

    add_spectral_features(
        features=features,
        prefix="last_token_layer_drift_trajectory",
        values=last_token_layer_drifts,
    )

    # Explicit last5 layer drift trajectory
    last5_layer_drifts = []

    for left_layer, right_layer in LAYER_PAIRS:
        drift = (
            layer_means["last5"][right_layer]
            - layer_means["last5"][left_layer]
        )

        last5_layer_drifts.append(safe_l2(drift))

    add_trajectory_features(
        features=features,
        prefix="last5_layer_drift_trajectory",
        values=last5_layer_drifts,
    )

    add_spectral_features(
        features=features,
        prefix="last5_layer_drift_trajectory",
        values=last5_layer_drifts,
    )

    # BLOCK 16: Hand-made uncertainty scores

    for zone_name in ["all", "first70", "last40", "last30", "last20", "last5", "last_token"]:
        jump_std = features.get(f"{zone_name}_jump_std", 0.0)
        curvature_std = features.get(f"{zone_name}_curvature_std", 0.0)
        jump_roughness = features.get(f"{zone_name}_jump_roughness", 0.0)
        jump_second_diff_abs_mean = abs(
            features.get(f"{zone_name}_jump_second_diff_mean", 0.0)
        )
        norm_std = features.get(f"{zone_name}_norm_std", 0.0)
        norm_roughness = features.get(f"{zone_name}_norm_roughness", 0.0)

        token_disagreements = [
            features.get(f"{zone_name}_l{layer}_token_disagreement", 0.0)
            for layer in LAYERS
        ]

        token_disagreement = float(np.mean(token_disagreements))

        features[f"{zone_name}_instability_score"] = (
            jump_std + curvature_std + token_disagreement
        )

        features[f"{zone_name}_trajectory_noise_score"] = (
            jump_roughness + jump_second_diff_abs_mean
        )

        features[f"{zone_name}_norm_instability_score"] = (
            norm_std + norm_roughness
        )

    all_disagreement = float(np.mean([
        features.get(f"all_l{layer}_token_disagreement", 0.0)
        for layer in LAYERS
    ]))

    last5_disagreement = float(np.mean([
        features.get(f"last5_l{layer}_token_disagreement", 0.0)
        for layer in LAYERS
    ]))

    features["late_collapse_score"] = (
        last5_disagreement - all_disagreement
    )

    semantic_shift_distances = []

    for layer in LAYERS:
        distance = safe_l2(
            layer_means["last30"][layer]
            - layer_means["first70"][layer]
        )

        semantic_shift_distances.append(distance)

    features["semantic_shift_score"] = float(
        np.mean(semantic_shift_distances)
    )

    add_trajectory_features(
        features=features,
        prefix="semantic_shift_by_layer",
        values=semantic_shift_distances,
    )

    # BLOCK 17: Simple text + geometry interactions

    response_text = str(response_text)
    prompt_text = str(prompt_text)

    response_length = len(response_text)
    prompt_length = len(prompt_text)
    total_length = max(response_length + prompt_length, 1)

    response_ratio = response_length / total_length

    number_of_sentences = max(
        1,
        response_text.count(".")
        + response_text.count("!")
        + response_text.count("?"),
    )

    punctuation_count = sum(
        char in ".,!?;:"
        for char in response_text
    )

    word_count = len(response_text.split())

    features["text_response_length"] = response_length
    features["text_prompt_length"] = prompt_length
    features["text_total_length"] = total_length
    features["text_response_ratio"] = response_ratio
    features["text_number_of_sentences"] = number_of_sentences
    features["text_punctuation_count"] = punctuation_count
    features["text_word_count"] = word_count

    for zone_name in ["all", "first70", "last40", "last30", "last20", "last5", "last_token"]:
        jump_std = features.get(f"{zone_name}_jump_std", 0.0)
        token_disagreement = float(np.mean([
            features.get(f"{zone_name}_l{layer}_token_disagreement", 0.0)
            for layer in LAYERS
        ]))
        trajectory_roughness = features.get(f"{zone_name}_jump_roughness", 0.0)
        uncertainty_score = features.get(f"{zone_name}_instability_score", 0.0)
        late_instability = features.get(f"{zone_name}_late_minus_early_jump", 0.0)

        features[f"{zone_name}_response_length_x_jump_std"] = (
            np.log1p(response_length) * jump_std
        )

        features[f"{zone_name}_response_length_x_token_disagreement"] = (
            np.log1p(response_length) * token_disagreement
        )

        features[f"{zone_name}_response_ratio_x_late_instability"] = (
            response_ratio * late_instability
        )

        features[f"{zone_name}_sentences_x_trajectory_roughness"] = (
            number_of_sentences * trajectory_roughness
        )

        features[f"{zone_name}_punctuation_x_uncertainty_score"] = (
            punctuation_count * uncertainty_score
        )

        features[f"{zone_name}_word_count_x_uncertainty_score"] = (
            np.log1p(word_count) * uncertainty_score
        )
    
    # BLOCK 18: Weighted pooling and activation entropy
    
    for zone_name in ["all", "first70", "last40", "last30", "last20", "last5"]:
        weighted_means = []
        normal_means = []
        entropy_means = []
        max_activation_means = []
        variance_means = []

        for layer in LAYERS:
            zone_tokens = hidden[layer][masks[zone_name]].float()

            if zone_tokens.shape[0] == 0:
                zone_tokens = hidden[layer][masks["all"]].float()

            weighted_mean = weighted_token_mean(zone_tokens)
            normal_mean = layer_means[zone_name][layer]

            weighted_means.append(weighted_mean)
            normal_means.append(normal_mean)

            features[f"{zone_name}_l{layer}_weighted_mean_norm"] = safe_l2(
                weighted_mean
            )

            features[f"{zone_name}_l{layer}_weighted_vs_normal_dist"] = safe_l2(
                weighted_mean - normal_mean
            )

            features[f"{zone_name}_l{layer}_weighted_vs_normal_cosine"] = safe_cosine(
                weighted_mean,
                normal_mean,
            )

            entropy_stats = activation_entropy_stats(zone_tokens)

            for stat_name, stat_value in entropy_stats.items():
                features[f"{zone_name}_l{layer}_activation_{stat_name}"] = stat_value

            entropy_means.append(entropy_stats["entropy_mean"])

            max_stats = activation_max_stats(zone_tokens)

            for stat_name, stat_value in max_stats.items():
                features[f"{zone_name}_l{layer}_activation_{stat_name}"] = stat_value

            max_activation_means.append(max_stats["max_mean"])

            variance_stats = activation_variance_stats(zone_tokens)

            for stat_name, stat_value in variance_stats.items():
                features[f"{zone_name}_l{layer}_activation_{stat_name}"] = stat_value

            variance_means.append(variance_stats["var_mean"])

        # weighted drift energy
        weighted_drift_energy = []

        for i in range(len(LAYERS) - 1):
            drift = weighted_means[i + 1] - weighted_means[i]
            weighted_drift_energy.append(safe_l2(drift))

        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_weighted_drift_energy",
            values=weighted_drift_energy,
        )

        add_spectral_features(
            features=features,
            prefix=f"{zone_name}_weighted_drift_energy",
            values=weighted_drift_energy,
        )

        # entropy trajectory
        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_activation_entropy",
            values=entropy_means,
        )

        # max activation trajectory
        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_activation_max",
            values=max_activation_means,
        )

        # variance trajectory
        add_trajectory_features(
            features=features,
            prefix=f"{zone_name}_activation_variance",
            values=variance_means,
        )

    # entropy late_minus_early / last5_vs_all
    all_entropy_by_layer = []
    last5_entropy_by_layer = []

    for layer in LAYERS:
        all_entropy_by_layer.append(
            features.get(f"all_l{layer}_activation_entropy_mean", 0.0)
        )
        last5_entropy_by_layer.append(
            features.get(f"last5_l{layer}_activation_entropy_mean", 0.0)
        )

    all_entropy_by_layer = np.asarray(all_entropy_by_layer, dtype=np.float32)
    last5_entropy_by_layer = np.asarray(last5_entropy_by_layer, dtype=np.float32)

    features["activation_entropy_last5_vs_all_mean_diff"] = float(
        last5_entropy_by_layer.mean() - all_entropy_by_layer.mean()
    )

    features["activation_entropy_last5_vs_all_mean_ratio"] = float(
        last5_entropy_by_layer.mean() / (all_entropy_by_layer.mean() + EPS)
    )

    features["activation_entropy_late_minus_early"] = float(
        all_entropy_by_layer[-2:].mean() - all_entropy_by_layer[:2].mean()
    )

    features["activation_entropy_late_div_early"] = float(
        all_entropy_by_layer[-2:].mean() /
        (abs(all_entropy_by_layer[:2].mean()) + EPS)
    )

    # BLOCK 19: Final / mid / early layer consistency
    consistency_layers = {
        "early": LAYERS[0],        # 11
        "mid": LAYERS[len(LAYERS) // 2],  # 14
        "final": LAYERS[-1],       # 16
    }

    # layer-level local stats
    for layer_name, layer in consistency_layers.items():
        all_tokens = hidden[layer][masks["all"]].float()
        last_token = layer_means["last_token"][layer]
        mean_repr = layer_means["all"][layer]

        if all_tokens.shape[0] == 0:
            all_tokens = torch.zeros((1, hidden.shape[-1]), dtype=hidden.dtype)

        features[f"{layer_name}_l{layer}_last_token_norm"] = safe_l2(last_token)
        features[f"{layer_name}_l{layer}_mean_norm"] = safe_l2(mean_repr)

        if all_tokens.shape[0] > 1:
            features[f"{layer_name}_l{layer}_token_variance"] = float(
                all_tokens.var(dim=0, unbiased=False).mean().item()
            )
        else:
            features[f"{layer_name}_l{layer}_token_variance"] = 0.0

        first_token = all_tokens[0]
        last_token_real = all_tokens[-1]

        features[f"{layer_name}_l{layer}_first_last_token_cosine"] = safe_cosine(
            first_token,
            last_token_real,
        )

        features[f"{layer_name}_l{layer}_first_last_token_distance"] = safe_l2(
            last_token_real - first_token
        )

    early_layer = consistency_layers["early"]
    mid_layer = consistency_layers["mid"]
    final_layer = consistency_layers["final"]

    early_last = layer_means["last_token"][early_layer]
    mid_last = layer_means["last_token"][mid_layer]
    final_last = layer_means["last_token"][final_layer]

    early_mean = layer_means["all"][early_layer]
    mid_mean = layer_means["all"][mid_layer]
    final_mean = layer_means["all"][final_layer]

    # last-token consistency
    features["mid_last_token_vs_final_last_token_cosine"] = safe_cosine(
        mid_last,
        final_last,
    )

    features["mid_last_token_vs_final_last_token_l2_distance"] = safe_l2(
        final_last - mid_last
    )

    features["early_last_token_vs_final_last_token_cosine"] = safe_cosine(
        early_last,
        final_last,
    )

    features["early_last_token_vs_final_last_token_l2_distance"] = safe_l2(
        final_last - early_last
    )

    # mean-representation consistency
    features["early_mean_vs_final_mean_cosine"] = safe_cosine(
        early_mean,
        final_mean,
    )

    features["early_mean_vs_final_mean_l2_distance"] = safe_l2(
        final_mean - early_mean
    )

    features["mid_mean_vs_final_mean_cosine"] = safe_cosine(
        mid_mean,
        final_mean,
    )

    features["mid_mean_vs_final_mean_l2_distance"] = safe_l2(
        final_mean - mid_mean
    )

    # norm ratios
    features["final_minus_mid_last_token_norm_ratio"] = (
        safe_l2(final_last - mid_last) /
        (safe_l2(mid_last) + EPS)
    )

    features["final_minus_early_last_token_norm_ratio"] = (
        safe_l2(final_last - early_last) /
        (safe_l2(early_last) + EPS)
    )

    features["final_minus_mid_mean_norm_ratio"] = (
        safe_l2(final_mean - mid_mean) /
        (safe_l2(mid_mean) + EPS)
    )

    features["final_minus_early_mean_norm_ratio"] = (
        safe_l2(final_mean - early_mean) /
        (safe_l2(early_mean) + EPS)
    )

    # BLOCK 20: Response validity & format features

    response_text_lower = response_text.lower()

    response_words = response_text.split()
    prompt_words = prompt_text.split()

    response_len = len(response_text)
    prompt_len = len(prompt_text)

    response_word_count = len(response_words)
    prompt_word_count = len(prompt_words)

    unique_words = len(set(w.lower() for w in response_words))

    special_chars = re.findall(r"[^a-zA-Z0-9\s]", response_text)
    digits = re.findall(r"\d", response_text)

    urls = re.findall(r"http[s]?://|www\.", response_text_lower)

    markdown_patterns = re.findall(
        r"```|`|\*\*|__|#+ |\* ",
        response_text,
    )

    code_blocks = re.findall(r"```.*?```", response_text, flags=re.DOTALL)

    bullet_lines = re.findall(
        r"^\s*[-•*]\s",
        response_text,
        flags=re.MULTILINE,
    )

    capital_words = [
        w for w in response_words
        if len(w) >= 2 and w.isupper()
    ]

    avg_word_len = (
        np.mean([len(w) for w in response_words])
        if response_words
        else 0.0
    )

    uncertainty_hits = sum(
        word in response_text_lower
        for word in UNCERTAINTY_WORDS
    )

    refusal_hits = sum(
        word in response_text_lower
        for word in REFUSAL_WORDS
    )

    citation_like_patterns = re.findall(
        r"\[\d+\]|\(\d{4}\)|et al\.|doi:",
        response_text_lower,
    )

    features["response_len_chars"] = response_len
    features["prompt_len_chars"] = prompt_len

    features["response_word_count"] = response_word_count
    features["prompt_word_count"] = prompt_word_count

    features["response_to_prompt_len_ratio"] = safe_div(
        response_len,
        prompt_len,
    )

    features["response_to_prompt_word_ratio"] = safe_div(
        response_word_count,
        prompt_word_count,
    )

    features["has_empty_response"] = float(response_len == 0)

    features["num_digits"] = len(digits)
    features["num_special_chars"] = len(special_chars)

    features["num_quotes"] = (
        response_text.count('"')
        + response_text.count("'")
    )

    features["num_parentheses"] = (
        response_text.count("(")
        + response_text.count(")")
    )

    features["num_brackets"] = (
        response_text.count("[")
        + response_text.count("]")
        + response_text.count("{")
        + response_text.count("}")
    )

    features["num_urls"] = len(urls)
    features["num_markdown_patterns"] = len(markdown_patterns)
    features["num_code_blocks"] = len(code_blocks)
    features["num_bullets"] = len(bullet_lines)

    features["num_newlines"] = response_text.count("\n")

    features["num_capital_words"] = len(capital_words)

    features["avg_word_len"] = float(avg_word_len)

    features["unique_word_ratio"] = safe_div(
        unique_words,
        response_word_count,
    )

    features["repetition_ratio"] = repetition_ratio(
        [w.lower() for w in response_words]
    )

    features["ends_with_terminal_punctuation"] = float(
        response_text.strip().endswith((".", "!", "?"))
    )

    features["has_uncertainty_words"] = float(
        uncertainty_hits > 0
    )

    features["uncertainty_word_count"] = uncertainty_hits

    features["has_refusal_words"] = float(
        refusal_hits > 0
    )

    features["refusal_word_count"] = refusal_hits

    features["has_citation_like_patterns"] = float(
        len(citation_like_patterns) > 0
    )

    features["citation_like_pattern_count"] = len(
        citation_like_patterns
    )

    # structural validity
    features["unmatched_parentheses"] = unmatched_symbol_count(
        response_text,
        "(",
        ")",
    )

    features["unmatched_square_brackets"] = unmatched_symbol_count(
        response_text,
        "[",
        "]",
    )

    features["unmatched_curly_brackets"] = unmatched_symbol_count(
        response_text,
        "{",
        "}",
    )

    features["response_format_validity_score"] = float(
        (response_len > 0)
        + (response_word_count >= 3)
        + (features["unmatched_parentheses"] == 0)
        + (features["unmatched_square_brackets"] == 0)
        + (features["unmatched_curly_brackets"] == 0)
    )
    
    if not features:
        features["dummy_feature"] = 0.0

    return {
        key: clean_value(value)
        for key, value in features.items()
    }


def extract_features_for_dataframe(df, model, tokenizer, device, has_label):
    texts = [
        str(prompt) + str(response)
        for prompt, response in zip(
            df["prompt"].astype(str),
            df["response"].astype(str),
        )
    ]

    rows = []

    for start in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="Extract geometric uncertainty features",
    ):
        batch_texts = texts[start:start + BATCH_SIZE]

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden_batch = torch.stack(
            outputs.hidden_states,
            dim=1,
        ).float().cpu()

        mask_batch = attention_mask.cpu().bool()

        for i in range(hidden_batch.shape[0]):
            row_id = start + i

            features = extract_features_one_sample(
                hidden=hidden_batch[i],
                valid_mask=mask_batch[i],
                prompt_text=df.iloc[row_id]["prompt"],
                response_text=df.iloc[row_id]["response"],
            )

            rows.append(features)

    out = pd.DataFrame(rows)
    out.insert(0, "source_index", df.index.to_numpy())

    if has_label:
        out["label"] = df["label"].astype(float).astype(int).to_numpy()

    out["prompt"] = df["prompt"].astype(str).to_numpy()
    out["response"] = df["response"].astype(str).to_numpy()

    return out


def main():
    device = get_device()

    print("=" * 80)
    print("BUILD GEOMETRIC UNCERTAINTY V2 FEATURES")
    print("=" * 80)
    print("Device:", device)
    print("Data file:", DATA_FILE)
    print("Test file:", TEST_FILE)
    print("Output dir:", OUTPUT_DIR)
    print("Layers:", LAYERS)
    print("Batch size:", BATCH_SIZE)

    model, tokenizer = get_model_and_tokenizer()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    train_df = pd.read_csv(DATA_FILE)

    train_features = extract_features_for_dataframe(
        df=train_df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        has_label=True,
    )

    train_features.to_parquet(TRAIN_OUTPUT, index=False)

    print()
    print("Saved train features:", TRAIN_OUTPUT)
    print("Train shape:", train_features.shape)

    if Path(TEST_FILE).exists():
        test_df = pd.read_csv(TEST_FILE)

        test_features = extract_features_for_dataframe(
            df=test_df,
            model=model,
            tokenizer=tokenizer,
            device=device,
            has_label=False,
        )

        test_features.to_parquet(TEST_OUTPUT, index=False)

        print()
        print("Saved test features:", TEST_OUTPUT)
        print("Test shape:", test_features.shape)

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()