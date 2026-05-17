"""
Build extra smart features for hallucination detection.

This file creates a separate parquet family for features that DO NOT require
prompt_len, logits, or forward hooks.

It intentionally DOES NOT duplicate geometric_uncertainty_v2 features.
The focus here is on extra feature families:

B. Additional heuristic token-position dynamics not covered by v2
C. Heuristic collapse/divergence
D. Heuristic self-contradiction and semantic divergence trajectories
E. Generation degeneration from raw text
F. Layer localization summaries and expanded layerwise spectral energy
G. Residual-like hidden deltas

Outputs:
./artifacts/extra_smart_features/features_dataset_extra_smart.parquet
./artifacts/extra_smart_features/features_test_extra_smart.parquet
"""

from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model import MAX_LENGTH, get_model_and_tokenizer


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"

OUTPUT_DIR = Path("./artifacts/extra_smart_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUTPUT = OUTPUT_DIR / "features_dataset_extra_smart.parquet"
TEST_OUTPUT = OUTPUT_DIR / "features_test_extra_smart.parquet"

BATCH_SIZE = 1
EXPORT_TEST = True

LAYERS = [10, 11, 12, 13, 14, 15, 16]
LAYER_PAIRS = list(zip(LAYERS[:-1], LAYERS[1:]))
LONG_LAYER_PAIRS = [(10, 12), (11, 13), (12, 14), (13, 15), (14, 16), (10, 16), (11, 16)]

EPS = 1e-8


# ============================================================
# BASIC HELPERS
# ============================================================


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def clean_value(value) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return value


def safe_l2(x) -> float:
    return clean_value(np.linalg.norm(to_numpy(x)))


def safe_cosine(a, b) -> float:
    a = to_numpy(a)
    b = to_numpy(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + EPS
    return clean_value(np.dot(a, b) / denom)


def safe_mean(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.shape[0] == 0:
        return torch.zeros(tokens.shape[-1], dtype=tokens.dtype)
    return tokens.mean(dim=0)


def entropy_from_values(values) -> float:
    values = np.abs(np.asarray(values, dtype=np.float32))
    total = values.sum()
    if total <= EPS:
        return 0.0
    probs = values / total
    return clean_value(-(probs * np.log(probs + EPS)).sum())


def add_stats(features: Dict[str, float], prefix: str, values: Iterable[float]) -> None:
    values = np.asarray(list(values), dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)

    features[f"{prefix}_mean"] = clean_value(values.mean())
    features[f"{prefix}_std"] = clean_value(values.std())
    features[f"{prefix}_min"] = clean_value(values.min())
    features[f"{prefix}_max"] = clean_value(values.max())
    features[f"{prefix}_range"] = clean_value(values.max() - values.min())
    features[f"{prefix}_p10"] = clean_value(np.percentile(values, 10))
    features[f"{prefix}_p25"] = clean_value(np.percentile(values, 25))
    features[f"{prefix}_p50"] = clean_value(np.percentile(values, 50))
    features[f"{prefix}_p75"] = clean_value(np.percentile(values, 75))
    features[f"{prefix}_p90"] = clean_value(np.percentile(values, 90))
    features[f"{prefix}_entropy"] = entropy_from_values(values)


def add_trajectory_features(features: Dict[str, float], prefix: str, values: Iterable[float]) -> None:
    values = np.asarray(list(values), dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)

    add_stats(features, prefix, values)

    diff1 = np.diff(values) if len(values) >= 2 else np.array([0.0], dtype=np.float32)
    diff2 = np.diff(values, n=2) if len(values) >= 3 else np.array([0.0], dtype=np.float32)

    x = np.arange(len(values), dtype=np.float32)
    slope = np.polyfit(x, values, 1)[0] if len(values) >= 2 else 0.0

    early = values[:2].mean() if len(values) >= 2 else values.mean()
    late = values[-2:].mean() if len(values) >= 2 else values.mean()

    features[f"{prefix}_slope"] = clean_value(slope)
    features[f"{prefix}_roughness"] = clean_value(np.abs(diff1).sum())
    features[f"{prefix}_smoothness"] = clean_value(1.0 / (1.0 + np.abs(diff1).sum()))
    features[f"{prefix}_acceleration_mean"] = clean_value(np.abs(diff2).mean())
    features[f"{prefix}_acceleration_max"] = clean_value(np.abs(diff2).max())
    features[f"{prefix}_late_minus_early"] = clean_value(late - early)
    features[f"{prefix}_late_div_early"] = clean_value(late / (abs(early) + EPS))
    features[f"{prefix}_num_increases"] = clean_value((diff1 > 0).sum())
    features[f"{prefix}_num_decreases"] = clean_value((diff1 < 0).sum())

    signs = np.sign(diff1)
    signs = signs[signs != 0]
    if len(signs) >= 2:
        sign_changes = (signs[1:] != signs[:-1]).sum()
    else:
        sign_changes = 0
    features[f"{prefix}_sign_changes"] = clean_value(sign_changes)


def add_spectral_features(features: Dict[str, float], prefix: str, values: Iterable[float]) -> None:
    values = np.asarray(list(values), dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)

    centered = values - values.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    total = power.sum()
    split = max(1, len(power) // 2)
    low = power[:split].sum()
    high = power[split:].sum()

    features[f"{prefix}_fft_energy"] = clean_value(total)
    features[f"{prefix}_fft_low_energy"] = clean_value(low)
    features[f"{prefix}_fft_high_energy"] = clean_value(high)
    features[f"{prefix}_fft_high_low_ratio"] = clean_value(high / (low + EPS))
    features[f"{prefix}_fft_entropy"] = entropy_from_values(power)
    features[f"{prefix}_fft_dominant_freq"] = clean_value(np.argmax(power) / max(len(power) - 1, 1))


def valid_token_positions(valid_mask: torch.Tensor) -> torch.Tensor:
    return torch.where(valid_mask.bool().cpu())[0]


def make_heuristic_zones(valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    idx = valid_token_positions(valid_mask)
    n = int(idx.numel())
    if n == 0:
        return {name: idx for name in ["all", "first70", "last40", "last30", "last20", "last5", "last_token", "early", "middle", "late"]}

    def last_frac(frac: float) -> torch.Tensor:
        keep = max(1, int(round(n * frac)))
        return idx[-keep:]

    first70_end = max(1, int(round(n * 0.70)))
    third = max(1, n // 3)
    middle_start = third
    late_start = min(n, 2 * third)

    return {
        "all": idx,
        "first70": idx[:first70_end],
        "last40": last_frac(0.40),
        "last30": last_frac(0.30),
        "last20": last_frac(0.20),
        "last5": idx[-min(5, n):],
        "last_token": idx[-1:],
        "early": idx[:third],
        "middle": idx[middle_start:late_start] if late_start > middle_start else idx,
        "late": idx[late_start:] if n > late_start else idx[-third:],
    }


def zone_tokens(hidden: torch.Tensor, layer: int, idx: torch.Tensor) -> torch.Tensor:
    if idx.numel() == 0:
        return torch.zeros((1, hidden.shape[-1]), dtype=hidden.dtype)
    return hidden[layer, idx].float()


def pairwise_cosine_stats(vectors: np.ndarray) -> Dict[str, float]:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape[0] <= 1:
        return {"mean": 1.0, "std": 0.0, "min": 1.0, "p10": 1.0, "p90": 1.0}

    normed = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + EPS)
    sim = normed @ normed.T
    tri = sim[np.triu_indices_from(sim, k=1)]

    return {
        "mean": clean_value(tri.mean()),
        "std": clean_value(tri.std()),
        "min": clean_value(tri.min()),
        "p10": clean_value(np.percentile(tri, 10)),
        "p90": clean_value(np.percentile(tri, 90)),
    }



def add_pairwise_hidden_features(features: Dict[str, float], prefix: str, tokens: torch.Tensor) -> None:
    vectors = to_numpy(tokens)
    pcs = pairwise_cosine_stats(vectors)
    features[f"{prefix}_pairwise_cosine_mean"] = pcs["mean"]
    features[f"{prefix}_pairwise_cosine_std"] = pcs["std"]
    features[f"{prefix}_pairwise_cosine_min"] = pcs["min"]
    features[f"{prefix}_pairwise_cosine_p10"] = pcs["p10"]
    features[f"{prefix}_pairwise_cosine_p90"] = pcs["p90"]
    features[f"{prefix}_collapse_score"] = clean_value(pcs["mean"])
    features[f"{prefix}_divergence_score"] = clean_value(1.0 - pcs["mean"])


def add_layerwise_spectral_energy_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
    zone_means: Dict[str, Dict[int, torch.Tensor]],
) -> None:
    """Expanded spectral summaries over token positions and over layer trajectories."""
    zone_names = ["all", "first70", "last30", "last20", "last5", "early", "middle", "late"]

    for zone_name in zone_names:
        energy_by_layer = []
        high_ratio_by_layer = []
        entropy_by_layer = []
        dominant_by_layer = []

        for layer in LAYERS:
            tokens = zone_tokens(hidden, layer, zones[zone_name])
            token_norms = torch.linalg.norm(tokens.float(), dim=1).detach().cpu().numpy()
            prefix = f"layer_spectral_l{layer}_{zone_name}"

            add_spectral_features(features, f"{prefix}_token_norm", token_norms)
            add_trajectory_features(features, f"{prefix}_token_norm_position", token_norms)

            centered_tokens = to_numpy(tokens - tokens.mean(dim=0, keepdim=True))
            fft_power = np.abs(np.fft.rfft(centered_tokens, axis=0)) ** 2
            freq_power = fft_power.sum(axis=1)
            split = max(1, len(freq_power) // 2)
            total = freq_power.sum()
            low = freq_power[:split].sum()
            high = freq_power[split:].sum()

            features[f"{prefix}_hidden_fft_energy"] = clean_value(total)
            features[f"{prefix}_hidden_fft_low_energy"] = clean_value(low)
            features[f"{prefix}_hidden_fft_high_energy"] = clean_value(high)
            features[f"{prefix}_hidden_fft_high_low_ratio"] = clean_value(high / (low + EPS))
            features[f"{prefix}_hidden_fft_entropy"] = entropy_from_values(freq_power)
            features[f"{prefix}_hidden_fft_dominant_freq"] = clean_value(
                np.argmax(freq_power) / max(len(freq_power) - 1, 1)
            )

            energy_by_layer.append(features[f"{prefix}_hidden_fft_energy"])
            high_ratio_by_layer.append(features[f"{prefix}_hidden_fft_high_low_ratio"])
            entropy_by_layer.append(features[f"{prefix}_hidden_fft_entropy"])
            dominant_by_layer.append(features[f"{prefix}_hidden_fft_dominant_freq"])

        add_trajectory_features(features, f"layer_spectral_{zone_name}_energy_by_layer", energy_by_layer)
        add_spectral_features(features, f"layer_spectral_{zone_name}_energy_by_layer", energy_by_layer)
        add_trajectory_features(features, f"layer_spectral_{zone_name}_high_ratio_by_layer", high_ratio_by_layer)
        add_trajectory_features(features, f"layer_spectral_{zone_name}_entropy_by_layer", entropy_by_layer)
        add_trajectory_features(features, f"layer_spectral_{zone_name}_dominant_freq_by_layer", dominant_by_layer)

    # Spectral energy of the zone mean trajectory across layers.
    for zone_name in zone_names:
        mean_vectors = np.stack([to_numpy(zone_means[zone_name][layer]) for layer in LAYERS], axis=0)
        centered = mean_vectors - mean_vectors.mean(axis=0, keepdims=True)
        layer_power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
        freq_power = layer_power.sum(axis=1)
        prefix = f"layer_spectral_{zone_name}_mean_vector_across_layers"
        add_spectral_features(features, prefix, freq_power)
        features[f"{prefix}_total_energy"] = clean_value(freq_power.sum())
        features[f"{prefix}_entropy"] = entropy_from_values(freq_power)


# ============================================================
# B-C-D-F-G. HIDDEN EXTRA FEATURES WITHOUT DUPLICATING V2
# ============================================================


def add_hidden_extra_features(features: Dict[str, float], hidden: torch.Tensor, valid_mask: torch.Tensor) -> None:
    zones = make_heuristic_zones(valid_mask)

    zone_means: Dict[str, Dict[int, torch.Tensor]] = {}
    for zone_name in zones:
        zone_means[zone_name] = {}
        for layer in LAYERS:
            zone_means[zone_name][layer] = safe_mean(zone_tokens(hidden, layer, zones[zone_name]))

    # B. Heuristic token-position dynamics.
    for layer in LAYERS:
        first70 = zone_means["first70"][layer]
        last30 = zone_means["last30"][layer]
        last20 = zone_means["last20"][layer]
        last5 = zone_means["last5"][layer]
        all_mean = zone_means["all"][layer]
        last_token = zone_means["last_token"][layer]
        early = zone_means["early"][layer]
        middle = zone_means["middle"][layer]
        late = zone_means["late"][layer]

        features[f"heur_l{layer}_first70_last30_distance"] = safe_l2(last30 - first70)
        features[f"heur_l{layer}_first70_last30_cosine"] = safe_cosine(first70, last30)
        features[f"heur_l{layer}_last20_all_distance"] = safe_l2(last20 - all_mean)
        features[f"heur_l{layer}_last20_all_cosine"] = safe_cosine(last20, all_mean)
        features[f"heur_l{layer}_last5_all_distance"] = safe_l2(last5 - all_mean)
        features[f"heur_l{layer}_last5_all_cosine"] = safe_cosine(last5, all_mean)
        features[f"heur_l{layer}_last_token_all_distance"] = safe_l2(last_token - all_mean)
        features[f"heur_l{layer}_last_token_last5_cosine"] = safe_cosine(last_token, last5)
        features[f"heur_l{layer}_late_instability"] = safe_l2(late - middle) + safe_l2(last5 - last20)
        features[f"heur_l{layer}_response_ending_drift"] = safe_l2(last_token - last20)

        # C. Explicit heuristic collapse/divergence block.
        all_tokens = zone_tokens(hidden, layer, zones["all"])
        last20_tokens = zone_tokens(hidden, layer, zones["last20"])
        last5_tokens = zone_tokens(hidden, layer, zones["last5"])
        late_tokens = zone_tokens(hidden, layer, zones["late"])

        all_norms = torch.linalg.norm(all_tokens.float(), dim=1).detach().cpu().numpy()
        last20_norms = torch.linalg.norm(last20_tokens.float(), dim=1).detach().cpu().numpy()
        last5_norms = torch.linalg.norm(last5_tokens.float(), dim=1).detach().cpu().numpy()
        late_norms = torch.linalg.norm(late_tokens.float(), dim=1).detach().cpu().numpy()

        add_pairwise_hidden_features(features, f"collapse_l{layer}_all", all_tokens)
        add_pairwise_hidden_features(features, f"collapse_l{layer}_late", late_tokens)
        add_pairwise_hidden_features(features, f"collapse_l{layer}_last20", last20_tokens)
        add_pairwise_hidden_features(features, f"collapse_l{layer}_last5", last5_tokens)

        features[f"collapse_l{layer}_last5_norm_std_ratio"] = clean_value(last5_norms.std() / (all_norms.std() + EPS))
        features[f"collapse_l{layer}_last20_norm_std_ratio"] = clean_value(last20_norms.std() / (all_norms.std() + EPS))
        features[f"collapse_l{layer}_late_norm_std_ratio"] = clean_value(late_norms.std() / (all_norms.std() + EPS))
        features[f"collapse_l{layer}_last5_norm_mean_ratio"] = clean_value(last5_norms.mean() / (all_norms.mean() + EPS))
        features[f"collapse_l{layer}_late_norm_mean_ratio"] = clean_value(late_norms.mean() / (all_norms.mean() + EPS))
        features[f"diverg_l{layer}_first70_last30_distance"] = features[f"heur_l{layer}_first70_last30_distance"]
        features[f"diverg_l{layer}_late_all_distance"] = safe_l2(late - all_mean)
        features[f"diverg_l{layer}_last5_last20_distance"] = safe_l2(last5 - last20)
        features[f"diverg_l{layer}_last_token_last20_distance"] = safe_l2(last_token - last20)

        # D. Heuristic self-contradiction and semantic divergence.
        features[f"contrad_l{layer}_early_late_cosine"] = safe_cosine(early, late)
        features[f"contrad_l{layer}_early_late_distance"] = safe_l2(late - early)
        features[f"contrad_l{layer}_middle_late_distance"] = safe_l2(late - middle)
        features[f"contrad_l{layer}_early_middle_distance"] = safe_l2(middle - early)
        features[f"contrad_l{layer}_semantic_divergence"] = safe_l2(late - early) / (safe_l2(early) + EPS)
        features[f"contrad_l{layer}_semantic_divergence_last30"] = safe_l2(last30 - first70) / (safe_l2(first70) + EPS)
        features[f"contrad_l{layer}_semantic_divergence_last20"] = safe_l2(last20 - all_mean) / (safe_l2(all_mean) + EPS)
        features[f"contrad_l{layer}_semantic_divergence_last5"] = safe_l2(last5 - all_mean) / (safe_l2(all_mean) + EPS)
        features[f"contrad_l{layer}_consistency_decay"] = safe_cosine(early, middle) - safe_cosine(middle, late)
        features[f"contrad_l{layer}_semantic_curvature"] = safe_l2((late - middle) - (middle - early))

    # Layer trajectories for token-position dynamics.
    for metric_prefix in [
        "first70_last30_distance",
        "last20_all_distance",
        "last5_all_distance",
        "last_token_all_distance",
        "late_instability",
        "response_ending_drift",
    ]:
        values = [features[f"heur_l{layer}_{metric_prefix}"] for layer in LAYERS]
        add_trajectory_features(features, f"heur_{metric_prefix}_by_layer", values)
        add_spectral_features(features, f"heur_{metric_prefix}_by_layer", values)

    # Explicit collapse/divergence trajectories.
    collapse_series = {
        "all_pairwise_cosine": [features[f"collapse_l{layer}_all_pairwise_cosine_mean"] for layer in LAYERS],
        "late_pairwise_cosine": [features[f"collapse_l{layer}_late_pairwise_cosine_mean"] for layer in LAYERS],
        "last20_pairwise_cosine": [features[f"collapse_l{layer}_last20_pairwise_cosine_mean"] for layer in LAYERS],
        "last5_pairwise_cosine": [features[f"collapse_l{layer}_last5_pairwise_cosine_mean"] for layer in LAYERS],
        "last5_norm_std_ratio": [features[f"collapse_l{layer}_last5_norm_std_ratio"] for layer in LAYERS],
        "last20_norm_std_ratio": [features[f"collapse_l{layer}_last20_norm_std_ratio"] for layer in LAYERS],
        "late_norm_std_ratio": [features[f"collapse_l{layer}_late_norm_std_ratio"] for layer in LAYERS],
    }
    for name, values in collapse_series.items():
        add_trajectory_features(features, f"heuristic_collapse_{name}_trajectory", values)
        add_spectral_features(features, f"heuristic_collapse_{name}_trajectory", values)

    divergence_series = {
        "first70_last30": [features[f"diverg_l{layer}_first70_last30_distance"] for layer in LAYERS],
        "late_all": [features[f"diverg_l{layer}_late_all_distance"] for layer in LAYERS],
        "last5_last20": [features[f"diverg_l{layer}_last5_last20_distance"] for layer in LAYERS],
        "last_token_last20": [features[f"diverg_l{layer}_last_token_last20_distance"] for layer in LAYERS],
    }
    for name, values in divergence_series.items():
        add_trajectory_features(features, f"heuristic_divergence_{name}_trajectory", values)
        add_spectral_features(features, f"heuristic_divergence_{name}_trajectory", values)

    # Expanded semantic divergence trajectories without prompt_len.
    semantic_series = {
        "early_late_distance": [features[f"contrad_l{layer}_early_late_distance"] for layer in LAYERS],
        "early_middle_distance": [features[f"contrad_l{layer}_early_middle_distance"] for layer in LAYERS],
        "middle_late_distance": [features[f"contrad_l{layer}_middle_late_distance"] for layer in LAYERS],
        "semantic_divergence": [features[f"contrad_l{layer}_semantic_divergence"] for layer in LAYERS],
        "semantic_divergence_last30": [features[f"contrad_l{layer}_semantic_divergence_last30"] for layer in LAYERS],
        "semantic_divergence_last20": [features[f"contrad_l{layer}_semantic_divergence_last20"] for layer in LAYERS],
        "semantic_divergence_last5": [features[f"contrad_l{layer}_semantic_divergence_last5"] for layer in LAYERS],
        "semantic_curvature": [features[f"contrad_l{layer}_semantic_curvature"] for layer in LAYERS],
        "consistency_decay": [features[f"contrad_l{layer}_consistency_decay"] for layer in LAYERS],
    }
    for name, values in semantic_series.items():
        add_trajectory_features(features, f"semantic_divergence_{name}_trajectory", values)
        add_spectral_features(features, f"semantic_divergence_{name}_trajectory", values)

    # Backward-compatible names from the first version.
    add_trajectory_features(features, "contrad_growth_trajectory", semantic_series["early_late_distance"])
    add_spectral_features(features, "contrad_growth_trajectory", semantic_series["early_late_distance"])
    add_trajectory_features(features, "contrad_consistency_decay_trajectory", semantic_series["consistency_decay"])

    # F. Expanded layerwise spectral energy.
    add_layerwise_spectral_energy_features(features, hidden, zones, zone_means)

    # F. Layer localization and G. residual-like hidden deltas.
    per_transition_energy = []
    per_transition_instability = []
    per_transition_explosion = []
    per_transition_collapse = []

    for left, right in LAYER_PAIRS + LONG_LAYER_PAIRS:
        all_left = zone_tokens(hidden, left, zones["all"])
        all_right = zone_tokens(hidden, right, zones["all"])
        delta_tokens = all_right - all_left
        delta_norms = torch.linalg.norm(delta_tokens.float(), dim=1).detach().cpu().numpy()

        prefix = f"residual_l{left}_to_l{right}"
        add_stats(features, f"{prefix}_token_delta_norm", delta_norms)
        add_trajectory_features(features, f"{prefix}_token_delta_norm_position", delta_norms)
        add_spectral_features(features, f"{prefix}_token_delta_norm_position", delta_norms)

        start_mean = zone_means["all"][left]
        end_mean = zone_means["all"][right]
        delta_mean = end_mean - start_mean

        energy = safe_l2(delta_mean)
        per_transition_energy.append(energy)
        per_transition_instability.append(np.std(delta_norms))
        per_transition_explosion.append(max(0.0, safe_l2(end_mean) - safe_l2(start_mean)))
        per_transition_collapse.append(max(0.0, safe_l2(start_mean) - safe_l2(end_mean)))

        features[f"{prefix}_mean_delta_energy"] = energy
        features[f"{prefix}_mean_delta_cosine_with_start"] = safe_cosine(delta_mean, start_mean)
        features[f"{prefix}_residual_explosion"] = per_transition_explosion[-1]
        features[f"{prefix}_residual_collapse"] = per_transition_collapse[-1]

    add_trajectory_features(features, "layer_localization_transition_energy", per_transition_energy)
    add_spectral_features(features, "layer_localization_transition_energy", per_transition_energy)
    add_trajectory_features(features, "layer_localization_transition_instability", per_transition_instability)
    add_spectral_features(features, "layer_localization_transition_instability", per_transition_instability)
    add_trajectory_features(features, "residual_explosion_trajectory", per_transition_explosion)
    add_spectral_features(features, "residual_explosion_trajectory", per_transition_explosion)
    add_trajectory_features(features, "residual_collapse_trajectory", per_transition_collapse)
    add_spectral_features(features, "residual_collapse_trajectory", per_transition_collapse)

    if per_transition_energy:
        strongest_idx = int(np.argmax(per_transition_energy))
        features["layer_localization_strongest_transition_idx"] = clean_value(strongest_idx / max(len(per_transition_energy) - 1, 1))
        features["layer_localization_strongest_transition_energy"] = clean_value(max(per_transition_energy))

    # Layerwise disagreement/collapse via heuristic zones, lighter than geometric_v2.
    layer_disagreement = []
    layer_collapse = []
    for layer in LAYERS:
        last5 = zone_tokens(hidden, layer, zones["last5"])
        all_tokens = zone_tokens(hidden, layer, zones["all"])
        last5_norms = torch.linalg.norm(last5.float(), dim=1).detach().cpu().numpy()
        all_norms = torch.linalg.norm(all_tokens.float(), dim=1).detach().cpu().numpy()
        layer_disagreement.append(abs(last5_norms.std() - all_norms.std()))
        layer_collapse.append(abs(last5_norms.mean() - all_norms.mean()))

    add_trajectory_features(features, "layerwise_disagreement_extra", layer_disagreement)
    add_spectral_features(features, "layerwise_disagreement_extra", layer_disagreement)
    add_trajectory_features(features, "layerwise_collapse_extra", layer_collapse)
    add_spectral_features(features, "layerwise_collapse_extra", layer_collapse)


# ============================================================
# E. GENERATION DEGENERATION / TEXT FEATURES
# ============================================================


def ngram_repetition(words: List[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(v for v in counts.values() if v > 1)
    return repeated / max(len(ngrams), 1)


def add_generation_degradation_features(features: Dict[str, float], prompt: str, response: str) -> None:
    response = str(response)
    prompt = str(prompt)
    text = response.strip()
    lower = text.lower()
    words = re.findall(r"\b\w+\b", lower)
    chars = list(text)

    features["gen_response_len"] = clean_value(len(response))
    features["gen_prompt_len"] = clean_value(len(prompt))
    features["gen_response_prompt_len_ratio"] = clean_value(len(response) / (len(prompt) + EPS))
    features["gen_word_count"] = clean_value(len(words))
    features["gen_unique_word_ratio"] = clean_value(len(set(words)) / (len(words) + EPS))

    counts = Counter(words)
    if words:
        max_word_freq = max(counts.values())
        repeated_words = sum(v for v in counts.values() if v > 1)
    else:
        max_word_freq = 0
        repeated_words = 0

    features["gen_max_word_frequency_ratio"] = clean_value(max_word_freq / (len(words) + EPS))
    features["gen_repeated_word_ratio"] = clean_value(repeated_words / (len(words) + EPS))
    features["gen_bigram_repetition_ratio"] = ngram_repetition(words, 2)
    features["gen_trigram_repetition_ratio"] = ngram_repetition(words, 3)
    features["gen_4gram_repetition_ratio"] = ngram_repetition(words, 4)

    # Looping behavior: repeated adjacent words / repeated sentence starts.
    adjacent_repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    features["gen_adjacent_word_repeat_ratio"] = clean_value(adjacent_repeats / (len(words) + EPS))

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    starts = [" ".join(re.findall(r"\b\w+\b", s.lower())[:3]) for s in sentences]
    starts = [s for s in starts if s]
    features["gen_sentence_count"] = clean_value(len(sentences))
    features["gen_sentence_start_repetition"] = clean_value(
        1.0 - len(set(starts)) / (len(starts) + EPS) if starts else 0.0
    )

    sentence_lengths = [len(s.split()) for s in sentences]
    add_stats(features, "gen_sentence_length", sentence_lengths)

    # Character and punctuation degeneration.
    punct = re.findall(r"[^\w\s]", text)
    features["gen_punctuation_count"] = clean_value(len(punct))
    features["gen_punctuation_ratio"] = clean_value(len(punct) / (len(text) + EPS))
    features["gen_newline_count"] = clean_value(text.count("\n"))
    features["gen_digit_count"] = clean_value(len(re.findall(r"\d", text)))
    features["gen_uppercase_word_count"] = clean_value(sum(1 for w in text.split() if len(w) > 1 and w.isupper()))

    if chars:
        char_counts = Counter(chars)
        char_probs = np.array(list(char_counts.values()), dtype=np.float32) / len(chars)
        features["gen_char_entropy"] = clean_value(-(char_probs * np.log(char_probs + EPS)).sum())
    else:
        features["gen_char_entropy"] = 0.0

    if words:
        word_probs = np.array(list(counts.values()), dtype=np.float32) / len(words)
        features["gen_word_entropy"] = clean_value(-(word_probs * np.log(word_probs + EPS)).sum())
        features["gen_lexical_collapse_score"] = clean_value(1.0 - len(set(words)) / len(words))
    else:
        features["gen_word_entropy"] = 0.0
        features["gen_lexical_collapse_score"] = 0.0

    # Simple format degeneration flags.
    features["gen_has_unclosed_parentheses"] = clean_value(abs(text.count("(") - text.count(")")) > 0)
    features["gen_has_unclosed_brackets"] = clean_value(abs(text.count("[") - text.count("]")) > 0)
    features["gen_markdown_marker_count"] = clean_value(len(re.findall(r"```|`|\*\*|__|#+", text)))


# ============================================================
# SAMPLE FEATURE EXTRACTION
# ============================================================


def extract_features_one_sample(
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
    prompt_text: str,
    response_text: str,
) -> Dict[str, float]:
    hidden = hidden.float().cpu()
    valid_mask = valid_mask.bool().cpu()

    features: Dict[str, float] = {}

    # A is intentionally not duplicated here: geometric_uncertainty_v2 is another parquet.
    # This file adds only non-duplicated extra smart features.
    add_hidden_extra_features(features, hidden, valid_mask)
    add_generation_degradation_features(features, prompt_text, response_text)

    return {key: clean_value(value) for key, value in features.items()}


# ============================================================
# DATASET EXTRACTION
# ============================================================


def extract_dataset_features(df: pd.DataFrame, model, tokenizer, device: torch.device, has_label: bool) -> pd.DataFrame:
    prompts = df["prompt"].astype(str).tolist()
    responses = df["response"].astype(str).tolist()
    texts = [p + r for p, r in zip(prompts, responses)]

    rows: List[Dict[str, float]] = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extract extra smart features"):
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
                output_hidden_states=True,
            )

        hidden_batch = torch.stack(outputs.hidden_states, dim=1).float().cpu()
        mask_batch = attention_mask.cpu().bool()

        for i in range(hidden_batch.shape[0]):
            row_idx = start + i
            rows.append(
                extract_features_one_sample(
                    hidden=hidden_batch[i],
                    valid_mask=mask_batch[i],
                    prompt_text=prompts[row_idx],
                    response_text=responses[row_idx],
                )
            )

        del outputs, hidden_batch, mask_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.insert(0, "source_index", df.index.to_numpy())

    if has_label:
        out["label"] = df["label"].astype(float).astype(int).to_numpy()

    out["prompt"] = df["prompt"].astype(str).to_numpy()
    out["response"] = df["response"].astype(str).to_numpy()

    return out


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    t0 = time.time()
    device = get_device()

    print("=" * 80)
    print("BUILD EXTRA SMART FEATURES")
    print("=" * 80)
    print(f"Device      : {device}")
    print(f"Data file   : {DATA_FILE}")
    print(f"Test file   : {TEST_FILE}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"Batch size  : {BATCH_SIZE}")
    print("Note        : no prompt_len, no logits, no hooks, no attentions")

    model, tokenizer = get_model_and_tokenizer()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    train_df = pd.read_csv(DATA_FILE)
    print(f"\nDataset rows: {len(train_df)}")

    train_features = extract_dataset_features(
        df=train_df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        has_label=True,
    )
    train_features.to_parquet(TRAIN_OUTPUT, index=False)

    print(f"Saved train: {TRAIN_OUTPUT}")
    print(f"Train shape: {train_features.shape}")

    if EXPORT_TEST and Path(TEST_FILE).exists():
        test_df = pd.read_csv(TEST_FILE)
        print(f"\nTest rows: {len(test_df)}")

        test_features = extract_dataset_features(
            df=test_df,
            model=model,
            tokenizer=tokenizer,
            device=device,
            has_label=False,
        )
        test_features.to_parquet(TEST_OUTPUT, index=False)

        print(f"Saved test: {TEST_OUTPUT}")
        print(f"Test shape: {test_features.shape}")

    print(f"\nDone in {time.time() - t0:.1f} sec")
    print("=" * 80)


if __name__ == "__main__":
    main()
