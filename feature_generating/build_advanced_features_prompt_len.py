"""
Build advanced prompt-length-aware hidden-state features.

This script uses only one additional boundary signal: prompt_len.
It does NOT require output_attentions, logits, hooks, or any external model.

Outputs:
./artifacts/advanced_features_prompt_len/features_dataset_advanced_prompt_len.parquet
./artifacts/advanced_features_prompt_len/features_test_advanced_prompt_len.parquet
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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

OUTPUT_DIR = Path("./artifacts/advanced_features_prompt_len")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUTPUT = OUTPUT_DIR / "features_dataset_advanced_prompt_len.parquet"
TEST_OUTPUT = OUTPUT_DIR / "features_test_advanced_prompt_len.parquet"

BATCH_SIZE = 1
EXPORT_TEST = True
EPS = 1e-8

MAX_POOL_LAYERS = [12, 13, 16]
MEAN_POOL_LAYERS = [14, 15, 16]
COMPACT_LAYERS = list(range(10, 20))
TEMPORAL_LAYERS = [12, 13, 14, 15, 16]
ICR_LAYERS = [0, 6, 12, 18, 23]
CENTROID_LAYERS = [1, 6, 12, 18, 23]


# ============================================================
# BASIC HELPERS
# ============================================================


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clean_value(value) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return value


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def safe_layer_index(layer: int, n_layers: int) -> int:
    if layer < 0:
        layer = n_layers + layer
    return int(min(max(layer, 0), n_layers - 1))


def safe_l2(x) -> float:
    arr = to_numpy(x).reshape(-1)
    return clean_value(np.linalg.norm(arr))


def safe_cosine(a, b) -> float:
    a_arr = to_numpy(a).reshape(-1)
    b_arr = to_numpy(b).reshape(-1)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + EPS
    return clean_value(np.dot(a_arr, b_arr) / denom)


def safe_angle(a, b) -> float:
    cos = np.clip(safe_cosine(a, b), -1.0, 1.0)
    return clean_value(math.acos(cos))


def safe_mean(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.numel() == 0 or tokens.shape[0] == 0:
        return torch.zeros(tokens.shape[-1], dtype=torch.float32)
    return tokens.float().mean(dim=0)


def safe_max(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.numel() == 0 or tokens.shape[0] == 0:
        return torch.zeros(tokens.shape[-1], dtype=torch.float32)
    return tokens.float().max(dim=0).values


def add_scalar(features: Dict[str, float], name: str, value) -> None:
    features[name] = clean_value(value)


def add_vector(features: Dict[str, float], prefix: str, vector: torch.Tensor | np.ndarray) -> None:
    arr = to_numpy(vector).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    for dim, value in enumerate(arr):
        features[f"{prefix}_d{dim}"] = clean_value(value)


def add_basic_stats(features: Dict[str, float], prefix: str, values: Iterable[float]) -> None:
    arr = np.asarray(list(values), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        arr = np.array([0.0], dtype=np.float32)
    add_scalar(features, f"{prefix}_mean", arr.mean())
    add_scalar(features, f"{prefix}_std", arr.std())
    add_scalar(features, f"{prefix}_min", arr.min())
    add_scalar(features, f"{prefix}_max", arr.max())


def valid_positions(valid_mask: torch.Tensor) -> torch.Tensor:
    return torch.where(valid_mask.bool().cpu())[0]


def ensure_non_empty(idx: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    if idx.numel() > 0:
        return idx
    if fallback.numel() > 0:
        return fallback[-1:]
    return fallback


def first_fraction(idx: torch.Tensor, frac: float) -> torch.Tensor:
    n = int(idx.numel())
    if n == 0:
        return idx
    keep = max(1, int(round(n * frac)))
    return idx[:keep]


def middle_fraction(idx: torch.Tensor, start_frac: float, end_frac: float) -> torch.Tensor:
    n = int(idx.numel())
    if n == 0:
        return idx
    start = min(n - 1, max(0, int(round(n * start_frac))))
    end = min(n, max(start + 1, int(round(n * end_frac))))
    return idx[start:end]


def last_fraction(idx: torch.Tensor, frac: float) -> torch.Tensor:
    n = int(idx.numel())
    if n == 0:
        return idx
    keep = max(1, int(round(n * frac)))
    return idx[-keep:]


def get_prompt_lengths(tokenizer, prompts: Sequence[str], max_length: int) -> List[int]:
    lengths = []
    for prompt in prompts:
        enc = tokenizer(
            str(prompt),
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        lengths.append(int(len(enc["input_ids"])))
    return lengths


def get_prompt_len_column_or_tokenize(df: pd.DataFrame, tokenizer) -> List[int]:
    for candidate in ["prompt_len", "prompt_length", "prompt_len_tokens"]:
        if candidate in df.columns:
            return df[candidate].astype(int).tolist()
    return get_prompt_lengths(tokenizer, df["prompt"].astype(str).tolist(), MAX_LENGTH)


def build_exact_zones(valid_mask: torch.Tensor, prompt_len: int) -> Dict[str, torch.Tensor]:
    valid_mask = valid_mask.bool().cpu()
    valid_idx = valid_positions(valid_mask)
    seq_len = int(valid_mask.numel())

    if valid_idx.numel() == 0:
        empty = valid_idx
        return {
            "all": empty,
            "prompt": empty,
            "response": empty,
            "response_wo_eos": empty,
            "response_first30": empty,
            "response_middle40": empty,
            "response_late30": empty,
            "response_last5": empty,
            "response_last10": empty,
            "prompt_last5": empty,
            "prompt_last10": empty,
            "last_token": empty,
        }

    prompt_len = min(max(int(prompt_len), 0), seq_len)
    pos = torch.arange(seq_len)

    prompt_idx = torch.where(valid_mask & (pos < prompt_len))[0]
    response_idx = torch.where(valid_mask & (pos >= prompt_len))[0]

    # Fallbacks requested in the spec.
    if response_idx.numel() == 0:
        response_idx = valid_idx[-1:]
    if prompt_idx.numel() == 0:
        prompt_idx = valid_idx[: max(1, min(int(prompt_len), int(valid_idx.numel())))]
        prompt_idx = ensure_non_empty(prompt_idx, valid_idx)

    response_wo_eos = response_idx[:-1] if response_idx.numel() >= 2 else response_idx.new_empty((0,))
    if response_wo_eos.numel() == 0:
        response_wo_eos = response_idx

    zones = {
        "all": valid_idx,
        "prompt": ensure_non_empty(prompt_idx, valid_idx),
        "response": ensure_non_empty(response_idx, valid_idx),
        "response_wo_eos": ensure_non_empty(response_wo_eos, response_idx),
        "response_first30": ensure_non_empty(first_fraction(response_idx, 0.30), response_idx),
        "response_middle40": ensure_non_empty(middle_fraction(response_idx, 0.30, 0.70), response_idx),
        "response_late30": ensure_non_empty(last_fraction(response_idx, 0.30), response_idx),
        "response_last5": ensure_non_empty(response_idx[-min(5, int(response_idx.numel())):], response_idx),
        "response_last10": ensure_non_empty(response_idx[-min(10, int(response_idx.numel())):], response_idx),
        "prompt_last5": ensure_non_empty(prompt_idx[-min(5, int(prompt_idx.numel())):], prompt_idx),
        "prompt_last10": ensure_non_empty(prompt_idx[-min(10, int(prompt_idx.numel())):], prompt_idx),
        "last_token": valid_idx[-1:],
    }
    return zones


def zone_tokens(hidden: torch.Tensor, layer: int, idx: torch.Tensor) -> torch.Tensor:
    layer_idx = safe_layer_index(layer, hidden.shape[0])
    if idx.numel() == 0:
        return torch.zeros((1, hidden.shape[-1]), dtype=torch.float32)
    return hidden[layer_idx, idx].float().cpu()


def layer_vector(hidden: torch.Tensor, layer: int, pos: int) -> torch.Tensor:
    layer_idx = safe_layer_index(layer, hidden.shape[0])
    return hidden[layer_idx, pos].float().cpu()


# ============================================================
# 1. EXACT RESPONSE MASKS + 10. LENGTH/META FEATURES
# ============================================================


def add_length_meta_features(
    features: Dict[str, float],
    zones: Dict[str, torch.Tensor],
    prompt_len: int,
    max_length: int,
) -> None:
    n_valid = int(zones["all"].numel())
    prompt_tokens = int(zones["prompt"].numel())
    response_tokens = int(zones["response"].numel())
    response_wo_eos_tokens = int(zones["response_wo_eos"].numel())

    add_scalar(features, "n_valid", n_valid)
    add_scalar(features, "prompt_len_tokens", prompt_tokens)
    add_scalar(features, "raw_prompt_len_tokens", int(prompt_len))
    add_scalar(features, "response_len_tokens", response_tokens)
    add_scalar(features, "response_wo_eos_len", response_wo_eos_tokens)
    add_scalar(features, "log1p_prompt_len", np.log1p(prompt_tokens))
    add_scalar(features, "log1p_response_len", np.log1p(response_tokens))
    add_scalar(features, "response_len_div_n_valid", response_tokens / (n_valid + EPS))
    add_scalar(features, "response_len_div_prompt_len", response_tokens / (prompt_tokens + EPS))
    add_scalar(features, "response_len_div_512", response_tokens / 512.0)
    add_scalar(features, "is_response_short", int(response_tokens <= 2))
    add_scalar(
        features,
        "is_response_truncated_or_empty",
        int(response_tokens <= 1 or n_valid >= max_length - 2),
    )


# ============================================================
# 2-3. EXACT MAX/MEAN + L16 HYBRID
# ============================================================


def add_exact_pooling_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    for layer in MAX_POOL_LAYERS:
        vec = safe_max(zone_tokens(hidden, layer, zones["response_wo_eos"]))
        add_vector(features, f"max_l{layer}_response_wo_eos", vec)

    for layer in MEAN_POOL_LAYERS:
        vec = safe_mean(zone_tokens(hidden, layer, zones["response_wo_eos"]))
        add_vector(features, f"mean_l{layer}_response_wo_eos", vec)

    # Ablation variants without EOS stripping.
    for layer in [12, 13]:
        add_vector(features, f"max_l{layer}_response", safe_max(zone_tokens(hidden, layer, zones["response"])))
    for layer in [14, 15]:
        add_vector(features, f"mean_l{layer}_response", safe_mean(zone_tokens(hidden, layer, zones["response"])))

    # Exact layer-16 hybrid block.
    add_vector(features, "l16_response_max", safe_max(zone_tokens(hidden, 16, zones["response"])))
    add_vector(features, "l16_response_mean", safe_mean(zone_tokens(hidden, 16, zones["response"])))
    last_response_pos = int(zones["response"][-1].item())
    add_vector(features, "l16_response_last_token", layer_vector(hidden, 16, last_response_pos))
    add_vector(features, "l16_response_last5_mean", safe_mean(zone_tokens(hidden, 16, zones["response_last5"])))


# ============================================================
# 4. EXACT COMPACT GEOMETRY L10-L19
# ============================================================


def add_exact_compact_geometry_l10_l19(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    response_idx = zones["response_wo_eos"]
    last_response_pos = int(response_idx[-1].item())

    response_means: Dict[int, torch.Tensor] = {}
    last_vectors: Dict[int, torch.Tensor] = {}
    l2_mean_values = []
    l2_std_values = []
    l2_max_values = []
    cos_values = []
    l2_between_values = []
    norm_drift_values = []
    mean_cos_values = []
    mean_l2_values = []

    for layer in COMPACT_LAYERS:
        tokens = zone_tokens(hidden, layer, response_idx)
        norms = torch.linalg.norm(tokens.float(), dim=1).detach().cpu().numpy()
        response_means[layer] = safe_mean(tokens)
        last_vectors[layer] = layer_vector(hidden, layer, last_response_pos)

        add_scalar(features, f"response_l2_mean_l{layer}", norms.mean() if norms.size else 0.0)
        add_scalar(features, f"response_l2_std_l{layer}", norms.std() if norms.size else 0.0)
        add_scalar(features, f"response_l2_max_l{layer}", norms.max() if norms.size else 0.0)

        l2_mean_values.append(clean_value(norms.mean() if norms.size else 0.0))
        l2_std_values.append(clean_value(norms.std() if norms.size else 0.0))
        l2_max_values.append(clean_value(norms.max() if norms.size else 0.0))

    for left, right in zip(COMPACT_LAYERS[:-1], COMPACT_LAYERS[1:]):
        left_last = last_vectors[left]
        right_last = last_vectors[right]
        left_mean = response_means[left]
        right_mean = response_means[right]

        cos_lr = safe_cosine(left_last, right_last)
        l2_lr = safe_l2(right_last - left_last)
        norm_drift = abs(safe_l2(right_last) - safe_l2(left_last))
        mean_cos = safe_cosine(left_mean, right_mean)
        mean_l2 = safe_l2(right_mean - left_mean)

        pair = f"l{left}_to_l{right}"
        add_scalar(features, f"last_response_token_cos_{pair}", cos_lr)
        add_scalar(features, f"last_response_token_l2_{pair}", l2_lr)
        add_scalar(features, f"abs_norm_drift_{pair}", norm_drift)
        add_scalar(features, f"response_mean_cos_{pair}", mean_cos)
        add_scalar(features, f"response_mean_l2_{pair}", mean_l2)

        cos_values.append(cos_lr)
        l2_between_values.append(l2_lr)
        norm_drift_values.append(norm_drift)
        mean_cos_values.append(mean_cos)
        mean_l2_values.append(mean_l2)

    add_basic_stats(features, "compact_response_l2_mean_per_layer", l2_mean_values)
    add_basic_stats(features, "compact_response_l2_std_per_layer", l2_std_values)
    add_basic_stats(features, "compact_response_l2_max_per_layer", l2_max_values)
    add_basic_stats(features, "compact_last_response_token_cos_between_layers", cos_values)
    add_basic_stats(features, "compact_last_response_token_l2_between_layers", l2_between_values)
    add_basic_stats(features, "compact_abs_norm_drift_between_layers", norm_drift_values)
    add_basic_stats(features, "compact_response_mean_cos_between_layers", mean_cos_values)
    add_basic_stats(features, "compact_response_mean_l2_between_layers", mean_l2_values)

    add_scalar(features, "response_fraction", zones["response"].numel() / (zones["all"].numel() + EPS))
    add_scalar(features, "log_response_len", np.log1p(zones["response"].numel()))


# ============================================================
# 5. EXACT SGI
# ============================================================


def add_exact_sgi_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    sgi_values = []
    last_response_pos = int(zones["response_wo_eos"][-1].item())
    embedding_ref = safe_mean(zone_tokens(hidden, 0, zones["all"]))

    for layer in COMPACT_LAYERS:
        prompt_center = safe_mean(zone_tokens(hidden, layer, zones["prompt"]))
        response_last = layer_vector(hidden, layer, last_response_pos)

        angle_to_prompt = safe_angle(response_last, prompt_center)
        angle_to_embedding = safe_angle(response_last, embedding_ref)
        sgi = angle_to_prompt / (angle_to_embedding + EPS)

        add_scalar(features, f"sgi_l{layer}", sgi)
        sgi_values.append(sgi)

    add_scalar(features, "sgi_mean", np.mean(sgi_values))
    add_scalar(features, "sgi_std", np.std(sgi_values))
    add_scalar(features, "sgi_min", np.min(sgi_values))
    add_scalar(features, "sgi_max", np.max(sgi_values))
    add_scalar(features, "sgi_late_minus_early", np.mean(sgi_values[-3:]) - np.mean(sgi_values[:3]))


# ============================================================
# 6. EXACT PROMPT-RESPONSE CENTROID SIMILARITY
# ============================================================


def add_exact_centroid_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    cos_values = []
    l2_values = []
    norm_ratio_values = []

    for layer in CENTROID_LAYERS:
        prompt_center = safe_mean(zone_tokens(hidden, layer, zones["prompt"]))
        response_center = safe_mean(zone_tokens(hidden, layer, zones["response_wo_eos"]))

        cos = safe_cosine(prompt_center, response_center)
        l2 = safe_l2(response_center - prompt_center)
        angle = safe_angle(prompt_center, response_center)
        norm_ratio = safe_l2(response_center) / (safe_l2(prompt_center) + EPS)
        drift = l2

        add_scalar(features, f"centroid_cos_l{layer}", cos)
        add_scalar(features, f"centroid_l2_l{layer}", l2)
        add_scalar(features, f"centroid_angle_l{layer}", angle)
        add_scalar(features, f"centroid_norm_ratio_l{layer}", norm_ratio)
        add_scalar(features, f"centroid_drift_l{layer}", drift)

        cos_values.append(cos)
        l2_values.append(l2)
        norm_ratio_values.append(norm_ratio)

    add_basic_stats(features, "centroid_cos", cos_values)
    add_basic_stats(features, "centroid_l2", l2_values)
    add_basic_stats(features, "centroid_norm_ratio", norm_ratio_values)


# ============================================================
# 7. EXACT RESPONSE TEMPORAL DYNAMICS
# ============================================================


def add_temporal_dynamics_for_tokens(
    features: Dict[str, float],
    prefix: str,
    tokens: torch.Tensor,
) -> None:
    arr = to_numpy(tokens)
    if arr.ndim != 2 or arr.shape[0] < 2:
        for name in [
            "velocity_norm_mean", "velocity_norm_std", "velocity_norm_max",
            "acceleration_norm_mean", "acceleration_norm_std", "acceleration_norm_max",
            "curvature_cos_mean", "curvature_cos_std", "curvature_cos_min",
            "direction_reversal_rate", "path_length", "endpoint_distance",
            "path_efficiency", "trajectory_roughness", "late_velocity_spike_ratio",
        ]:
            add_scalar(features, f"{prefix}_{name}", 0.0)
        return

    velocity = np.diff(arr, axis=0)
    velocity_norms = np.linalg.norm(velocity, axis=1)

    if velocity.shape[0] >= 2:
        acceleration = np.diff(velocity, axis=0)
        v1 = velocity[:-1]
        v2 = velocity[1:]
        denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + EPS
        curvature_cos = np.sum(v1 * v2, axis=1) / denom
    else:
        acceleration = np.zeros((1, arr.shape[1]), dtype=np.float32)
        curvature_cos = np.array([1.0], dtype=np.float32)

    acceleration_norms = np.linalg.norm(acceleration, axis=1)
    path_length = clean_value(velocity_norms.sum())
    endpoint_distance = clean_value(np.linalg.norm(arr[-1] - arr[0]))
    split = max(1, len(velocity_norms) // 2)
    early_vel = velocity_norms[:split].mean()
    late_vel = velocity_norms[split:].mean() if split < len(velocity_norms) else velocity_norms[-1]

    add_scalar(features, f"{prefix}_velocity_norm_mean", velocity_norms.mean())
    add_scalar(features, f"{prefix}_velocity_norm_std", velocity_norms.std())
    add_scalar(features, f"{prefix}_velocity_norm_max", velocity_norms.max())
    add_scalar(features, f"{prefix}_acceleration_norm_mean", acceleration_norms.mean())
    add_scalar(features, f"{prefix}_acceleration_norm_std", acceleration_norms.std())
    add_scalar(features, f"{prefix}_acceleration_norm_max", acceleration_norms.max())
    add_scalar(features, f"{prefix}_curvature_cos_mean", curvature_cos.mean())
    add_scalar(features, f"{prefix}_curvature_cos_std", curvature_cos.std())
    add_scalar(features, f"{prefix}_curvature_cos_min", curvature_cos.min())
    add_scalar(features, f"{prefix}_direction_reversal_rate", np.mean(curvature_cos < 0.0))
    add_scalar(features, f"{prefix}_path_length", path_length)
    add_scalar(features, f"{prefix}_endpoint_distance", endpoint_distance)
    add_scalar(features, f"{prefix}_path_efficiency", endpoint_distance / (path_length + EPS))
    add_scalar(features, f"{prefix}_trajectory_roughness", acceleration_norms.sum() / (path_length + EPS))
    add_scalar(features, f"{prefix}_late_velocity_spike_ratio", late_vel / (early_vel + EPS))


def add_exact_temporal_dynamics(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    for layer in TEMPORAL_LAYERS:
        for zone_name in ["response_wo_eos", "response_late30", "response_last10"]:
            tokens = zone_tokens(hidden, layer, zones[zone_name])
            add_temporal_dynamics_for_tokens(features, f"temporal_l{layer}_{zone_name}", tokens)


# ============================================================
# 8. EXACT CROSS-LAYER UPDATE NORMS
# ============================================================


def add_exact_cross_layer_update_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    for zone_name in ["response_wo_eos", "response_last10", "response_last5"]:
        all_mean_norms = []
        all_std_norms = []
        all_max_norms = []
        all_min_norms = []
        all_cv_norms = []
        all_cos_consistency = []
        all_anisotropy = []
        previous_mean_update = None

        for left, right in zip(COMPACT_LAYERS[:-1], COMPACT_LAYERS[1:]):
            left_tokens = zone_tokens(hidden, left, zones[zone_name])
            right_tokens = zone_tokens(hidden, right, zones[zone_name])
            min_len = min(left_tokens.shape[0], right_tokens.shape[0])
            update = right_tokens[:min_len] - left_tokens[:min_len]
            norms = torch.linalg.norm(update.float(), dim=1).detach().cpu().numpy()
            mean_update = safe_mean(update)

            mean_norm = clean_value(norms.mean() if norms.size else 0.0)
            std_norm = clean_value(norms.std() if norms.size else 0.0)
            max_norm = clean_value(norms.max() if norms.size else 0.0)
            min_norm = clean_value(norms.min() if norms.size else 0.0)
            cv_norm = clean_value(std_norm / (mean_norm + EPS))
            anisotropy = clean_value(max_norm / (mean_norm + EPS))
            cos_consistency = 1.0 if previous_mean_update is None else safe_cosine(mean_update, previous_mean_update)

            pair = f"l{left}_to_l{right}"
            prefix = f"cross_update_{zone_name}_{pair}"
            add_scalar(features, f"{prefix}_update_norm_mean", mean_norm)
            add_scalar(features, f"{prefix}_update_norm_std", std_norm)
            add_scalar(features, f"{prefix}_update_norm_max", max_norm)
            add_scalar(features, f"{prefix}_update_norm_min", min_norm)
            add_scalar(features, f"{prefix}_update_norm_cv", cv_norm)
            add_scalar(features, f"{prefix}_update_cosine_consistency", cos_consistency)
            add_scalar(features, f"{prefix}_update_anisotropy", anisotropy)

            all_mean_norms.append(mean_norm)
            all_std_norms.append(std_norm)
            all_max_norms.append(max_norm)
            all_min_norms.append(min_norm)
            all_cv_norms.append(cv_norm)
            all_cos_consistency.append(cos_consistency)
            all_anisotropy.append(anisotropy)
            previous_mean_update = mean_update

        add_basic_stats(features, f"cross_update_{zone_name}_update_norm_mean", all_mean_norms)
        add_basic_stats(features, f"cross_update_{zone_name}_update_norm_std", all_std_norms)
        add_basic_stats(features, f"cross_update_{zone_name}_update_norm_max", all_max_norms)
        add_basic_stats(features, f"cross_update_{zone_name}_update_norm_min", all_min_norms)
        add_basic_stats(features, f"cross_update_{zone_name}_update_norm_cv", all_cv_norms)
        add_basic_stats(features, f"cross_update_{zone_name}_update_cosine_consistency", all_cos_consistency)
        add_basic_stats(features, f"cross_update_{zone_name}_update_anisotropy", all_anisotropy)
        add_scalar(
            features,
            f"cross_update_{zone_name}_update_late_minus_early",
            np.mean(all_mean_norms[-3:]) - np.mean(all_mean_norms[:3]),
        )


# ============================================================
# 9. EXACT ICR-LITE
# ============================================================


def spectral_entropy_from_eigvals(eigvals: np.ndarray) -> float:
    eigvals = np.maximum(np.asarray(eigvals, dtype=np.float64), 0.0)
    total = eigvals.sum()
    if total <= EPS:
        return 0.0
    probs = eigvals / total
    return clean_value(-(probs * np.log(probs + EPS)).sum())


def eigvals_from_token_matrix(tokens: torch.Tensor) -> np.ndarray:
    arr = to_numpy(tokens)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return np.array([0.0], dtype=np.float32)
    arr = arr - arr.mean(axis=0, keepdims=True)
    try:
        singular_values = np.linalg.svd(arr, full_matrices=False, compute_uv=False)
        eigvals = (singular_values ** 2) / max(arr.shape[0] - 1, 1)
    except np.linalg.LinAlgError:
        eigvals = np.array([0.0], dtype=np.float32)
    eigvals = np.nan_to_num(eigvals, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(eigvals.astype(np.float32), 0.0)


def add_exact_icr_lite_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    for layer in ICR_LAYERS:
        for zone_name in ["prompt", "response_wo_eos", "response_last10"]:
            tokens = zone_tokens(hidden, layer, zones[zone_name])
            eigvals = eigvals_from_token_matrix(tokens)
            total = eigvals.sum()
            sq_sum = np.square(eigvals).sum()
            sorted_eigs = np.sort(eigvals)[::-1]
            positive = sorted_eigs[sorted_eigs > EPS]

            participation_ratio = (total ** 2) / (sq_sum + EPS)
            top1_ratio = sorted_eigs[0] / (total + EPS) if sorted_eigs.size else 0.0
            top3_ratio = sorted_eigs[:3].sum() / (total + EPS) if sorted_eigs.size else 0.0
            entropy = spectral_entropy_from_eigvals(sorted_eigs)
            effective_rank = math.exp(entropy) if entropy > 0 else 0.0
            condition_proxy = (positive[0] / (positive[-1] + EPS)) if positive.size >= 2 else 0.0

            prefix = f"icr_l{layer}_{zone_name}"
            add_scalar(features, f"{prefix}_participation_ratio", participation_ratio)
            add_scalar(features, f"{prefix}_top1_ratio", top1_ratio)
            add_scalar(features, f"{prefix}_top3_ratio", top3_ratio)
            add_scalar(features, f"{prefix}_spectral_entropy", entropy)
            add_scalar(features, f"{prefix}_effective_rank", effective_rank)
            add_scalar(features, f"{prefix}_condition_proxy", condition_proxy)


# ============================================================
# SAMPLE EXTRACTION
# ============================================================


def extract_features_one_sample(
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
    prompt_len: int,
) -> Dict[str, float]:
    hidden = hidden.float().cpu()
    valid_mask = valid_mask.bool().cpu()
    zones = build_exact_zones(valid_mask, prompt_len)

    features: Dict[str, float] = {}
    add_length_meta_features(features, zones, prompt_len, MAX_LENGTH)
    add_exact_pooling_features(features, hidden, zones)
    add_exact_compact_geometry_l10_l19(features, hidden, zones)
    add_exact_sgi_features(features, hidden, zones)
    add_exact_centroid_features(features, hidden, zones)
    add_exact_temporal_dynamics(features, hidden, zones)
    add_exact_cross_layer_update_features(features, hidden, zones)
    add_exact_icr_lite_features(features, hidden, zones)

    return {key: clean_value(value) for key, value in features.items()}


# ============================================================
# DATASET EXTRACTION
# ============================================================


def extract_dataset_features(
    df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    has_label: bool,
) -> pd.DataFrame:
    prompts = df["prompt"].astype(str).tolist()
    responses = df["response"].astype(str).tolist()
    prompt_lengths = get_prompt_len_column_or_tokenize(df, tokenizer)
    texts = [p + r for p, r in zip(prompts, responses)]

    rows: List[Dict[str, float]] = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extract prompt-len advanced features"):
        batch_texts = texts[start:start + BATCH_SIZE]
        batch_prompt_lengths = prompt_lengths[start:start + BATCH_SIZE]

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
            rows.append(
                extract_features_one_sample(
                    hidden=hidden_batch[i],
                    valid_mask=mask_batch[i],
                    prompt_len=batch_prompt_lengths[i],
                )
            )

        del outputs, hidden_batch, mask_batch, input_ids, attention_mask, encoding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
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
    print("BUILD ADVANCED PROMPT-LENGTH-AWARE FEATURES")
    print("=" * 80)
    print(f"Device      : {device}")
    print(f"Data file   : {DATA_FILE}")
    print(f"Test file   : {TEST_FILE}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"Batch size  : {BATCH_SIZE}")
    print("Note        : uses prompt_len only; no logits, hooks, or attentions")

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
