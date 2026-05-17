"""aggregation.py — full Track A advanced feature extractor.

This file mirrors the research feature builder used for
A__advanced_all__top1250__pca256. It intentionally uses only Track A inputs:
hidden states and the ordinary attention mask. It does not use prompt_len,
attention maps, logits, token text, labels, or any other extra infrastructure.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

import numpy as np
import torch

try:
    from model import MAX_LENGTH
except Exception:
    MAX_LENGTH = 512


# ============================================================
# CONFIG
# ============================================================

EPS = 1e-8

MAX_POOL_LAYERS = [12, 13, 16]
MEAN_POOL_LAYERS = [14, 15, 16]
COMPACT_LAYERS = list(range(10, 20))
TEMPORAL_LAYERS = [12, 13, 14, 15, 16]
ICR_LAYERS = [0, 6, 12, 18, 23]
CENTROID_LAYERS = [1, 6, 12, 18, 23]

MODEL_TOP3_LAYERS = [-3, -2, -1]
MID_TOP3_LAYERS = [14, 15, 16]
COMPETITOR_TOP3_LAYERS = [12, 13, 16]
TOP3_WEIGHTS = [0.3, 0.3, 0.4]


# ============================================================
# BASIC HELPERS
# ============================================================


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
    """Support both positive and negative layer indices safely."""
    if layer < 0:
        layer = n_layers + layer
    return int(min(max(layer, 0), n_layers - 1))


def safe_l2(x) -> float:
    arr = to_numpy(x)
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


def add_vector_stats(features: Dict[str, float], prefix: str, vector: torch.Tensor | np.ndarray) -> None:
    arr = to_numpy(vector).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    add_scalar(features, f"{prefix}_norm", np.linalg.norm(arr))
    add_scalar(features, f"{prefix}_mean", arr.mean() if arr.size else 0.0)
    add_scalar(features, f"{prefix}_std", arr.std() if arr.size else 0.0)
    add_scalar(features, f"{prefix}_abs_mean", np.abs(arr).mean() if arr.size else 0.0)


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


def last_fraction(idx: torch.Tensor, frac: float) -> torch.Tensor:
    n = int(idx.numel())
    if n == 0:
        return idx
    keep = max(1, int(round(n * frac)))
    return idx[-keep:]


def ensure_non_empty(idx: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    if idx.numel() > 0:
        return idx
    return fallback[-1:] if fallback.numel() > 0 else fallback


def build_zones(valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Heuristic response zones without prompt_len."""
    idx = valid_positions(valid_mask)
    n = int(idx.numel())

    if n == 0:
        empty = idx
        return {
            "all": empty,
            "first70": empty,
            "last30": empty,
            "last20": empty,
            "last10": empty,
            "last5": empty,
            "last_token": empty,
            "last30_wo_last": empty,
            "last20_wo_last": empty,
            "last_without_eos_30": empty,
            "last_without_eos_20": empty,
        }

    idx_wo_last = idx[:-1] if n >= 2 else idx[-1:]
    first70_end = max(1, int(round(n * 0.70)))

    zones = {
        "all": idx,
        "first70": idx[:first70_end],
        "last30": last_fraction(idx, 0.30),
        "last20": last_fraction(idx, 0.20),
        "last10": last_fraction(idx, 0.10),
        "last5": idx[-min(5, n):],
        "last_token": idx[-1:],
        "last30_wo_last": last_fraction(idx_wo_last, 0.30),
        "last20_wo_last": last_fraction(idx_wo_last, 0.20),
    }
    zones["last_without_eos_30"] = zones["last30_wo_last"]
    zones["last_without_eos_20"] = zones["last20_wo_last"]

    for key, value in list(zones.items()):
        zones[key] = ensure_non_empty(value, idx)

    return zones


def zone_tokens(hidden: torch.Tensor, layer: int, idx: torch.Tensor) -> torch.Tensor:
    n_layers = hidden.shape[0]
    layer_idx = safe_layer_index(layer, n_layers)
    if idx.numel() == 0:
        return torch.zeros((1, hidden.shape[-1]), dtype=torch.float32)
    return hidden[layer_idx, idx].float().cpu()


def layer_vector(hidden: torch.Tensor, layer: int, pos: int) -> torch.Tensor:
    layer_idx = safe_layer_index(layer, hidden.shape[0])
    return hidden[layer_idx, pos].float().cpu()


# ============================================================
# 1-3. MAX/MEAN + EOS + LAYER 16
# ============================================================


def add_length_and_fallback_features(
    features: Dict[str, float],
    zones: Dict[str, torch.Tensor],
    max_length: int,
) -> None:
    n_valid = int(zones["all"].numel())
    n_valid_wo_last = max(0, n_valid - 1)
    last30_len = int(zones["last30"].numel())
    last30_wo_last_len = int(zones["last30_wo_last"].numel())

    add_scalar(features, "n_valid", n_valid)
    add_scalar(features, "n_valid_wo_last", n_valid_wo_last)
    add_scalar(features, "heur_response_len_last30", last30_len)
    add_scalar(features, "heur_response_len_last30_wo_last", last30_wo_last_len)
    add_scalar(features, "heur_response_ratio_last30", last30_len / (n_valid + EPS))
    add_scalar(features, "heur_response_ratio_last30_wo_last", last30_wo_last_len / (n_valid_wo_last + EPS))

    add_scalar(features, "maybe_truncated", int(n_valid >= max_length - 2))
    add_scalar(features, "response_proxy_too_short", int(last30_len <= 2))
    add_scalar(features, "fallback_used_last_token", int(last30_len <= 2))


def add_max_mean_pooling_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    # 1.2 competitor-style max vectors.
    max_zone_names = ["last30", "last20", "last_without_eos_30"]
    for layer in MAX_POOL_LAYERS:
        for zone_name in max_zone_names:
            vec = safe_max(zone_tokens(hidden, layer, zones[zone_name]))
            add_vector(features, f"max_l{layer}_{zone_name}", vec)

    # 1.3 competitor-specific mean vectors.
    mean_specs = [
        (14, "last_without_eos_30"),
        (15, "last_without_eos_30"),
        (16, "last_without_eos_30"),
        (14, "last20"),
        (15, "last20"),
        (16, "last20"),
    ]
    for layer, zone_name in mean_specs:
        vec = safe_mean(zone_tokens(hidden, layer, zones[zone_name]))
        add_vector(features, f"mean_l{layer}_{zone_name}", vec)

    # 3. Explicit layer-16 block, with names requested in the spec.
    l16_max_last30 = safe_max(zone_tokens(hidden, 16, zones["last30"]))
    l16_mean_last30 = safe_mean(zone_tokens(hidden, 16, zones["last30"]))
    l16_max_last20 = safe_max(zone_tokens(hidden, 16, zones["last20"]))
    l16_mean_last20 = safe_mean(zone_tokens(hidden, 16, zones["last20"]))
    l16_last_token = safe_mean(zone_tokens(hidden, 16, zones["last_token"]))

    add_vector(features, "l16_max_last30", l16_max_last30)
    add_vector(features, "l16_mean_last30", l16_mean_last30)
    add_vector(features, "l16_max_last20", l16_max_last20)
    add_vector(features, "l16_mean_last20", l16_mean_last20)
    add_vector(features, "l16_last_token", l16_last_token)


# ============================================================
# 4-5. WEIGHTED TOP-3 LAST TOKEN + COMPACT GEOMETRY
# ============================================================


def weighted_layers_last_token(
    hidden: torch.Tensor,
    last_pos: int,
    layers: Sequence[int],
    weights: Sequence[float],
) -> torch.Tensor:
    vectors = [layer_vector(hidden, layer, last_pos) * float(weight) for layer, weight in zip(layers, weights)]
    return torch.stack(vectors, dim=0).sum(dim=0)


def add_weighted_top3_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    last_pos = int(zones["last_token"][-1].item()) if zones["last_token"].numel() else 0

    specs = {
        "weighted_top3_model_last_token": MODEL_TOP3_LAYERS,
        "weighted_top3_mid_last_token": MID_TOP3_LAYERS,
        "weighted_top3_competitor_last_token": COMPETITOR_TOP3_LAYERS,
    }

    for prefix, layers in specs.items():
        vec = weighted_layers_last_token(hidden, last_pos, layers, TOP3_WEIGHTS)
        add_vector(features, prefix, vec)
        add_vector_stats(features, prefix, vec)


def add_compact_last_token_geometry_for_layers(
    features: Dict[str, float],
    hidden: torch.Tensor,
    last_pos: int,
    layers: Sequence[int],
    prefix: str,
) -> None:
    vectors = [layer_vector(hidden, layer, last_pos) for layer in layers]
    norms = [safe_l2(vec) for vec in vectors]

    final_vec = vectors[-1]
    add_scalar(features, f"{prefix}_last_token_final_norm", norms[-1])
    add_scalar(features, f"{prefix}_last_token_top3_mean_norm", np.mean(norms))
    add_scalar(features, f"{prefix}_last_token_top3_std_norm", np.std(norms))

    add_scalar(features, f"{prefix}_last_token_cos_last2", safe_cosine(vectors[-1], vectors[-2]))
    add_scalar(features, f"{prefix}_last_token_cos_last13", safe_cosine(vectors[-1], vectors[0]))
    add_scalar(features, f"{prefix}_last_token_l2_last12", safe_l2(vectors[-1] - vectors[-2]))
    add_scalar(features, f"{prefix}_last_token_l2_last13", safe_l2(vectors[-1] - vectors[0]))
    add_scalar(features, f"{prefix}_last_token_norm_ratio", norms[-1] / (norms[0] + EPS))
    add_scalar(features, f"{prefix}_last_token_abs_norm_drift", abs(norms[-1] - norms[0]))


def add_compact_last_token_geometry(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    n_valid = int(zones["all"].numel())
    last_pos = int(zones["last_token"][-1].item()) if zones["last_token"].numel() else 0

    # Global names requested in the spec: model top-3 version.
    model_vectors = [layer_vector(hidden, layer, last_pos) for layer in MODEL_TOP3_LAYERS]
    model_norms = [safe_l2(vec) for vec in model_vectors]
    add_scalar(features, "last_token_seq_len", n_valid)
    add_scalar(features, "last_token_final_norm", model_norms[-1])
    add_scalar(features, "last_token_top3_mean_norm", np.mean(model_norms))
    add_scalar(features, "last_token_top3_std_norm", np.std(model_norms))
    add_scalar(features, "last_token_cos_last2", safe_cosine(model_vectors[-1], model_vectors[-2]))
    add_scalar(features, "last_token_cos_last13", safe_cosine(model_vectors[-1], model_vectors[0]))
    add_scalar(features, "last_token_l2_last12", safe_l2(model_vectors[-1] - model_vectors[-2]))
    add_scalar(features, "last_token_l2_last13", safe_l2(model_vectors[-1] - model_vectors[0]))
    add_scalar(features, "last_token_norm_ratio", model_norms[-1] / (model_norms[0] + EPS))

    # Explicit versions for both requested sets.
    add_compact_last_token_geometry_for_layers(
        features, hidden, last_pos, MODEL_TOP3_LAYERS, "model_top3"
    )
    add_compact_last_token_geometry_for_layers(
        features, hidden, last_pos, MID_TOP3_LAYERS, "mid_top3"
    )


# ============================================================
# 6. COMPACT GEOMETRY L10-L19 WITHOUT PROMPT_LEN
# ============================================================


def add_compact_geometry_l10_l19(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    zone_names = ["last30_wo_last", "last20_wo_last", "last5", "last_token"]

    for zone_name in zone_names:
        cos_values = []
        norm_drift_values = []

        zone_means: Dict[int, torch.Tensor] = {}
        last_token_vectors: Dict[int, torch.Tensor] = {}
        l2_means = []
        l2_stds = []
        l2_maxs = []
        l2_mins = []

        for layer in COMPACT_LAYERS:
            tokens = zone_tokens(hidden, layer, zones[zone_name])
            norms = torch.linalg.norm(tokens.float(), dim=1).detach().cpu().numpy()
            zone_means[layer] = safe_mean(tokens)
            last_token_vectors[layer] = zone_tokens(hidden, layer, zones["last_token"])[-1]

            prefix = f"l{layer}_{zone_name}_heur_resp_l2"
            add_scalar(features, f"{prefix}_mean", norms.mean() if norms.size else 0.0)
            add_scalar(features, f"{prefix}_std", norms.std() if norms.size else 0.0)
            add_scalar(features, f"{prefix}_max", norms.max() if norms.size else 0.0)
            add_scalar(features, f"{prefix}_min", norms.min() if norms.size else 0.0)

            l2_means.append(clean_value(norms.mean() if norms.size else 0.0))
            l2_stds.append(clean_value(norms.std() if norms.size else 0.0))
            l2_maxs.append(clean_value(norms.max() if norms.size else 0.0))
            l2_mins.append(clean_value(norms.min() if norms.size else 0.0))

        for left, right in zip(COMPACT_LAYERS[:-1], COMPACT_LAYERS[1:]):
            left_last = last_token_vectors[left]
            right_last = last_token_vectors[right]
            left_mean = zone_means[left]
            right_mean = zone_means[right]

            cos_lr = safe_cosine(left_last, right_last)
            l2_lr = safe_l2(right_last - left_last)
            norm_drift = abs(safe_l2(right_last) - safe_l2(left_last))
            mean_cos = safe_cosine(left_mean, right_mean)
            mean_l2 = safe_l2(right_mean - left_mean)

            pair_prefix = f"{zone_name}_l{left}_to_l{right}"
            add_scalar(features, f"last_token_cos_{pair_prefix}", cos_lr)
            add_scalar(features, f"last_token_l2_{pair_prefix}", l2_lr)
            add_scalar(features, f"last_token_abs_norm_drift_{pair_prefix}", norm_drift)
            add_scalar(features, f"heur_resp_mean_cos_{pair_prefix}", mean_cos)
            add_scalar(features, f"heur_resp_mean_l2_{pair_prefix}", mean_l2)

            cos_values.append(cos_lr)
            norm_drift_values.append(norm_drift)

        add_scalar(features, f"compact_l10_l19_{zone_name}_cos_mean", np.mean(cos_values))
        add_scalar(features, f"compact_l10_l19_{zone_name}_cos_min", np.min(cos_values))
        add_scalar(features, f"compact_l10_l19_{zone_name}_cos_std", np.std(cos_values))
        add_scalar(features, f"compact_l10_l19_{zone_name}_norm_drift_mean", np.mean(norm_drift_values))
        add_scalar(features, f"compact_l10_l19_{zone_name}_norm_drift_max", np.max(norm_drift_values))
        add_scalar(features, f"compact_l10_l19_{zone_name}_norm_drift_std", np.std(norm_drift_values))

        # Extra trajectory summaries for per-layer token norms.
        add_basic_stats(features, f"compact_l10_l19_{zone_name}_l2_mean_by_layer", l2_means)
        add_basic_stats(features, f"compact_l10_l19_{zone_name}_l2_std_by_layer", l2_stds)
        add_basic_stats(features, f"compact_l10_l19_{zone_name}_l2_max_by_layer", l2_maxs)
        add_basic_stats(features, f"compact_l10_l19_{zone_name}_l2_min_by_layer", l2_mins)


# ============================================================
# 7. SGI-LIKE WITHOUT PROMPT_LEN
# ============================================================


def add_sgi_proxy_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    sgi_values = []
    response_idx = zones["last30_wo_last"]
    last_response_pos = int(response_idx[-1].item()) if response_idx.numel() else int(zones["last_token"][-1].item())

    # Layer-0 reference: mean over all valid tokens. This is stable and avoids
    # depending on a specific special-token position.
    embedding_reference = safe_mean(zone_tokens(hidden, 0, zones["all"]))

    for layer in COMPACT_LAYERS:
        prompt_proxy_center = safe_mean(zone_tokens(hidden, layer, zones["first70"]))
        response_last = layer_vector(hidden, layer, last_response_pos)

        angle_to_prompt = safe_angle(response_last, prompt_proxy_center)
        angle_to_embedding = safe_angle(response_last, embedding_reference)
        sgi_proxy = angle_to_prompt / (angle_to_embedding + EPS)

        add_scalar(features, f"sgi_proxy_l{layer}", sgi_proxy)
        sgi_values.append(sgi_proxy)

    add_scalar(features, "sgi_proxy_mean", np.mean(sgi_values))
    add_scalar(features, "sgi_proxy_std", np.std(sgi_values))
    add_scalar(features, "sgi_proxy_min", np.min(sgi_values))
    add_scalar(features, "sgi_proxy_max", np.max(sgi_values))
    early = np.mean(sgi_values[:3])
    late = np.mean(sgi_values[-3:])
    add_scalar(features, "sgi_proxy_late_minus_early", late - early)


# ============================================================
# 8. CROSS-LAYER UPDATE NORM FEATURES
# ============================================================


def add_cross_layer_update_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    zone_names = ["last30_wo_last", "last20_wo_last", "last5", "last_token"]

    for zone_name in zone_names:
        all_update_norm_means = []
        all_update_norm_stds = []
        all_update_norm_maxs = []
        all_update_cos_with_prev = []
        previous_mean_update = None

        for left, right in zip(COMPACT_LAYERS[:-1], COMPACT_LAYERS[1:]):
            left_tokens = zone_tokens(hidden, left, zones[zone_name])
            right_tokens = zone_tokens(hidden, right, zones[zone_name])

            min_len = min(left_tokens.shape[0], right_tokens.shape[0])
            left_tokens = left_tokens[:min_len]
            right_tokens = right_tokens[:min_len]
            update = right_tokens - left_tokens

            norms = torch.linalg.norm(update.float(), dim=1).detach().cpu().numpy()
            mean_update = safe_mean(update)

            mean_norm = clean_value(norms.mean() if norms.size else 0.0)
            std_norm = clean_value(norms.std() if norms.size else 0.0)
            max_norm = clean_value(norms.max() if norms.size else 0.0)
            min_norm = clean_value(norms.min() if norms.size else 0.0)
            cv_norm = clean_value(std_norm / (mean_norm + EPS))
            anisotropy = clean_value(max_norm / (mean_norm + EPS))
            cos_prev = 1.0 if previous_mean_update is None else safe_cosine(mean_update, previous_mean_update)

            pair_prefix = f"cross_update_{zone_name}_l{left}_to_l{right}"
            add_scalar(features, f"{pair_prefix}_update_norm_mean", mean_norm)
            add_scalar(features, f"{pair_prefix}_update_norm_std", std_norm)
            add_scalar(features, f"{pair_prefix}_update_norm_max", max_norm)
            add_scalar(features, f"{pair_prefix}_update_norm_min", min_norm)
            add_scalar(features, f"{pair_prefix}_update_norm_cv", cv_norm)
            add_scalar(features, f"{pair_prefix}_update_cosine_with_prev_update", cos_prev)
            add_scalar(features, f"{pair_prefix}_update_anisotropy", anisotropy)

            all_update_norm_means.append(mean_norm)
            all_update_norm_stds.append(std_norm)
            all_update_norm_maxs.append(max_norm)
            all_update_cos_with_prev.append(cos_prev)
            previous_mean_update = mean_update

        add_scalar(features, f"cross_layer_update_{zone_name}_norm_mean", np.mean(all_update_norm_means))
        add_scalar(features, f"cross_layer_update_{zone_name}_norm_std", np.mean(all_update_norm_stds))
        add_scalar(features, f"cross_layer_update_{zone_name}_norm_max", np.max(all_update_norm_maxs))
        add_scalar(
            features,
            f"cross_layer_update_{zone_name}_cos_consistency_mean",
            np.mean(all_update_cos_with_prev),
        )
        add_scalar(
            features,
            f"cross_layer_update_{zone_name}_cos_consistency_min",
            np.min(all_update_cos_with_prev),
        )


# ============================================================
# 9. RESPONSE TEMPORAL DYNAMICS WITHOUT PROMPT_LEN
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
    acceleration = np.diff(velocity, axis=0) if velocity.shape[0] >= 2 else np.zeros((1, arr.shape[1]), dtype=np.float32)
    acceleration_norms = np.linalg.norm(acceleration, axis=1)

    if velocity.shape[0] >= 2:
        v1 = velocity[:-1]
        v2 = velocity[1:]
        denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + EPS
        curvature_cos = np.sum(v1 * v2, axis=1) / denom
    else:
        curvature_cos = np.array([1.0], dtype=np.float32)

    path_length = clean_value(velocity_norms.sum())
    endpoint_distance = clean_value(np.linalg.norm(arr[-1] - arr[0]))
    early_vel = velocity_norms[: max(1, len(velocity_norms) // 2)].mean()
    late_vel = velocity_norms[max(0, len(velocity_norms) // 2):].mean()

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


def add_response_temporal_dynamics(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    for layer in TEMPORAL_LAYERS:
        for zone_name in ["last30_wo_last", "last20_wo_last"]:
            tokens = zone_tokens(hidden, layer, zones[zone_name])
            add_temporal_dynamics_for_tokens(
                features,
                f"temporal_l{layer}_{zone_name}",
                tokens,
            )


# ============================================================
# 10. ICR-LITE VECTOR
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

    # Use singular values of token matrix instead of explicit D x D covariance.
    # This is faster and numerically safer when hidden_dim is large.
    try:
        _, singular_values, _ = np.linalg.svd(arr, full_matrices=False)
        eigvals = (singular_values ** 2) / max(arr.shape[0] - 1, 1)
    except np.linalg.LinAlgError:
        eigvals = np.array([0.0], dtype=np.float32)

    eigvals = np.nan_to_num(eigvals, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(eigvals.astype(np.float32), 0.0)


def add_icr_lite_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    for layer in ICR_LAYERS:
        for zone_name in ["all", "last30_wo_last", "last20_wo_last"]:
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
# 11. PROMPT-RESPONSE CENTROID SIMILARITY PROXY
# ============================================================


def add_centroid_proxy_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    cos_values = []
    l2_values = []
    norm_ratio_values = []

    for layer in CENTROID_LAYERS:
        prompt_center = safe_mean(zone_tokens(hidden, layer, zones["first70"]))
        response_center = safe_mean(zone_tokens(hidden, layer, zones["last30_wo_last"]))

        cos = safe_cosine(prompt_center, response_center)
        l2 = safe_l2(response_center - prompt_center)
        angle = safe_angle(prompt_center, response_center)
        norm_ratio = safe_l2(response_center) / (safe_l2(prompt_center) + EPS)
        drift = safe_l2(response_center - prompt_center)

        add_scalar(features, f"centroid_proxy_cos_l{layer}", cos)
        add_scalar(features, f"centroid_proxy_l2_l{layer}", l2)
        add_scalar(features, f"centroid_proxy_angle_l{layer}", angle)
        add_scalar(features, f"centroid_proxy_norm_ratio_l{layer}", norm_ratio)
        add_scalar(features, f"centroid_proxy_response_minus_prompt_norm_l{layer}", drift)

        cos_values.append(cos)
        l2_values.append(l2)
        norm_ratio_values.append(norm_ratio)

    add_scalar(features, "centroid_proxy_cos_mean", np.mean(cos_values))
    add_scalar(features, "centroid_proxy_cos_min", np.min(cos_values))
    add_scalar(features, "centroid_proxy_cos_std", np.std(cos_values))
    add_scalar(features, "centroid_proxy_l2_mean", np.mean(l2_values))
    add_scalar(features, "centroid_proxy_l2_max", np.max(l2_values))
    add_scalar(features, "centroid_proxy_l2_std", np.std(l2_values))
    add_scalar(features, "centroid_proxy_norm_ratio_mean", np.mean(norm_ratio_values))
    add_scalar(features, "centroid_proxy_norm_ratio_std", np.std(norm_ratio_values))


# ============================================================
# SAMPLE FEATURE EXTRACTION
# ============================================================


def extract_features_one_sample(hidden: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, float]:
    hidden = hidden.float().cpu()
    valid_mask = valid_mask.bool().cpu()

    features: Dict[str, float] = {}
    zones = build_zones(valid_mask)

    add_length_and_fallback_features(features, zones, MAX_LENGTH)
    add_max_mean_pooling_features(features, hidden, zones)
    add_weighted_top3_features(features, hidden, zones)
    add_compact_last_token_geometry(features, hidden, zones)
    add_compact_geometry_l10_l19(features, hidden, zones)
    add_sgi_proxy_features(features, hidden, zones)
    add_cross_layer_update_features(features, hidden, zones)
    add_response_temporal_dynamics(features, hidden, zones)
    add_icr_lite_features(features, hidden, zones)
    add_centroid_proxy_features(features, hidden, zones)

    return {key: clean_value(value) for key, value in features.items()}




def _ordered_feature_tensor(features: Dict[str, float]) -> torch.Tensor:
    """Convert insertion-ordered feature dictionary to a feature tensor."""
    values = [clean_value(value) for value in features.values()]
    return torch.tensor(values, dtype=torch.float32)


def aggregate(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Build the full Track A advanced feature vector.

    This function mirrors build_advanced_features.py exactly: no prompt_len,
    no attentions, no logits, only hidden states and the attention mask.
    """
    return _ordered_feature_tensor(
        extract_features_one_sample(hidden_states, attention_mask)
    )


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compatibility hook for the official pipeline.

    All geometric Track A features are already included in aggregate(), so this
    returns an empty tensor to avoid duplicating features when use_geometric=True.
    """
    del hidden_states, attention_mask
    return torch.zeros(0, dtype=torch.float32)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    features = aggregate(hidden_states, attention_mask)
    if use_geometric:
        return torch.cat(
            [features, extract_geometric_features(hidden_states, attention_mask)],
            dim=0,
        )
    return features
