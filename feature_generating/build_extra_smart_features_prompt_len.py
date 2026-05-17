"""Build prompt-len-aware smart features for hallucination detection.

Creates:
    ./artifacts/extra_smart_features_prompt_len/features_dataset_extra_smart_prompt_len.parquet
    ./artifacts/extra_smart_features_prompt_len/features_test_extra_smart_prompt_len.parquet

This script intentionally contains only features that require exact prompt/response
separation. The only extra input needed versus a hidden-state aggregation pipeline
is prompt_len. It uses only hidden states, valid masks, and prompt_len.
"""

from __future__ import annotations

import time
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

OUTPUT_DIR = Path("./artifacts/extra_smart_features_prompt_len")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUTPUT = OUTPUT_DIR / "features_dataset_extra_smart_prompt_len.parquet"
TEST_OUTPUT = OUTPUT_DIR / "features_test_extra_smart_prompt_len.parquet"

BATCH_SIZE = 2
EXPORT_TEST = True

LAYERS = [10, 11, 12, 13, 14, 15, 16]
RICH_LAYERS = [11, 12, 13, 14, 15, 16]
MIDDLE4_LAYERS = [11, 12, 13, 14]

LAYER_PAIRS = list(zip(LAYERS[:-1], LAYERS[1:]))
RICH_LAYER_PAIRS = list(zip(RICH_LAYERS[:-1], RICH_LAYERS[1:]))
LONG_LAYER_PAIRS = [
    (10, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
    (10, 16),
    (11, 16),
]

EPS = 1e-8


# ============================================================
# DEVICE / MODEL
# ============================================================


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# BASIC NUMERIC HELPERS
# ============================================================


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
    return tokens.float().mean(dim=0)


def safe_var_mean(tokens: torch.Tensor) -> float:
    if tokens.shape[0] <= 1:
        return 0.0
    return clean_value(tokens.float().var(dim=0, unbiased=False).mean().item())


def entropy(values) -> float:
    values = np.abs(np.asarray(values, dtype=np.float32))
    total = float(values.sum())
    if total <= EPS:
        return 0.0
    p = values / total
    return clean_value(-(p * np.log(p + EPS)).sum())


def add_stats(features: Dict[str, float], prefix: str, values) -> None:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) == 0:
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
    features[f"{prefix}_iqr"] = clean_value(np.percentile(values, 75) - np.percentile(values, 25))
    features[f"{prefix}_entropy"] = entropy(values)


def add_trajectory(features: Dict[str, float], prefix: str, values) -> None:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) == 0:
        values = np.array([0.0], dtype=np.float32)

    add_stats(features, prefix, values)

    diff1 = np.diff(values) if len(values) >= 2 else np.array([0.0], dtype=np.float32)
    diff2 = np.diff(values, n=2) if len(values) >= 3 else np.array([0.0], dtype=np.float32)

    x = np.arange(len(values), dtype=np.float32)
    slope = float(np.polyfit(x, values, 1)[0]) if len(values) >= 2 else 0.0

    early = float(values[:2].mean()) if len(values) >= 2 else float(values.mean())
    late = float(values[-2:].mean()) if len(values) >= 2 else float(values.mean())
    roughness = float(np.abs(diff1).sum())

    features[f"{prefix}_slope"] = clean_value(slope)
    features[f"{prefix}_roughness"] = clean_value(roughness)
    features[f"{prefix}_smoothness"] = clean_value(1.0 / (1.0 + roughness))
    features[f"{prefix}_acceleration_mean"] = clean_value(np.abs(diff2).mean())
    features[f"{prefix}_acceleration_max"] = clean_value(np.abs(diff2).max())
    features[f"{prefix}_late_minus_early"] = clean_value(late - early)
    features[f"{prefix}_late_div_early"] = clean_value(late / (abs(early) + EPS))
    features[f"{prefix}_num_increases"] = clean_value((diff1 > 0).sum())
    features[f"{prefix}_num_decreases"] = clean_value((diff1 < 0).sum())

    signs = np.sign(diff1)
    signs = signs[signs != 0]
    features[f"{prefix}_sign_changes"] = clean_value((signs[1:] != signs[:-1]).sum()) if len(signs) >= 2 else 0.0


def add_spectral(features: Dict[str, float], prefix: str, values) -> None:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) == 0:
        values = np.array([0.0], dtype=np.float32)

    centered = values - values.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    total = float(power.sum())
    split = max(1, len(power) // 2)
    low = float(power[:split].sum())
    high = float(power[split:].sum())

    features[f"{prefix}_fft_energy"] = clean_value(total)
    features[f"{prefix}_spectral_low_energy"] = clean_value(low)
    features[f"{prefix}_spectral_high_energy"] = clean_value(high)
    features[f"{prefix}_spectral_high_low_ratio"] = clean_value(high / (low + EPS))
    features[f"{prefix}_spectral_entropy"] = entropy(power)
    features[f"{prefix}_dominant_frequency"] = clean_value(np.argmax(power) / max(len(power) - 1, 1))


def add_vector(features: Dict[str, float], prefix: str, vec: torch.Tensor) -> None:
    arr = to_numpy(vec)
    for i, value in enumerate(arr):
        features[f"{prefix}_d{i}"] = clean_value(value)


# ============================================================
# TOKEN / COVARIANCE HELPERS
# ============================================================


def pairwise_cosine_stats(tokens: torch.Tensor) -> Dict[str, float]:
    arr = to_numpy(tokens)
    if arr.shape[0] <= 1:
        return {"mean": 1.0, "std": 0.0, "min": 1.0, "p10": 1.0, "p90": 1.0}

    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + EPS)
    sim = arr @ arr.T
    tri = sim[np.triu_indices_from(sim, k=1)]
    return {
        "mean": clean_value(tri.mean()),
        "std": clean_value(tri.std()),
        "min": clean_value(tri.min()),
        "p10": clean_value(np.percentile(tri, 10)),
        "p90": clean_value(np.percentile(tri, 90)),
    }


def covariance_stats(tokens: torch.Tensor) -> Dict[str, float]:
    arr = to_numpy(tokens)
    if arr.shape[0] <= 2:
        return {
            "trace": 0.0,
            "top1_ratio": 0.0,
            "top3_ratio": 0.0,
            "top5_ratio": 0.0,
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "spectral_entropy": 0.0,
        }

    centered = arr - arr.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eig = singular_values ** 2
    total = float(eig.sum()) + EPS
    p = eig / total
    ent = float(-(p * np.log(p + EPS)).sum())
    return {
        "trace": clean_value(total),
        "top1_ratio": clean_value(p[:1].sum()),
        "top3_ratio": clean_value(p[:3].sum()),
        "top5_ratio": clean_value(p[:5].sum()),
        "effective_rank": clean_value(np.exp(ent)),
        "participation_ratio": clean_value((eig.sum() ** 2) / ((eig ** 2).sum() + EPS)),
        "spectral_entropy": clean_value(ent),
    }


def add_zone_token_stats(features: Dict[str, float], prefix: str, tokens: torch.Tensor) -> None:
    tokens = tokens.float()
    if tokens.shape[0] == 0:
        tokens = torch.zeros((1, tokens.shape[-1]), dtype=tokens.dtype)

    norms = torch.linalg.norm(tokens, dim=1).detach().cpu().numpy()
    add_stats(features, f"{prefix}_token_norm", norms)

    features[f"{prefix}_activation_mean"] = clean_value(tokens.mean().item())
    features[f"{prefix}_activation_std"] = clean_value(tokens.std(unbiased=False).item()) if tokens.numel() > 1 else 0.0
    features[f"{prefix}_activation_abs_mean"] = clean_value(tokens.abs().mean().item())
    features[f"{prefix}_activation_max"] = clean_value(tokens.max().item())
    features[f"{prefix}_activation_min"] = clean_value(tokens.min().item())
    features[f"{prefix}_feature_variance_mean"] = safe_var_mean(tokens)

    pcs = pairwise_cosine_stats(tokens)
    for key, value in pcs.items():
        features[f"{prefix}_pairwise_cosine_{key}"] = value
    features[f"{prefix}_disagreement"] = clean_value(1.0 - pcs["mean"])

    cov = covariance_stats(tokens)
    for key, value in cov.items():
        features[f"{prefix}_cov_{key}"] = value


def add_response_spectral_tokens(features: Dict[str, float], prefix: str, tokens: torch.Tensor) -> None:
    """Spectral dynamics over exact response-token positions using hidden states only."""
    tokens = tokens.float()
    if tokens.shape[0] == 0:
        tokens = torch.zeros((1, tokens.shape[-1]), dtype=tokens.dtype)

    token_norms = torch.linalg.norm(tokens, dim=1).detach().cpu().numpy()
    add_spectral(features, f"{prefix}_token_norm", token_norms)

    centered = to_numpy(tokens - tokens.mean(dim=0, keepdim=True))
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    freq_power = power.sum(axis=1)
    split = max(1, len(freq_power) // 2)
    low = float(freq_power[:split].sum())
    high = float(freq_power[split:].sum())
    total = float(freq_power.sum())

    features[f"{prefix}_hidden_fft_energy"] = clean_value(total)
    features[f"{prefix}_hidden_fft_low_energy"] = clean_value(low)
    features[f"{prefix}_hidden_fft_high_energy"] = clean_value(high)
    features[f"{prefix}_hidden_fft_high_low_ratio"] = clean_value(high / (low + EPS))
    features[f"{prefix}_hidden_fft_entropy"] = entropy(freq_power)
    features[f"{prefix}_hidden_fft_dominant_freq"] = clean_value(np.argmax(freq_power) / max(len(freq_power) - 1, 1))


# ============================================================
# PROMPT LENGTH / EXACT ZONES
# ============================================================


def get_prompt_lengths(tokenizer, prompts: List[str], max_length: int) -> List[int]:
    lengths = []
    for prompt in prompts:
        enc = tokenizer(
            str(prompt),
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        lengths.append(len(enc["input_ids"]))
    return lengths


def split_indices(idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    n = int(idx.numel())
    if n == 0:
        return {
            "all": idx,
            "early": idx,
            "middle": idx,
            "late": idx,
            "first_5": idx,
            "first_10": idx,
            "last_5": idx,
            "last_10": idx,
        }

    third = max(1, n // 3)
    mid_start = third
    late_start = min(n, 2 * third)
    return {
        "all": idx,
        "early": idx[:third],
        "middle": idx[mid_start:late_start] if late_start > mid_start else idx,
        "late": idx[late_start:] if n > late_start else idx[-third:],
        "first_5": idx[:min(5, n)],
        "first_10": idx[:min(10, n)],
        "last_5": idx[-min(5, n):],
        "last_10": idx[-min(10, n):],
    }


def build_prompt_len_zones(hidden: torch.Tensor, valid_mask: torch.Tensor, prompt_len: int):
    valid_mask = valid_mask.bool().cpu()
    seq_len = hidden.shape[1]
    prompt_len = min(max(int(prompt_len), 0), seq_len)
    pos = torch.arange(seq_len)

    prompt_mask = valid_mask & (pos < prompt_len)
    response_mask = valid_mask & (pos >= prompt_len)

    # Fallback keeps the feature generator numerically safe for truncated or empty responses.
    if prompt_mask.sum().item() == 0:
        prompt_mask = valid_mask.clone()
    if response_mask.sum().item() == 0:
        response_mask = valid_mask.clone()

    prompt_idx = torch.where(prompt_mask)[0]
    response_idx = torch.where(response_mask)[0]
    all_idx = torch.where(valid_mask)[0]

    zones = {
        "all": all_idx,
        "prompt": prompt_idx,
        "response": response_idx,
    }

    for name, idx in split_indices(prompt_idx).items():
        zones[f"prompt_{name}"] = idx
    for name, idx in split_indices(response_idx).items():
        zones[f"response_{name}"] = idx

    return zones, prompt_mask, response_mask


def zone_tokens(hidden: torch.Tensor, layer: int, idx: torch.Tensor) -> torch.Tensor:
    if idx.numel() == 0:
        return torch.zeros((1, hidden.shape[-1]), dtype=hidden.dtype)
    return hidden[layer, idx].float()


def build_layer_means(hidden: torch.Tensor, zones: Dict[str, torch.Tensor], zone_names: Iterable[str]):
    means = {}
    for zone_name in zone_names:
        means[zone_name] = {}
        for layer in LAYERS:
            means[zone_name][layer] = safe_mean(zone_tokens(hidden, layer, zones[zone_name]))
    return means


# ============================================================
# FEATURE EXTRACTION FOR ONE SAMPLE
# ============================================================


def extract_features_one_sample(
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
    prompt_len: int,
) -> Dict[str, float]:
    hidden = hidden.float().cpu()
    valid_mask = valid_mask.bool().cpu()

    zones, _, _ = build_prompt_len_zones(
        hidden=hidden,
        valid_mask=valid_mask,
        prompt_len=prompt_len,
    )

    zone_names = [
        "prompt",
        "response",
        "response_early",
        "response_middle",
        "response_late",
        "response_last_5",
        "response_last_10",
    ]
    means = build_layer_means(hidden, zones, zone_names)

    features: Dict[str, float] = {}

    # ========================================================
    # A. selected_rich_features
    # ========================================================

    for layer in RICH_LAYERS:
        add_vector(features, f"pl_rich_response_mean_l{layer}", means["response"][layer])
        add_vector(
            features,
            f"pl_rich_response_minus_prompt_l{layer}",
            means["response"][layer] - means["prompt"][layer],
        )

    middle4 = torch.stack([means["response"][layer] for layer in MIDDLE4_LAYERS]).mean(dim=0)
    add_vector(features, "pl_rich_response_middle4", middle4)

    for left, right in RICH_LAYER_PAIRS:
        drift = means["response"][right] - means["response"][left]
        add_vector(features, f"pl_rich_response_drift_l{left}_to_l{right}", drift)
        add_vector(features, f"pl_rich_abs_response_drift_l{left}_to_l{right}", drift.abs())

    # ========================================================
    # B. Exact token position dynamics
    # ========================================================

    exact_zones = [
        "response",
        "response_early",
        "response_middle",
        "response_late",
        "response_last_5",
        "response_last_10",
    ]

    for zone_name in exact_zones:
        jumps = []
        cosines = []
        drift_norm_means = []
        drift_norm_stds = []
        for left, right in LAYER_PAIRS:
            drift = means[zone_name][right] - means[zone_name][left]
            jumps.append(safe_l2(drift))
            cosines.append(safe_cosine(means[zone_name][left], means[zone_name][right]))

            zone_idx = zones[zone_name]
            token_drift = zone_tokens(hidden, right, zone_idx) - zone_tokens(hidden, left, zone_idx)
            token_drift_norms = torch.linalg.norm(token_drift.float(), dim=1).detach().cpu().numpy()
            drift_norm_means.append(float(token_drift_norms.mean()) if len(token_drift_norms) else 0.0)
            drift_norm_stds.append(float(token_drift_norms.std()) if len(token_drift_norms) else 0.0)

        add_trajectory(features, f"pl_{zone_name}_jump", jumps)
        add_spectral(features, f"pl_{zone_name}_jump", jumps)
        add_trajectory(features, f"pl_{zone_name}_layer_cosine", cosines)
        add_trajectory(features, f"pl_{zone_name}_drift_norm_mean", drift_norm_means)
        add_trajectory(features, f"pl_{zone_name}_drift_norm_std", drift_norm_stds)

        curvatures = [
            safe_cosine(
                means[zone_name][LAYER_PAIRS[i][1]] - means[zone_name][LAYER_PAIRS[i][0]],
                means[zone_name][LAYER_PAIRS[i + 1][1]] - means[zone_name][LAYER_PAIRS[i + 1][0]],
            )
            for i in range(len(LAYER_PAIRS) - 1)
        ]
        add_trajectory(features, f"pl_{zone_name}_drift_curvature", curvatures)

    for layer in RICH_LAYERS:
        early = means["response_early"][layer]
        middle = means["response_middle"][layer]
        late = means["response_late"][layer]
        last5 = means["response_last_5"][layer]
        last10 = means["response_last_10"][layer]
        response = means["response"][layer]
        prompt = means["prompt"][layer]

        pairs = {
            "late_minus_early": late - early,
            "middle_minus_early": middle - early,
            "late_minus_middle": late - middle,
            "last5_minus_response": last5 - response,
            "last10_minus_response": last10 - response,
            "response_minus_prompt": response - prompt,
            "last5_minus_prompt": last5 - prompt,
        }
        for name, vec in pairs.items():
            add_stats(features, f"pl_pos_l{layer}_{name}", to_numpy(vec))

        features[f"pl_pos_l{layer}_early_late_cosine"] = safe_cosine(early, late)
        features[f"pl_pos_l{layer}_middle_late_cosine"] = safe_cosine(middle, late)
        features[f"pl_pos_l{layer}_last5_response_cosine"] = safe_cosine(last5, response)
        features[f"pl_pos_l{layer}_last10_response_cosine"] = safe_cosine(last10, response)
        features[f"pl_pos_l{layer}_response_prompt_cosine"] = safe_cosine(response, prompt)

        # Explicit response-ending collapse and exact late instability.
        last5_tokens = zone_tokens(hidden, layer, zones["response_last_5"])
        last10_tokens = zone_tokens(hidden, layer, zones["response_last_10"])
        late_tokens = zone_tokens(hidden, layer, zones["response_late"])
        response_tokens = zone_tokens(hidden, layer, zones["response"])

        last5_pcs = pairwise_cosine_stats(last5_tokens)
        last10_pcs = pairwise_cosine_stats(last10_tokens)
        last5_cov = covariance_stats(last5_tokens)
        late_pcs = pairwise_cosine_stats(late_tokens)

        features[f"pl_exact_response_ending_collapse_l{layer}_last5_pairwise_cosine"] = last5_pcs["mean"]
        features[f"pl_exact_response_ending_collapse_l{layer}_last10_pairwise_cosine"] = last10_pcs["mean"]
        features[f"pl_exact_response_ending_collapse_l{layer}_last5_cov_top1"] = last5_cov["top1_ratio"]
        features[f"pl_exact_response_ending_collapse_l{layer}_last5_vs_response_disagreement"] = clean_value(
            abs((1.0 - last5_pcs["mean"]) - (1.0 - pairwise_cosine_stats(response_tokens)["mean"]))
        )
        features[f"pl_exact_late_instability_l{layer}"] = clean_value((1.0 - late_pcs["mean"]) + safe_var_mean(late_tokens))
        features[f"pl_exact_late_stage_semantic_drift_l{layer}"] = safe_l2(last5 - middle)

    response_idx = zones["response"]
    for left, right in LAYER_PAIRS + LONG_LAYER_PAIRS:
        drift_tokens = hidden[right, response_idx] - hidden[left, response_idx]
        if drift_tokens.shape[0] == 0:
            drift_tokens = torch.zeros((1, hidden.shape[-1]), dtype=hidden.dtype)
        norms = torch.linalg.norm(drift_tokens.float(), dim=1).detach().cpu().numpy()
        add_stats(features, f"pl_tokenwise_response_drift_l{left}_to_l{right}", norms)
        add_trajectory(features, f"pl_tokenwise_response_drift_l{left}_to_l{right}_position", norms)
        add_spectral(features, f"pl_tokenwise_response_drift_l{left}_to_l{right}_position", norms)

    for zone_name in exact_zones:
        for layer in RICH_LAYERS:
            tokens = zone_tokens(hidden, layer, zones[zone_name])
            add_zone_token_stats(features, f"pl_scalar_{zone_name}_l{layer}", tokens)
            add_response_spectral_tokens(features, f"pl_spectral_{zone_name}_l{layer}", tokens)

    ending_collapse_series = [
        features[f"pl_exact_response_ending_collapse_l{layer}_last5_pairwise_cosine"]
        for layer in RICH_LAYERS
    ]
    late_instability_series = [features[f"pl_exact_late_instability_l{layer}"] for layer in RICH_LAYERS]
    late_stage_semantic_drift_series = [
        features[f"pl_exact_late_stage_semantic_drift_l{layer}"] for layer in RICH_LAYERS
    ]
    add_trajectory(features, "pl_exact_response_ending_collapse_by_layer", ending_collapse_series)
    add_spectral(features, "pl_exact_response_ending_collapse_by_layer", ending_collapse_series)
    add_trajectory(features, "pl_exact_late_instability_by_layer", late_instability_series)
    add_spectral(features, "pl_exact_late_instability_by_layer", late_instability_series)
    add_trajectory(features, "pl_exact_late_stage_semantic_drift_by_layer", late_stage_semantic_drift_series)
    add_spectral(features, "pl_exact_late_stage_semantic_drift_by_layer", late_stage_semantic_drift_series)

    # ========================================================
    # C. Exact self-contradiction
    # ========================================================

    contradiction_distances = []
    semantic_divergences = []
    consistency_cosines = []
    consistency_decays = []
    late_stage_drifts = []

    for layer in RICH_LAYERS:
        early = means["response_early"][layer]
        middle = means["response_middle"][layer]
        late = means["response_late"][layer]
        last5 = means["response_last_5"][layer]

        early_late_dist = safe_l2(late - early)
        middle_late_dist = safe_l2(late - middle)
        last5_early_dist = safe_l2(last5 - early)
        early_late_cos = safe_cosine(early, late)
        early_middle_cos = safe_cosine(early, middle)
        middle_late_cos = safe_cosine(middle, late)
        consistency_decay = early_middle_cos - middle_late_cos
        semantic_divergence = early_late_dist / (safe_l2(early) + EPS)
        late_stage_drift = safe_l2(last5 - late)

        features[f"pl_exact_contradiction_l{layer}_early_late_distance"] = early_late_dist
        features[f"pl_exact_contradiction_l{layer}_middle_late_distance"] = middle_late_dist
        features[f"pl_exact_contradiction_l{layer}_last5_early_distance"] = last5_early_dist
        features[f"pl_exact_contradiction_l{layer}_early_late_cosine"] = early_late_cos
        features[f"pl_exact_contradiction_l{layer}_semantic_divergence"] = semantic_divergence
        features[f"pl_exact_contradiction_l{layer}_consistency_decay"] = consistency_decay
        features[f"pl_exact_contradiction_l{layer}_late_stage_semantic_drift"] = late_stage_drift

        contradiction_distances.append(early_late_dist)
        semantic_divergences.append(semantic_divergence)
        consistency_cosines.append(early_late_cos)
        consistency_decays.append(consistency_decay)
        late_stage_drifts.append(late_stage_drift)

    add_trajectory(features, "pl_exact_contradiction_growth_by_layer", contradiction_distances)
    add_spectral(features, "pl_exact_contradiction_growth_by_layer", contradiction_distances)
    add_trajectory(features, "pl_exact_response_semantic_divergence_by_layer", semantic_divergences)
    add_spectral(features, "pl_exact_response_semantic_divergence_by_layer", semantic_divergences)
    add_trajectory(features, "pl_exact_response_consistency_by_layer", consistency_cosines)
    add_trajectory(features, "pl_exact_response_consistency_decay_by_layer", consistency_decays)
    add_spectral(features, "pl_exact_response_consistency_decay_by_layer", consistency_decays)
    add_trajectory(features, "pl_exact_late_stage_semantic_drift_contradiction_by_layer", late_stage_drifts)
    add_spectral(features, "pl_exact_late_stage_semantic_drift_contradiction_by_layer", late_stage_drifts)

    # ========================================================
    # D. Exact layer localization
    # ========================================================

    layer_instability = []
    layer_disagreement = []
    layer_collapse = []
    layer_uncertainty_spike = []
    layer_drift_energy = []

    for layer in RICH_LAYERS:
        response_tokens = zone_tokens(hidden, layer, zones["response"])
        pcs = pairwise_cosine_stats(response_tokens)
        cov = covariance_stats(response_tokens)

        disagreement = clean_value(1.0 - pcs["mean"])
        collapse = cov["top1_ratio"]
        instability = clean_value(disagreement + safe_var_mean(response_tokens))
        uncertainty_spike = clean_value(instability + collapse)

        features[f"pl_layer_l{layer}_response_instability"] = instability
        features[f"pl_layer_l{layer}_response_disagreement"] = disagreement
        features[f"pl_layer_l{layer}_response_collapse"] = collapse
        features[f"pl_layer_l{layer}_response_uncertainty_spike"] = uncertainty_spike

        layer_instability.append(instability)
        layer_disagreement.append(disagreement)
        layer_collapse.append(collapse)
        layer_uncertainty_spike.append(uncertainty_spike)

    for left, right in RICH_LAYER_PAIRS:
        drift = means["response"][right] - means["response"][left]
        energy = safe_l2(drift)
        features[f"pl_layer_transition_l{left}_to_l{right}_response_drift_energy"] = energy
        layer_drift_energy.append(energy)

    add_trajectory(features, "pl_layer_response_instability", layer_instability)
    add_spectral(features, "pl_layer_response_instability", layer_instability)
    add_trajectory(features, "pl_layer_response_disagreement", layer_disagreement)
    add_spectral(features, "pl_layer_response_disagreement", layer_disagreement)
    add_trajectory(features, "pl_layer_response_collapse", layer_collapse)
    add_spectral(features, "pl_layer_response_collapse", layer_collapse)
    add_trajectory(features, "pl_layer_response_uncertainty_spike", layer_uncertainty_spike)
    add_spectral(features, "pl_layer_response_uncertainty_spike", layer_uncertainty_spike)
    add_trajectory(features, "pl_layer_response_drift_localization", layer_drift_energy)
    add_spectral(features, "pl_layer_response_drift_localization", layer_drift_energy)

    if layer_drift_energy:
        max_idx = int(np.argmax(layer_drift_energy))
        features["pl_strongest_response_drift_transition_idx"] = clean_value(max_idx)
        features["pl_strongest_response_drift_transition_energy"] = clean_value(layer_drift_energy[max_idx])

    return {key: clean_value(value) for key, value in features.items()}


# ============================================================
# DATASET EXTRACTION
# ============================================================


def extract_dataset_features(df: pd.DataFrame, model, tokenizer, device: torch.device, has_label: bool) -> pd.DataFrame:
    prompts = df["prompt"].astype(str).tolist()
    responses = df["response"].astype(str).tolist()
    texts = [p + r for p, r in zip(prompts, responses)]
    prompt_lengths = get_prompt_lengths(tokenizer, prompts, MAX_LENGTH)

    rows = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extract prompt-len smart features"):
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
            features = extract_features_one_sample(
                hidden=hidden_batch[i],
                valid_mask=mask_batch[i],
                prompt_len=batch_prompt_lengths[i],
            )
            rows.append(features)

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
    print("BUILD EXTRA SMART PROMPT-LEN FEATURES")
    print("=" * 80)
    print("Device:", device)
    print("Output dir:", OUTPUT_DIR)
    print("Batch size:", BATCH_SIZE)
    print("Note: hidden states + valid masks + prompt_len only")

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    train_df = pd.read_csv(DATA_FILE)
    train_features = extract_dataset_features(
        df=train_df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        has_label=True,
    )
    train_features.to_parquet(TRAIN_OUTPUT, index=False)
    print("Saved train:", TRAIN_OUTPUT)
    print("Train shape:", train_features.shape)

    if EXPORT_TEST and Path(TEST_FILE).exists():
        test_df = pd.read_csv(TEST_FILE)
        test_features = extract_dataset_features(
            df=test_df,
            model=model,
            tokenizer=tokenizer,
            device=device,
            has_label=False,
        )
        test_features.to_parquet(TEST_OUTPUT, index=False)
        print("Saved test:", TEST_OUTPUT)
        print("Test shape:", test_features.shape)

    print(f"Done in {time.time() - t0:.1f} sec")
    print("=" * 80)


if __name__ == "__main__":
    main()
