"""
Build MORE extra features for hallucination detection.

This script is intentionally solution.py-compatible:
- no prompt_len
- no logits
- no attention maps
- no forward hooks
- no repository infrastructure changes

Feature families included in one parquet:
1. Long-range layer instability
2. Token trajectory dynamics
3. Last-token / last-5 focused features
4. Text-only degradation features
5. Cross-zone heuristic features

Outputs:
./artifacts/more_extra_features/features_dataset_more_extra.parquet
./artifacts/more_extra_features/features_test_more_extra.parquet
"""

from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from model import MAX_LENGTH, get_model_and_tokenizer


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"

OUTPUT_DIR = Path("./artifacts/more_extra_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUTPUT = OUTPUT_DIR / "features_dataset_more_extra.parquet"
TEST_OUTPUT = OUTPUT_DIR / "features_test_more_extra.parquet"

BATCH_SIZE = 1
EXPORT_TEST = True

LAYERS = [8, 9, 10, 11, 12, 13, 14, 15, 16]
FOCUS_LAYERS = [10, 11, 12, 13, 14, 15, 16]
LONG_LAYER_PAIRS = [
    (8, 16),
    (10, 16),
    (12, 16),
    (8, 12),
    (8, 14),
    (10, 14),
    (11, 16),
    (13, 16),
]
ADJACENT_LAYER_PAIRS = list(zip(FOCUS_LAYERS[:-1], FOCUS_LAYERS[1:]))
ALL_LAYER_PAIRS = LONG_LAYER_PAIRS + ADJACENT_LAYER_PAIRS

EPS = 1e-8

UNCERTAINTY_MARKERS = [
    "maybe",
    "perhaps",
    "probably",
    "possibly",
    "i think",
    "i guess",
    "not sure",
    "unclear",
    "it seems",
    "appears to",
    "as far as i know",
    "likely",
    "might",
    "could",
    "approximately",
]

REFUSAL_OR_LIMIT_MARKERS = [
    "i don't know",
    "i do not know",
    "i can't",
    "i cannot",
    "i'm unable",
    "i am unable",
    "cannot determine",
    "no information",
    "not enough information",
]


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
    a = to_numpy(a).reshape(-1)
    b = to_numpy(b).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + EPS
    return clean_value(np.dot(a, b) / denom)


def safe_mean(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.shape[0] == 0:
        return torch.zeros(tokens.shape[-1], dtype=tokens.dtype)
    return tokens.float().mean(dim=0)


def safe_std(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.shape[0] <= 1:
        return torch.zeros(tokens.shape[-1], dtype=tokens.dtype)
    return tokens.float().std(dim=0, unbiased=False)


def entropy_from_values(values: Iterable[float]) -> float:
    values = np.abs(np.asarray(list(values), dtype=np.float32))
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
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


def add_trajectory_stats(features: Dict[str, float], prefix: str, values: Iterable[float]) -> None:
    values = np.asarray(list(values), dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)

    add_stats(features, prefix, values)

    diff1 = np.diff(values) if len(values) >= 2 else np.array([0.0], dtype=np.float32)
    diff2 = np.diff(values, n=2) if len(values) >= 3 else np.array([0.0], dtype=np.float32)

    if len(values) >= 2:
        x = np.arange(len(values), dtype=np.float32)
        slope = np.polyfit(x, values, 1)[0]
    else:
        slope = 0.0

    split = max(1, len(values) // 3)
    early = values[:split].mean()
    late = values[-split:].mean()

    features[f"{prefix}_slope"] = clean_value(slope)
    features[f"{prefix}_roughness"] = clean_value(np.abs(diff1).sum())
    features[f"{prefix}_roughness_mean"] = clean_value(np.abs(diff1).mean())
    features[f"{prefix}_acceleration_mean"] = clean_value(np.abs(diff2).mean())
    features[f"{prefix}_acceleration_max"] = clean_value(np.abs(diff2).max())
    features[f"{prefix}_late_minus_early"] = clean_value(late - early)
    features[f"{prefix}_late_div_early"] = clean_value(late / (abs(early) + EPS))
    features[f"{prefix}_num_spikes_p90"] = clean_value((values > np.percentile(values, 90)).sum())

    signs = np.sign(diff1)
    signs = signs[signs != 0]
    features[f"{prefix}_sign_changes"] = clean_value((signs[1:] != signs[:-1]).sum() if len(signs) >= 2 else 0)


def add_fft_stats(features: Dict[str, float], prefix: str, values: Iterable[float]) -> None:
    values = np.asarray(list(values), dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)

    centered = values - values.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    split = max(1, len(power) // 2)
    low = power[:split].sum()
    high = power[split:].sum()
    total = power.sum()

    features[f"{prefix}_fft_energy"] = clean_value(total)
    features[f"{prefix}_fft_low_energy"] = clean_value(low)
    features[f"{prefix}_fft_high_energy"] = clean_value(high)
    features[f"{prefix}_fft_high_low_ratio"] = clean_value(high / (low + EPS))
    features[f"{prefix}_fft_entropy"] = entropy_from_values(power)
    features[f"{prefix}_fft_dominant_freq"] = clean_value(np.argmax(power) / max(len(power) - 1, 1))


def valid_positions(valid_mask: torch.Tensor) -> torch.Tensor:
    return torch.where(valid_mask.bool().cpu())[0]


def make_zones(valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    idx = valid_positions(valid_mask)
    n = int(idx.numel())

    zone_names = [
        "all",
        "first70",
        "last30",
        "last20",
        "last10",
        "last5",
        "last_token",
        "early",
        "middle",
        "late",
    ]

    if n == 0:
        return {name: idx for name in zone_names}

    def last_frac(frac: float) -> torch.Tensor:
        keep = max(1, int(round(n * frac)))
        return idx[-keep:]

    third = max(1, n // 3)
    first70_end = max(1, int(round(n * 0.70)))

    return {
        "all": idx,
        "first70": idx[:first70_end],
        "last30": last_frac(0.30),
        "last20": last_frac(0.20),
        "last10": last_frac(0.10),
        "last5": idx[-min(5, n):],
        "last_token": idx[-1:],
        "early": idx[:third],
        "middle": idx[third: 2 * third] if 2 * third > third else idx,
        "late": idx[2 * third:] if n > 2 * third else idx[-third:],
    }


def zone_tokens(hidden: torch.Tensor, layer: int, idx: torch.Tensor) -> torch.Tensor:
    if idx.numel() == 0:
        return torch.zeros((1, hidden.shape[-1]), dtype=hidden.dtype)
    return hidden[layer, idx].float()


def pairwise_cosine_values(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape[0] <= 1:
        return np.array([1.0], dtype=np.float32)
    normed = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + EPS)
    sim = normed @ normed.T
    return sim[np.triu_indices_from(sim, k=1)]


# ============================================================
# 1. LONG-RANGE LAYER INSTABILITY
# ============================================================


def add_layer_instability_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    zone_names = ["all", "first70", "last30", "last20", "last5", "last_token", "late"]
    pair_distance_series = []
    pair_cosine_series = []

    for zone_name in zone_names:
        distance_values = []
        cosine_values = []
        norm_ratio_values = []

        for left, right in ALL_LAYER_PAIRS:
            left_mean = safe_mean(zone_tokens(hidden, left, zones[zone_name]))
            right_mean = safe_mean(zone_tokens(hidden, right, zones[zone_name]))
            delta = right_mean - left_mean

            prefix = f"layer_instability_{zone_name}_l{left}_to_l{right}"
            distance = safe_l2(delta)
            cosine_value = safe_cosine(left_mean, right_mean)
            norm_ratio = safe_l2(right_mean) / (safe_l2(left_mean) + EPS)

            features[f"{prefix}_l2"] = distance
            features[f"{prefix}_cosine"] = cosine_value
            features[f"{prefix}_cosine_drop"] = clean_value(1.0 - cosine_value)
            features[f"{prefix}_norm_ratio"] = clean_value(norm_ratio)
            features[f"{prefix}_delta_start_cosine"] = safe_cosine(delta, left_mean)
            features[f"{prefix}_delta_end_cosine"] = safe_cosine(delta, right_mean)

            distance_values.append(distance)
            cosine_values.append(cosine_value)
            norm_ratio_values.append(norm_ratio)
            pair_distance_series.append(distance)
            pair_cosine_series.append(cosine_value)

        add_trajectory_stats(features, f"layer_instability_{zone_name}_l2_by_pair", distance_values)
        add_fft_stats(features, f"layer_instability_{zone_name}_l2_by_pair", distance_values)
        add_trajectory_stats(features, f"layer_instability_{zone_name}_cosine_by_pair", cosine_values)
        add_trajectory_stats(features, f"layer_instability_{zone_name}_norm_ratio_by_pair", norm_ratio_values)

    add_stats(features, "layer_instability_global_l2", pair_distance_series)
    add_stats(features, "layer_instability_global_cosine", pair_cosine_series)


# ============================================================
# 2. TOKEN TRAJECTORY DYNAMICS
# ============================================================


def add_token_trajectory_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    for layer in FOCUS_LAYERS:
        for zone_name in ["all", "last30", "last20", "last10", "last5", "late"]:
            tokens = zone_tokens(hidden, layer, zones[zone_name])
            token_np = to_numpy(tokens)
            norms = np.linalg.norm(token_np, axis=1)

            if token_np.shape[0] >= 2:
                prev = token_np[:-1]
                nxt = token_np[1:]
                step_delta = nxt - prev
                step_l2 = np.linalg.norm(step_delta, axis=1)
                step_cos = np.sum(prev * nxt, axis=1) / (
                    np.linalg.norm(prev, axis=1) * np.linalg.norm(nxt, axis=1) + EPS
                )
            else:
                step_l2 = np.array([0.0], dtype=np.float32)
                step_cos = np.array([1.0], dtype=np.float32)

            prefix = f"token_traj_l{layer}_{zone_name}"
            add_trajectory_stats(features, f"{prefix}_norm", norms)
            add_fft_stats(features, f"{prefix}_norm", norms)
            add_trajectory_stats(features, f"{prefix}_step_l2", step_l2)
            add_fft_stats(features, f"{prefix}_step_l2", step_l2)
            add_trajectory_stats(features, f"{prefix}_step_cosine", step_cos)

            p90 = np.percentile(step_l2, 90) if len(step_l2) else 0.0
            p95 = np.percentile(step_l2, 95) if len(step_l2) else 0.0
            features[f"{prefix}_num_step_spikes_p90"] = clean_value((step_l2 > p90).sum())
            features[f"{prefix}_num_step_spikes_p95"] = clean_value((step_l2 > p95).sum())
            features[f"{prefix}_late_spike_ratio"] = clean_value(
                step_l2[-max(1, len(step_l2) // 3):].mean() / (step_l2.mean() + EPS)
            )
            features[f"{prefix}_max_step_to_mean"] = clean_value(step_l2.max() / (step_l2.mean() + EPS))

            pairwise_cos = pairwise_cosine_values(token_np)
            add_stats(features, f"{prefix}_pairwise_cosine", pairwise_cos)
            features[f"{prefix}_collapse_score"] = clean_value(pairwise_cos.mean())
            features[f"{prefix}_divergence_score"] = clean_value(1.0 - pairwise_cos.mean())


# ============================================================
# 3. LAST TOKEN / LAST-5 FOCUSED FEATURES
# ============================================================


def add_last_token_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    last_token_norms = []
    last5_norms = []
    last_token_to_all_cos = []
    last_token_to_last30_cos = []
    last5_to_all_cos = []

    for layer in FOCUS_LAYERS:
        all_tokens = zone_tokens(hidden, layer, zones["all"])
        last30_tokens = zone_tokens(hidden, layer, zones["last30"])
        last5_tokens = zone_tokens(hidden, layer, zones["last5"])
        last_token = zone_tokens(hidden, layer, zones["last_token"])

        all_mean = safe_mean(all_tokens)
        last30_mean = safe_mean(last30_tokens)
        last5_mean = safe_mean(last5_tokens)
        last_token_vec = safe_mean(last_token)

        last5_std = safe_std(last5_tokens)
        all_std = safe_std(all_tokens)

        prefix = f"last_focus_l{layer}"
        last_token_norm = safe_l2(last_token_vec)
        last5_norm = safe_l2(last5_mean)
        lt_all_cos = safe_cosine(last_token_vec, all_mean)
        lt_last30_cos = safe_cosine(last_token_vec, last30_mean)
        l5_all_cos = safe_cosine(last5_mean, all_mean)

        features[f"{prefix}_last_token_norm"] = last_token_norm
        features[f"{prefix}_last5_mean_norm"] = last5_norm
        features[f"{prefix}_last5_std_norm"] = safe_l2(last5_std)
        features[f"{prefix}_last5_std_to_all_std"] = clean_value(safe_l2(last5_std) / (safe_l2(all_std) + EPS))
        features[f"{prefix}_last_token_to_all_cosine"] = lt_all_cos
        features[f"{prefix}_last_token_to_last30_cosine"] = lt_last30_cos
        features[f"{prefix}_last_token_to_last5_cosine"] = safe_cosine(last_token_vec, last5_mean)
        features[f"{prefix}_last5_to_all_cosine"] = l5_all_cos
        features[f"{prefix}_last_token_to_all_l2"] = safe_l2(last_token_vec - all_mean)
        features[f"{prefix}_last_token_to_last30_l2"] = safe_l2(last_token_vec - last30_mean)
        features[f"{prefix}_last_token_to_last5_l2"] = safe_l2(last_token_vec - last5_mean)
        features[f"{prefix}_last5_to_all_l2"] = safe_l2(last5_mean - all_mean)
        features[f"{prefix}_last_token_outlier_score"] = clean_value(
            safe_l2(last_token_vec - all_mean) / (safe_l2(all_std) + EPS)
        )

        last5_token_norms = np.linalg.norm(to_numpy(last5_tokens), axis=1)
        add_stats(features, f"{prefix}_last5_token_norm", last5_token_norms)
        add_trajectory_stats(features, f"{prefix}_last5_token_norm_position", last5_token_norms)

        if last5_tokens.shape[0] >= 2:
            last5_np = to_numpy(last5_tokens)
            step_l2 = np.linalg.norm(last5_np[1:] - last5_np[:-1], axis=1)
        else:
            step_l2 = np.array([0.0], dtype=np.float32)
        add_stats(features, f"{prefix}_last5_step_l2", step_l2)

        last_token_norms.append(last_token_norm)
        last5_norms.append(last5_norm)
        last_token_to_all_cos.append(lt_all_cos)
        last_token_to_last30_cos.append(lt_last30_cos)
        last5_to_all_cos.append(l5_all_cos)

    add_trajectory_stats(features, "last_focus_last_token_norm_by_layer", last_token_norms)
    add_trajectory_stats(features, "last_focus_last5_norm_by_layer", last5_norms)
    add_trajectory_stats(features, "last_focus_last_token_to_all_cosine_by_layer", last_token_to_all_cos)
    add_trajectory_stats(features, "last_focus_last_token_to_last30_cosine_by_layer", last_token_to_last30_cos)
    add_trajectory_stats(features, "last_focus_last5_to_all_cosine_by_layer", last5_to_all_cos)

    for left, right in ALL_LAYER_PAIRS:
        lt_left = safe_mean(zone_tokens(hidden, left, zones["last_token"]))
        lt_right = safe_mean(zone_tokens(hidden, right, zones["last_token"]))
        l5_left = safe_mean(zone_tokens(hidden, left, zones["last5"]))
        l5_right = safe_mean(zone_tokens(hidden, right, zones["last5"]))

        prefix = f"last_focus_layer_drift_l{left}_to_l{right}"
        features[f"{prefix}_last_token_l2"] = safe_l2(lt_right - lt_left)
        features[f"{prefix}_last_token_cosine"] = safe_cosine(lt_left, lt_right)
        features[f"{prefix}_last5_l2"] = safe_l2(l5_right - l5_left)
        features[f"{prefix}_last5_cosine"] = safe_cosine(l5_left, l5_right)


# ============================================================
# 4. TEXT-ONLY DEGRADATION FEATURES
# ============================================================


def ngram_repetition_ratio(words: List[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count for count in counts.values() if count > 1)
    return clean_value(repeated / max(len(ngrams), 1))


def add_text_degradation_features(features: Dict[str, float], prompt: str, response: str) -> None:
    prompt = str(prompt)
    response = str(response)
    text = response.strip()
    lower = text.lower()

    words = re.findall(r"\b\w+\b", lower)
    alpha_words = [word for word in words if any(ch.isalpha() for ch in word)]
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    chars = list(text)

    features["text_response_chars"] = clean_value(len(response))
    features["text_response_stripped_chars"] = clean_value(len(text))
    features["text_prompt_chars"] = clean_value(len(prompt))
    features["text_response_prompt_char_ratio"] = clean_value(len(response) / (len(prompt) + EPS))
    features["text_word_count"] = clean_value(len(words))
    features["text_alpha_word_count"] = clean_value(len(alpha_words))
    features["text_unique_word_ratio"] = clean_value(len(set(words)) / (len(words) + EPS))
    features["text_sentence_count"] = clean_value(len(sentences))

    word_lengths = [len(word) for word in words]
    sentence_lengths = [len(re.findall(r"\b\w+\b", sentence)) for sentence in sentences]
    add_stats(features, "text_word_length", word_lengths)
    add_stats(features, "text_sentence_length", sentence_lengths)

    counts = Counter(words)
    repeated_words = sum(count for count in counts.values() if count > 1)
    max_word_count = max(counts.values()) if counts else 0
    features["text_repeated_word_ratio"] = clean_value(repeated_words / (len(words) + EPS))
    features["text_max_word_frequency_ratio"] = clean_value(max_word_count / (len(words) + EPS))
    features["text_bigram_repetition_ratio"] = ngram_repetition_ratio(words, 2)
    features["text_trigram_repetition_ratio"] = ngram_repetition_ratio(words, 3)
    features["text_4gram_repetition_ratio"] = ngram_repetition_ratio(words, 4)
    features["text_5gram_repetition_ratio"] = ngram_repetition_ratio(words, 5)

    adjacent_repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    features["text_adjacent_word_repeat_ratio"] = clean_value(adjacent_repeats / (len(words) + EPS))

    sentence_starts = [" ".join(re.findall(r"\b\w+\b", sentence.lower())[:3]) for sentence in sentences]
    sentence_starts = [start for start in sentence_starts if start]
    features["text_sentence_start_repeat_ratio"] = clean_value(
        1.0 - len(set(sentence_starts)) / (len(sentence_starts) + EPS) if sentence_starts else 0.0
    )

    digits = re.findall(r"\d", text)
    numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", text)
    punctuation = re.findall(r"[^\w\s]", text)
    quotes = re.findall(r"[\"'“”‘’]", text)

    features["text_digit_ratio"] = clean_value(len(digits) / (len(text) + EPS))
    features["text_number_count"] = clean_value(len(numbers))
    features["text_number_word_ratio"] = clean_value(len(numbers) / (len(words) + EPS))
    features["text_punctuation_ratio"] = clean_value(len(punctuation) / (len(text) + EPS))
    features["text_quote_count"] = clean_value(len(quotes))
    features["text_parentheses_count"] = clean_value(text.count("(") + text.count(")"))
    features["text_bracket_count"] = clean_value(text.count("[") + text.count("]"))
    features["text_unbalanced_parentheses"] = clean_value(abs(text.count("(") - text.count(")")) > 0)
    features["text_unbalanced_brackets"] = clean_value(abs(text.count("[") - text.count("]")) > 0)
    features["text_ellipsis_count"] = clean_value(text.count("..."))
    features["text_newline_count"] = clean_value(text.count("\n"))
    features["text_markdown_marker_count"] = clean_value(len(re.findall(r"```|`|\*\*|__|#+|\||^-\s", text, flags=re.M)))
    features["text_uppercase_word_ratio"] = clean_value(
        sum(1 for token in text.split() if len(token) > 1 and token.isupper()) / (len(text.split()) + EPS)
    )

    if chars:
        char_counts = Counter(chars)
        char_probs = np.asarray(list(char_counts.values()), dtype=np.float32) / len(chars)
        features["text_char_entropy"] = clean_value(-(char_probs * np.log(char_probs + EPS)).sum())
    else:
        features["text_char_entropy"] = 0.0

    if words:
        word_probs = np.asarray(list(counts.values()), dtype=np.float32) / len(words)
        features["text_word_entropy"] = clean_value(-(word_probs * np.log(word_probs + EPS)).sum())
    else:
        features["text_word_entropy"] = 0.0

    features["text_uncertainty_marker_count"] = clean_value(sum(lower.count(marker) for marker in UNCERTAINTY_MARKERS))
    features["text_uncertainty_marker_ratio"] = clean_value(features["text_uncertainty_marker_count"] / (len(words) + EPS))
    features["text_refusal_or_limit_marker_count"] = clean_value(sum(lower.count(marker) for marker in REFUSAL_OR_LIMIT_MARKERS))
    features["text_ends_without_terminal_punctuation"] = clean_value(bool(text) and text[-1] not in ".!?")
    features["text_has_many_commas"] = clean_value(text.count(",") >= 5)
    features["text_comma_ratio"] = clean_value(text.count(",") / (len(text) + EPS))
    features["text_colon_semicolon_ratio"] = clean_value((text.count(":") + text.count(";")) / (len(text) + EPS))

    prompt_words = set(re.findall(r"\b\w+\b", prompt.lower()))
    response_words = set(words)
    features["text_prompt_response_word_overlap"] = clean_value(
        len(prompt_words & response_words) / (len(response_words) + EPS)
    )
    features["text_new_words_vs_prompt_ratio"] = clean_value(
        len(response_words - prompt_words) / (len(response_words) + EPS)
    )


# ============================================================
# 5. CROSS-ZONE HEURISTIC FEATURES
# ============================================================


def add_cross_zone_features(
    features: Dict[str, float],
    hidden: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    zone_pairs = [
        ("first70", "last30"),
        ("last20", "all"),
        ("last5", "all"),
        ("last5", "last30"),
        ("last_token", "all"),
        ("last_token", "last30"),
        ("late", "early"),
        ("late", "middle"),
    ]

    for left_zone, right_zone in zone_pairs:
        cosine_series = []
        distance_series = []
        norm_ratio_series = []
        var_ratio_series = []

        for layer in FOCUS_LAYERS:
            left_tokens = zone_tokens(hidden, layer, zones[left_zone])
            right_tokens = zone_tokens(hidden, layer, zones[right_zone])
            left_mean = safe_mean(left_tokens)
            right_mean = safe_mean(right_tokens)
            left_std = safe_std(left_tokens)
            right_std = safe_std(right_tokens)

            prefix = f"cross_zone_l{layer}_{left_zone}_vs_{right_zone}"
            cosine_value = safe_cosine(left_mean, right_mean)
            distance_value = safe_l2(left_mean - right_mean)
            norm_ratio = safe_l2(left_mean) / (safe_l2(right_mean) + EPS)
            var_ratio = safe_l2(left_std) / (safe_l2(right_std) + EPS)

            features[f"{prefix}_cosine"] = cosine_value
            features[f"{prefix}_distance"] = distance_value
            features[f"{prefix}_norm_ratio"] = clean_value(norm_ratio)
            features[f"{prefix}_std_ratio"] = clean_value(var_ratio)
            features[f"{prefix}_collapse_divergence"] = clean_value(1.0 - cosine_value)
            features[f"{prefix}_relative_distance"] = clean_value(distance_value / (safe_l2(right_mean) + EPS))

            cosine_series.append(cosine_value)
            distance_series.append(distance_value)
            norm_ratio_series.append(norm_ratio)
            var_ratio_series.append(var_ratio)

        pair_name = f"cross_zone_{left_zone}_vs_{right_zone}"
        add_trajectory_stats(features, f"{pair_name}_cosine_by_layer", cosine_series)
        add_trajectory_stats(features, f"{pair_name}_distance_by_layer", distance_series)
        add_fft_stats(features, f"{pair_name}_distance_by_layer", distance_series)
        add_trajectory_stats(features, f"{pair_name}_norm_ratio_by_layer", norm_ratio_series)
        add_trajectory_stats(features, f"{pair_name}_std_ratio_by_layer", var_ratio_series)


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
    zones = make_zones(valid_mask)

    features: Dict[str, float] = {}

    add_layer_instability_features(features, hidden, zones)
    add_token_trajectory_features(features, hidden, zones)
    add_last_token_features(features, hidden, zones)
    add_text_degradation_features(features, prompt_text, response_text)
    add_cross_zone_features(features, hidden, zones)

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
    texts = [prompt + response for prompt, response in zip(prompts, responses)]

    rows: List[Dict[str, float]] = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extract more extra features"):
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
    print("BUILD MORE EXTRA FEATURES")
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
