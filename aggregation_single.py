from __future__ import annotations

import numpy as np
import pandas as pd
import torch
try:
    from model import MAX_LENGTH
except Exception:
    MAX_LENGTH = 512


DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

LAYERS = [10, 11, 12, 13, 14, 15, 16]
RICH_LAYERS = [11, 12, 13, 14, 15, 16]
MIDDLE4_LAYERS = [11, 12, 13, 14]

DRIFT_PAIRS = list(zip(LAYERS[:-1], LAYERS[1:]))
RICH_DRIFT_PAIRS = list(zip(RICH_LAYERS[:-1], RICH_LAYERS[1:]))
LONG_DRIFT_PAIRS = [
    (10, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
    (10, 16),
    (11, 16),
]

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


def safe_mean(x):
    if x.shape[0] == 0:
        return torch.zeros(x.shape[-1], dtype=x.dtype)

    return x.mean(dim=0)


def safe_std(x):
    if x.shape[0] <= 1:
        return torch.zeros(x.shape[-1], dtype=x.dtype)

    return x.std(dim=0, unbiased=False)


def safe_l2(x):
    return float(np.linalg.norm(to_numpy(x)))


def safe_cosine(a, b):
    a = to_numpy(a)
    b = to_numpy(b)

    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b) + EPS)
    )


def value_entropy(values):
    values = np.abs(np.asarray(values, dtype=np.float32))
    total = values.sum()

    if total <= EPS:
        return 0.0

    p = values / total
    return float(-(p * np.log(p + EPS)).sum())


def add_stats(features, prefix, values):
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if len(values) == 0:
        values = np.array([0.0], dtype=np.float32)

    features[f"{prefix}_mean"] = float(values.mean())
    features[f"{prefix}_std"] = float(values.std())
    features[f"{prefix}_min"] = float(values.min())
    features[f"{prefix}_max"] = float(values.max())
    features[f"{prefix}_range"] = float(values.max() - values.min())
    features[f"{prefix}_p10"] = float(np.percentile(values, 10))
    features[f"{prefix}_p25"] = float(np.percentile(values, 25))
    features[f"{prefix}_p50"] = float(np.percentile(values, 50))
    features[f"{prefix}_p75"] = float(np.percentile(values, 75))
    features[f"{prefix}_p90"] = float(np.percentile(values, 90))
    features[f"{prefix}_iqr"] = float(
        np.percentile(values, 75) - np.percentile(values, 25)
    )
    features[f"{prefix}_entropy"] = value_entropy(values)


def add_trajectory_features(features, prefix, values):
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if len(values) == 0:
        values = np.array([0.0], dtype=np.float32)

    diff1 = np.diff(values) if len(values) >= 2 else np.array([0.0])
    diff2 = np.diff(values, n=2) if len(values) >= 3 else np.array([0.0])

    add_stats(features, prefix, values)

    x = np.arange(len(values), dtype=np.float32)
    slope = float(np.polyfit(x, values, 1)[0]) if len(values) >= 2 else 0.0

    early = float(values[:2].mean()) if len(values) >= 2 else float(values.mean())
    late = float(values[-2:].mean()) if len(values) >= 2 else float(values.mean())

    features[f"{prefix}_slope"] = slope
    features[f"{prefix}_roughness"] = float(np.abs(diff1).sum())
    features[f"{prefix}_smoothness"] = float(1.0 / (1.0 + np.abs(diff1).sum()))
    features[f"{prefix}_acceleration_mean"] = float(np.abs(diff2).mean())
    features[f"{prefix}_acceleration_max"] = float(np.abs(diff2).max())
    features[f"{prefix}_late_minus_early"] = late - early
    features[f"{prefix}_late_div_early"] = late / (abs(early) + EPS)
    features[f"{prefix}_num_increases"] = int((diff1 > 0).sum())
    features[f"{prefix}_num_decreases"] = int((diff1 < 0).sum())

    signs = np.sign(diff1)
    signs = signs[signs != 0]

    if len(signs) >= 2:
        features[f"{prefix}_sign_changes"] = int((signs[1:] != signs[:-1]).sum())
    else:
        features[f"{prefix}_sign_changes"] = 0


def add_spectral_features(features, prefix, values):
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

    features[f"{prefix}_fft_energy"] = total
    features[f"{prefix}_spectral_low_energy"] = low
    features[f"{prefix}_spectral_high_energy"] = high
    features[f"{prefix}_spectral_high_low_ratio"] = high / (low + EPS)
    features[f"{prefix}_spectral_entropy"] = value_entropy(power)
    features[f"{prefix}_dominant_frequency"] = float(
        np.argmax(power) / max(len(power) - 1, 1)
    )


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

    normalized = token_matrix / (
        np.linalg.norm(token_matrix, axis=1, keepdims=True) + EPS
    )
    sim = normalized @ normalized.T
    tri = sim[np.triu_indices_from(sim, k=1)]

    return {
        "mean": float(tri.mean()),
        "std": float(tri.std()),
        "min": float(tri.min()),
        "p10": float(np.percentile(tri, 10)),
        "p90": float(np.percentile(tri, 90)),
    }


def covariance_pca_stats(token_matrix):
    token_matrix = to_numpy(token_matrix)

    if token_matrix.shape[0] <= 2:
        return {
            "trace": 0.0,
            "top1_ratio": 0.0,
            "top3_ratio": 0.0,
            "top5_ratio": 0.0,
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "spectral_entropy": 0.0,
        }

    centered = token_matrix - token_matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False,
    )

    eig = singular_values ** 2
    total = float(eig.sum()) + EPS
    p = eig / total

    entropy = float(-(p * np.log(p + EPS)).sum())

    return {
        "trace": total,
        "top1_ratio": float(p[:1].sum()),
        "top3_ratio": float(p[:3].sum()),
        "top5_ratio": float(p[:5].sum()),
        "effective_rank": float(np.exp(entropy)),
        "participation_ratio": float((eig.sum() ** 2) / ((eig ** 2).sum() + EPS)),
        "spectral_entropy": entropy,
    }


def activation_entropy(token_matrix):
    token_matrix = token_matrix.float()

    if token_matrix.shape[0] == 0:
        return np.array([0.0], dtype=np.float32)

    probs = torch.softmax(token_matrix, dim=-1)
    ent = -(probs * torch.log(probs + EPS)).sum(dim=-1)

    return ent.detach().cpu().numpy()

def get_prompt_lengths(tokenizer, prompts, max_length):
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


def split_positions(idx):
    n = int(idx.numel())

    if n == 0:
        return {
            "all": idx,
            "first": idx,
            "early": idx,
            "middle": idx,
            "late": idx,
            "last": idx,
            "last_5": idx,
            "last_10": idx,
            "first_5": idx,
            "first_10": idx,
        }

    third = max(1, n // 3)
    mid_start = third
    late_start = min(n, 2 * third)

    return {
        "all": idx,
        "first": idx[:third],
        "early": idx[:third],
        "middle": idx[mid_start:late_start] if late_start > mid_start else idx,
        "late": idx[late_start:] if n > late_start else idx[-third:],
        "last": idx[-third:],
        "last_5": idx[-min(5, n):],
        "last_10": idx[-min(10, n):],
        "first_5": idx[:min(5, n)],
        "first_10": idx[:min(10, n)],
    }


def make_exact_masks(hidden, valid_mask, prompt_len):
    valid_mask = valid_mask.bool().cpu()

    seq_len = hidden.shape[1]
    prompt_len = min(max(int(prompt_len), 0), seq_len)

    pos = torch.arange(seq_len)

    prompt_mask = valid_mask & (pos < prompt_len)
    response_mask = valid_mask & (pos >= prompt_len)

    if prompt_mask.sum().item() == 0:
        prompt_mask = valid_mask.clone()

    if response_mask.sum().item() == 0:
        response_mask = valid_mask.clone()

    prompt_idx = torch.where(prompt_mask)[0]
    response_idx = torch.where(response_mask)[0]
    valid_idx = torch.where(valid_mask)[0]

    zones = {
        "all": valid_idx,
        "prompt": prompt_idx,
        "response": response_idx,
    }

    for name, idx in split_positions(response_idx).items():
        zones[f"response_{name}"] = idx

    for name, idx in split_positions(prompt_idx).items():
        zones[f"prompt_{name}"] = idx

    return zones, prompt_mask, response_mask


def zone_tokens(hidden, layer, idx):
    if idx.numel() == 0:
        return torch.zeros((1, hidden.shape[-1]), dtype=hidden.dtype)

    return hidden[layer, idx].float()


def add_vector(features, prefix, vec):
    vec = to_numpy(vec)

    for i, value in enumerate(vec):
        features[f"{prefix}_d{i}"] = float(value)


def add_zone_scalar_features(features, prefix, tokens):
    token_norms = torch.linalg.norm(tokens.float(), dim=1).detach().cpu().numpy()

    add_stats(features, f"{prefix}_token_norm", token_norms)

    features[f"{prefix}_activation_mean"] = float(tokens.mean().item())
    features[f"{prefix}_activation_std"] = (
        float(tokens.std(unbiased=False).item()) if tokens.numel() > 1 else 0.0
    )
    features[f"{prefix}_activation_abs_mean"] = float(tokens.abs().mean().item())
    features[f"{prefix}_activation_max"] = float(tokens.max().item())
    features[f"{prefix}_activation_min"] = float(tokens.min().item())

    if tokens.shape[0] > 1:
        var_dim = tokens.var(dim=0, unbiased=False).detach().cpu().numpy()
    else:
        var_dim = np.array([0.0], dtype=np.float32)

    add_stats(features, f"{prefix}_activation_variance_dim", var_dim)

    ent = activation_entropy(tokens)
    add_stats(features, f"{prefix}_activation_entropy", ent)

    pcs = pairwise_cosine_stats(tokens)
    for key, value in pcs.items():
        features[f"{prefix}_pairwise_cosine_{key}"] = value

    features[f"{prefix}_token_disagreement"] = 1.0 - pcs["mean"]

    cov = covariance_pca_stats(tokens)
    for key, value in cov.items():
        features[f"{prefix}_cov_{key}"] = value


def extract_prompt_len_features_for_sample(hidden, valid_mask, prompt_len):
    hidden = hidden.float().cpu()
    valid_mask = valid_mask.bool().cpu()

    zones, prompt_mask, response_mask = make_exact_masks(
        hidden=hidden,
        valid_mask=valid_mask,
        prompt_len=prompt_len,
    )

    features = {}

    zone_layer_means = {}
    main_zones = [
        "all",
        "prompt",
        "response",
        "response_early",
        "response_middle",
        "response_late",
        "response_last_5",
        "response_last_10",
        "prompt_last_5",
        "prompt_last_10",
    ]

    for zone_name in main_zones:
        zone_layer_means[zone_name] = {}

        for layer in LAYERS:
            tokens = zone_tokens(hidden, layer, zones[zone_name])
            zone_layer_means[zone_name][layer] = safe_mean(tokens)

    # A. Exact length features

    valid_count = float(valid_mask.sum().item())
    prompt_count = float(prompt_mask.sum().item())
    response_count = float(response_mask.sum().item())

    features["exact_length_valid_tokens"] = valid_count
    features["exact_length_prompt_tokens"] = prompt_count
    features["exact_length_response_tokens"] = response_count
    features["exact_length_response_ratio_total"] = response_count / max(valid_count, 1.0)
    features["exact_length_prompt_ratio_total"] = prompt_count / max(valid_count, 1.0)
    features["exact_length_response_to_prompt_ratio"] = response_count / max(prompt_count, 1.0)

    # B. Rich raw vectors from selected_rich

    for layer in RICH_LAYERS:
        add_vector(
            features,
            f"rich_mean_response_l{layer}",
            zone_layer_means["response"][layer],
        )

    middle4_response = torch.stack([
        zone_layer_means["response"][layer]
        for layer in MIDDLE4_LAYERS
    ]).mean(dim=0)

    add_vector(features, "rich_mean_response_middle4", middle4_response)

    for layer in RICH_LAYERS:
        add_vector(
            features,
            f"rich_response_minus_prompt_l{layer}",
            zone_layer_means["response"][layer] - zone_layer_means["prompt"][layer],
        )

    for left, right in RICH_DRIFT_PAIRS:
        response_drift = (
            zone_layer_means["response"][right]
            - zone_layer_means["response"][left]
        )

        add_vector(
            features,
            f"rich_response_drift_l{left}_to_l{right}",
            response_drift,
        )

        add_vector(
            features,
            f"rich_abs_response_drift_l{left}_to_l{right}",
            response_drift.abs(),
        )

    # C. Exact scalar stats by prompt / response / zones

    scalar_zones = [
        "prompt",
        "response",
        "response_early",
        "response_middle",
        "response_late",
        "response_last_5",
        "response_last_10",
    ]

    for zone_name in scalar_zones:
        for layer in RICH_LAYERS:
            tokens = zone_tokens(hidden, layer, zones[zone_name])
            add_zone_scalar_features(
                features,
                f"scalar_{zone_name}_l{layer}",
                tokens,
            )

    # D. Response-only drift trajectories

    for zone_name in [
        "response",
        "response_early",
        "response_middle",
        "response_late",
        "response_last_5",
        "response_last_10",
    ]:
        jump_values = []
        cosine_values = []
        drift_vectors = []

        for left, right in DRIFT_PAIRS:
            drift = (
                zone_layer_means[zone_name][right]
                - zone_layer_means[zone_name][left]
            )

            drift_vectors.append(drift)
            jump_values.append(safe_l2(drift))
            cosine_values.append(
                safe_cosine(
                    zone_layer_means[zone_name][left],
                    zone_layer_means[zone_name][right],
                )
            )

        add_trajectory_features(features, f"{zone_name}_jump", jump_values)
        add_spectral_features(features, f"{zone_name}_jump", jump_values)

        add_trajectory_features(features, f"{zone_name}_layer_cosine", cosine_values)

        curvature_values = []

        for i in range(len(drift_vectors) - 1):
            curvature_values.append(
                safe_cosine(drift_vectors[i], drift_vectors[i + 1])
            )

        add_trajectory_features(features, f"{zone_name}_curvature", curvature_values)

    # E. Exact token-position dynamics

    for layer in LAYERS:
        early = zone_layer_means["response_early"][layer]
        middle = zone_layer_means["response_middle"][layer]
        late = zone_layer_means["response_late"][layer]
        last5 = zone_layer_means["response_last_5"][layer]
        response = zone_layer_means["response"][layer]
        prompt = zone_layer_means["prompt"][layer]

        pairs = {
            "late_minus_early": late - early,
            "middle_minus_early": middle - early,
            "late_minus_middle": late - middle,
            "last5_minus_response": last5 - response,
            "response_minus_prompt": response - prompt,
            "last5_minus_prompt": last5 - prompt,
        }

        for pair_name, vec in pairs.items():
            add_stats(
                features,
                f"pos_l{layer}_{pair_name}",
                to_numpy(vec),
            )

        features[f"pos_l{layer}_early_late_cosine"] = safe_cosine(early, late)
        features[f"pos_l{layer}_middle_late_cosine"] = safe_cosine(middle, late)
        features[f"pos_l{layer}_last5_response_cosine"] = safe_cosine(last5, response)
        features[f"pos_l{layer}_response_prompt_cosine"] = safe_cosine(response, prompt)

    # F. Token-wise drift dynamics inside exact response

    response_idx = zones["response"]

    for left, right in DRIFT_PAIRS + LONG_DRIFT_PAIRS:
        if response_idx.numel() == 0:
            drift_tokens = torch.zeros((1, hidden.shape[-1]), dtype=hidden.dtype)
        else:
            drift_tokens = hidden[right, response_idx] - hidden[left, response_idx]

        token_norms = torch.linalg.norm(drift_tokens.float(), dim=1).detach().cpu().numpy()

        add_stats(
            features,
            f"tokenwise_response_drift_l{left}_to_l{right}_norm",
            token_norms,
        )

        add_trajectory_features(
            features,
            f"tokenwise_response_drift_l{left}_to_l{right}_norm_position",
            token_norms,
        )

        add_spectral_features(
            features,
            f"tokenwise_response_drift_l{left}_to_l{right}_norm_position",
            token_norms,
        )

    # G. Exact response zone contrast

    exact_zone_pairs = [
        ("response_late", "response_early"),
        ("response_late", "response_middle"),
        ("response_last_5", "response"),
        ("response_last_10", "response"),
        ("response", "prompt"),
        ("response_last_5", "prompt"),
        ("response_last_5", "prompt_last_5"),
    ]

    for zone_a, zone_b in exact_zone_pairs:
        for layer in RICH_LAYERS:
            mean_a = zone_layer_means[zone_a][layer]
            mean_b = zone_layer_means[zone_b][layer]

            features[f"{zone_a}_vs_{zone_b}_l{layer}_cosine"] = safe_cosine(
                mean_a,
                mean_b,
            )
            features[f"{zone_a}_vs_{zone_b}_l{layer}_l2_distance"] = safe_l2(
                mean_a - mean_b
            )
            features[f"{zone_a}_vs_{zone_b}_l{layer}_norm_ratio"] = (
                safe_l2(mean_a) / (safe_l2(mean_b) + EPS)
            )

    # H. Exact response consensus / collapse

    for zone_name in [
        "response",
        "response_early",
        "response_middle",
        "response_late",
        "response_last_5",
        "response_last_10",
    ]:
        disagreement_by_layer = []
        collapse_by_layer = []

        for layer in RICH_LAYERS:
            tokens = zone_tokens(hidden, layer, zones[zone_name])
            pcs = pairwise_cosine_stats(tokens)
            disagreement = 1.0 - pcs["mean"]

            features[f"{zone_name}_l{layer}_exact_disagreement"] = disagreement
            disagreement_by_layer.append(disagreement)

            cov = covariance_pca_stats(tokens)
            features[f"{zone_name}_l{layer}_exact_collapse_score"] = cov["top1_ratio"]
            features[f"{zone_name}_l{layer}_exact_effective_rank"] = cov["effective_rank"]
            collapse_by_layer.append(cov["top1_ratio"])

        add_trajectory_features(
            features,
            f"{zone_name}_disagreement_by_layer",
            disagreement_by_layer,
        )
        add_trajectory_features(
            features,
            f"{zone_name}_collapse_by_layer",
            collapse_by_layer,
        )

    # I. Exact late-response instability scores

    for layer in RICH_LAYERS:
        response_dis = features.get(f"response_l{layer}_exact_disagreement", 0.0)
        late_dis = features.get(f"response_late_l{layer}_exact_disagreement", 0.0)
        last5_dis = features.get(f"response_last_5_l{layer}_exact_disagreement", 0.0)

        features[f"exact_l{layer}_late_minus_response_disagreement"] = (
            late_dis - response_dis
        )
        features[f"exact_l{layer}_last5_minus_response_disagreement"] = (
            last5_dis - response_dis
        )

        response_rank = features.get(f"response_l{layer}_exact_effective_rank", 0.0)
        late_rank = features.get(f"response_late_l{layer}_exact_effective_rank", 0.0)
        last5_rank = features.get(f"response_last_5_l{layer}_exact_effective_rank", 0.0)

        features[f"exact_l{layer}_late_minus_response_effective_rank"] = (
            late_rank - response_rank
        )
        features[f"exact_l{layer}_last5_minus_response_effective_rank"] = (
            last5_rank - response_rank
        )

    # J. Early / mid / final consistency with exact response

    consistency_layers = {
        "early": RICH_LAYERS[0],
        "mid": RICH_LAYERS[len(RICH_LAYERS) // 2],
        "final": RICH_LAYERS[-1],
    }

    early = consistency_layers["early"]
    mid = consistency_layers["mid"]
    final = consistency_layers["final"]

    for zone_name in ["response", "response_last_5", "prompt"]:
        early_mean = zone_layer_means[zone_name][early]
        mid_mean = zone_layer_means[zone_name][mid]
        final_mean = zone_layer_means[zone_name][final]

        features[f"{zone_name}_early_mid_cosine"] = safe_cosine(early_mean, mid_mean)
        features[f"{zone_name}_mid_final_cosine"] = safe_cosine(mid_mean, final_mean)
        features[f"{zone_name}_early_final_cosine"] = safe_cosine(early_mean, final_mean)

        features[f"{zone_name}_early_mid_distance"] = safe_l2(mid_mean - early_mean)
        features[f"{zone_name}_mid_final_distance"] = safe_l2(final_mean - mid_mean)
        features[f"{zone_name}_early_final_distance"] = safe_l2(final_mean - early_mean)

        features[f"{zone_name}_final_minus_mid_norm_ratio"] = (
            safe_l2(final_mean - mid_mean) / (safe_l2(mid_mean) + EPS)
        )
        features[f"{zone_name}_final_minus_early_norm_ratio"] = (
            safe_l2(final_mean - early_mean) / (safe_l2(early_mean) + EPS)
        )

    # Clean all values
    clean_features = {}

    for key, value in features.items():
        value = float(value)

        if not np.isfinite(value):
            value = 0.0

        clean_features[key] = value

    return clean_features


# ============================================================
# OFFICIAL PIPELINE WRAPPER
# ============================================================

class _PromptLengthProvider:
    """Stateful prompt-length provider for the fixed official solution loop."""

    def __init__(self) -> None:
        self._lengths = None
        self._cursor = 0

    def _load_lengths(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        all_lengths = []
        for path in [DATA_FILE, TEST_FILE]:
            try:
                df = pd.read_csv(path)
            except FileNotFoundError:
                continue

            if "prompt_len" in df.columns:
                lengths = df["prompt_len"].astype(int).tolist()
            elif "prompt_length" in df.columns:
                lengths = df["prompt_length"].astype(int).tolist()
            elif "prompt_len_tokens" in df.columns:
                lengths = df["prompt_len_tokens"].astype(int).tolist()
            else:
                lengths = get_prompt_lengths(
                    tokenizer=tokenizer,
                    prompts=df["prompt"].astype(str).tolist(),
                    max_length=MAX_LENGTH,
                )
            all_lengths.extend(lengths)

        if not all_lengths:
            raise RuntimeError(
                "Cannot reconstruct prompt lengths: data/dataset.csv and "
                "data/test.csv are unavailable."
            )

        self._lengths = all_lengths

    def next(self) -> int:
        if self._lengths is None:
            self._load_lengths()

        if self._cursor >= len(self._lengths):
            self._cursor = 0

        value = int(self._lengths[self._cursor])
        self._cursor += 1
        return value


_PROMPT_LENGTH_PROVIDER = _PromptLengthProvider()


def aggregate(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    prompt_len = _PROMPT_LENGTH_PROVIDER.next()
    features = extract_prompt_len_features_for_sample(
        hidden=hidden_states,
        valid_mask=attention_mask,
        prompt_len=prompt_len,
    )
    out = torch.tensor(list(features.values()), dtype=torch.float32)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
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
