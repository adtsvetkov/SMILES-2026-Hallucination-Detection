"""
Build selected rich hidden-state features for hallucination detection.

This script is intentionally separate from solution.py.
It creates parquet datasets for experimentation in notebooks.

Selected feature groups are based on diagnostics:
1. mean_response layers 11-16 concat
2. mean_response middle4 mean
3. response_minus_prompt layers 11-16 concat
4. response drift layers 11->12 ... 15->16
5. abs response drift layers 11->12 ... 15->16
6. scalar norm/stat features for layers 11-16
7. response length / response ratio scalar features
8. quantile features for response tokens, prompt/response norms, and response drifts
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model import MAX_LENGTH, get_model_and_tokenizer

# ============================================================
# CONFIG — change paths here if needed
# ============================================================

DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"
OUTPUT_DIR = "./artifacts/selected_rich_features"

DATASET_OUTPUT = "features_dataset_selected_rich_quantiles.parquet"
TEST_OUTPUT = "features_test_selected_rich_quantiles.parquet"

BATCH_SIZE = 4
EXPORT_TEST = True

# Hidden-state indices. In transformers, hidden_states[0] is embeddings,
# hidden_states[1:] are transformer layers. Diagnostics used these same indices.
SELECTED_LAYERS = [11, 12, 13, 14, 15, 16]
MIDDLE4_LAYERS = [11, 12, 13, 14]
DRIFT_PAIRS = [(11, 12), (12, 13), (13, 14), (14, 15), (15, 16)]

EPS = 1e-8


# ============================================================
# DEVICE
# ============================================================


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# MASK HELPERS
# ============================================================


def get_prompt_lengths(
    tokenizer,
    prompts: List[str],
    max_length: int,
) -> List[int]:
    """Tokenize prompts only to estimate where response tokens start."""
    lengths = []
    for prompt in prompts:
        enc = tokenizer(
            prompt,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        lengths.append(len(enc["input_ids"]))
    return lengths


def safe_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over token dimension using a boolean mask."""
    if mask.sum().item() == 0:
        return torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
    return x[mask].mean(dim=0)


def safe_std(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Std over token dimension using a boolean mask."""
    if mask.sum().item() <= 1:
        return torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
    return x[mask].std(dim=0, unbiased=False)


def safe_quantile(x: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    """Quantile over token dimension using a boolean mask."""
    if mask.sum().item() == 0:
        return torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
    return torch.quantile(x[mask].float(), q=q, dim=0)


def safe_quantile_block(x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
    """Return q25, q50, q75 and IQR vectors over token dimension."""
    q25 = safe_quantile(x, mask, 0.25)
    q50 = safe_quantile(x, mask, 0.50)
    q75 = safe_quantile(x, mask, 0.75)
    iqr = q75 - q25
    return torch.cat([q25, q50, q75, iqr]), ["q25", "q50", "q75", "iqr"]


def scalar_quantile_stats_for_zone(
    layer_hidden: torch.Tensor,
    zone_mask: torch.Tensor,
) -> List[float]:
    """Compact scalar quantile diagnostics for one layer and one token zone."""
    if zone_mask.sum().item() == 0:
        return [0.0] * 8

    z = layer_hidden[zone_mask].float()
    token_l2 = torch.linalg.norm(z, dim=1)

    if token_l2.numel() == 0 or z.numel() == 0:
        return [0.0] * 8

    token_l2_q25 = torch.quantile(token_l2, 0.25)
    token_l2_q50 = torch.quantile(token_l2, 0.50)
    token_l2_q75 = torch.quantile(token_l2, 0.75)

    z_flat = z.reshape(-1)
    activation_q25 = torch.quantile(z_flat, 0.25)
    activation_q50 = torch.quantile(z_flat, 0.50)
    activation_q75 = torch.quantile(z_flat, 0.75)

    return [
        float(token_l2_q25.cpu()),
        float(token_l2_q50.cpu()),
        float(token_l2_q75.cpu()),
        float((token_l2_q75 - token_l2_q25).cpu()),
        float(activation_q25.cpu()),
        float(activation_q50.cpu()),
        float(activation_q75.cpu()),
        float((activation_q75 - activation_q25).cpu()),
    ]


def scalar_stats_for_zone(
    layer_hidden: torch.Tensor,
    zone_mask: torch.Tensor,
) -> List[float]:
    """Compact scalar diagnostics for one layer and one token zone."""
    if zone_mask.sum().item() == 0:
        return [0.0] * 8

    z = layer_hidden[zone_mask].float()
    token_l2 = torch.linalg.norm(z, dim=1)

    return [
        float(token_l2.mean().cpu()),
        float(token_l2.std(unbiased=False).cpu()) if token_l2.numel() > 1 else 0.0,
        float(z.mean().cpu()),
        float(z.std(unbiased=False).cpu()) if z.numel() > 1 else 0.0,
        float(z.abs().mean().cpu()),
        float(z.var(dim=0, unbiased=False).mean().cpu()) if z.shape[0] > 1 else 0.0,
        float(z.max().cpu()),
        float(z.min().cpu()),
    ]


# ============================================================
# FEATURE EXTRACTION FOR ONE SAMPLE
# ============================================================


def extract_selected_features_for_sample(
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
    prompt_len: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    hidden shape: (n_layers, seq_len, hidden_dim)
    valid_mask shape: (seq_len,)
    """
    hidden = hidden.float().cpu()
    valid_mask = valid_mask.bool().cpu()

    seq_len = hidden.shape[1]
    prompt_len = min(max(int(prompt_len), 0), seq_len)

    position_ids = torch.arange(seq_len)
    prompt_mask = valid_mask & (position_ids < prompt_len)
    response_mask = valid_mask & (position_ids >= prompt_len)

    # Fallback if truncation removed the response.
    if response_mask.sum().item() == 0:
        response_mask = valid_mask.clone()

    vectors: List[torch.Tensor] = []
    names: List[str] = []

    # Precompute means.
    mean_response: Dict[int, torch.Tensor] = {}
    mean_prompt: Dict[int, torch.Tensor] = {}

    for layer in SELECTED_LAYERS:
        mean_response[layer] = safe_mean(hidden[layer], response_mask)
        mean_prompt[layer] = safe_mean(hidden[layer], prompt_mask)

    # 1. mean_response layers 11-16 concat.
    for layer in SELECTED_LAYERS:
        vec = mean_response[layer]
        vectors.append(vec)
        names.extend([f"mean_response_l{layer}_d{d}" for d in range(vec.numel())])

    # 2. mean_response middle4 mean.
    middle4_mean = torch.stack([mean_response[layer] for layer in MIDDLE4_LAYERS]).mean(dim=0)
    vectors.append(middle4_mean)
    names.extend([f"mean_response_middle4_mean_d{d}" for d in range(middle4_mean.numel())])

    # 3. response_minus_prompt layers 11-16 concat.
    for layer in SELECTED_LAYERS:
        vec = mean_response[layer] - mean_prompt[layer]
        vectors.append(vec)
        names.extend([f"response_minus_prompt_l{layer}_d{d}" for d in range(vec.numel())])

    # 4. response drift layers 11->12 ... 15->16.
    for left, right in DRIFT_PAIRS:
        vec = mean_response[right] - mean_response[left]
        vectors.append(vec)
        names.extend([f"response_drift_l{left}_to_l{right}_d{d}" for d in range(vec.numel())])

    # 5. abs response drift.
    for left, right in DRIFT_PAIRS:
        vec = (mean_response[right] - mean_response[left]).abs()
        vectors.append(vec)
        names.extend([f"abs_response_drift_l{left}_to_l{right}_d{d}" for d in range(vec.numel())])

    # 6. scalar norm/stat features for layers 11-16.
    scalar_values: List[float] = []
    scalar_names: List[str] = []
    stat_names = [
        "token_l2_mean",
        "token_l2_std",
        "activation_mean",
        "activation_std",
        "activation_abs_mean",
        "feature_variance_mean",
        "activation_max",
        "activation_min",
    ]

    zones = {
        "all": valid_mask,
        "prompt": prompt_mask,
        "response": response_mask,
    }

    for layer in SELECTED_LAYERS:
        for zone_name, zone_mask in zones.items():
            stats = scalar_stats_for_zone(hidden[layer], zone_mask)
            scalar_values.extend(stats)
            scalar_names.extend([f"scalar_l{layer}_{zone_name}_{s}" for s in stat_names])

    # Additional scalar drift statistics.
    for left, right in DRIFT_PAIRS:
        diff = hidden[right] - hidden[left]
        for zone_name, zone_mask in zones.items():
            stats = scalar_stats_for_zone(diff, zone_mask)
            scalar_values.extend(stats)
            scalar_names.extend([
                f"scalar_drift_l{left}_to_l{right}_{zone_name}_{s}" for s in stat_names
            ])

    # 7. quantile vector features for response tokens in layers 11-16.
    # These capture distribution shape over response tokens, not only the mean.
    for layer in SELECTED_LAYERS:
        quantile_vec, quantile_names = safe_quantile_block(hidden[layer], response_mask)
        vectors.append(quantile_vec)
        hidden_dim = hidden.shape[-1]
        for q_name in quantile_names:
            names.extend([f"response_quantile_{q_name}_l{layer}_d{d}" for d in range(hidden_dim)])

    # 8. quantile vector features for response drift layers 11->12 ... 15->16.
    for left, right in DRIFT_PAIRS:
        diff = hidden[right] - hidden[left]
        quantile_vec, quantile_names = safe_quantile_block(diff, response_mask)
        vectors.append(quantile_vec)
        hidden_dim = hidden.shape[-1]
        for q_name in quantile_names:
            names.extend([
                f"response_drift_quantile_{q_name}_l{left}_to_l{right}_d{d}"
                for d in range(hidden_dim)
            ])

    # 9. scalar quantile features for layers 11-16 and drift pairs.
    quantile_scalar_names = [
        "token_l2_q25",
        "token_l2_q50",
        "token_l2_q75",
        "token_l2_iqr",
        "activation_q25",
        "activation_q50",
        "activation_q75",
        "activation_iqr",
    ]

    for layer in SELECTED_LAYERS:
        for zone_name, zone_mask in zones.items():
            stats = scalar_quantile_stats_for_zone(hidden[layer], zone_mask)
            scalar_values.extend(stats)
            scalar_names.extend([
                f"scalar_quantile_l{layer}_{zone_name}_{s}"
                for s in quantile_scalar_names
            ])

    for left, right in DRIFT_PAIRS:
        diff = hidden[right] - hidden[left]
        for zone_name, zone_mask in zones.items():
            stats = scalar_quantile_stats_for_zone(diff, zone_mask)
            scalar_values.extend(stats)
            scalar_names.extend([
                f"scalar_quantile_drift_l{left}_to_l{right}_{zone_name}_{s}"
                for s in quantile_scalar_names
            ])

    # 10. response length / response ratio.
    valid_count = float(valid_mask.sum().item())
    prompt_count = float(prompt_mask.sum().item())
    response_count = float(response_mask.sum().item())
    scalar_values.extend([
        valid_count,
        prompt_count,
        response_count,
        response_count / max(valid_count, 1.0),
        prompt_count / max(valid_count, 1.0),
        response_count / max(prompt_count, 1.0),
    ])
    scalar_names.extend([
        "length_valid_tokens",
        "length_prompt_tokens",
        "length_response_tokens",
        "length_response_ratio_total",
        "length_prompt_ratio_total",
        "length_response_to_prompt_ratio",
    ])

    scalar_tensor = torch.tensor(scalar_values, dtype=torch.float32)
    vectors.append(scalar_tensor)
    names.extend(scalar_names)

    feature_vector = torch.cat(vectors).numpy().astype(np.float32)
    return feature_vector, names


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
    texts = [p + r for p, r in zip(prompts, responses)]
    prompt_lengths = get_prompt_lengths(tokenizer, prompts, MAX_LENGTH)

    all_features: List[np.ndarray] = []
    feature_names: Optional[List[str]] = None

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extract selected rich features"):
        batch_texts = texts[start : start + BATCH_SIZE]
        batch_prompt_lengths = prompt_lengths[start : start + BATCH_SIZE]

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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden_batch = torch.stack(outputs.hidden_states, dim=1).float().cpu()
        mask_batch = attention_mask.cpu().bool()

        for i in range(hidden_batch.shape[0]):
            features, names = extract_selected_features_for_sample(
                hidden=hidden_batch[i],
                valid_mask=mask_batch[i],
                prompt_len=batch_prompt_lengths[i],
            )
            all_features.append(features)
            if feature_names is None:
                feature_names = names

    assert feature_names is not None
    X = np.vstack(all_features)

    out = pd.DataFrame(X, columns=feature_names)
    out.insert(0, "source_index", df.index.to_numpy())

    if has_label:
        out["label"] = df["label"].astype(float).astype(int).to_numpy()

    # Keep raw text for debugging/EDA. Drop these later before modeling.
    out["prompt"] = df["prompt"].astype(str).to_numpy()
    out["response"] = df["response"].astype(str).to_numpy()

    return out


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("=" * 80)
    print("BUILD SELECTED RICH FEATURES V2 WITH QUANTILES")
    print("=" * 80)
    print(f"Device      : {device}")
    print(f"Data file   : {DATA_FILE}")
    print(f"Test file   : {TEST_FILE}")
    print(f"Output dir  : {output_dir}")
    print(f"Layers      : {SELECTED_LAYERS}")
    print(f"Batch size  : {BATCH_SIZE}")

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()

    t0 = time.time()

    df = pd.read_csv(DATA_FILE)
    print(f"\nDataset rows: {len(df)}")
    dataset_features = extract_dataset_features(
        df=df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        has_label=True,
    )
    dataset_path = output_dir / DATASET_OUTPUT
    dataset_features.to_parquet(dataset_path, index=False)
    print(f"Saved dataset features: {dataset_path}")
    print(f"Dataset feature table shape: {dataset_features.shape}")

    if EXPORT_TEST and Path(TEST_FILE).exists():
        df_test = pd.read_csv(TEST_FILE)
        print(f"\nTest rows: {len(df_test)}")
        test_features = extract_dataset_features(
            df=df_test,
            model=model,
            tokenizer=tokenizer,
            device=device,
            has_label=False,
        )
        test_path = output_dir / TEST_OUTPUT
        test_features.to_parquet(test_path, index=False)
        print(f"Saved test features: {test_path}")
        print(f"Test feature table shape: {test_features.shape}")

    print(f"\nDone in {(time.time() - t0):.1f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
