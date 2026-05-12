"""
Diagnostic hidden-state analysis for SMILES-2026 Hallucination Detection.

Run this file from the repository root by pressing Run in your IDE.
It does NOT modify solution.py / aggregation.py / probe.py / splitting.py.

What it does:
1. Loads dataset.csv.
2. Runs Qwen once to obtain hidden states.
3. Builds lightweight diagnostic feature blocks from layers / token zones.
4. Computes probe-independent signal metrics:
   - Pearson correlation with label
   - univariate AUROC, orientation-invariant
   - Cohen's d effect size
   - optional mutual information on top dimensions
   - unsupervised PCA-1 separability
5. Saves CSV reports and a text summary that you can paste back into ChatGPT.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.feature_selection import mutual_info_classif
    HAS_MI = True
except Exception:
    HAS_MI = False

from model import MAX_LENGTH, get_model_and_tokenizer


# ============================================================
# CONFIG — change paths/flags here manually if needed
# ============================================================

DATA_FILE = "./data/dataset.csv"
OUTPUT_DIR = "./artifacts/diagnostics_hidden_states"

BATCH_SIZE = 4
RANDOM_STATE = 42

# If True, mutual information is computed for top dimensions of each vector block.
# This is useful but can make the script slower.
COMPUTE_MUTUAL_INFO = True
MAX_MI_DIMS_PER_BLOCK = 150

# Keep vector blocks that are useful for diagnostics but avoid producing enormous outputs.
TOP_FEATURES_PER_BLOCK_TO_SAVE = 20
TOP_BLOCKS_TO_PRINT = 40
TOP_SCALARS_TO_PRINT = 40
TOP_FEATURES_TO_PRINT = 80

# Optional 2D PCA coordinates for selected top blocks are saved for plotting later.
SAVE_PCA_COORDINATES_FOR_TOP_N_BLOCKS = 12

EPS = 1e-8


# ============================================================
# Utility functions
# ============================================================


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_mean(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if x.numel() == 0 or x.shape[dim] == 0:
        shape = list(x.shape)
        del shape[dim]
        return torch.zeros(shape, dtype=x.dtype, device=x.device)
    return x.mean(dim=dim)


def safe_std(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if x.numel() == 0 or x.shape[dim] <= 1:
        shape = list(x.shape)
        del shape[dim]
        return torch.zeros(shape, dtype=x.dtype, device=x.device)
    return x.std(dim=dim, unbiased=False)


def safe_last(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if x.numel() == 0 or x.shape[dim] == 0:
        shape = list(x.shape)
        del shape[dim]
        return torch.zeros(shape, dtype=x.dtype, device=x.device)
    index = [slice(None)] * x.ndim
    index[dim] = -1
    return x[tuple(index)]


def contiguous_bucket_indices(start: int, end: int, bucket: str) -> np.ndarray:
    """Return token indices for first/middle/last part of [start, end)."""
    length = max(0, end - start)
    if length == 0:
        return np.array([], dtype=np.int64)

    a = start
    b = end
    one_third = max(1, length // 3)

    if bucket == "first":
        return np.arange(a, min(b, a + one_third), dtype=np.int64)
    if bucket == "middle":
        mid_start = a + length // 3
        mid_end = a + 2 * length // 3
        if mid_end <= mid_start:
            mid_end = min(b, mid_start + 1)
        return np.arange(mid_start, mid_end, dtype=np.int64)
    if bucket == "last":
        return np.arange(max(a, b - one_third), b, dtype=np.int64)
    raise ValueError(f"Unknown bucket: {bucket}")


def pearson_corr_vector(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X_centered = X - X.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    numerator = (X_centered * y_centered[:, None]).sum(axis=0)
    denominator = np.sqrt((X_centered ** 2).sum(axis=0) * (y_centered ** 2).sum())
    corr = numerator / (denominator + EPS)
    corr[~np.isfinite(corr)] = 0.0
    return corr


def cohen_d_vector(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X0 = X[y == 0]
    X1 = X[y == 1]
    if len(X0) < 2 or len(X1) < 2:
        return np.zeros(X.shape[1], dtype=np.float64)
    mean0 = X0.mean(axis=0)
    mean1 = X1.mean(axis=0)
    var0 = X0.var(axis=0, ddof=1)
    var1 = X1.var(axis=0, ddof=1)
    pooled = np.sqrt(((len(X0) - 1) * var0 + (len(X1) - 1) * var1) / max(1, len(X0) + len(X1) - 2))
    d = (mean1 - mean0) / (pooled + EPS)
    d[~np.isfinite(d)] = 0.0
    return d


def auc_vector(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    aucs = np.zeros(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.nanstd(col) < EPS:
            aucs[j] = 0.5
            continue
        try:
            auc = roc_auc_score(y, col)
            aucs[j] = max(auc, 1.0 - auc)  # orientation-invariant separability
        except Exception:
            aucs[j] = 0.5
    return aucs


def pca_pc1_auc(X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return PC1 oriented AUROC, explained variance ratio PC1, PC2."""
    try:
        X_scaled = StandardScaler().fit_transform(X)
        n_components = 2 if X.shape[1] >= 2 else 1
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        coords = pca.fit_transform(X_scaled)
        auc = roc_auc_score(y, coords[:, 0])
        auc = max(auc, 1.0 - auc)
        evr1 = float(pca.explained_variance_ratio_[0])
        evr2 = float(pca.explained_variance_ratio_[1]) if n_components > 1 else 0.0
        return float(auc), evr1, evr2
    except Exception:
        return 0.5, 0.0, 0.0


def summarize_vector_block(block_name: str, X: np.ndarray, y: np.ndarray) -> Tuple[dict, pd.DataFrame]:
    """Compute probe-independent metrics for a feature block."""
    X = np.asarray(X, dtype=np.float32)
    corr = pearson_corr_vector(X, y)
    abs_corr = np.abs(corr)
    d = cohen_d_vector(X, y)
    abs_d = np.abs(d)
    aucs = auc_vector(X, y)
    auc_delta = np.abs(aucs - 0.5)

    mi_mean = np.nan
    mi_max = np.nan
    mi_top20_mean = np.nan

    if COMPUTE_MUTUAL_INFO and HAS_MI and X.shape[1] > 0:
        # Compute MI only on dimensions that already look somewhat promising by correlation.
        mi_dim_count = min(MAX_MI_DIMS_PER_BLOCK, X.shape[1])
        candidate_dims = np.argsort(abs_corr)[-mi_dim_count:]
        try:
            mi_values = mutual_info_classif(
                X[:, candidate_dims],
                y,
                discrete_features=False,
                random_state=RANDOM_STATE,
            )
            mi_mean = float(np.mean(mi_values))
            mi_max = float(np.max(mi_values))
            mi_top20_mean = float(np.mean(np.sort(mi_values)[-min(20, len(mi_values)):]))
        except Exception:
            pass

    pca_auc, pca_evr1, pca_evr2 = pca_pc1_auc(X, y)

    def top_mean(values: np.ndarray, k: int) -> float:
        k = min(k, len(values))
        if k <= 0:
            return 0.0
        return float(np.mean(np.sort(values)[-k:]))

    summary = {
        "block": block_name,
        "n_features": int(X.shape[1]),
        "mean_abs_corr": float(np.mean(abs_corr)),
        "max_abs_corr": float(np.max(abs_corr)),
        "top20_abs_corr_mean": top_mean(abs_corr, 20),
        "top50_abs_corr_mean": top_mean(abs_corr, 50),
        "mean_auc": float(np.mean(aucs)),
        "max_auc": float(np.max(aucs)),
        "top20_auc_mean": top_mean(aucs, 20),
        "top50_auc_mean": top_mean(aucs, 50),
        "mean_auc_delta": float(np.mean(auc_delta)),
        "max_auc_delta": float(np.max(auc_delta)),
        "top20_auc_delta_mean": top_mean(auc_delta, 20),
        "mean_abs_cohen_d": float(np.mean(abs_d)),
        "max_abs_cohen_d": float(np.max(abs_d)),
        "top20_abs_cohen_d_mean": top_mean(abs_d, 20),
        "mi_mean_top_dims": mi_mean,
        "mi_max_top_dims": mi_max,
        "mi_top20_mean_top_dims": mi_top20_mean,
        "pca_pc1_auc": pca_auc,
        "pca_evr1": pca_evr1,
        "pca_evr2": pca_evr2,
    }

    # A single combined heuristic score for ranking. It is not a model metric.
    summary["diagnostic_score"] = float(
        0.35 * summary["top20_auc_delta_mean"]
        + 0.25 * summary["top20_abs_corr_mean"]
        + 0.25 * min(summary["top20_abs_cohen_d_mean"] / 2.0, 1.0)
        + 0.15 * abs(summary["pca_pc1_auc"] - 0.5)
    )

    top_k = min(TOP_FEATURES_PER_BLOCK_TO_SAVE, X.shape[1])
    top_dims = np.argsort(
        0.45 * auc_delta + 0.30 * abs_corr + 0.25 * np.minimum(abs_d / 2.0, 1.0)
    )[-top_k:][::-1]

    top_features = pd.DataFrame({
        "block": block_name,
        "dim": top_dims.astype(int),
        "corr": corr[top_dims],
        "abs_corr": abs_corr[top_dims],
        "auc_oriented": aucs[top_dims],
        "auc_delta": auc_delta[top_dims],
        "cohen_d": d[top_dims],
        "abs_cohen_d": abs_d[top_dims],
    })
    return summary, top_features


def scalar_metric_row(name: str, values: np.ndarray, y: np.ndarray, meta: dict) -> dict:
    values = np.asarray(values, dtype=np.float64)
    corr = pearson_corr_vector(values.reshape(-1, 1), y)[0]
    d = cohen_d_vector(values.reshape(-1, 1), y)[0]
    try:
        auc = roc_auc_score(y, values)
        auc = max(auc, 1.0 - auc)
    except Exception:
        auc = 0.5
    row = {
        "metric": name,
        **meta,
        "mean_label0": float(np.mean(values[y == 0])),
        "mean_label1": float(np.mean(values[y == 1])),
        "std_label0": float(np.std(values[y == 0])),
        "std_label1": float(np.std(values[y == 1])),
        "corr": float(corr),
        "abs_corr": float(abs(corr)),
        "auc_oriented": float(auc),
        "auc_delta": float(abs(auc - 0.5)),
        "cohen_d": float(d),
        "abs_cohen_d": float(abs(d)),
    }
    row["diagnostic_score"] = float(
        0.4 * row["auc_delta"]
        + 0.3 * row["abs_corr"]
        + 0.3 * min(row["abs_cohen_d"] / 2.0, 1.0)
    )
    return row


def stack_list(values: List[np.ndarray]) -> np.ndarray:
    return np.stack(values, axis=0).astype(np.float32)


# ============================================================
# Hidden-state extraction
# ============================================================


def extract_diagnostic_arrays() -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray], pd.DataFrame]:
    """Run the model once and collect compact per-layer arrays."""
    output_dir = ensure_dir(OUTPUT_DIR)

    df = pd.read_csv(DATA_FILE)
    y = df["label"].astype(float).astype(int).to_numpy()
    texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    prompts = df["prompt"].astype(str).tolist()

    device = get_device()
    print("=" * 80)
    print("HIDDEN STATE DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    print(f"Device      : {device}")
    print(f"Data        : {DATA_FILE}")
    print(f"Rows        : {len(df)}")
    print(f"Labels      : {dict(pd.Series(y).value_counts().sort_index())}")
    print(f"Max length  : {MAX_LENGTH}")
    print(f"Output dir  : {output_dir}")
    print(f"MI enabled  : {COMPUTE_MUTUAL_INFO and HAS_MI}")
    print()

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()

    arrays: Dict[str, List[np.ndarray]] = {
        "last_token": [],
        "mean_all": [],
        "std_all": [],
        "mean_prompt": [],
        "std_prompt": [],
        "mean_response": [],
        "std_response": [],
        "mean_response_first": [],
        "mean_response_middle": [],
        "mean_response_last": [],
        "mean_prompt_first": [],
        "mean_prompt_last": [],
    }

    scalar_rows_raw: List[dict] = []
    sample_rows: List[dict] = []

    t0 = time.time()

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting diagnostics", unit="batch"):
        batch_texts = texts[start:start + BATCH_SIZE]
        batch_prompts = prompts[start:start + BATCH_SIZE]

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Prompt lengths under the same tokenizer, used to separate prompt/response zones.
        prompt_lengths = []
        for prompt in batch_prompts:
            prompt_ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            )["input_ids"][0]
            prompt_lengths.append(int(prompt_ids.numel()))

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # (batch, n_layers, seq_len, hidden_dim), includes embedding layer at index 0.
        hidden = torch.stack(outputs.hidden_states, dim=1).float().cpu()
        mask_cpu = attention_mask.cpu()

        batch_size, n_layers, seq_len, hidden_dim = hidden.shape

        for i in range(batch_size):
            global_idx = start + i
            valid_len = int(mask_cpu[i].sum().item())
            prompt_len = min(prompt_lengths[i], valid_len)
            response_len = max(0, valid_len - prompt_len)

            sample_rows.append({
                "sample_idx": global_idx,
                "label": int(y[global_idx]),
                "valid_tokens": valid_len,
                "prompt_tokens": prompt_len,
                "response_tokens": response_len,
                "response_ratio": response_len / max(valid_len, 1),
            })

            sample_hidden = hidden[i, :, :valid_len, :]  # (layers, valid_tokens, dim)
            prompt_hidden = sample_hidden[:, :prompt_len, :]
            response_hidden = sample_hidden[:, prompt_len:valid_len, :]

            response_first_idx = contiguous_bucket_indices(prompt_len, valid_len, "first")
            response_mid_idx = contiguous_bucket_indices(prompt_len, valid_len, "middle")
            response_last_idx = contiguous_bucket_indices(prompt_len, valid_len, "last")
            prompt_first_idx = contiguous_bucket_indices(0, prompt_len, "first")
            prompt_last_idx = contiguous_bucket_indices(0, prompt_len, "last")

            arrays["last_token"].append(sample_hidden[:, -1, :].numpy())
            arrays["mean_all"].append(sample_hidden.mean(dim=1).numpy())
            arrays["std_all"].append(sample_hidden.std(dim=1, unbiased=False).numpy())
            arrays["mean_prompt"].append(safe_mean(prompt_hidden, dim=1).numpy())
            arrays["std_prompt"].append(safe_std(prompt_hidden, dim=1).numpy())
            arrays["mean_response"].append(safe_mean(response_hidden, dim=1).numpy())
            arrays["std_response"].append(safe_std(response_hidden, dim=1).numpy())

            arrays["mean_response_first"].append(safe_mean(sample_hidden[:, response_first_idx, :], dim=1).numpy())
            arrays["mean_response_middle"].append(safe_mean(sample_hidden[:, response_mid_idx, :], dim=1).numpy())
            arrays["mean_response_last"].append(safe_mean(sample_hidden[:, response_last_idx, :], dim=1).numpy())
            arrays["mean_prompt_first"].append(safe_mean(sample_hidden[:, prompt_first_idx, :], dim=1).numpy())
            arrays["mean_prompt_last"].append(safe_mean(sample_hidden[:, prompt_last_idx, :], dim=1).numpy())

            # Scalar diagnostics by layer and zone.
            zones = {
                "all": sample_hidden,
                "prompt": prompt_hidden,
                "response": response_hidden,
                "response_first": sample_hidden[:, response_first_idx, :],
                "response_middle": sample_hidden[:, response_mid_idx, :],
                "response_last": sample_hidden[:, response_last_idx, :],
                "prompt_first": sample_hidden[:, prompt_first_idx, :],
                "prompt_last": sample_hidden[:, prompt_last_idx, :],
            }

            for zone_name, zone_tensor in zones.items():
                # zone_tensor shape: (layers, tokens, dim), tokens may be zero.
                for layer_idx in range(n_layers):
                    layer_tokens = zone_tensor[layer_idx]
                    if layer_tokens.numel() == 0 or layer_tokens.shape[0] == 0:
                        vals = {
                            "token_l2_mean": 0.0,
                            "token_l2_std": 0.0,
                            "activation_mean": 0.0,
                            "activation_std": 0.0,
                            "activation_abs_mean": 0.0,
                            "activation_abs_max": 0.0,
                            "feature_variance_mean": 0.0,
                        }
                    else:
                        token_norms = torch.linalg.norm(layer_tokens, dim=-1)
                        feature_var = layer_tokens.var(dim=0, unbiased=False)
                        vals = {
                            "token_l2_mean": float(token_norms.mean().item()),
                            "token_l2_std": float(token_norms.std(unbiased=False).item()),
                            "activation_mean": float(layer_tokens.mean().item()),
                            "activation_std": float(layer_tokens.std(unbiased=False).item()),
                            "activation_abs_mean": float(layer_tokens.abs().mean().item()),
                            "activation_abs_max": float(layer_tokens.abs().max().item()),
                            "feature_variance_mean": float(feature_var.mean().item()),
                        }

                    for metric_name, metric_value in vals.items():
                        scalar_rows_raw.append({
                            "sample_idx": global_idx,
                            "label": int(y[global_idx]),
                            "layer": layer_idx,
                            "zone": zone_name,
                            "metric": metric_name,
                            "value": metric_value,
                        })

    elapsed = time.time() - t0
    print(f"Extraction done in {elapsed:.1f} seconds")

    compact_arrays = {name: stack_list(vals) for name, vals in arrays.items()}
    sample_stats = pd.DataFrame(sample_rows)
    raw_scalar_df = pd.DataFrame(scalar_rows_raw)

    # Save compact arrays as compressed npz for quick reuse.
    npz_path = output_dir / "diagnostic_compact_arrays.npz"
    np.savez_compressed(npz_path, y=y, **compact_arrays)
    print(f"Saved compact arrays: {npz_path}")

    sample_stats.to_csv(output_dir / "sample_token_lengths.csv", index=False)
    print(f"Saved sample token lengths: {output_dir / 'sample_token_lengths.csv'}")

    return df, y, compact_arrays, raw_scalar_df


# ============================================================
# Diagnostic scoring
# ============================================================


def build_vector_blocks(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Create many vector blocks from compact arrays.

    Each base array shape: (N, L, D). Each block shape: (N, features).
    """
    blocks: Dict[str, np.ndarray] = {}

    base_names = [
        "last_token",
        "mean_all",
        "std_all",
        "mean_prompt",
        "std_prompt",
        "mean_response",
        "std_response",
        "mean_response_first",
        "mean_response_middle",
        "mean_response_last",
        "mean_prompt_first",
        "mean_prompt_last",
    ]

    # Per-layer blocks for each base representation.
    for base in base_names:
        A = arrays[base]
        n_layers = A.shape[1]
        for layer_idx in range(n_layers):
            blocks[f"{base}__layer_{layer_idx:02d}"] = A[:, layer_idx, :]

    # Layer groups: early / middle / late / last k.
    for base in ["last_token", "mean_all", "std_all", "mean_response", "std_response", "mean_prompt"]:
        A = arrays[base]
        n_layers = A.shape[1]
        groups = {
            "first4": list(range(0, min(4, n_layers))),
            "middle4": list(range(max(0, n_layers // 2 - 2), min(n_layers, n_layers // 2 + 2))),
            "last2": list(range(max(0, n_layers - 2), n_layers)),
            "last4": list(range(max(0, n_layers - 4), n_layers)),
            "last8": list(range(max(0, n_layers - 8), n_layers)),
            "all_layers_mean": list(range(n_layers)),
        }
        for group_name, layers in groups.items():
            if not layers:
                continue
            if group_name == "all_layers_mean":
                blocks[f"{base}__{group_name}"] = A.mean(axis=1)
            else:
                blocks[f"{base}__{group_name}_concat"] = A[:, layers, :].reshape(A.shape[0], -1)
                blocks[f"{base}__{group_name}_mean"] = A[:, layers, :].mean(axis=1)

    # Prompt-response contrasts.
    for rep in ["mean", "std"]:
        prompt = arrays[f"{rep}_prompt"]
        response = arrays[f"{rep}_response"]
        diff = response - prompt
        abs_diff = np.abs(diff)
        for layer_idx in range(diff.shape[1]):
            blocks[f"{rep}_response_minus_prompt__layer_{layer_idx:02d}"] = diff[:, layer_idx, :]
            blocks[f"{rep}_abs_response_minus_prompt__layer_{layer_idx:02d}"] = abs_diff[:, layer_idx, :]
        blocks[f"{rep}_response_minus_prompt__last4_mean"] = diff[:, -4:, :].mean(axis=1)
        blocks[f"{rep}_abs_response_minus_prompt__last4_mean"] = abs_diff[:, -4:, :].mean(axis=1)
        blocks[f"{rep}_response_minus_prompt__last4_concat"] = diff[:, -4:, :].reshape(diff.shape[0], -1)

    # Token-position contrasts inside response.
    response_first = arrays["mean_response_first"]
    response_middle = arrays["mean_response_middle"]
    response_last = arrays["mean_response_last"]
    contrasts = {
        "response_last_minus_first": response_last - response_first,
        "response_last_minus_middle": response_last - response_middle,
        "response_middle_minus_first": response_middle - response_first,
    }
    for name, A in contrasts.items():
        for layer_idx in range(A.shape[1]):
            blocks[f"{name}__layer_{layer_idx:02d}"] = A[:, layer_idx, :]
        blocks[f"{name}__last4_mean"] = A[:, -4:, :].mean(axis=1)
        blocks[f"{name}__last4_concat"] = A[:, -4:, :].reshape(A.shape[0], -1)

    # Layer drift: adjacent layer changes.
    for base in ["last_token", "mean_all", "mean_response", "mean_prompt"]:
        A = arrays[base]
        drift = A[:, 1:, :] - A[:, :-1, :]
        abs_drift = np.abs(drift)
        for transition_idx in range(drift.shape[1]):
            blocks[f"{base}__drift_layer_{transition_idx:02d}_to_{transition_idx + 1:02d}"] = drift[:, transition_idx, :]
            blocks[f"{base}__abs_drift_layer_{transition_idx:02d}_to_{transition_idx + 1:02d}"] = abs_drift[:, transition_idx, :]
        blocks[f"{base}__last4_drift_mean"] = drift[:, -4:, :].mean(axis=1)
        blocks[f"{base}__last4_abs_drift_mean"] = abs_drift[:, -4:, :].mean(axis=1)
        blocks[f"{base}__last8_drift_mean"] = drift[:, -8:, :].mean(axis=1)
        blocks[f"{base}__last8_abs_drift_mean"] = abs_drift[:, -8:, :].mean(axis=1)

    # Long-range layer contrasts.
    for base in ["last_token", "mean_all", "mean_response"]:
        A = arrays[base]
        n_layers = A.shape[1]
        layer_pairs = [
            (0, n_layers - 1),
            (max(0, n_layers // 2), n_layers - 1),
            (max(0, n_layers - 4), n_layers - 1),
            (max(0, n_layers - 8), n_layers - 1),
        ]
        for left, right in layer_pairs:
            if left == right:
                continue
            blocks[f"{base}__layer_{right:02d}_minus_{left:02d}"] = A[:, right, :] - A[:, left, :]
            blocks[f"{base}__abs_layer_{right:02d}_minus_{left:02d}"] = np.abs(A[:, right, :] - A[:, left, :])

    # Compact scalar-like vector blocks: norms per layer / cosine drifts.
    scalar_vector_blocks = {}
    for base in ["last_token", "mean_all", "std_all", "mean_prompt", "mean_response", "std_response"]:
        A = arrays[base]
        scalar_vector_blocks[f"{base}__l2_norm_by_layer"] = np.linalg.norm(A, axis=2)
        scalar_vector_blocks[f"{base}__mean_by_layer"] = A.mean(axis=2)
        scalar_vector_blocks[f"{base}__std_by_layer"] = A.std(axis=2)
        scalar_vector_blocks[f"{base}__abs_mean_by_layer"] = np.abs(A).mean(axis=2)

    for base in ["last_token", "mean_all", "mean_response", "mean_prompt"]:
        A = arrays[base]
        left = A[:, :-1, :]
        right = A[:, 1:, :]
        dot = (left * right).sum(axis=2)
        denom = np.linalg.norm(left, axis=2) * np.linalg.norm(right, axis=2) + EPS
        cos = dot / denom
        scalar_vector_blocks[f"{base}__adjacent_layer_cosine"] = cos
        scalar_vector_blocks[f"{base}__adjacent_layer_l2_drift"] = np.linalg.norm(right - left, axis=2)

    blocks.update(scalar_vector_blocks)
    return blocks


def analyze_scalar_rows(raw_scalar_df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    rows = []
    grouped = raw_scalar_df.groupby(["layer", "zone", "metric"], sort=False)
    for (layer, zone, metric), g in grouped:
        g_sorted = g.sort_values("sample_idx")
        values = g_sorted["value"].to_numpy()
        rows.append(scalar_metric_row(
            name=f"layer_{layer:02d}__{zone}__{metric}",
            values=values,
            y=y,
            meta={"layer": int(layer), "zone": zone, "stat": metric},
        ))
    return pd.DataFrame(rows).sort_values("diagnostic_score", ascending=False)


def analyze_sample_lengths(df: pd.DataFrame, y: np.ndarray, output_dir: Path) -> pd.DataFrame:
    rows = []
    for col in ["valid_tokens", "prompt_tokens", "response_tokens", "response_ratio"]:
        rows.append(scalar_metric_row(
            name=f"sample_length__{col}",
            values=df[col].to_numpy(),
            y=y,
            meta={"layer": -1, "zone": "sample", "stat": col},
        ))
    out = pd.DataFrame(rows).sort_values("diagnostic_score", ascending=False)
    out.to_csv(output_dir / "diagnostic_sample_length_metrics.csv", index=False)
    return out


def save_pca_coordinates(blocks: Dict[str, np.ndarray], y: np.ndarray, top_block_names: List[str], output_dir: Path) -> None:
    pca_dir = ensure_dir(output_dir / "pca_coordinates")
    for block_name in top_block_names[:SAVE_PCA_COORDINATES_FOR_TOP_N_BLOCKS]:
        X = blocks[block_name]
        try:
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            coords = pca.fit_transform(X_scaled)
            out = pd.DataFrame({
                "sample_idx": np.arange(len(y)),
                "label": y,
                "pc1": coords[:, 0],
                "pc2": coords[:, 1],
                "explained_var_pc1": pca.explained_variance_ratio_[0],
                "explained_var_pc2": pca.explained_variance_ratio_[1],
                "block": block_name,
            })
            safe_name = block_name.replace("/", "_").replace(" ", "_").replace(":", "_")
            out.to_csv(pca_dir / f"pca__{safe_name}.csv", index=False)
        except Exception:
            continue


def write_report(
    output_dir: Path,
    y: np.ndarray,
    vector_scores: pd.DataFrame,
    scalar_scores: pd.DataFrame,
    sample_length_scores: pd.DataFrame,
    top_features: pd.DataFrame,
    elapsed_total: float,
) -> None:
    report_path = output_dir / "DIAGNOSTIC_REPORT.txt"

    def df_to_text(df: pd.DataFrame, cols: List[str], n: int) -> str:
        present_cols = [c for c in cols if c in df.columns]
        return df[present_cols].head(n).to_string(index=False)

    lines = []
    lines.append("=" * 100)
    lines.append("HIDDEN STATE DIAGNOSTIC REPORT")
    lines.append("=" * 100)
    lines.append(f"Samples: {len(y)}")
    lines.append(f"Class counts: {dict(pd.Series(y).value_counts().sort_index())}")
    lines.append(f"Positive rate: {float(np.mean(y)):.4f}")
    lines.append(f"Total time: {elapsed_total:.1f} sec")
    lines.append("")
    lines.append("INTERPRETATION GUIDE")
    lines.append("- diagnostic_score is a heuristic for ranking blocks/scalars. Higher means more class signal.")
    lines.append("- auc_oriented / max_auc / top20_auc_mean are orientation-invariant: 0.5 = no signal, 1.0 = perfect separation.")
    lines.append("- abs_corr shows linear relation to label.")
    lines.append("- abs_cohen_d shows standardized class mean gap; roughly 0.2 small, 0.5 medium, 0.8+ large.")
    lines.append("- pca_pc1_auc checks whether unsupervised PC1 separates classes.")
    lines.append("")

    lines.append("TOP VECTOR BLOCKS")
    lines.append("- These are candidate aggregation blocks to consider for aggregation.py.")
    lines.append(df_to_text(vector_scores, [
        "block", "n_features", "diagnostic_score", "top20_auc_mean", "max_auc",
        "top20_abs_corr_mean", "max_abs_corr", "top20_abs_cohen_d_mean",
        "pca_pc1_auc", "mi_top20_mean_top_dims",
    ], TOP_BLOCKS_TO_PRINT))
    lines.append("")

    lines.append("TOP SCALAR DIAGNOSTICS")
    lines.append("- These show layer/zone/statistics where hallucinated vs truthful examples differ.")
    lines.append(df_to_text(scalar_scores, [
        "metric", "layer", "zone", "stat", "diagnostic_score", "auc_oriented",
        "abs_corr", "abs_cohen_d", "mean_label0", "mean_label1",
    ], TOP_SCALARS_TO_PRINT))
    lines.append("")

    lines.append("SAMPLE LENGTH SIGNAL")
    lines.append("- Useful to detect whether labels are partly predictable from prompt/response length.")
    lines.append(df_to_text(sample_length_scores, [
        "metric", "diagnostic_score", "auc_oriented", "abs_corr", "abs_cohen_d", "mean_label0", "mean_label1",
    ], 20))
    lines.append("")

    lines.append("TOP INDIVIDUAL FEATURE DIMENSIONS ACROSS BLOCKS")
    lines.append("- Use this for insight, not directly for final selection without CV.")
    lines.append(df_to_text(top_features, [
        "block", "dim", "combined_feature_score", "auc_oriented", "abs_corr", "abs_cohen_d", "corr", "cohen_d",
    ], TOP_FEATURES_TO_PRINT))
    lines.append("")

    lines.append("FILES SAVED")
    for filename in [
        "diagnostic_vector_block_scores.csv",
        "diagnostic_scalar_layer_zone_scores.csv",
        "diagnostic_sample_length_metrics.csv",
        "diagnostic_top_individual_features.csv",
        "sample_token_lengths.csv",
        "diagnostic_compact_arrays.npz",
        "pca_coordinates/*.csv",
    ]:
        lines.append(f"- {filename}")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"\nSaved report: {report_path}")


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    total_t0 = time.time()
    output_dir = ensure_dir(OUTPUT_DIR)

    df, y, arrays, raw_scalar_df = extract_diagnostic_arrays()

    print("\nBuilding vector blocks...")
    blocks = build_vector_blocks(arrays)
    print(f"Vector blocks built: {len(blocks)}")

    print("\nScoring vector blocks...")
    vector_rows = []
    all_top_features = []

    for block_name, X_block in tqdm(blocks.items(), desc="Scoring blocks", unit="block"):
        summary, top_features = summarize_vector_block(block_name, X_block, y)
        vector_rows.append(summary)
        all_top_features.append(top_features)

    vector_scores = pd.DataFrame(vector_rows).sort_values("diagnostic_score", ascending=False)
    top_features_df = pd.concat(all_top_features, ignore_index=True)
    top_features_df["combined_feature_score"] = (
        0.45 * top_features_df["auc_delta"]
        + 0.30 * top_features_df["abs_corr"]
        + 0.25 * np.minimum(top_features_df["abs_cohen_d"] / 2.0, 1.0)
    )
    top_features_df = top_features_df.sort_values("combined_feature_score", ascending=False)

    print("\nScoring scalar layer/zone diagnostics...")
    scalar_scores = analyze_scalar_rows(raw_scalar_df, y)

    sample_stats = pd.read_csv(output_dir / "sample_token_lengths.csv")
    sample_length_scores = analyze_sample_lengths(sample_stats, y, output_dir)

    # Save outputs.
    vector_scores.to_csv(output_dir / "diagnostic_vector_block_scores.csv", index=False)
    scalar_scores.to_csv(output_dir / "diagnostic_scalar_layer_zone_scores.csv", index=False)
    top_features_df.to_csv(output_dir / "diagnostic_top_individual_features.csv", index=False)

    # Save compact JSON with top block names for the next stage.
    top_block_names = vector_scores["block"].head(TOP_BLOCKS_TO_PRINT).tolist()
    (output_dir / "top_blocks.json").write_text(
        json.dumps(top_block_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSaving PCA coordinates for top blocks...")
    save_pca_coordinates(blocks, y, top_block_names, output_dir)

    elapsed_total = time.time() - total_t0
    write_report(
        output_dir=output_dir,
        y=y,
        vector_scores=vector_scores,
        scalar_scores=scalar_scores,
        sample_length_scores=sample_length_scores,
        top_features=top_features_df,
        elapsed_total=elapsed_total,
    )
