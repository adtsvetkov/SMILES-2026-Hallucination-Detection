"""
Build token-position dynamics features for hallucination detection.

Creates:
./artifacts/token_position_dynamics/features_dataset_token_position_dynamics.parquet
./artifacts/token_position_dynamics/features_test_token_position_dynamics.parquet
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


DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"
OUTPUT_DIR = "./artifacts/token_position_dynamics"

DATASET_OUTPUT = "features_dataset_token_position_dynamics.parquet"
TEST_OUTPUT = "features_test_token_position_dynamics.parquet"

BATCH_SIZE = 4
EXPORT_TEST = True

LAYERS = [10, 11, 12, 13, 14, 15, 16]
DRIFT_PAIRS = [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16)]
LONG_DRIFT_PAIRS = [(10, 12), (11, 13), (12, 14), (13, 15), (14, 16), (10, 16), (11, 16)]

EPS = 1e-8


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_prompt_lengths(tokenizer, prompts: List[str], max_length: int) -> List[int]:
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


def safe_mean(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] == 0:
        return torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
    return x.mean(dim=0)


def safe_std(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] <= 1:
        return torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
    return x.std(dim=0, unbiased=False)


def split_response_positions(response_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    response_idx: positions of response tokens.
    Returns first/middle/last thirds + ending windows.
    """
    n = int(response_idx.numel())

    if n == 0:
        return {
            "early": response_idx,
            "middle": response_idx,
            "late": response_idx,
            "last_5": response_idx,
            "last_10": response_idx,
        }

    a = max(1, n // 3)
    b = max(a + 1, 2 * n // 3) if n >= 3 else n

    return {
        "early": response_idx[:a],
        "middle": response_idx[a:b] if b > a else response_idx,
        "late": response_idx[b:] if n > b else response_idx[-a:],
        "last_5": response_idx[-min(5, n):],
        "last_10": response_idx[-min(10, n):],
    }


def l2_features(vec: torch.Tensor, prefix: str, names: List[str], values: List[float]) -> None:
    values.extend([
        float(torch.linalg.norm(vec).cpu()),
        float(vec.abs().mean().cpu()),
        float(vec.pow(2).mean().cpu()),
        float(vec.std(unbiased=False).cpu()) if vec.numel() > 1 else 0.0,
        float(vec.max().cpu()),
        float(vec.min().cpu()),
    ])
    names.extend([
        f"{prefix}_l2",
        f"{prefix}_abs_mean",
        f"{prefix}_energy",
        f"{prefix}_std",
        f"{prefix}_max",
        f"{prefix}_min",
    ])


def token_norm_stats(token_vectors: torch.Tensor, prefix: str, names: List[str], values: List[float]) -> None:
    if token_vectors.shape[0] == 0:
        values.extend([0.0] * 8)
    else:
        norms = torch.linalg.norm(token_vectors, dim=1)
        probs = norms / (norms.sum() + EPS)
        entropy = -(probs * torch.log(probs + EPS)).sum()

        values.extend([
            float(norms.mean().cpu()),
            float(norms.std(unbiased=False).cpu()) if norms.numel() > 1 else 0.0,
            float(norms.max().cpu()),
            float(norms.min().cpu()),
            float((norms[-1] - norms[0]).cpu()) if norms.numel() > 1 else 0.0,
            float(entropy.cpu()),
            float((norms[-min(5, norms.numel()):].mean()).cpu()),
            float((norms[:min(5, norms.numel())].mean()).cpu()),
        ])

    names.extend([
        f"{prefix}_token_norm_mean",
        f"{prefix}_token_norm_std",
        f"{prefix}_token_norm_max",
        f"{prefix}_token_norm_min",
        f"{prefix}_token_norm_last_minus_first",
        f"{prefix}_token_norm_entropy",
        f"{prefix}_token_norm_last5_mean",
        f"{prefix}_token_norm_first5_mean",
    ])


def slope_features(token_vectors: torch.Tensor, prefix: str, names: List[str], values: List[float]) -> None:
    """
    Lightweight slope features over response positions.
    Computes slope of token L2 norm across response tokens.
    """
    if token_vectors.shape[0] <= 1:
        values.extend([0.0, 0.0, 0.0])
    else:
        norms = torch.linalg.norm(token_vectors, dim=1).float()
        t = torch.linspace(0.0, 1.0, steps=norms.numel(), device=norms.device)
        t_centered = t - t.mean()
        y_centered = norms - norms.mean()

        slope = (t_centered * y_centered).sum() / ((t_centered ** 2).sum() + EPS)
        first = norms[: max(1, norms.numel() // 3)].mean()
        last = norms[-max(1, norms.numel() // 3):].mean()

        values.extend([
            float(slope.cpu()),
            float((last - first).cpu()),
            float((last / (first + EPS)).cpu()),
        ])

    names.extend([
        f"{prefix}_norm_slope",
        f"{prefix}_late_minus_early_norm",
        f"{prefix}_late_over_early_norm",
    ])


def extract_features_for_sample(
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
    prompt_len: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    hidden: n_layers x seq_len x hidden_dim
    """
    hidden = hidden.float().cpu()
    valid_mask = valid_mask.bool().cpu()

    seq_len = hidden.shape[1]
    prompt_len = min(max(int(prompt_len), 0), seq_len)

    pos = torch.arange(seq_len)
    response_mask = valid_mask & (pos >= prompt_len)

    if response_mask.sum().item() == 0:
        response_mask = valid_mask.clone()

    response_idx = torch.where(response_mask)[0]
    zones = split_response_positions(response_idx)

    values: List[float] = []
    names: List[str] = []

    # ============================================================
    # 1. Early vs late response divergence per layer
    # ============================================================
    for layer in LAYERS:
        early = hidden[layer, zones["early"]]
        late = hidden[layer, zones["late"]]

        early_mean = safe_mean(early)
        late_mean = safe_mean(late)

        diff = late_mean - early_mean
        l2_features(diff, f"pos_l{layer}_late_minus_early", names, values)

        cosine = torch.nn.functional.cosine_similarity(
            early_mean.unsqueeze(0),
            late_mean.unsqueeze(0),
            dim=1,
        )[0]
        values.append(float(cosine.cpu()))
        names.append(f"pos_l{layer}_early_late_cosine")

    # ============================================================
    # 2. Response ending collapse
    # last tokens vs earlier/whole response
    # ============================================================
    for layer in LAYERS:
        all_resp = hidden[layer, response_idx]
        early = hidden[layer, zones["early"]]
        last_5 = hidden[layer, zones["last_5"]]
        last_10 = hidden[layer, zones["last_10"]]

        all_mean = safe_mean(all_resp)
        early_mean = safe_mean(early)
        last5_mean = safe_mean(last_5)
        last10_mean = safe_mean(last_10)

        l2_features(last5_mean - all_mean, f"pos_l{layer}_last5_minus_all", names, values)
        l2_features(last10_mean - all_mean, f"pos_l{layer}_last10_minus_all", names, values)
        l2_features(last5_mean - early_mean, f"pos_l{layer}_last5_minus_early", names, values)

        token_norm_stats(last_5, f"pos_l{layer}_last5", names, values)
        token_norm_stats(all_resp, f"pos_l{layer}_all_response", names, values)

    # ============================================================
    # 3. Entropy-like drift over tokens
    # and slope across token positions
    # ============================================================
    for left, right in DRIFT_PAIRS:
        drift_tokens = hidden[right, response_idx] - hidden[left, response_idx]

        token_norm_stats(
            drift_tokens,
            f"pos_drift_l{left}_to_l{right}_response",
            names,
            values,
        )

        slope_features(
            drift_tokens,
            f"pos_drift_l{left}_to_l{right}_response",
            names,
            values,
        )

        # early/middle/late drift comparison
        zone_means = {}
        for zone_name in ["early", "middle", "late"]:
            z = hidden[right, zones[zone_name]] - hidden[left, zones[zone_name]]
            zone_means[zone_name] = safe_mean(z)

        l2_features(
            zone_means["late"] - zone_means["early"],
            f"pos_drift_l{left}_to_l{right}_late_minus_early",
            names,
            values,
        )

        l2_features(
            zone_means["middle"] - zone_means["early"],
            f"pos_drift_l{left}_to_l{right}_middle_minus_early",
            names,
            values,
        )

        l2_features(
            zone_means["late"] - zone_means["middle"],
            f"pos_drift_l{left}_to_l{right}_late_minus_middle",
            names,
            values,
        )

    # ============================================================
    # 4. Long-pair token drift dynamics
    # ============================================================
    for left, right in LONG_DRIFT_PAIRS:
        drift_tokens = hidden[right, response_idx] - hidden[left, response_idx]

        token_norm_stats(
            drift_tokens,
            f"pos_long_drift_l{left}_to_l{right}_response",
            names,
            values,
        )

        slope_features(
            drift_tokens,
            f"pos_long_drift_l{left}_to_l{right}_response",
            names,
            values,
        )

    # ============================================================
    # 5. Global scalar response-length context
    # ============================================================
    response_count = float(response_idx.numel())
    valid_count = float(valid_mask.sum().item())

    values.extend([
        response_count,
        response_count / max(valid_count, 1.0),
    ])
    names.extend([
        "pos_length_response_tokens",
        "pos_length_response_ratio",
    ])

    return np.asarray(values, dtype=np.float32), names


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

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extract token-position dynamics"):
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
            features, names = extract_features_for_sample(
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

    out["prompt"] = df["prompt"].astype(str).to_numpy()
    out["response"] = df["response"].astype(str).to_numpy()

    return out


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 80)
    print("BUILD TOKEN-POSITION DYNAMICS FEATURES")
    print("=" * 80)
    print(f"Device     : {device}")
    print(f"Data file  : {DATA_FILE}")
    print(f"Test file  : {TEST_FILE}")
    print(f"Output dir : {output_dir}")
    print(f"Layers     : {LAYERS}")
    print(f"Batch size : {BATCH_SIZE}")

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
    print(f"Dataset shape: {dataset_features.shape}")

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
        print(f"Test shape: {test_features.shape}")

    print(f"\nDone in {(time.time() - t0):.1f} sec")
    print("=" * 80)


if __name__ == "__main__":
    main()