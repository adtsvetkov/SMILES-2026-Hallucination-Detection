"""
Build advanced attention-based features for hallucination detection.

This script intentionally lives outside the fixed solution.py feature path,
because it requires an additional forward-pass configuration:
- output_attentions=True
- eager attention implementation when supported

It does NOT use logits, generation scores, hooks, external verifiers, or labels.
The only boundary signal is prompt length, computed from a prompt_len column when
available or by tokenizing the prompt text.

Outputs:
./artifacts/advanced_features_infrastructure_change/features_dataset_advanced_infrastructure_change.parquet
./artifacts/advanced_features_infrastructure_change/features_test_advanced_infrastructure_change.parquet
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from model import MAX_LENGTH
except Exception:
    MAX_LENGTH = 512


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = os.environ.get("PRIMARY_MODEL_NAME", "Qwen/Qwen2.5-0.5B")

DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"

OUTPUT_DIR = Path("./artifacts/advanced_features_infrastructure_change")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUTPUT = OUTPUT_DIR / "features_dataset_advanced_infrastructure_change.parquet"
TEST_OUTPUT = OUTPUT_DIR / "features_test_advanced_infrastructure_change.parquet"

BATCH_SIZE = 1
EXPORT_TEST = True
EPS = 1e-8

ATTENTION_LAYERS = [11, 12, 13, 14, 15, 16]


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


def add_scalar(features: Dict[str, float], name: str, value) -> None:
    features[name] = clean_value(value)


def as_np(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().float().numpy()
    arr = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def add_stats(
    features: Dict[str, float],
    prefix: str,
    values: Iterable[float],
    include_std: bool = True,
) -> None:
    arr = as_np(list(values)).reshape(-1)
    if arr.size == 0:
        arr = np.array([0.0], dtype=np.float32)
    add_scalar(features, f"{prefix}_mean", arr.mean())
    add_scalar(features, f"{prefix}_min", arr.min())
    add_scalar(features, f"{prefix}_max", arr.max())
    if include_std:
        add_scalar(features, f"{prefix}_std", arr.std())


def entropy_rows(prob_matrix: np.ndarray) -> np.ndarray:
    probs = np.asarray(prob_matrix, dtype=np.float32)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.clip(probs, 0.0, 1.0)
    return -(probs * np.log(probs + EPS)).sum(axis=1)


def slope(values: np.ndarray) -> float:
    values = as_np(values).reshape(-1)
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float32)
    return clean_value(np.polyfit(x, values, 1)[0])


def safe_nonempty_array(values, fallback: float = 0.0) -> np.ndarray:
    """Return a finite non-empty 1D numpy array.

    Some samples can have an extremely short response span after truncation or
    EOS stripping. This helper prevents empty-slice warnings and reductions on
    empty arrays while keeping feature dimensions stable.
    """
    arr = as_np(values).reshape(-1)
    if arr.size == 0:
        return np.array([fallback], dtype=np.float32)
    return arr


def safe_segment(values, start: int | None = None, end: int | None = None, fallback: float = 0.0) -> np.ndarray:
    arr = safe_nonempty_array(values, fallback=fallback)
    segment = arr[start:end]
    if segment.size == 0:
        return np.array([fallback], dtype=np.float32)
    return segment


def valid_positions(valid_mask: torch.Tensor) -> torch.Tensor:
    return torch.where(valid_mask.bool().cpu())[0]


def ensure_non_empty(idx: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    if idx.numel() > 0:
        return idx
    if fallback.numel() > 0:
        return fallback[-1:]
    return fallback


def last_fraction(idx: torch.Tensor, frac: float) -> torch.Tensor:
    n = int(idx.numel())
    if n == 0:
        return idx
    keep = max(1, int(round(n * frac)))
    return idx[-keep:]


def get_prompt_lengths(tokenizer, prompts: Sequence[str], max_length: int) -> List[int]:
    lengths: List[int] = []
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
            "response_late30": empty,
            "response_last5": empty,
            "first_valid": empty,
        }

    prompt_len = min(max(int(prompt_len), 0), seq_len)
    pos = torch.arange(seq_len)
    prompt_idx = torch.where(valid_mask & (pos < prompt_len))[0]
    response_idx = torch.where(valid_mask & (pos >= prompt_len))[0]

    if response_idx.numel() == 0:
        response_idx = valid_idx[-1:]
    if prompt_idx.numel() == 0:
        prompt_idx = valid_idx[:1]

    response_wo_eos = response_idx[:-1] if response_idx.numel() >= 2 else response_idx.new_empty((0,))
    if response_wo_eos.numel() == 0:
        response_wo_eos = response_idx

    return {
        "all": valid_idx,
        "prompt": ensure_non_empty(prompt_idx, valid_idx),
        "response": ensure_non_empty(response_idx, valid_idx),
        "response_wo_eos": ensure_non_empty(response_wo_eos, response_idx),
        "response_late30": ensure_non_empty(last_fraction(response_idx, 0.30), response_idx),
        "response_last5": ensure_non_empty(response_idx[-min(5, int(response_idx.numel())):], response_idx),
        "first_valid": valid_idx[:1],
    }


def safe_layer_index(layer: int, n_layers: int) -> int:
    if layer < 0:
        layer = n_layers + layer
    return int(min(max(layer, 0), n_layers - 1))


# ============================================================
# MODEL LOADING
# ============================================================


def load_attention_model_and_tokenizer(device: torch.device):
    print(f"[Model] Loading '{MODEL_NAME}' with attention outputs enabled ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "output_hidden_states": False,
        "output_attentions": True,
        "torch_dtype": torch.bfloat16,
    }
    # Some attention implementations do not return attention weights. For Qwen,
    # eager attention is the safest option when available.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            attn_implementation="eager",
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)

    model.to(device)
    model.eval()
    return model, tokenizer


# ============================================================
# ATTENTION FEATURES
# ============================================================


def attention_masses_for_head(
    head_attn: np.ndarray,
    query_idx: np.ndarray,
    prompt_idx: np.ndarray,
    response_idx: np.ndarray,
    first_valid_idx: int,
    valid_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute prompt/response/sink masses for one attention head.

    head_attn shape is [seq_len, seq_len]. Rows are query positions, columns are
    key positions.
    """
    if query_idx.size == 0:
        query_idx = response_idx[-1:] if response_idx.size else valid_idx[-1:]
    if prompt_idx.size == 0:
        prompt_idx = valid_idx[:1]
    if response_idx.size == 0:
        response_idx = valid_idx[-1:]

    q_attn = head_attn[query_idx]
    prompt_mass = q_attn[:, prompt_idx].sum(axis=1)
    response_mass = q_attn[:, response_idx].sum(axis=1)
    sink_mass = q_attn[:, first_valid_idx]
    valid_probs = q_attn[:, valid_idx]
    entropy = entropy_rows(valid_probs)
    lookback = prompt_mass / (prompt_mass + response_mass + EPS)

    return {
        "prompt_mass": prompt_mass,
        "response_mass": response_mass,
        "sink_mass": sink_mass,
        "entropy": entropy,
        "lookback": lookback,
    }


def add_layer_head_attention_features(
    features: Dict[str, float],
    layer_idx: int,
    attn_layer: torch.Tensor,
    zones: Dict[str, torch.Tensor],
) -> None:
    """Add per-layer/head attention features and layer summaries."""
    attn_np = as_np(attn_layer)  # [heads, seq, seq]
    n_heads = int(attn_np.shape[0])

    valid_idx = zones["all"].numpy().astype(int)
    prompt_idx = zones["prompt"].numpy().astype(int)
    response_idx = zones["response"].numpy().astype(int)
    response_query_idx = zones["response_wo_eos"].numpy().astype(int)
    first_valid_idx = int(zones["first_valid"][0].item()) if zones["first_valid"].numel() else 0

    head_lookback_means = []
    head_entropy_means = []
    layer_prompt_masses = []
    layer_response_masses = []
    layer_sink_masses = []

    for head_idx in range(n_heads):
        masses = attention_masses_for_head(
            head_attn=attn_np[head_idx],
            query_idx=response_query_idx,
            prompt_idx=prompt_idx,
            response_idx=response_idx,
            first_valid_idx=first_valid_idx,
            valid_idx=valid_idx,
        )

        prefix = f"attn_l{layer_idx}_h{head_idx}"

        # 1. Attention Lookback-Lens per layer/head.
        add_stats(features, f"{prefix}_lookback", masses["lookback"], include_std=True)
        add_stats(features, f"{prefix}_attention_entropy", masses["entropy"], include_std=True)
        add_scalar(features, f"{prefix}_attention_to_sink_mean", masses["sink_mass"].mean())
        add_scalar(features, f"{prefix}_attention_to_sink_max", masses["sink_mass"].max())
        add_scalar(features, f"{prefix}_attention_to_response_mean", masses["response_mass"].mean())
        add_scalar(features, f"{prefix}_attention_to_response_max", masses["response_mass"].max())

        # 2. Attention trajectory / grounding decay over response positions.
        lookback_by_pos = safe_nonempty_array(masses["lookback"], fallback=0.0)
        late_start = min(len(lookback_by_pos) - 1, max(0, int(round(len(lookback_by_pos) * 0.70))))
        late_vals = safe_segment(lookback_by_pos, late_start, None, fallback=lookback_by_pos[-1])
        first_end = max(1, min(5, len(lookback_by_pos)))
        first_vals = safe_segment(lookback_by_pos, 0, first_end, fallback=lookback_by_pos[0])
        last5_vals = safe_segment(lookback_by_pos, -min(5, len(lookback_by_pos)), None, fallback=lookback_by_pos[-1])
        diffs = np.diff(lookback_by_pos) if lookback_by_pos.size >= 2 else np.array([0.0], dtype=np.float32)

        lookback_mean = float(lookback_by_pos.mean())
        late_mean = float(late_vals.mean())
        first_mean = float(first_vals.mean())
        last5_mean = float(last5_vals.mean())

        add_scalar(features, f"{prefix}_grounding_slope", slope(lookback_by_pos))
        add_scalar(features, f"{prefix}_grounding_late_minus_early", late_mean - first_mean)
        add_scalar(features, f"{prefix}_grounding_decay_ratio", late_mean / (first_mean + EPS))
        add_scalar(features, f"{prefix}_grounding_roughness", np.abs(diffs).sum())
        add_scalar(features, f"{prefix}_grounding_min_late", late_vals.min())
        add_scalar(features, f"{prefix}_grounding_last5_mean", last5_mean)
        add_scalar(features, f"{prefix}_grounding_last5_vs_all", last5_mean / (lookback_mean + EPS))

        head_lookback_means.append(clean_value(masses["lookback"].mean()))
        head_entropy_means.append(clean_value(masses["entropy"].mean()))
        layer_prompt_masses.extend(as_np(masses["prompt_mass"]).tolist())
        layer_response_masses.extend(as_np(masses["response_mass"]).tolist())
        layer_sink_masses.extend(as_np(masses["sink_mass"]).tolist())

    # 3. Head disagreement per layer.
    head_lookback_values = as_np(head_lookback_means)
    head_entropy_values = as_np(head_entropy_means)
    add_scalar(features, f"attn_l{layer_idx}_head_lookback_mean", head_lookback_values.mean())
    add_scalar(features, f"attn_l{layer_idx}_head_lookback_std", head_lookback_values.std())
    add_scalar(features, f"attn_l{layer_idx}_head_lookback_min", head_lookback_values.min())
    add_scalar(features, f"attn_l{layer_idx}_head_lookback_max", head_lookback_values.max())
    add_scalar(features, f"attn_l{layer_idx}_head_disagreement", head_lookback_values.std())
    add_scalar(features, f"attn_l{layer_idx}_head_collapse", head_lookback_values.max() - head_lookback_values.mean())
    add_scalar(features, f"attn_l{layer_idx}_head_entropy_mean", head_entropy_values.mean())

    # 4. Prompt / response / sink mass summaries per layer.
    add_stats(features, f"attn_l{layer_idx}_prompt_mass_layer", layer_prompt_masses, include_std=True)
    add_stats(features, f"attn_l{layer_idx}_response_mass_layer", layer_response_masses, include_std=True)
    sink_arr = as_np(layer_sink_masses)
    add_scalar(features, f"attn_l{layer_idx}_sink_mass_layer_mean", sink_arr.mean())
    add_scalar(features, f"attn_l{layer_idx}_sink_mass_layer_std", sink_arr.std())
    add_scalar(features, f"attn_l{layer_idx}_sink_mass_layer_max", sink_arr.max())


def add_attention_features(
    features: Dict[str, float],
    attentions: Sequence[torch.Tensor],
    zones: Dict[str, torch.Tensor],
) -> None:
    layer_level_lookback = []
    layer_level_prompt_mass = []
    layer_level_response_mass = []
    layer_level_sink_mass = []

    for requested_layer in ATTENTION_LAYERS:
        attn_pos = safe_layer_index(requested_layer, len(attentions))
        # attentions[attn_pos] shape for one sample: [heads, seq, seq]
        attn_layer = attentions[attn_pos]
        add_layer_head_attention_features(features, requested_layer, attn_layer, zones)

        layer_level_lookback.append(features.get(f"attn_l{requested_layer}_head_lookback_mean", 0.0))
        layer_level_prompt_mass.append(features.get(f"attn_l{requested_layer}_prompt_mass_layer_mean", 0.0))
        layer_level_response_mass.append(features.get(f"attn_l{requested_layer}_response_mass_layer_mean", 0.0))
        layer_level_sink_mass.append(features.get(f"attn_l{requested_layer}_sink_mass_layer_mean", 0.0))

    add_stats(features, "attn_l11_l16_layer_lookback", layer_level_lookback, include_std=True)
    add_scalar(features, "attn_l11_l16_layer_lookback_slope", slope(as_np(layer_level_lookback)))
    add_scalar(
        features,
        "attn_l11_l16_layer_lookback_late_minus_early",
        np.mean(layer_level_lookback[-2:]) - np.mean(layer_level_lookback[:2]),
    )

    add_stats(features, "attn_l11_l16_prompt_mass", layer_level_prompt_mass, include_std=True)
    add_stats(features, "attn_l11_l16_response_mass", layer_level_response_mass, include_std=True)
    add_stats(features, "attn_l11_l16_sink_mass", layer_level_sink_mass, include_std=True)


def add_length_meta_features(
    features: Dict[str, float],
    zones: Dict[str, torch.Tensor],
    prompt_len: int,
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
    add_scalar(features, "response_fraction", response_tokens / (n_valid + EPS))
    add_scalar(features, "log1p_prompt_len", math.log1p(prompt_tokens))
    add_scalar(features, "log1p_response_len", math.log1p(response_tokens))
    add_scalar(features, "is_response_short", int(response_tokens <= 2))
    add_scalar(features, "is_maybe_truncated", int(n_valid >= MAX_LENGTH - 2))


# ============================================================
# SAMPLE AND DATASET EXTRACTION
# ============================================================


def extract_features_one_sample(
    attentions: Sequence[torch.Tensor],
    valid_mask: torch.Tensor,
    prompt_len: int,
) -> Dict[str, float]:
    valid_mask = valid_mask.bool().cpu()
    zones = build_exact_zones(valid_mask, prompt_len)

    features: Dict[str, float] = {}
    add_length_meta_features(features, zones, prompt_len)
    add_attention_features(features, attentions, zones)
    return {key: clean_value(value) for key, value in features.items()}


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

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extract attention features"):
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
                output_attentions=True,
                output_hidden_states=False,
                use_cache=False,
            )

        if outputs.attentions is None:
            raise RuntimeError(
                "Model did not return attentions. Try using an eager attention implementation."
            )

        mask_batch = attention_mask.detach().cpu().bool()
        # Convert from tuple[layer][batch, head, seq, seq] to per-sample tuple.
        attentions_cpu = [att.detach().cpu().float() for att in outputs.attentions]

        for i in range(input_ids.shape[0]):
            sample_attentions = [att[i] for att in attentions_cpu]
            rows.append(
                extract_features_one_sample(
                    attentions=sample_attentions,
                    valid_mask=mask_batch[i],
                    prompt_len=batch_prompt_lengths[i],
                )
            )

        del outputs, attentions_cpu, mask_batch, input_ids, attention_mask, encoding
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
    print("BUILD ADVANCED ATTENTION-BASED FEATURES")
    print("=" * 80)
    print(f"Device      : {device}")
    print(f"Model       : {MODEL_NAME}")
    print(f"Data file   : {DATA_FILE}")
    print(f"Test file   : {TEST_FILE}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"Batch size  : {BATCH_SIZE}")
    print("Note        : uses attentions; no logits, hooks, or external verifier")

    model, tokenizer = load_attention_model_and_tokenizer(device)

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
