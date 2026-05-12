"""
Build extended drift hidden-state features for hallucination detection.

This script is intentionally separate from solution.py.
It creates several parquet datasets so we can test drift hypotheses independently:

1. drift_transforms:
   signed drift, abs drift, squared drift, sign drift, normalized drift
   on adjacent response-layer transitions.

2. drift_long_pairs:
   response drift across wider layer transitions.

3. drift_token_zones:
   response drift separately for first/middle/last response tokens.

4. drift_extended_all:
   combination of all extended drift groups.

No labels are used during extraction.
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
OUTPUT_DIR = "./artifacts/drift_extended_features"

BATCH_SIZE = 4
EXPORT_TEST = True

# hidden_states[0] = embeddings, hidden_states[1:] = transformer layers
ADJACENT_PAIRS = [(11, 12), (12, 13), (13, 14), (14, 15), (15, 16)]
LONG_PAIRS = [(10, 12), (11, 13), (12, 14), (13, 15), (14, 16), (10, 14), (11, 15), (12, 16), (10, 16), (11, 16)]
ALL_PAIRS = sorted(set(ADJACENT_PAIRS + LONG_PAIRS))

EPS = 1e-8

OUTPUT_SPECS = {
    "drift_transforms": "features_dataset_drift_transforms.parquet",
    "drift_long_pairs": "features_dataset_drift_long_pairs.parquet",
    "drift_token_zones": "features_dataset_drift_token_zones.parquet",
    "drift_extended_all": "features_dataset_drift_extended_all.parquet",
}

TEST_OUTPUT_SPECS = {
    "drift_transforms": "features_test_drift_transforms.parquet",
    "drift_long_pairs": "features_test_drift_long_pairs.parquet",
    "drift_token_zones": "features_test_drift_token_zones.parquet",
    "drift_extended_all": "features_test_drift_extended_all.parquet",
}


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


def safe_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum().item() == 0:
        return torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
    return x[mask].mean(dim=0)


def make_response_zone_masks(
    valid_mask: torch.Tensor,
    prompt_len: int,
) -> Dict[str, torch.Tensor]:
    seq_len = valid_mask.shape[0]
    prompt_len = min(max(int(prompt_len), 0), seq_len)

    position_ids = torch.arange(seq_len)
    response_mask = valid_mask & (position_ids >= prompt_len)

    if response_mask.sum().item() == 0:
        response_mask = valid_mask.clone()

    response_positions = torch.where(response_mask)[0]

    zones = {"response_all": response_mask}

    if response_positions.numel() < 3:
        zones["response_first"] = response_mask
        zones["response_middle"] = response_mask
        zones["response_last"] = response_mask
        return zones

    n = response_positions.numel()
    first_end = max(n // 3, 1)
    second_end = max((2 * n) // 3, first_end + 1)

    first_pos = response_positions[:first_end]
    middle_pos = response_positions[first_end:second_end]
    last_pos = response_positions[second_end:]

    def positions_to_mask(pos: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        if pos.numel() > 0:
            mask[pos] = True
        else:
            mask[response_positions] = True
        return mask

    zones["response_first"] = positions_to_mask(first_pos)
    zones["response_middle"] = positions_to_mask(middle_pos)
    zones["response_last"] = positions_to_mask(last_pos)

    return zones


# ============================================================
# FEATURE BLOCKS
# ============================================================


def append_vector(vectors: List[torch.Tensor], names: List[str], prefix: str, vec: torch.Tensor) -> None:
    vectors.append(vec.float())
    names.extend([f"{prefix}_d{d}" for d in range(vec.numel())])


def normalized_diff(right: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
    diff = right - left
    scale = torch.linalg.norm(right, dim=0) + torch.linalg.norm(left, dim=0) + EPS
    # scale is scalar for vectors because dim=0 over hidden dimension would be wrong if 1D.
    # For a 1D vector, torch.linalg.norm(..., dim=0) returns abs per coordinate.
    # We want coordinatewise magnitude normalization to keep signs stable.
    coord_scale = right.abs() + left.abs() + EPS
    return diff / coord_scale


def build_drift_transforms(mean_by_layer: Dict[int, torch.Tensor]) -> Tuple[List[torch.Tensor], List[str]]:
    vectors: List[torch.Tensor] = []
    names: List[str] = []

    for left, right in ADJACENT_PAIRS:
        diff = mean_by_layer[right] - mean_by_layer[left]
        abs_diff = diff.abs()
        sq_diff = diff.pow(2)
        sign_diff = torch.sign(diff)
        norm_diff = normalized_diff(mean_by_layer[right], mean_by_layer[left])

        append_vector(vectors, names, f"drift_signed_l{left}_to_l{right}", diff)
        append_vector(vectors, names, f"drift_abs_l{left}_to_l{right}", abs_diff)
        append_vector(vectors, names, f"drift_squared_l{left}_to_l{right}", sq_diff)
        append_vector(vectors, names, f"drift_sign_l{left}_to_l{right}", sign_diff)
        append_vector(vectors, names, f"drift_normed_l{left}_to_l{right}", norm_diff)

    return vectors, names


def build_drift_long_pairs(mean_by_layer: Dict[int, torch.Tensor]) -> Tuple[List[torch.Tensor], List[str]]:
    vectors: List[torch.Tensor] = []
    names: List[str] = []

    for left, right in LONG_PAIRS:
        diff = mean_by_layer[right] - mean_by_layer[left]
        abs_diff = diff.abs()
        norm_diff = normalized_diff(mean_by_layer[right], mean_by_layer[left])

        append_vector(vectors, names, f"long_drift_signed_l{left}_to_l{right}", diff)
        append_vector(vectors, names, f"long_drift_abs_l{left}_to_l{right}", abs_diff)
        append_vector(vectors, names, f"long_drift_normed_l{left}_to_l{right}", norm_diff)

    return vectors, names


def build_drift_token_zones(
    hidden: torch.Tensor,
    zone_masks: Dict[str, torch.Tensor],
) -> Tuple[List[torch.Tensor], List[str]]:
    vectors: List[torch.Tensor] = []
    names: List[str] = []

    for zone_name, zone_mask in zone_masks.items():
        if zone_name == "response_all":
            # Already covered in other blocks; keep this file focused on token zones.
            continue

        zone_mean_by_layer = {
            layer: safe_mean(hidden[layer], zone_mask)
            for layer in sorted({x for pair in ALL_PAIRS for x in pair})
        }

        for left, right in ADJACENT_PAIRS:
            diff = zone_mean_by_layer[right] - zone_mean_by_layer[left]
            abs_diff = diff.abs()
            norm_diff = normalized_diff(zone_mean_by_layer[right], zone_mean_by_layer[left])

            append_vector(vectors, names, f"{zone_name}_drift_signed_l{left}_to_l{right}", diff)
            append_vector(vectors, names, f"{zone_name}_drift_abs_l{left}_to_l{right}", abs_diff)
            append_vector(vectors, names, f"{zone_name}_drift_normed_l{left}_to_l{right}", norm_diff)

    return vectors, names


def extract_feature_blocks_for_sample(
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
    prompt_len: int,
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    hidden = hidden.float().cpu()
    valid_mask = valid_mask.bool().cpu()

    zone_masks = make_response_zone_masks(valid_mask, prompt_len)
    response_mask = zone_masks["response_all"]

    layers_needed = sorted({x for pair in ALL_PAIRS for x in pair})
    mean_response = {layer: safe_mean(hidden[layer], response_mask) for layer in layers_needed}

    blocks: Dict[str, Tuple[List[torch.Tensor], List[str]]] = {}

    blocks["drift_transforms"] = build_drift_transforms(mean_response)
    blocks["drift_long_pairs"] = build_drift_long_pairs(mean_response)
    blocks["drift_token_zones"] = build_drift_token_zones(hidden, zone_masks)

    all_vectors: List[torch.Tensor] = []
    all_names: List[str] = []
    for block_name in ["drift_transforms", "drift_long_pairs", "drift_token_zones"]:
        vectors, names = blocks[block_name]
        all_vectors.extend(vectors)
        all_names.extend(names)
    blocks["drift_extended_all"] = (all_vectors, all_names)

    out: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for block_name, (vectors, names) in blocks.items():
        feature_vector = torch.cat(vectors).numpy().astype(np.float32)
        out[block_name] = (feature_vector, names)

    return out


# ============================================================
# DATASET EXTRACTION
# ============================================================


def extract_dataset_features(
    df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    has_label: bool,
) -> Dict[str, pd.DataFrame]:
    prompts = df["prompt"].astype(str).tolist()
    responses = df["response"].astype(str).tolist()
    texts = [p + r for p, r in zip(prompts, responses)]
    prompt_lengths = get_prompt_lengths(tokenizer, prompts, MAX_LENGTH)

    features_by_block: Dict[str, List[np.ndarray]] = {name: [] for name in OUTPUT_SPECS}
    names_by_block: Dict[str, Optional[List[str]]] = {name: None for name in OUTPUT_SPECS}

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extract extended drift features"):
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
            blocks = extract_feature_blocks_for_sample(
                hidden=hidden_batch[i],
                valid_mask=mask_batch[i],
                prompt_len=batch_prompt_lengths[i],
            )

            for block_name, (features, names) in blocks.items():
                features_by_block[block_name].append(features)
                if names_by_block[block_name] is None:
                    names_by_block[block_name] = names

    out: Dict[str, pd.DataFrame] = {}

    for block_name, all_features in features_by_block.items():
        feature_names = names_by_block[block_name]
        assert feature_names is not None

        X = np.vstack(all_features)
        block_df = pd.DataFrame(X, columns=feature_names)
        block_df.insert(0, "source_index", df.index.to_numpy())

        if has_label:
            block_df["label"] = df["label"].astype(float).astype(int).to_numpy()

        block_df["prompt"] = df["prompt"].astype(str).to_numpy()
        block_df["response"] = df["response"].astype(str).to_numpy()

        out[block_name] = block_df

    return out


# ============================================================
# MAIN
# ============================================================


def save_blocks(blocks: Dict[str, pd.DataFrame], output_dir: Path, specs: Dict[str, str]) -> None:
    for block_name, block_df in blocks.items():
        path = output_dir / specs[block_name]
        block_df.to_parquet(path, index=False)
        print(f"Saved {block_name}: {path} | shape={block_df.shape}")


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("=" * 80)
    print("BUILD EXTENDED DRIFT FEATURES")
    print("=" * 80)
    print(f"Device      : {device}")
    print(f"Data file   : {DATA_FILE}")
    print(f"Test file   : {TEST_FILE}")
    print(f"Output dir  : {output_dir}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Adjacent pairs: {ADJACENT_PAIRS}")
    print(f"Long pairs    : {LONG_PAIRS}")

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()

    t0 = time.time()

    df = pd.read_csv(DATA_FILE)
    print(f"\nDataset rows: {len(df)}")
    dataset_blocks = extract_dataset_features(
        df=df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        has_label=True,
    )
    save_blocks(dataset_blocks, output_dir, OUTPUT_SPECS)

    if EXPORT_TEST and Path(TEST_FILE).exists():
        df_test = pd.read_csv(TEST_FILE)
        print(f"\nTest rows: {len(df_test)}")
        test_blocks = extract_dataset_features(
            df=df_test,
            model=model,
            tokenizer=tokenizer,
            device=device,
            has_label=False,
        )
        save_blocks(test_blocks, output_dir, TEST_OUTPUT_SPECS)

    print(f"\nDone in {(time.time() - t0):.1f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
