# ============================================================
# Build compact solution.py-compatible parquet feature spaces
# ============================================================

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from model import MAX_LENGTH, get_model_and_tokenizer


DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"

OUTPUT_DIR = Path("./artifacts/solution_compatible_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4

SELECTED_LAYERS = [11, 12, 13, 14, 15, 16]
MIDDLE4_LAYERS = [11, 12, 13, 14]
DRIFT_PAIRS = [(11, 12), (12, 13), (13, 14), (14, 15), (15, 16)]

EPS = 1e-8

# ============================================================
# Helpers
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum().item() == 0:
        return torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
    return x[mask].mean(dim=0)


def make_solution_response_masks(valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    solution.py-compatible approximation:
    no prompt_len is available, so response = final 30% of valid tokens.
    """
    valid_mask = valid_mask.bool().cpu()
    valid_positions = torch.where(valid_mask)[0]

    if valid_positions.numel() == 0:
        return valid_mask.clone(), valid_mask.clone()

    n_valid = int(valid_positions.numel())
    response_len = max(1, int(round(n_valid * 0.30)))

    response_positions = valid_positions[-response_len:]

    response_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    response_mask[response_positions] = True

    prompt_mask = valid_mask & (~response_mask)

    if prompt_mask.sum().item() == 0:
        prompt_mask = valid_mask.clone()

    return prompt_mask, response_mask


def normed_diff(right: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
    diff = right - left
    scale = right.abs() + left.abs() + EPS
    return diff / scale

# ============================================================
# Feature extraction for one sample
# ============================================================

def extract_middle4_response_features(
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[str]]:
    _, response_mask = make_solution_response_masks(valid_mask)

    vectors = []
    names = []

    mean_response: Dict[int, torch.Tensor] = {}

    for layer in MIDDLE4_LAYERS:
        mean_response[layer] = safe_mean(hidden[layer], response_mask)

    # middle4 concat
    for layer in MIDDLE4_LAYERS:
        vec = mean_response[layer]
        vectors.append(vec)
        names.extend([f"middle4_response_l{layer}_d{d}" for d in range(vec.numel())])

    # middle4 mean
    middle4_mean = torch.stack([mean_response[layer] for layer in MIDDLE4_LAYERS]).mean(dim=0)
    vectors.append(middle4_mean)
    names.extend([f"middle4_response_mean_d{d}" for d in range(middle4_mean.numel())])

    return vectors, names


def extract_drift_transforms_late_features(
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[str]]:
    _, response_mask = make_solution_response_masks(valid_mask)

    vectors = []
    names = []

    mean_response: Dict[int, torch.Tensor] = {}

    for layer in SELECTED_LAYERS:
        mean_response[layer] = safe_mean(hidden[layer], response_mask)

    for left, right in DRIFT_PAIRS:
        diff = mean_response[right] - mean_response[left]

        feature_variants = {
            "signed": diff,
            "abs": diff.abs(),
            "squared": diff.pow(2),
            "sign": torch.sign(diff),
            "normed": normed_diff(mean_response[right], mean_response[left]),
        }

        for variant_name, vec in feature_variants.items():
            vectors.append(vec)
            names.extend([
                f"drift_{variant_name}_l{left}_to_l{right}_d{d}"
                for d in range(vec.numel())
            ])

    return vectors, names


def extract_middle4_plus_drift_features(
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[np.ndarray, List[str]]:
    vectors = []
    names = []

    v1, n1 = extract_middle4_response_features(hidden, valid_mask)
    v2, n2 = extract_drift_transforms_late_features(hidden, valid_mask)

    vectors.extend(v1)
    vectors.extend(v2)
    names.extend(n1)
    names.extend(n2)

    features = torch.cat([v.float().cpu() for v in vectors], dim=0).numpy().astype(np.float32)
    return features, names


def extract_one_sample_all_spaces(
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    hidden = hidden.float().cpu()
    valid_mask = valid_mask.bool().cpu()

    spaces = {}

    v_mid, n_mid = extract_middle4_response_features(hidden, valid_mask)
    spaces["middle4_response"] = (
        torch.cat([v.float().cpu() for v in v_mid], dim=0).numpy().astype(np.float32),
        n_mid,
    )

    v_drift, n_drift = extract_drift_transforms_late_features(hidden, valid_mask)
    spaces["drift_transforms_late"] = (
        torch.cat([v.float().cpu() for v in v_drift], dim=0).numpy().astype(np.float32),
        n_drift,
    )

    features_combo, names_combo = extract_middle4_plus_drift_features(hidden, valid_mask)
    spaces["middle4_plus_drift"] = (features_combo, names_combo)

    return spaces

# ============================================================
# Dataset extraction
# ============================================================

def extract_feature_spaces_for_df(
    df: pd.DataFrame,
    model,
    tokenizer,
    device,
    has_label: bool,
) -> Dict[str, pd.DataFrame]:
    texts = [
        str(p) + str(r)
        for p, r in zip(df["prompt"].astype(str), df["response"].astype(str))
    ]

    all_features: Dict[str, List[np.ndarray]] = {
        "middle4_response": [],
        "drift_transforms_late": [],
        "middle4_plus_drift": [],
    }

    feature_names: Dict[str, Optional[List[str]]] = {
        "middle4_response": None,
        "drift_transforms_late": None,
        "middle4_plus_drift": None,
    }

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extract feature spaces"):
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden_batch = torch.stack(outputs.hidden_states, dim=1).float().cpu()
        mask_batch = attention_mask.cpu().bool()

        for i in range(hidden_batch.shape[0]):
            spaces = extract_one_sample_all_spaces(
                hidden=hidden_batch[i],
                valid_mask=mask_batch[i],
            )

            for space_name, (features, names) in spaces.items():
                all_features[space_name].append(features)

                if feature_names[space_name] is None:
                    feature_names[space_name] = names

    output_tables = {}

    for space_name, rows in all_features.items():
        names = feature_names[space_name]
        assert names is not None

        X = np.vstack(rows)
        out = pd.DataFrame(X, columns=names)

        out.insert(0, "source_index", df.index.to_numpy())

        if has_label:
            out["label"] = df["label"].astype(float).astype(int).to_numpy()

        out["prompt"] = df["prompt"].astype(str).to_numpy()
        out["response"] = df["response"].astype(str).to_numpy()

        output_tables[space_name] = out

    return output_tables

# ============================================================
# Run extraction and save parquets
# ============================================================

device = get_device()
print("Device:", device)

model, tokenizer = get_model_and_tokenizer()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(device)
model.eval()

df_train = pd.read_csv(DATA_FILE)

train_spaces = extract_feature_spaces_for_df(
    df=df_train,
    model=model,
    tokenizer=tokenizer,
    device=device,
    has_label=True,
)

for space_name, table in train_spaces.items():
    path = OUTPUT_DIR / f"features_dataset_{space_name}.parquet"
    table.to_parquet(path, index=False)
    print(space_name, table.shape, "->", path)


df_test = pd.read_csv(TEST_FILE)

test_spaces = extract_feature_spaces_for_df(
    df=df_test,
    model=model,
    tokenizer=tokenizer,
    device=device,
    has_label=False,
)

for space_name, table in test_spaces.items():
    path = OUTPUT_DIR / f"features_test_{space_name}.parquet"
    table.to_parquet(path, index=False)
    print(space_name, table.shape, "->", path)