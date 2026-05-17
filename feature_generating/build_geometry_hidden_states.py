from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from model import MAX_LENGTH, get_model_and_tokenizer

DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"

OUT_DIR = Path("./artifacts/geometric_uncertainty_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = OUT_DIR / "features_dataset_geometric_uncertainty.parquet"
TEST_OUT = OUT_DIR / "features_test_geometric_uncertainty.parquet"

BATCH_SIZE = 4

LAYERS = [11, 12, 13, 14, 15, 16]
PAIRS = list(zip(LAYERS[:-1], LAYERS[1:]))

EPS = 1e-8

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_mean(x, mask):
    if mask.sum().item() == 0:
        return torch.zeros(x.shape[-1], dtype=x.dtype)
    return x[mask].mean(dim=0)


def safe_std(x, mask):
    if mask.sum().item() <= 1:
        return torch.zeros(x.shape[-1], dtype=x.dtype)
    return x[mask].std(dim=0, unbiased=False)


def l2(x):
    return torch.linalg.norm(x.float(), dim=-1)


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().unsqueeze(0),
        b.float().unsqueeze(0),
        dim=1,
    )[0]


def scalar_stats(values, prefix, out):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        values = np.array([0.0], dtype=np.float32)

    out[f"{prefix}_mean"] = float(values.mean())
    out[f"{prefix}_std"] = float(values.std())
    out[f"{prefix}_min"] = float(values.min())
    out[f"{prefix}_max"] = float(values.max())
    out[f"{prefix}_range"] = float(values.max() - values.min())
    out[f"{prefix}_median"] = float(np.median(values))
    out[f"{prefix}_p25"] = float(np.percentile(values, 25))
    out[f"{prefix}_p75"] = float(np.percentile(values, 75))


def make_solution_masks(valid_mask):
    valid_mask = valid_mask.bool().cpu()
    positions = torch.where(valid_mask)[0]

    if positions.numel() == 0:
        return {
            "all": valid_mask,
            "last30": valid_mask,
            "last20": valid_mask,
            "last40": valid_mask,
            "last_token": valid_mask,
            "last5": valid_mask,
            "first70": valid_mask,
        }

    n = int(positions.numel())

    def last_frac(frac):
        m = max(1, int(round(n * frac)))
        mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        mask[positions[-m:]] = True
        return mask

    def last_n(k):
        m = max(1, min(k, n))
        mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        mask[positions[-m:]] = True
        return mask

    last30 = last_frac(0.30)

    first70 = valid_mask & (~last30)
    if first70.sum().item() == 0:
        first70 = valid_mask.clone()

    return {
        "all": valid_mask,
        "last20": last_frac(0.20),
        "last30": last30,
        "last40": last_frac(0.40),
        "last_token": last_n(1),
        "last5": last_n(5),
        "first70": first70,
    }

def extract_uncertainty_features_one(hidden, valid_mask):
    """
    hidden: (n_layers, seq_len, hidden_dim)
    valid_mask: (seq_len,)
    """
    hidden = hidden.float().cpu()
    masks = make_solution_masks(valid_mask)

    out = {}

    means: Dict[str, Dict[int, torch.Tensor]] = {}

    for zone_name, mask in masks.items():
        means[zone_name] = {}
        for layer in LAYERS:
            means[zone_name][layer] = safe_mean(hidden[layer], mask)

    for zone_name in ["all", "last30", "last20", "last40", "last_token", "last5"]:
        jumps = []

        for left, right in PAIRS:
            diff = means[zone_name][right] - means[zone_name][left]
            value = float(torch.linalg.norm(diff).item())
            out[f"jump_energy_{zone_name}_l{left}_to_l{right}"] = value
            jumps.append(value)

        scalar_stats(jumps, f"jump_energy_{zone_name}_trajectory", out)

    for zone_name in ["all", "last30", "last20", "last40"]:
        drift_vectors = []

        for left, right in PAIRS:
            drift_vectors.append((means[zone_name][right] - means[zone_name][left]).numpy())

        drift_matrix = np.vstack(drift_vectors)

        out[f"drift_dim_variance_{zone_name}_mean"] = float(drift_matrix.var(axis=0).mean())
        out[f"drift_dim_variance_{zone_name}_max"] = float(drift_matrix.var(axis=0).max())
        out[f"drift_dim_variance_{zone_name}_std"] = float(drift_matrix.var(axis=0).std())

    for zone_name in ["all", "last30", "last20", "last40", "last_token", "last5"]:
        drift_vecs = [
            means[zone_name][right] - means[zone_name][left]
            for left, right in PAIRS
        ]

        curvatures = []

        for i in range(len(drift_vecs) - 1):
            c = float(cosine(drift_vecs[i], drift_vecs[i + 1]).item())
            out[f"curvature_cos_{zone_name}_{i}"] = c
            curvatures.append(c)

        scalar_stats(curvatures, f"curvature_cos_{zone_name}_trajectory", out)

    for zone_name in ["all", "last30", "last20", "last40", "last_token", "last5"]:
        norms = []

        for layer in LAYERS:
            value = float(torch.linalg.norm(means[zone_name][layer]).item())
            out[f"layer_norm_{zone_name}_l{layer}"] = value
            norms.append(value)

        scalar_stats(norms, f"layer_norm_{zone_name}_trajectory", out)

        for left, right in PAIRS:
            ratio = norms[LAYERS.index(right)] / max(norms[LAYERS.index(left)], EPS)
            out[f"layer_norm_ratio_{zone_name}_l{left}_to_l{right}"] = float(ratio)

    for zone_name in ["all", "last30", "last20", "last40", "last_token", "last5"]:
        cos_values = []

        for left, right in PAIRS:
            value = float(cosine(means[zone_name][left], means[zone_name][right]).item())
            out[f"layer_cosine_{zone_name}_l{left}_to_l{right}"] = value
            cos_values.append(value)

        scalar_stats(cos_values, f"layer_cosine_{zone_name}_trajectory", out)

    for zone_name in ["all", "last30", "last20", "last40", "last5"]:
        mask = masks[zone_name]

        for layer in LAYERS:
            z = hidden[layer][mask].float()

            if z.shape[0] == 0:
                z = hidden[layer][masks["all"]].float()

            if z.shape[0] == 0:
                out[f"token_spread_{zone_name}_l{layer}_mean_var"] = 0.0
                out[f"token_spread_{zone_name}_l{layer}_mean_norm"] = 0.0
                out[f"token_spread_{zone_name}_l{layer}_std_norm"] = 0.0
                continue

            token_norms = torch.linalg.norm(z, dim=1).numpy()

            out[f"token_spread_{zone_name}_l{layer}_mean_var"] = (
                float(z.var(dim=0, unbiased=False).mean().item())
                if z.shape[0] > 1 else 0.0
            )
            out[f"token_spread_{zone_name}_l{layer}_mean_norm"] = float(token_norms.mean())
            out[f"token_spread_{zone_name}_l{layer}_std_norm"] = float(token_norms.std())


    for zone_name in ["all", "last30", "last20", "last40", "last_token", "last5"]:
        jumps = []

        for left, right in PAIRS:
            jumps.append(float(torch.linalg.norm(means[zone_name][right] - means[zone_name][left]).item()))

        early = np.mean(jumps[:2])
        late = np.mean(jumps[-2:])

        out[f"late_instability_{zone_name}_late_minus_early"] = float(late - early)
        out[f"late_instability_{zone_name}_late_div_early"] = float(late / max(early, EPS))
        out[f"late_instability_{zone_name}_late_mean"] = float(late)
        out[f"late_instability_{zone_name}_early_mean"] = float(early)


    for zone_name in ["all", "last30", "last20", "last40", "last_token", "last5"]:
        jump_series = []

        for left, right in PAIRS:
            jump_series.append(float(torch.linalg.norm(means[zone_name][right] - means[zone_name][left]).item()))

        jump_series = np.asarray(jump_series, dtype=np.float32)

        x = np.arange(len(jump_series), dtype=np.float32)
        if len(jump_series) > 1:
            slope = np.polyfit(x, jump_series, 1)[0]
            second_diff = np.diff(jump_series, n=2)
        else:
            slope = 0.0
            second_diff = np.array([0.0], dtype=np.float32)

        out[f"trajectory_slope_{zone_name}"] = float(slope)
        out[f"trajectory_roughness_{zone_name}"] = float(np.abs(np.diff(jump_series)).sum())
        out[f"trajectory_second_diff_abs_mean_{zone_name}"] = float(np.abs(second_diff).mean())
        out[f"trajectory_second_diff_abs_max_{zone_name}"] = float(np.abs(second_diff).max())

    return out

def extract_uncertainty_features_df(df, model, tokenizer, device, has_label):
    texts = [
        str(p) + str(r)
        for p, r in zip(df["prompt"].astype(str), df["response"].astype(str))
    ]

    rows = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extract geometric uncertainty"):
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
            row = extract_uncertainty_features_one(
                hidden=hidden_batch[i],
                valid_mask=mask_batch[i],
            )
            rows.append(row)

    out = pd.DataFrame(rows)
    out.insert(0, "source_index", df.index.to_numpy())

    if has_label:
        out["label"] = df["label"].astype(float).astype(int).to_numpy()

    out["prompt"] = df["prompt"].astype(str).to_numpy()
    out["response"] = df["response"].astype(str).to_numpy()

    return out

device = get_device()
print("Device:", device)

model, tokenizer = get_model_and_tokenizer()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(device)
model.eval()

df_train = pd.read_csv(DATA_FILE)

geom_train = extract_uncertainty_features_df(
    df=df_train,
    model=model,
    tokenizer=tokenizer,
    device=device,
    has_label=True,
)

geom_train.to_parquet(TRAIN_OUT, index=False)
print("Saved:", TRAIN_OUT)
print("Train shape:", geom_train.shape)

df_test = pd.read_csv(TEST_FILE)

geom_test = extract_uncertainty_features_df(
    df=df_test,
    model=model,
    tokenizer=tokenizer,
    device=device,
    has_label=False,
)

geom_test.to_parquet(TEST_OUT, index=False)
print("Saved:", TEST_OUT)
print("Test shape:", geom_test.shape)