"""
Build experimental infrastructure-change features for hallucination detection.

This script intentionally DOES NOT use model.py. It is an advanced feature
family that may require infrastructure changes: eager attention, logits,
forward hooks, and optional second-model / verifier signals.

It creates:
./artifacts/extra_smart_features_infrastructure_change/features_dataset_extra_smart_infrastructure_change.parquet
./artifacts/extra_smart_features_infrastructure_change/features_test_extra_smart_infrastructure_change.parquet

Feature groups covered:
1. Attention features without prompt_len
2. Attention maps with exact prompt_len split
3. Attention grounding decay
4. Retrieval / memory failure proxy
5. Logits entropy
6. Logit competition
7. Token surprise trajectory
8. Confidence trajectory
9. Probability geometry
10. Response-ending uncertainty
11. MLP activation explosion/collapse via hooks
12. True residual stream hooks via hooks
13. Optional cross-model disagreement
14. Lightweight external verifier proxy signals

Notes:
- The primary model is loaded here with attn_implementation="eager".
- prompt_len is computed inside this script from prompt text.
- No labels are used during feature extraction.
- Optional second model can be enabled through SECOND_MODEL_NAME.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = os.environ.get("PRIMARY_MODEL_NAME", "Qwen/Qwen2.5-0.5B")
SECOND_MODEL_NAME = os.environ.get("SECOND_MODEL_NAME", "")

DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"

OUTPUT_DIR = Path("./artifacts/extra_smart_features_infrastructure_change")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUTPUT = OUTPUT_DIR / "features_dataset_extra_smart_infrastructure_change.parquet"
TEST_OUTPUT = OUTPUT_DIR / "features_test_extra_smart_infrastructure_change.parquet"

MAX_LENGTH = 512
BATCH_SIZE = 1
EXPORT_TEST = True

# hidden_states[0] is embeddings; these are the same layer ids we used elsewhere
LAYERS = [10, 11, 12, 13, 14, 15, 16]
LAYER_PAIRS = list(zip(LAYERS[:-1], LAYERS[1:]))
LONG_LAYER_PAIRS = [(10, 12), (11, 13), (12, 14), (13, 15), (14, 16), (10, 16), (11, 16)]

EPS = 1e-8
TOPK = 10


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


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


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
    return tokens.mean(dim=0)


def entropy_from_values(values) -> float:
    values = np.abs(np.asarray(values, dtype=np.float32))
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
    features[f"{prefix}_p95"] = clean_value(np.percentile(values, 95))
    features[f"{prefix}_entropy"] = entropy_from_values(values)


def add_trajectory_features(features: Dict[str, float], prefix: str, values: Iterable[float]) -> None:
    values = np.asarray(list(values), dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)

    add_stats(features, prefix, values)

    diff1 = np.diff(values) if len(values) >= 2 else np.array([0.0], dtype=np.float32)
    diff2 = np.diff(values, n=2) if len(values) >= 3 else np.array([0.0], dtype=np.float32)
    x = np.arange(len(values), dtype=np.float32)
    slope = np.polyfit(x, values, 1)[0] if len(values) >= 2 else 0.0
    early = values[:2].mean() if len(values) >= 2 else values.mean()
    late = values[-2:].mean() if len(values) >= 2 else values.mean()

    features[f"{prefix}_slope"] = clean_value(slope)
    features[f"{prefix}_roughness"] = clean_value(np.abs(diff1).sum())
    features[f"{prefix}_smoothness"] = clean_value(1.0 / (1.0 + np.abs(diff1).sum()))
    features[f"{prefix}_acceleration_mean"] = clean_value(np.abs(diff2).mean())
    features[f"{prefix}_acceleration_max"] = clean_value(np.abs(diff2).max())
    features[f"{prefix}_late_minus_early"] = clean_value(late - early)
    features[f"{prefix}_late_div_early"] = clean_value(late / (abs(early) + EPS))
    features[f"{prefix}_num_increases"] = clean_value((diff1 > 0).sum())
    features[f"{prefix}_num_decreases"] = clean_value((diff1 < 0).sum())

    signs = np.sign(diff1)
    signs = signs[signs != 0]
    if len(signs) >= 2:
        sign_changes = (signs[1:] != signs[:-1]).sum()
    else:
        sign_changes = 0
    features[f"{prefix}_sign_changes"] = clean_value(sign_changes)


def add_spectral_features(features: Dict[str, float], prefix: str, values: Iterable[float]) -> None:
    values = np.asarray(list(values), dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)

    centered = values - values.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    total = power.sum()
    split = max(1, len(power) // 2)
    low = power[:split].sum()
    high = power[split:].sum()

    features[f"{prefix}_fft_energy"] = clean_value(total)
    features[f"{prefix}_fft_low_energy"] = clean_value(low)
    features[f"{prefix}_fft_high_energy"] = clean_value(high)
    features[f"{prefix}_fft_high_low_ratio"] = clean_value(high / (low + EPS))
    features[f"{prefix}_fft_entropy"] = entropy_from_values(power)
    features[f"{prefix}_fft_dominant_freq"] = clean_value(np.argmax(power) / max(len(power) - 1, 1))


def pairwise_cosine_stats(vectors: np.ndarray) -> Dict[str, float]:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape[0] <= 1:
        return {"mean": 1.0, "std": 0.0, "min": 1.0, "p10": 1.0, "p90": 1.0}

    normed = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + EPS)
    sim = normed @ normed.T
    tri = sim[np.triu_indices_from(sim, k=1)]
    return {
        "mean": clean_value(tri.mean()),
        "std": clean_value(tri.std()),
        "min": clean_value(tri.min()),
        "p10": clean_value(np.percentile(tri, 10)),
        "p90": clean_value(np.percentile(tri, 90)),
    }


def get_prompt_lengths(tokenizer, prompts: List[str]) -> List[int]:
    lengths = []
    for prompt in prompts:
        enc = tokenizer(
            str(prompt),
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        lengths.append(len(enc["input_ids"]))
    return lengths


def make_exact_indices(valid_mask: torch.Tensor, prompt_len: int) -> Dict[str, np.ndarray]:
    valid_mask = valid_mask.bool().cpu()
    seq_len = int(valid_mask.shape[0])
    prompt_len = min(max(int(prompt_len), 0), seq_len)
    positions = torch.arange(seq_len)

    prompt_mask = valid_mask & (positions < prompt_len)
    response_mask = valid_mask & (positions >= prompt_len)

    if prompt_mask.sum().item() == 0:
        prompt_mask = valid_mask.clone()
    if response_mask.sum().item() == 0:
        response_mask = valid_mask.clone()

    valid_idx = torch.where(valid_mask)[0].numpy()
    prompt_idx = torch.where(prompt_mask)[0].numpy()
    response_idx = torch.where(response_mask)[0].numpy()

    def split(idx: np.ndarray) -> Dict[str, np.ndarray]:
        n = len(idx)
        if n == 0:
            return {
                "all": idx,
                "early": idx,
                "middle": idx,
                "late": idx,
                "last5": idx,
                "last10": idx,
                "first5": idx,
                "first10": idx,
            }
        third = max(1, n // 3)
        mid_start = third
        late_start = min(n, 2 * third)
        return {
            "all": idx,
            "early": idx[:third],
            "middle": idx[mid_start:late_start] if late_start > mid_start else idx,
            "late": idx[late_start:] if n > late_start else idx[-third:],
            "last5": idx[-min(5, n):],
            "last10": idx[-min(10, n):],
            "first5": idx[:min(5, n)],
            "first10": idx[:min(10, n)],
        }

    zones = {
        "valid": valid_idx,
        "prompt": prompt_idx,
        "response": response_idx,
    }
    for name, idx in split(response_idx).items():
        zones[f"response_{name}"] = idx
    for name, idx in split(prompt_idx).items():
        zones[f"prompt_{name}"] = idx
    return zones


# ============================================================
# MODEL LOADING — autonomous, does not use model.py
# ============================================================


def load_primary_model(device: torch.device):
    print(f"[Primary model] Loading {MODEL_NAME} with eager attentions...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
        output_attentions=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def load_second_model(device: torch.device):
    if not SECOND_MODEL_NAME:
        return None, None
    print(f"[Second model] Loading {SECOND_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(SECOND_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        SECOND_MODEL_NAME,
        output_hidden_states=True,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


# ============================================================
# FORWARD HOOKS
# ============================================================


def discover_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise RuntimeError("Cannot discover transformer layers for hooks.")


def register_hooks(model):
    layers = discover_layers(model)
    captures = {
        "mlp_outputs": {},
        "residual_inputs": {},
        "residual_outputs": {},
    }
    handles = []

    def make_block_hook(layer_idx: int):
        def hook(module, inputs, output):
            if inputs:
                captures["residual_inputs"][layer_idx] = inputs[0].detach().cpu().float()
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output
            captures["residual_outputs"][layer_idx] = out_tensor.detach().cpu().float()
        return hook

    def make_mlp_hook(layer_idx: int):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output
            captures["mlp_outputs"][layer_idx] = out_tensor.detach().cpu().float()
        return hook

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_block_hook(idx)))
        if hasattr(layer, "mlp"):
            handles.append(layer.mlp.register_forward_hook(make_mlp_hook(idx)))

    return handles, captures


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


# ============================================================
# ATTENTION FEATURES
# ============================================================


def attention_vector_stats(features: Dict[str, float], prefix: str, vector: np.ndarray) -> None:
    vector = np.asarray(vector, dtype=np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    total = vector.sum() + EPS
    probs = vector / total
    sorted_probs = np.sort(probs)[::-1]
    features[f"{prefix}_entropy"] = clean_value(-(probs * np.log(probs + EPS)).sum())
    features[f"{prefix}_top1_mass"] = clean_value(sorted_probs[:1].sum())
    features[f"{prefix}_top5_mass"] = clean_value(sorted_probs[:5].sum())
    features[f"{prefix}_concentration"] = clean_value((probs ** 2).sum())
    features[f"{prefix}_effective_support"] = clean_value(1.0 / ((probs ** 2).sum() + EPS))


def local_positions(source_idx: np.ndarray, subset_idx: np.ndarray) -> np.ndarray:
    mapping = {int(pos): i for i, pos in enumerate(source_idx.tolist())}
    return np.array([mapping[int(pos)] for pos in subset_idx if int(pos) in mapping], dtype=np.int64)


def add_attention_features(
    features: Dict[str, float],
    attentions,
    valid_mask: torch.Tensor,
    prompt_len: int,
) -> None:
    if attentions is None:
        features["attention_available"] = 0.0
        return
    features["attention_available"] = 1.0

    zones = make_exact_indices(valid_mask, prompt_len)
    valid_idx = zones["valid"]
    if len(valid_idx) == 0:
        return

    prompt_local = local_positions(valid_idx, zones["prompt"])
    response_local = local_positions(valid_idx, zones["response"])
    late_local = local_positions(valid_idx, zones["response_late"])
    last5_local = local_positions(valid_idx, zones["response_last5"])
    early_local = local_positions(valid_idx, zones["response_early"])

    layer_entropy = []
    layer_concentration = []
    layer_top1 = []
    layer_top5 = []
    layer_head_disagreement = []
    layer_last_token_entropy = []
    layer_resp_to_prompt = []
    layer_late_to_prompt = []
    layer_last5_to_prompt = []
    layer_resp_to_resp = []
    layer_prompt_to_resp = []

    for layer_i, attn in enumerate(attentions):
        # attn shape: (1, heads, seq, seq)
        attn_np = attn.detach().cpu().float().numpy()[0]
        attn_np = attn_np[:, valid_idx][:, :, valid_idx]
        n_heads, n_seq, _ = attn_np.shape

        head_entropy = []
        head_concentration = []
        head_top1 = []
        head_top5 = []
        head_vectors = []
        head_last_entropy = []
        head_resp_prompt = []
        head_late_prompt = []
        head_last5_prompt = []
        head_resp_resp = []
        head_prompt_resp = []

        for head in range(n_heads):
            h = attn_np[head]
            h = h / (h.sum(axis=1, keepdims=True) + EPS)
            sorted_rows = np.sort(h, axis=1)[:, ::-1]
            ent_rows = -(h * np.log(h + EPS)).sum(axis=1)
            concentration_rows = (h ** 2).sum(axis=1)

            head_entropy.append(ent_rows.mean())
            head_concentration.append(concentration_rows.mean())
            head_top1.append(sorted_rows[:, 0].mean())
            head_top5.append(sorted_rows[:, :min(5, sorted_rows.shape[1])].sum(axis=1).mean())
            head_vectors.append(h.mean(axis=0))

            last_row = h[-1]
            attention_vector_stats(features, f"attn_l{layer_i}_h{head}_last_token", last_row)
            head_last_entropy.append(features[f"attn_l{layer_i}_h{head}_last_token_entropy"])

            def mass(query_local: np.ndarray, key_local: np.ndarray) -> float:
                if len(query_local) == 0 or len(key_local) == 0:
                    return 0.0
                return float(h[np.ix_(query_local, key_local)].sum(axis=1).mean())

            head_resp_prompt.append(mass(response_local, prompt_local))
            head_late_prompt.append(mass(late_local, prompt_local))
            head_last5_prompt.append(mass(last5_local, prompt_local))
            head_resp_resp.append(mass(response_local, response_local))
            head_prompt_resp.append(mass(prompt_local, response_local))

        add_stats(features, f"attn_l{layer_i}_entropy", head_entropy)
        add_stats(features, f"attn_l{layer_i}_concentration", head_concentration)
        add_stats(features, f"attn_l{layer_i}_top1_mass", head_top1)
        add_stats(features, f"attn_l{layer_i}_top5_mass", head_top5)
        add_stats(features, f"attn_l{layer_i}_last_token_entropy", head_last_entropy)

        head_vectors = np.asarray(head_vectors, dtype=np.float32)
        pcs = pairwise_cosine_stats(head_vectors)
        features[f"attn_l{layer_i}_head_disagreement"] = clean_value(1.0 - pcs["mean"])
        features[f"attn_l{layer_i}_head_diversity"] = pcs["std"]
        features[f"attn_l{layer_i}_head_collapse"] = pcs["mean"]

        add_stats(features, f"attn_l{layer_i}_response_to_prompt", head_resp_prompt)
        add_stats(features, f"attn_l{layer_i}_late_response_to_prompt", head_late_prompt)
        add_stats(features, f"attn_l{layer_i}_last5_to_prompt", head_last5_prompt)
        add_stats(features, f"attn_l{layer_i}_response_to_response", head_resp_resp)
        add_stats(features, f"attn_l{layer_i}_prompt_to_response", head_prompt_resp)

        # Zone entropy over prompt/response/later keys for response queries.
        mean_h = attn_np.mean(axis=0)
        mean_h = mean_h / (mean_h.sum(axis=1, keepdims=True) + EPS)
        if len(response_local) > 0:
            response_rows = mean_h[response_local]
            response_key_entropy = -(response_rows * np.log(response_rows + EPS)).sum(axis=1)
            add_stats(features, f"attn_l{layer_i}_response_query_entropy", response_key_entropy)
        if len(late_local) > 0:
            late_rows = mean_h[late_local]
            late_key_entropy = -(late_rows * np.log(late_rows + EPS)).sum(axis=1)
            add_stats(features, f"attn_l{layer_i}_late_response_query_entropy", late_key_entropy)

        layer_entropy.append(np.mean(head_entropy))
        layer_concentration.append(np.mean(head_concentration))
        layer_top1.append(np.mean(head_top1))
        layer_top5.append(np.mean(head_top5))
        layer_head_disagreement.append(features[f"attn_l{layer_i}_head_disagreement"])
        layer_last_token_entropy.append(np.mean(head_last_entropy))
        layer_resp_to_prompt.append(np.mean(head_resp_prompt))
        layer_late_to_prompt.append(np.mean(head_late_prompt))
        layer_last5_to_prompt.append(np.mean(head_last5_prompt))
        layer_resp_to_resp.append(np.mean(head_resp_resp))
        layer_prompt_to_resp.append(np.mean(head_prompt_resp))

    trajectories = {
        "attn_entropy_layers": layer_entropy,
        "attn_concentration_layers": layer_concentration,
        "attn_top1_layers": layer_top1,
        "attn_top5_layers": layer_top5,
        "attn_head_disagreement_layers": layer_head_disagreement,
        "attn_last_token_entropy_layers": layer_last_token_entropy,
        "attn_response_to_prompt_layers": layer_resp_to_prompt,
        "attn_late_to_prompt_layers": layer_late_to_prompt,
        "attn_last5_to_prompt_layers": layer_last5_to_prompt,
        "attn_response_to_response_layers": layer_resp_to_resp,
        "attn_prompt_to_response_layers": layer_prompt_to_resp,
    }

    for name, values in trajectories.items():
        add_trajectory_features(features, name, values)
        add_spectral_features(features, name, values)

    # Grounding decay / forgetting / re-grounding.
    response_to_prompt = np.asarray(layer_resp_to_prompt, dtype=np.float32)
    late_to_prompt = np.asarray(layer_late_to_prompt, dtype=np.float32)
    last5_to_prompt = np.asarray(layer_last5_to_prompt, dtype=np.float32)

    features["grounding_late_prompt_collapse"] = clean_value(response_to_prompt.mean() - late_to_prompt.mean())
    features["grounding_last5_prompt_collapse"] = clean_value(response_to_prompt.mean() - last5_to_prompt.mean())
    features["attention_forgetting_score"] = clean_value(response_to_prompt[:2].mean() - response_to_prompt[-2:].mean())
    features["attention_regrounding_score"] = clean_value(response_to_prompt[-1] - response_to_prompt.min()) if len(response_to_prompt) else 0.0
    features["attention_to_context_collapse"] = clean_value(layer_resp_to_prompt[-1] - layer_resp_to_prompt[0]) if len(layer_resp_to_prompt) >= 2 else 0.0

    # Attention drift across response positions: early/late query prompt mass.
    if len(early_local) > 0 and len(late_local) > 0 and len(prompt_local) > 0:
        drift_values = []
        for attn in attentions:
            attn_np = attn.detach().cpu().float().numpy()[0]
            attn_np = attn_np[:, valid_idx][:, :, valid_idx]
            h = attn_np.mean(axis=0)
            h = h / (h.sum(axis=1, keepdims=True) + EPS)
            early_mass = h[np.ix_(early_local, prompt_local)].sum(axis=1).mean()
            late_mass = h[np.ix_(late_local, prompt_local)].sum(axis=1).mean()
            drift_values.append(late_mass - early_mass)
        add_trajectory_features(features, "attn_response_position_grounding_drift", drift_values)
        add_spectral_features(features, "attn_response_position_grounding_drift", drift_values)


# ============================================================
# RETRIEVAL / MEMORY FAILURE PROXY
# ============================================================


ANCHOR_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on", "for", "with",
    "as", "by", "from", "at", "it", "its", "into", "about", "can", "could", "should", "would",
    "not", "no", "yes", "you", "your", "we", "our", "they", "their", "i", "he", "she", "them",
}


def extract_prompt_anchor_terms(prompt_text: str, max_terms: int = 16) -> List[str]:
    """Lightweight entity/noun proxy: named-looking spans, numbers, and long content words."""
    text = str(prompt_text)
    named = re.findall(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,3}\b", text)
    numbers = re.findall(r"\b\d+(?:[.,:/-]\d+)*\b", text)
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b", text)
    content_words = [w for w in words if w.lower() not in ANCHOR_STOPWORDS]

    # Keep frequency-rich content words as a noun/entity proxy without requiring POS tagging.
    frequent = [w for w, _ in Counter(w.lower() for w in content_words).most_common(max_terms)]
    candidates = named + numbers + frequent

    out = []
    seen = set()
    for term in candidates:
        norm = term.strip().lower()
        if len(norm) < 2 or norm in seen:
            continue
        seen.add(norm)
        out.append(term.strip())
        if len(out) >= max_terms:
            break
    return out


def find_token_subsequence_positions(haystack: List[int], needle: List[int]) -> List[int]:
    if not needle or not haystack or len(needle) > len(haystack):
        return []
    out = []
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            out.extend(range(i, i + n))
    return out


def prompt_anchor_token_positions(
    prompt_text: str,
    tokenizer,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> np.ndarray:
    prompt_len = max(0, min(int(prompt_len), int(input_ids.shape[0])))
    prompt_ids = input_ids.detach().cpu().tolist()[:prompt_len]
    anchors = extract_prompt_anchor_terms(prompt_text)
    positions = set()

    for term in anchors:
        encodings = [
            tokenizer(str(term), add_special_tokens=False).get("input_ids", []),
            tokenizer(" " + str(term), add_special_tokens=False).get("input_ids", []),
        ]
        for term_ids in encodings:
            for pos in find_token_subsequence_positions(prompt_ids, term_ids):
                positions.add(pos)

    if not positions:
        # Fallback to sparse prompt content positions. Still a prompt-anchor proxy, not a full prompt zone.
        keep = min(prompt_len, 12)
        positions.update(range(keep))

    return np.array(sorted(p for p in positions if 0 <= p < prompt_len), dtype=np.int64)


def add_retrieval_memory_proxy_features(
    features: Dict[str, float],
    attentions,
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    prompt_len: int,
    prompt_text: str,
    response_text: str,
    tokenizer,
) -> None:
    zones = make_exact_indices(valid_mask, prompt_len)
    valid_idx = zones["valid"]
    if len(valid_idx) == 0:
        features["retrieval_proxy_available"] = 0.0
        return

    features["retrieval_proxy_available"] = 1.0
    anchor_idx = prompt_anchor_token_positions(prompt_text, tokenizer, input_ids, prompt_len)
    anchor_idx = np.array([p for p in anchor_idx if p in set(zones["prompt"].tolist())], dtype=np.int64)
    if len(anchor_idx) == 0:
        anchor_idx = zones["prompt_first10"]

    features["retrieval_anchor_count"] = clean_value(len(anchor_idx))
    prompt_terms = extract_prompt_anchor_terms(prompt_text)
    response_lower = str(response_text).lower()
    if prompt_terms:
        matched = sum(1 for term in prompt_terms if str(term).lower() in response_lower)
        features["retrieval_entity_surface_recall"] = clean_value(matched / max(len(prompt_terms), 1))
    else:
        features["retrieval_entity_surface_recall"] = 0.0

    prompt_local = local_positions(valid_idx, zones["prompt"])
    anchor_local = local_positions(valid_idx, anchor_idx)
    response_local = local_positions(valid_idx, zones["response"])
    late_local = local_positions(valid_idx, zones["response_late"])
    last5_local = local_positions(valid_idx, zones["response_last5"])

    anchor_mass_series = []
    late_anchor_mass_series = []
    last5_anchor_mass_series = []
    prompt_mass_series = []
    anchor_entropy_series = []

    if attentions is not None:
        for layer_i, attn in enumerate(attentions):
            attn_np = attn.detach().cpu().float().numpy()[0]
            attn_np = attn_np[:, valid_idx][:, :, valid_idx]
            mean_h = attn_np.mean(axis=0)
            mean_h = mean_h / (mean_h.sum(axis=1, keepdims=True) + EPS)

            def mass(query_local: np.ndarray, key_local: np.ndarray) -> float:
                if len(query_local) == 0 or len(key_local) == 0:
                    return 0.0
                return float(mean_h[np.ix_(query_local, key_local)].sum(axis=1).mean())

            anchor_mass = mass(response_local, anchor_local)
            late_anchor_mass = mass(late_local, anchor_local)
            last5_anchor_mass = mass(last5_local, anchor_local)
            prompt_mass = mass(response_local, prompt_local)

            anchor_mass_series.append(anchor_mass)
            late_anchor_mass_series.append(late_anchor_mass)
            last5_anchor_mass_series.append(last5_anchor_mass)
            prompt_mass_series.append(prompt_mass)

            if len(response_local) > 0 and len(anchor_local) > 0:
                anchor_rows = mean_h[np.ix_(response_local, anchor_local)]
                row_sums = anchor_rows.sum(axis=1, keepdims=True) + EPS
                anchor_probs = anchor_rows / row_sums
                anchor_entropy = (-(anchor_probs * np.log(anchor_probs + EPS)).sum(axis=1)).mean()
            else:
                anchor_entropy = 0.0
            anchor_entropy_series.append(anchor_entropy)

            features[f"retrieval_l{layer_i}_response_to_anchor_mass"] = clean_value(anchor_mass)
            features[f"retrieval_l{layer_i}_late_to_anchor_mass"] = clean_value(late_anchor_mass)
            features[f"retrieval_l{layer_i}_last5_to_anchor_mass"] = clean_value(last5_anchor_mass)
            features[f"retrieval_l{layer_i}_anchor_attention_entropy"] = clean_value(anchor_entropy)

        add_trajectory_features(features, "retrieval_anchor_attention_persistence", anchor_mass_series)
        add_spectral_features(features, "retrieval_anchor_attention_persistence", anchor_mass_series)
        add_trajectory_features(features, "retrieval_late_anchor_attention_persistence", late_anchor_mass_series)
        add_trajectory_features(features, "retrieval_last5_anchor_attention_persistence", last5_anchor_mass_series)
        add_trajectory_features(features, "retrieval_anchor_attention_entropy", anchor_entropy_series)
        features["retrieval_prompt_anchor_decay"] = clean_value(anchor_mass_series[0] - anchor_mass_series[-1]) if len(anchor_mass_series) >= 2 else 0.0
        features["retrieval_factual_anchor_attention_collapse"] = clean_value(np.mean(anchor_mass_series) - anchor_mass_series[-1]) if len(anchor_mass_series) else 0.0
        features["retrieval_anchor_stability"] = clean_value(1.0 / (1.0 + np.std(anchor_mass_series))) if len(anchor_mass_series) else 0.0
        features["retrieval_entity_grounding_persistence"] = clean_value(np.mean(anchor_mass_series)) if len(anchor_mass_series) else 0.0
        features["retrieval_noun_entity_attention_persistence"] = clean_value(np.mean(late_anchor_mass_series)) if len(late_anchor_mass_series) else 0.0
    else:
        features["retrieval_prompt_anchor_decay"] = 0.0
        features["retrieval_factual_anchor_attention_collapse"] = 0.0
        features["retrieval_anchor_stability"] = 0.0
        features["retrieval_entity_grounding_persistence"] = 0.0
        features["retrieval_noun_entity_attention_persistence"] = 0.0

    hidden = hidden.detach().cpu().float()
    anchor_t = torch.tensor(anchor_idx, dtype=torch.long)
    response_t = torch.tensor(zones["response"], dtype=torch.long)
    late_t = torch.tensor(zones["response_late"], dtype=torch.long)
    last5_t = torch.tensor(zones["response_last5"], dtype=torch.long)

    semantic_anchor_drift = []
    semantic_late_anchor_drift = []
    semantic_last5_anchor_drift = []
    for layer in LAYERS:
        anchor_mean = safe_mean(hidden[layer, anchor_t]) if anchor_t.numel() else safe_mean(hidden[layer, torch.tensor(zones["prompt"], dtype=torch.long)])
        response_mean = safe_mean(hidden[layer, response_t]) if response_t.numel() else safe_mean(hidden[layer])
        late_mean = safe_mean(hidden[layer, late_t]) if late_t.numel() else response_mean
        last5_mean = safe_mean(hidden[layer, last5_t]) if last5_t.numel() else late_mean

        semantic_anchor_drift.append(safe_l2(response_mean - anchor_mean))
        semantic_late_anchor_drift.append(safe_l2(late_mean - anchor_mean))
        semantic_last5_anchor_drift.append(safe_l2(last5_mean - anchor_mean))
        features[f"retrieval_l{layer}_semantic_grounding_drift_response_anchor"] = semantic_anchor_drift[-1]
        features[f"retrieval_l{layer}_semantic_grounding_drift_late_anchor"] = semantic_late_anchor_drift[-1]
        features[f"retrieval_l{layer}_semantic_grounding_cos_late_anchor"] = safe_cosine(late_mean, anchor_mean)

    add_trajectory_features(features, "retrieval_semantic_grounding_drift_response_anchor", semantic_anchor_drift)
    add_spectral_features(features, "retrieval_semantic_grounding_drift_response_anchor", semantic_anchor_drift)
    add_trajectory_features(features, "retrieval_semantic_grounding_drift_late_anchor", semantic_late_anchor_drift)
    add_trajectory_features(features, "retrieval_semantic_grounding_drift_last5_anchor", semantic_last5_anchor_drift)


# ============================================================
# LOGITS / PROBABILITY FEATURES
# ============================================================


def logits_to_stats(
    features: Dict[str, float],
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    prompt_len: int,
) -> None:
    # logits shape for one sample: seq, vocab. logits[t] predicts token t+1.
    logits = logits.detach().cpu().float()
    input_ids = input_ids.detach().cpu()
    zones = make_exact_indices(valid_mask, prompt_len)

    # Predictive positions aligned to actual next token.
    seq_len = logits.shape[0]
    pred_pos = np.arange(0, seq_len - 1)
    actual_tokens = input_ids[1:seq_len]

    valid_next = valid_mask[1:seq_len].bool().cpu().numpy()
    response_next = np.isin(np.arange(1, seq_len), zones["response"])
    mask = valid_next & response_next
    selected_pos = pred_pos[mask]

    if len(selected_pos) == 0:
        selected_pos = pred_pos[valid_next]
        mask = valid_next

    if len(selected_pos) == 0:
        features["logits_available"] = 0.0
        return

    features["logits_available"] = 1.0

    selected_logits = logits[selected_pos]
    selected_actual = actual_tokens[mask]

    log_probs = torch.log_softmax(selected_logits, dim=-1)
    probs = torch.softmax(selected_logits, dim=-1)

    entropy_vals = (-(probs * log_probs).sum(dim=-1)).numpy()
    top_probs, top_idx = torch.topk(probs, k=min(TOPK, probs.shape[-1]), dim=-1)
    top_probs_np = top_probs.numpy()

    top1 = top_probs_np[:, 0]
    top2 = top_probs_np[:, 1] if top_probs_np.shape[1] > 1 else np.zeros_like(top1)
    margin = top1 - top2
    top5_mass = top_probs_np[:, :min(5, top_probs_np.shape[1])].sum(axis=1)
    topk_mass = top_probs_np.sum(axis=1)

    actual_log_probs = log_probs[torch.arange(len(selected_actual)), selected_actual].numpy()
    actual_probs = np.exp(actual_log_probs)
    surprisal = -actual_log_probs

    add_trajectory_features(features, "logits_entropy_response", entropy_vals)
    add_spectral_features(features, "logits_entropy_response", entropy_vals)
    add_trajectory_features(features, "logits_top1_prob_response", top1)
    add_trajectory_features(features, "logits_top2_prob_response", top2)
    add_trajectory_features(features, "logits_margin_response", margin)
    add_spectral_features(features, "logits_margin_response", margin)
    add_trajectory_features(features, "logits_top5_mass_response", top5_mass)
    add_trajectory_features(features, "logits_topk_mass_response", topk_mass)
    add_trajectory_features(features, "logits_surprisal_response", surprisal)
    add_spectral_features(features, "logits_surprisal_response", surprisal)
    add_trajectory_features(features, "logits_actual_token_prob_response", actual_probs)

    features["logits_entropy_last_token"] = clean_value(entropy_vals[-1])
    features["logits_entropy_last5_mean"] = clean_value(entropy_vals[-min(5, len(entropy_vals)):].mean())
    features["logits_entropy_late_spike"] = clean_value(entropy_vals[-min(5, len(entropy_vals)):].max() - entropy_vals.mean())
    features["logits_final_token_surprise"] = clean_value(surprisal[-1])
    features["logits_final5_surprise_mean"] = clean_value(surprisal[-min(5, len(surprisal)):].mean())
    features["logits_late_uncertainty_growth"] = clean_value(entropy_vals[-min(5, len(entropy_vals)):].mean() - entropy_vals[:min(5, len(entropy_vals))].mean())
    features["logits_late_confidence_collapse"] = clean_value(top1[:min(5, len(top1))].mean() - top1[-min(5, len(top1)):].mean())
    features["logits_winner_instability"] = clean_value((np.diff(top_idx[:, 0].numpy()) != 0).mean() if len(top_idx) > 1 else 0.0)
    features["logits_probability_spread_mean"] = clean_value(top1.mean() - top_probs_np[:, -1].mean())

    # Probability geometry across adjacent positions.
    if selected_logits.shape[0] >= 2:
        p1 = probs[:-1]
        p2 = probs[1:]
        log_p1 = torch.log(p1 + EPS)
        log_p2 = torch.log(p2 + EPS)
        kl12 = (p1 * (log_p1 - log_p2)).sum(dim=-1).numpy()
        kl21 = (p2 * (log_p2 - log_p1)).sum(dim=-1).numpy()
        m = 0.5 * (p1 + p2)
        js = 0.5 * (p1 * (torch.log(p1 + EPS) - torch.log(m + EPS))).sum(dim=-1)
        js += 0.5 * (p2 * (torch.log(p2 + EPS) - torch.log(m + EPS))).sum(dim=-1)
        js = js.numpy()
        add_trajectory_features(features, "prob_kl_adjacent", kl12)
        add_trajectory_features(features, "prob_symmetric_kl_adjacent", kl12 + kl21)
        add_trajectory_features(features, "prob_js_adjacent", js)
        add_spectral_features(features, "prob_js_adjacent", js)
    else:
        add_trajectory_features(features, "prob_kl_adjacent", [0.0])
        add_trajectory_features(features, "prob_symmetric_kl_adjacent", [0.0])
        add_trajectory_features(features, "prob_js_adjacent", [0.0])

    confidence = top1
    add_trajectory_features(features, "confidence_trajectory", confidence)
    add_spectral_features(features, "confidence_trajectory", confidence)
    features["confidence_decay"] = clean_value(confidence[:min(5, len(confidence))].mean() - confidence[-min(5, len(confidence)):].mean())
    features["confidence_collapse"] = clean_value(confidence.max() - confidence[-min(5, len(confidence)):].mean())
    features["confidence_instability"] = clean_value(np.diff(confidence).std() if len(confidence) > 1 else 0.0)


# ============================================================
# HOOK FEATURES
# ============================================================


def add_tensor_token_stats(features: Dict[str, float], prefix: str, tensor: torch.Tensor, valid_mask: torch.Tensor) -> None:
    tensor = tensor.detach().cpu().float()
    if tensor.dim() == 3:
        tensor = tensor[0]
    idx = torch.where(valid_mask.bool().cpu())[0]
    if idx.numel() == 0:
        idx = torch.arange(tensor.shape[0])
    tokens = tensor[idx]
    norms = torch.linalg.norm(tokens, dim=-1).numpy()
    add_stats(features, f"{prefix}_norm", norms)
    add_trajectory_features(features, f"{prefix}_norm_position", norms)
    add_spectral_features(features, f"{prefix}_norm_position", norms)

    flat = tokens.numpy().reshape(-1)
    abs_flat = np.abs(flat)
    features[f"{prefix}_activation_abs_mean"] = clean_value(abs_flat.mean())
    features[f"{prefix}_activation_sparsity_1e_3"] = clean_value((abs_flat < 1e-3).mean())
    features[f"{prefix}_activation_sparsity_1e_2"] = clean_value((abs_flat < 1e-2).mean())
    features[f"{prefix}_activation_entropy"] = entropy_from_values(abs_flat[: min(len(abs_flat), 20000)])


def add_hook_features(features: Dict[str, float], captures, valid_mask: torch.Tensor) -> None:
    mlp = captures.get("mlp_outputs", {})
    res_in = captures.get("residual_inputs", {})
    res_out = captures.get("residual_outputs", {})

    features["hooks_mlp_available"] = clean_value(len(mlp) > 0)
    features["hooks_residual_available"] = clean_value(len(res_in) > 0 and len(res_out) > 0)

    mlp_layer_norm_means = []
    mlp_explosion = []
    mlp_collapse = []

    for layer_idx in sorted(mlp.keys()):
        tensor = mlp[layer_idx]
        prefix = f"mlp_l{layer_idx}"
        add_tensor_token_stats(features, prefix, tensor, valid_mask)
        norms = torch.linalg.norm(tensor[0] if tensor.dim() == 3 else tensor, dim=-1).numpy()
        mean_norm = float(norms.mean())
        mlp_layer_norm_means.append(mean_norm)
        if len(mlp_layer_norm_means) > 1:
            prev = mlp_layer_norm_means[-2]
            mlp_explosion.append(max(0.0, mean_norm - prev))
            mlp_collapse.append(max(0.0, prev - mean_norm))

    add_trajectory_features(features, "mlp_layer_norm_trajectory", mlp_layer_norm_means)
    add_spectral_features(features, "mlp_layer_norm_trajectory", mlp_layer_norm_means)
    add_trajectory_features(features, "mlp_explosion_score_trajectory", mlp_explosion)
    add_trajectory_features(features, "mlp_collapse_score_trajectory", mlp_collapse)

    residual_update_norms = []
    residual_semantic_cosines = []
    residual_explosion = []
    residual_collapse = []

    for layer_idx in sorted(set(res_in.keys()) & set(res_out.keys())):
        before = res_in[layer_idx]
        after = res_out[layer_idx]
        update = after - before
        prefix = f"true_residual_l{layer_idx}"
        add_tensor_token_stats(features, f"{prefix}_update", update, valid_mask)
        add_tensor_token_stats(features, f"{prefix}_before", before, valid_mask)
        add_tensor_token_stats(features, f"{prefix}_after", after, valid_mask)

        before_mean = before[0].mean(dim=0) if before.dim() == 3 else before.mean(dim=0)
        after_mean = after[0].mean(dim=0) if after.dim() == 3 else after.mean(dim=0)
        update_mean = after_mean - before_mean
        update_norm = safe_l2(update_mean)
        before_norm = safe_l2(before_mean)
        after_norm = safe_l2(after_mean)
        residual_update_norms.append(update_norm)
        residual_semantic_cosines.append(safe_cosine(before_mean, after_mean))
        residual_explosion.append(max(0.0, after_norm - before_norm))
        residual_collapse.append(max(0.0, before_norm - after_norm))

        features[f"{prefix}_semantic_drift"] = update_norm
        features[f"{prefix}_before_after_cosine"] = safe_cosine(before_mean, after_mean)
        features[f"{prefix}_explosion"] = clean_value(max(0.0, after_norm - before_norm))
        features[f"{prefix}_collapse"] = clean_value(max(0.0, before_norm - after_norm))

    add_trajectory_features(features, "true_residual_update_norms", residual_update_norms)
    add_spectral_features(features, "true_residual_update_norms", residual_update_norms)
    add_trajectory_features(features, "true_residual_semantic_cosines", residual_semantic_cosines)
    add_trajectory_features(features, "true_residual_explosion", residual_explosion)
    add_trajectory_features(features, "true_residual_collapse", residual_collapse)

    if residual_update_norms:
        idx = int(np.argmax(residual_update_norms))
        features["true_residual_localization_idx"] = clean_value(idx / max(len(residual_update_norms) - 1, 1))
        features["true_residual_localization_max_update"] = clean_value(max(residual_update_norms))


# ============================================================
# CROSS-MODEL FEATURES
# ============================================================


def add_cross_model_features(
    features: Dict[str, float],
    primary_logits: torch.Tensor,
    primary_hidden: torch.Tensor,
    second_outputs,
    input_ids: torch.Tensor,
    second_input_ids: Optional[torch.Tensor],
    second_attention_mask: Optional[torch.Tensor],
    valid_mask: torch.Tensor,
    prompt_len: int,
    second_prompt_len: Optional[int],
) -> None:
    if second_outputs is None or second_input_ids is None or second_attention_mask is None:
        features["cross_model_available"] = 0.0
        return

    features["cross_model_available"] = 1.0
    second_logits = second_outputs.logits.detach().cpu().float()[0]
    primary_logits = primary_logits.detach().cpu().float()
    input_ids = input_ids.detach().cpu()
    second_input_ids = second_input_ids.detach().cpu()
    second_mask = second_attention_mask.detach().cpu().bool()

    primary_zones = make_exact_indices(valid_mask, prompt_len)
    second_zones = make_exact_indices(second_mask, int(second_prompt_len or prompt_len))

    # Primary model token-level uncertainty on the primary tokenization.
    p_seq_len = min(primary_logits.shape[0], input_ids.shape[0])
    p_pred_pos = np.arange(0, p_seq_len - 1)
    p_actual = input_ids[1:p_seq_len]
    p_valid_next = valid_mask[1:p_seq_len].bool().cpu().numpy()
    p_response_next = np.isin(np.arange(1, p_seq_len), primary_zones["response"])
    p_mask = p_valid_next & p_response_next
    p_selected = p_pred_pos[p_mask]

    # Second model perplexity/surprisal on its own tokenization. This avoids mixing vocabularies.
    s_seq_len = min(second_logits.shape[0], second_input_ids.shape[0])
    s_pred_pos = np.arange(0, s_seq_len - 1)
    s_actual = second_input_ids[1:s_seq_len]
    s_valid_next = second_mask[1:s_seq_len].numpy()
    s_response_next = np.isin(np.arange(1, s_seq_len), second_zones["response"])
    s_mask = s_valid_next & s_response_next
    s_selected = s_pred_pos[s_mask]

    if len(p_selected) == 0 or len(s_selected) == 0:
        features["cross_model_available"] = 0.0
        return

    p_log_probs = torch.log_softmax(primary_logits[p_selected], dim=-1)
    p_probs = p_log_probs.exp()
    p_surprisal = -p_log_probs[torch.arange(len(p_selected)), p_actual[p_mask]].numpy()
    p_entropy = (-(p_probs * p_log_probs).sum(dim=-1)).numpy()
    p_top = torch.argmax(p_probs, dim=-1).numpy()
    p_conf = torch.max(p_probs, dim=-1).values.numpy()

    s_log_probs = torch.log_softmax(second_logits[s_selected], dim=-1)
    s_probs = s_log_probs.exp()
    s_surprisal = -s_log_probs[torch.arange(len(s_selected)), s_actual[s_mask]].numpy()
    s_entropy = (-(s_probs * s_log_probs).sum(dim=-1)).numpy()
    s_top = torch.argmax(s_probs, dim=-1).numpy()
    s_conf = torch.max(s_probs, dim=-1).values.numpy()

    features["cross_model_primary_perplexity"] = clean_value(math.exp(float(np.mean(p_surprisal))))
    features["cross_model_second_perplexity"] = clean_value(math.exp(float(np.mean(s_surprisal))))
    features["cross_model_perplexity_ratio_second_primary"] = clean_value(
        features["cross_model_second_perplexity"] / (features["cross_model_primary_perplexity"] + EPS)
    )

    add_trajectory_features(features, "cross_model_primary_surprisal", p_surprisal)
    add_trajectory_features(features, "cross_model_second_surprisal", s_surprisal)

    # Compare trajectories after length-normalized interpolation.
    common_len = max(2, min(len(p_surprisal), len(s_surprisal)))
    grid = np.linspace(0.0, 1.0, common_len)
    p_grid = np.interp(grid, np.linspace(0.0, 1.0, len(p_surprisal)), p_surprisal)
    s_grid = np.interp(grid, np.linspace(0.0, 1.0, len(s_surprisal)), s_surprisal)
    p_entropy_grid = np.interp(grid, np.linspace(0.0, 1.0, len(p_entropy)), p_entropy)
    s_entropy_grid = np.interp(grid, np.linspace(0.0, 1.0, len(s_entropy)), s_entropy)
    p_conf_grid = np.interp(grid, np.linspace(0.0, 1.0, len(p_conf)), p_conf)
    s_conf_grid = np.interp(grid, np.linspace(0.0, 1.0, len(s_conf)), s_conf)

    add_trajectory_features(features, "cross_model_surprisal_diff", s_grid - p_grid)
    add_trajectory_features(features, "cross_model_entropy_diff", s_entropy_grid - p_entropy_grid)
    add_trajectory_features(features, "cross_model_confidence_disagreement", np.abs(p_conf_grid - s_conf_grid))

    # True KL is only valid if vocab dimensions align.
    if primary_logits.shape[-1] == second_logits.shape[-1]:
        aligned_len = min(len(p_selected), len(s_selected))
        p_lp = torch.log_softmax(primary_logits[p_selected[:aligned_len]], dim=-1)
        s_lp = torch.log_softmax(second_logits[s_selected[:aligned_len]], dim=-1)
        p_pr = p_lp.exp()
        kl = (p_pr * (p_lp - s_lp)).sum(dim=-1).numpy()
        add_trajectory_features(features, "cross_model_kl_primary_to_second", kl)
    else:
        features["cross_model_kl_primary_to_second_vocab_mismatch"] = 1.0

    aligned_top_len = min(len(p_top), len(s_top))
    if aligned_top_len:
        features["cross_model_token_disagreement_rate"] = clean_value(
            (p_top[:aligned_top_len] != s_top[:aligned_top_len]).mean()
        )
        features["cross_model_agreement_collapse_near_end"] = clean_value(
            (p_top[-min(5, aligned_top_len):] != s_top[-min(5, aligned_top_len):]).mean()
        )

    # Hidden-space disagreement. Exact cosine if dimensions match, norm-profile proxy otherwise.
    if hasattr(second_outputs, "hidden_states") and second_outputs.hidden_states is not None:
        features["cross_model_hidden_available"] = 1.0
        p_hidden = primary_hidden.detach().cpu().float()
        s_hidden_stack = torch.stack(second_outputs.hidden_states, dim=1).detach().cpu().float()[0]
        p_resp = torch.tensor(primary_zones["response"], dtype=torch.long)
        s_resp = torch.tensor(second_zones["response"], dtype=torch.long)
        hidden_disagreements = []
        hidden_norm_diffs = []
        for p_layer, s_layer in zip(LAYERS, LAYERS):
            if p_layer >= p_hidden.shape[0] or s_layer >= s_hidden_stack.shape[0]:
                continue
            p_tokens = p_hidden[p_layer, p_resp] if p_resp.numel() else p_hidden[p_layer]
            s_tokens = s_hidden_stack[s_layer, s_resp] if s_resp.numel() else s_hidden_stack[s_layer]
            p_mean = safe_mean(p_tokens)
            s_mean = safe_mean(s_tokens)
            hidden_norm_diffs.append(abs(safe_l2(p_mean) - safe_l2(s_mean)))
            if p_mean.shape[-1] == s_mean.shape[-1]:
                hidden_disagreements.append(1.0 - safe_cosine(p_mean, s_mean))
        if hidden_disagreements:
            add_trajectory_features(features, "cross_model_hidden_space_disagreement", hidden_disagreements)
            add_spectral_features(features, "cross_model_hidden_space_disagreement", hidden_disagreements)
        add_trajectory_features(features, "cross_model_hidden_norm_disagreement", hidden_norm_diffs)
    else:
        features["cross_model_hidden_available"] = 0.0


# ============================================================
# EXTERNAL VERIFIER PROXY SIGNALS
# ============================================================


def text_content_terms(text: str) -> List[str]:
    terms = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b", str(text).lower())
    return [t for t in terms if t not in ANCHOR_STOPWORDS]


def add_external_verifier_signals(
    features: Dict[str, float],
    prompt_text: str,
    response_text: str,
) -> None:
    """Lightweight local verifier proxies; no labels and no external API calls."""
    prompt_terms = text_content_terms(prompt_text)
    response_terms = text_content_terms(response_text)
    prompt_set = set(prompt_terms)
    response_set = set(response_terms)

    overlap = len(prompt_set & response_set)
    response_coverage = overlap / max(len(response_set), 1)
    prompt_recall = overlap / max(len(prompt_set), 1)

    anchors = extract_prompt_anchor_terms(prompt_text)
    anchor_hits = sum(1 for term in anchors if str(term).lower() in str(response_text).lower())
    anchor_recall = anchor_hits / max(len(anchors), 1)

    contradiction_patterns = [
        r"\bhowever\b", r"\bbut\b", r"\balthough\b", r"\bnevertheless\b",
        r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bincorrect\b", r"\bfalse\b",
        r"\bon the other hand\b", r"\bcontradict\w*\b",
    ]
    response_lower = str(response_text).lower()
    contradiction_hits = sum(len(re.findall(pattern, response_lower)) for pattern in contradiction_patterns)
    contradiction_score = contradiction_hits / max(len(response_terms), 1)

    # Entailment proxy: enough prompt anchor preservation and broad lexical overlap.
    entailment_proxy = 0.5 * anchor_recall + 0.5 * prompt_recall
    factual_consistency_proxy = 0.6 * anchor_recall + 0.4 * response_coverage
    retrieval_proxy = 0.7 * anchor_recall + 0.3 * prompt_recall

    features["external_verifier_available"] = 1.0
    features["external_retrieval_verifier_score"] = clean_value(retrieval_proxy)
    features["external_factual_consistency_score"] = clean_value(factual_consistency_proxy)
    features["external_contradiction_verifier_score"] = clean_value(contradiction_score)
    features["external_semantic_entailment_score"] = clean_value(entailment_proxy)
    features["external_prompt_term_recall"] = clean_value(prompt_recall)
    features["external_response_grounded_term_ratio"] = clean_value(response_coverage)
    features["external_anchor_recall"] = clean_value(anchor_recall)


# ============================================================
# SAMPLE EXTRACTION
# ============================================================


def extract_features_one_sample(
    hidden: torch.Tensor,
    attentions,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    prompt_len: int,
    prompt_text: str,
    response_text: str,
    tokenizer,
    captures,
    second_outputs,
    second_input_ids: Optional[torch.Tensor],
    second_attention_mask: Optional[torch.Tensor],
    second_prompt_len: Optional[int],
) -> Dict[str, float]:
    features: Dict[str, float] = {}

    add_attention_features(features, attentions, valid_mask, prompt_len)
    add_retrieval_memory_proxy_features(
        features=features,
        attentions=attentions,
        hidden=hidden,
        input_ids=input_ids,
        valid_mask=valid_mask,
        prompt_len=prompt_len,
        prompt_text=prompt_text,
        response_text=response_text,
        tokenizer=tokenizer,
    )
    logits_to_stats(features, logits, input_ids, valid_mask, prompt_len)
    add_hook_features(features, captures, valid_mask)
    add_cross_model_features(
        features=features,
        primary_logits=logits,
        primary_hidden=hidden,
        second_outputs=second_outputs,
        input_ids=input_ids,
        second_input_ids=second_input_ids,
        second_attention_mask=second_attention_mask,
        valid_mask=valid_mask,
        prompt_len=prompt_len,
        second_prompt_len=second_prompt_len,
    )
    add_external_verifier_signals(features, prompt_text, response_text)

    return {key: clean_value(value) for key, value in features.items()}


# ============================================================
# DATASET EXTRACTION
# ============================================================


def extract_dataset_features(
    df: pd.DataFrame,
    model,
    tokenizer,
    second_model,
    second_tokenizer,
    device: torch.device,
    has_label: bool,
) -> pd.DataFrame:
    prompts = df["prompt"].astype(str).tolist()
    responses = df["response"].astype(str).tolist()
    texts = [p + r for p, r in zip(prompts, responses)]
    prompt_lengths = get_prompt_lengths(tokenizer, prompts)
    second_prompt_lengths = get_prompt_lengths(second_tokenizer, prompts) if second_tokenizer is not None else [None] * len(prompts)
    rows = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extract infrastructure-change features"):
        batch_texts = texts[start:start + BATCH_SIZE]
        batch_prompt_lengths = prompt_lengths[start:start + BATCH_SIZE]
        batch_second_prompt_lengths = second_prompt_lengths[start:start + BATCH_SIZE]

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        handles, captures = register_hooks(model)
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                    use_cache=False,
                )
        finally:
            remove_hooks(handles)

        second_outputs = None
        second_input_ids_cpu = None
        second_attention_mask_cpu = None
        if second_model is not None and second_tokenizer is not None:
            second_encoding = second_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
            second_input_ids = second_encoding["input_ids"].to(device)
            second_attention_mask = second_encoding["attention_mask"].to(device)
            second_input_ids_cpu = second_input_ids.cpu()
            second_attention_mask_cpu = second_attention_mask.cpu()
            with torch.no_grad():
                second_outputs = second_model(
                    input_ids=second_input_ids,
                    attention_mask=second_attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

        hidden_batch = torch.stack(outputs.hidden_states, dim=1).float().cpu()
        attentions = outputs.attentions
        logits_batch = outputs.logits.detach().cpu().float()
        mask_batch = attention_mask.cpu().bool()
        input_ids_cpu = input_ids.cpu()

        for i in range(hidden_batch.shape[0]):
            sample_attentions = [layer_attn[i:i + 1].cpu() for layer_attn in attentions] if attentions is not None else None
            sample_second_outputs = None
            if second_outputs is not None:
                # Only BATCH_SIZE=1 is officially supported for clean second_outputs slicing.
                sample_second_outputs = second_outputs

            row = extract_features_one_sample(
                hidden=hidden_batch[i],
                attentions=sample_attentions,
                logits=logits_batch[i],
                input_ids=input_ids_cpu[i],
                valid_mask=mask_batch[i],
                prompt_len=batch_prompt_lengths[i],
                prompt_text=prompts[start + i],
                response_text=responses[start + i],
                tokenizer=tokenizer,
                captures=captures,
                second_outputs=sample_second_outputs,
                second_input_ids=second_input_ids_cpu[i] if second_input_ids_cpu is not None else None,
                second_attention_mask=second_attention_mask_cpu[i] if second_attention_mask_cpu is not None else None,
                second_prompt_len=batch_second_prompt_lengths[i],
            )
            rows.append(row)

        del outputs, hidden_batch, logits_batch
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
    print("BUILD EXTRA SMART FEATURES — INFRASTRUCTURE CHANGE")
    print("=" * 80)
    print(f"Primary model : {MODEL_NAME}")
    print(f"Second model  : {SECOND_MODEL_NAME or 'disabled'}")
    print(f"Device        : {device}")
    print(f"Output dir    : {OUTPUT_DIR}")
    print("Requires      : eager attention, logits, hooks")

    model, tokenizer = load_primary_model(device)
    second_model, second_tokenizer = load_second_model(device)

    train_df = pd.read_csv(DATA_FILE)
    print(f"\nDataset rows: {len(train_df)}")
    train_features = extract_dataset_features(
        df=train_df,
        model=model,
        tokenizer=tokenizer,
        second_model=second_model,
        second_tokenizer=second_tokenizer,
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
            second_model=second_model,
            second_tokenizer=second_tokenizer,
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
