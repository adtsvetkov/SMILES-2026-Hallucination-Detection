"""
Final hidden-state aggregation for hallucination detection.

This file implements the final `drift_extended_all` feature space used in the
notebook experiments.  It focuses on response-layer representation drift:

1. adjacent layer drift transforms: signed / abs / squared / sign / normalized;
2. long-pair layer drift transforms;
3. token-zone drift for first / middle / last response thirds.

Important:
    The exact final notebook pipeline used the true prompt length to isolate
    response tokens.  The public `solution.py` skeleton calls
    `aggregation_and_feature_extraction(hidden, mask, use_geometric=False)`
    without passing prompt length.  Therefore this implementation supports
    an optional `prompt_len` argument.  If `prompt_len` is not supplied, it
    falls back to a conservative response heuristic: use the final 30% of real
    tokens as the response span.  For exact reproduction, pass `prompt_len`
    from `solution.py`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


ADJACENT_PAIRS: list[tuple[int, int]] = [
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 16),
]

LONG_PAIRS: list[tuple[int, int]] = [
    (10, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
    (10, 14),
    (11, 15),
    (12, 16),
    (10, 16),
    (11, 16),
]

ALL_PAIRS = sorted(set(ADJACENT_PAIRS + LONG_PAIRS))
LAYERS_NEEDED = sorted({layer for pair in ALL_PAIRS for layer in pair})
EPS = 1e-8


def _as_bool_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.detach().cpu().bool()


def _safe_mean(layer_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum().item() == 0:
        return torch.zeros(
            layer_hidden.shape[-1],
            dtype=layer_hidden.dtype,
            device=layer_hidden.device,
        )
    return layer_hidden[mask].mean(dim=0)


def _normalised_diff(right: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
    diff = right - left
    coord_scale = right.abs() + left.abs() + EPS
    return diff / coord_scale


def _append_vector(vectors: List[torch.Tensor], prefix: str, vec: torch.Tensor) -> None:
    # Names are intentionally not returned by the public aggregation function,
    # but prefix mirrors the notebook feature names.
    del prefix
    vectors.append(vec.float().cpu())


def _response_mask_from_prompt_len(
    valid_mask: torch.Tensor,
    prompt_len: int | None,
) -> torch.Tensor:
    seq_len = valid_mask.shape[0]
    position_ids = torch.arange(seq_len)

    if prompt_len is None:
        valid_positions = torch.where(valid_mask)[0]
        if valid_positions.numel() == 0:
            return valid_mask.clone()

        # Fallback for the original skeleton interface: response is at the end
        # of prompt+response.  Use final 30% of valid tokens, at least 1 token.
        n_valid = int(valid_positions.numel())
        response_len = max(1, int(round(n_valid * 0.30)))
        response_positions = valid_positions[-response_len:]
        response_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        response_mask[response_positions] = True
        return response_mask

    prompt_len = min(max(int(prompt_len), 0), seq_len)
    response_mask = valid_mask & (position_ids >= prompt_len)

    # If truncation removes the response, fall back to the full valid span.
    if response_mask.sum().item() == 0:
        response_mask = valid_mask.clone()

    return response_mask


def _make_response_zone_masks(
    valid_mask: torch.Tensor,
    prompt_len: int | None,
) -> Dict[str, torch.Tensor]:
    response_mask = _response_mask_from_prompt_len(valid_mask, prompt_len)
    response_positions = torch.where(response_mask)[0]

    zones: Dict[str, torch.Tensor] = {"response_all": response_mask}

    if response_positions.numel() < 3:
        zones["response_first"] = response_mask
        zones["response_middle"] = response_mask
        zones["response_last"] = response_mask
        return zones

    n = int(response_positions.numel())
    first_end = max(n // 3, 1)
    second_end = max((2 * n) // 3, first_end + 1)

    zone_positions = {
        "response_first": response_positions[:first_end],
        "response_middle": response_positions[first_end:second_end],
        "response_last": response_positions[second_end:],
    }

    for zone_name, positions in zone_positions.items():
        zone_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        if positions.numel() > 0:
            zone_mask[positions] = True
        else:
            zone_mask = response_mask.clone()
        zones[zone_name] = zone_mask

    return zones


def _build_mean_response_by_layer(
    hidden_states: torch.Tensor,
    response_mask: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    return {
        layer: _safe_mean(hidden_states[layer], response_mask)
        for layer in LAYERS_NEEDED
    }


def _build_drift_transforms(mean_by_layer: Dict[int, torch.Tensor]) -> List[torch.Tensor]:
    vectors: List[torch.Tensor] = []

    for left, right in ADJACENT_PAIRS:
        diff = mean_by_layer[right] - mean_by_layer[left]
        _append_vector(vectors, f"drift_signed_l{left}_to_l{right}", diff)
        _append_vector(vectors, f"drift_abs_l{left}_to_l{right}", diff.abs())
        _append_vector(vectors, f"drift_squared_l{left}_to_l{right}", diff.pow(2))
        _append_vector(vectors, f"drift_sign_l{left}_to_l{right}", torch.sign(diff))
        _append_vector(
            vectors,
            f"drift_normed_l{left}_to_l{right}",
            _normalised_diff(mean_by_layer[right], mean_by_layer[left]),
        )

    return vectors


def _build_long_pair_drifts(mean_by_layer: Dict[int, torch.Tensor]) -> List[torch.Tensor]:
    vectors: List[torch.Tensor] = []

    for left, right in LONG_PAIRS:
        diff = mean_by_layer[right] - mean_by_layer[left]
        _append_vector(vectors, f"long_drift_signed_l{left}_to_l{right}", diff)
        _append_vector(vectors, f"long_drift_abs_l{left}_to_l{right}", diff.abs())
        _append_vector(
            vectors,
            f"long_drift_normed_l{left}_to_l{right}",
            _normalised_diff(mean_by_layer[right], mean_by_layer[left]),
        )

    return vectors


def _build_token_zone_drifts(
    hidden_states: torch.Tensor,
    zone_masks: Dict[str, torch.Tensor],
) -> List[torch.Tensor]:
    vectors: List[torch.Tensor] = []

    for zone_name, zone_mask in zone_masks.items():
        if zone_name == "response_all":
            continue

        zone_mean_by_layer = {
            layer: _safe_mean(hidden_states[layer], zone_mask)
            for layer in LAYERS_NEEDED
        }

        for left, right in ADJACENT_PAIRS:
            diff = zone_mean_by_layer[right] - zone_mean_by_layer[left]
            _append_vector(vectors, f"{zone_name}_drift_signed_l{left}_to_l{right}", diff)
            _append_vector(vectors, f"{zone_name}_drift_abs_l{left}_to_l{right}", diff.abs())
            _append_vector(
                vectors,
                f"{zone_name}_drift_normed_l{left}_to_l{right}",
                _normalised_diff(zone_mean_by_layer[right], zone_mean_by_layer[left]),
            )

    return vectors


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int | None = None,
) -> torch.Tensor:
    """Return the final `drift_extended_all` feature vector.

    Args:
        hidden_states: Tensor with shape (n_layers, seq_len, hidden_dim).
        attention_mask: Tensor with shape (seq_len,).
        prompt_len: Optional tokenized prompt length.  Passing it gives exact
            response-token masking.  If absent, a last-30%-tokens heuristic is
            used for compatibility with the original skeleton.
    """
    hidden_states = hidden_states.float().detach().cpu()
    valid_mask = _as_bool_mask(attention_mask)

    zone_masks = _make_response_zone_masks(valid_mask, prompt_len)
    response_mask = zone_masks["response_all"]
    mean_response = _build_mean_response_by_layer(hidden_states, response_mask)

    vectors: List[torch.Tensor] = []
    vectors.extend(_build_drift_transforms(mean_response))
    vectors.extend(_build_long_pair_drifts(mean_response))
    vectors.extend(_build_token_zone_drifts(hidden_states, zone_masks))

    return torch.cat(vectors, dim=0).float()


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compatibility hook.  Features are already produced by aggregate()."""
    del hidden_states, attention_mask
    return torch.zeros(0, dtype=torch.float32)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
    prompt_len: int | None = None,
) -> torch.Tensor:
    """Main entry point used by solution.py.

    The optional `prompt_len` argument is backward-compatible: old calls still
    work, exact calls can pass prompt length.
    """
    del use_geometric
    return aggregate(hidden_states, attention_mask, prompt_len=prompt_len)
