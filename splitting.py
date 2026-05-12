"""Fold-safe stratified splitting for the final hallucination pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    n_splits: int = 5,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Return 5 stratified outer folds with an inner validation split.

    Each sample appears exactly once in an outer test fold.  Validation indices
    are sampled only from the corresponding outer trainval part, preserving the
    label ratio.  This mirrors the evaluation protocol used in the experiments.
    """
    del df

    y = np.asarray(y).astype(int)
    idx = np.arange(len(y))

    outer = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for fold_id, (idx_trainval, idx_test) in enumerate(outer.split(idx, y), start=1):
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=val_size,
            random_state=random_state + fold_id,
            stratify=y[idx_trainval],
        )

        splits.append(
            (
                np.asarray(idx_train, dtype=int),
                np.asarray(idx_val, dtype=int),
                np.asarray(idx_test, dtype=int),
            )
        )

    return splits
