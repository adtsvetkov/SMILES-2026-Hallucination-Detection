"""
Submission-oriented probe.py

This file puts the notebook-best modelling idea inside the official
HallucinationProbe interface:

- CatBoost classifier
- recursive feature elimination over the high-dimensional drift features
- three RFE selectors: 250 -> 200 -> 125 -> 70 / 60 / 80
- weighted probability ensemble

The implementation is fold-safe: every feature selection step is performed
inside fit() using only the X, y passed by evaluate.py for the current training
split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split


RANDOM_STATE = 42

CATBOOST_PARAMS = {
    "iterations": 600,
    "depth": 4,
    "learning_rate": 0.0628238389168676,
    "l2_leaf_reg": 9.703703315819581,
    "random_strength": 6.728629794179622,
    "bagging_temperature": 1.3001972097295067,
    "border_count": 161,
    "auto_class_weights": "Balanced",
}

RFE_STEPS = {
    "rfe_70": (200, 125, 70),
    "rfe_60": (200, 125, 60),
    "rfe_80": (200, 125, 80),
}

ENSEMBLE_ORDER = ["rfe_70", "rfe_60", "rfe_80"]
ENSEMBLE_WEIGHTS = np.array(
    [
        0.8326522912543401,
        0.13906799872596168,
        0.14228215132046573,
    ],
    dtype=np.float64,
)
ENSEMBLE_WEIGHTS = ENSEMBLE_WEIGHTS / ENSEMBLE_WEIGHTS.sum()

START_K = 250
INNER_SPLITS = 3


@dataclass
class _Member:
    name: str
    feature_idx: np.ndarray
    model: CatBoostClassifier


def _clean_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _make_catboost(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        **CATBOOST_PARAMS,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def _fast_univariate_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fast supervised feature ranking.

    The notebook used AUC/correlation-driven preselection before CatBoost RFE.
    For the official runtime, absolute point-biserial correlation is a stable
    and much faster proxy for high-dimensional initial filtering.
    """
    y = y.astype(np.float32)
    y = y - y.mean()
    Xc = X - X.mean(axis=0, keepdims=True)

    numerator = np.abs(Xc.T @ y)
    denominator = (
        np.sqrt(np.sum(Xc * Xc, axis=0))
        * np.sqrt(float(np.sum(y * y)))
        + 1e-12
    )
    scores = numerator / denominator
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), scores.shape[0])
    if k <= 0:
        return np.arange(0, dtype=int)
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.astype(int)


def _inner_cv_indices(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y).astype(int)
    n_splits = min(INNER_SPLITS, np.bincount(y).min())

    if n_splits >= 2:
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        return [(tr, va) for tr, va in skf.split(np.arange(len(y)), y)]

    idx = np.arange(len(y))
    tr, va = train_test_split(
        idx,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    return [(tr, va)]


def _rfe_select_features(
    X: np.ndarray,
    y: np.ndarray,
    steps: Iterable[int],
    seed_offset: int = 0,
) -> np.ndarray:
    """Recursive CatBoost feature elimination, fold-safe inside fit()."""
    scores = _fast_univariate_scores(X, y)
    current_idx = _top_k(scores, START_K)
    inner_splits = _inner_cv_indices(y)

    for step_i, target_k in enumerate(steps):
        target_k = min(int(target_k), len(current_idx))
        importance_sum = np.zeros(len(current_idx), dtype=np.float64)

        for inner_i, (idx_train, idx_val) in enumerate(inner_splits):
            model = _make_catboost(
                seed=RANDOM_STATE + 1000 * seed_offset + 100 * step_i + inner_i
            )
            model.fit(
                X[idx_train][:, current_idx],
                y[idx_train],
                eval_set=(X[idx_val][:, current_idx], y[idx_val]),
                use_best_model=True,
            )
            imp = model.get_feature_importance().astype(np.float64)
            imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
            if imp.sum() > 0:
                imp = imp / imp.sum()
            importance_sum += imp

        local_top = np.argsort(importance_sum)[-target_k:][::-1]
        current_idx = current_idx[local_top]

    return current_idx.astype(int)


class HallucinationProbe(nn.Module):
    """Official evaluate.py-compatible CatBoost RFE ensemble."""

    def __init__(self) -> None:
        super().__init__()
        self.members_: list[_Member] = []
        self.threshold_: float = 0.5
        self.prior_: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X = _clean_X(X)
        y = np.asarray(y).astype(int)
        self.members_ = []
        self.prior_ = float(y.mean()) if len(y) else 0.5

        if len(y) < 20 or len(np.unique(y)) < 2:
            return self

        # Final model split used only for early stopping. Feature selection above
        # is already internal-CV based and uses only this fit() training split.
        idx = np.arange(len(y))
        idx_train, idx_val = train_test_split(
            idx,
            test_size=0.15,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        for member_i, name in enumerate(ENSEMBLE_ORDER):
            feature_idx = _rfe_select_features(
                X=X,
                y=y,
                steps=RFE_STEPS[name],
                seed_offset=member_i + 1,
            )

            model = _make_catboost(seed=RANDOM_STATE + 100 + member_i)
            model.fit(
                X[idx_train][:, feature_idx],
                y[idx_train],
                eval_set=(X[idx_val][:, feature_idx], y[idx_val]),
                use_best_model=True,
            )

            self.members_.append(_Member(name=name, feature_idx=feature_idx, model=model))

        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        X_val = _clean_X(X_val)
        y_val = np.asarray(y_val).astype(int)
        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.unique(
            np.concatenate([np.linspace(0.05, 0.95, 181), probs])
        )
        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in candidates:
            preds = (probs >= threshold).astype(int)
            score = f1_score(y_val, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        self.threshold_ = best_threshold
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _clean_X(X)

        if not self.members_:
            p = np.full(X.shape[0], self.prior_, dtype=np.float64)
            return np.column_stack([1.0 - p, p])

        p = np.zeros(X.shape[0], dtype=np.float64)
        for weight, member in zip(ENSEMBLE_WEIGHTS, self.members_):
            p += float(weight) * member.model.predict_proba(X[:, member.feature_idx])[:, 1]

        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)
