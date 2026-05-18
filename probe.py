"""
probe.py — Track C 4-view logistic-rank-fusion probe.

Expected feature layout from aggregation.py:
  [0:312]       B__extra_smart_prompt_len_all__top312_pca64
  [312:1178]    C__attention_all__top866_pca64
  [1178:2428]   B__advanced_prompt_len_max_mean__top1250_pca128
  [2428:2432]   C__attention_sink__selected4
"""
from __future__ import annotations

import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Four independent logistic probes with late rank fusion."""

    VIEW_SLICES = (
        slice(0, 312),
        slice(312, 1178),
        slice(1178, 2428),
        slice(2428, 2432),
    )
    PCA_COMPONENTS = (64, 64, 128, None)

    def __init__(self) -> None:
        super().__init__()
        self.models: list[Pipeline] = []
        self._threshold: float = 0.5
        self._is_fitted = False

    def _make_pipeline(self, n_features: int, n_samples: int, pca_components: int | None) -> Pipeline:
        steps = [("scaler", StandardScaler())]
        if pca_components is not None:
            n_components = min(int(pca_components), int(n_features), max(1, int(n_samples) - 1))
            steps.append(("pca", PCA(n_components=n_components, random_state=42)))
        steps.append((
            "logreg",
            LogisticRegression(
                C=0.003,
                penalty="l2",
                solver="lbfgs",
                max_iter=3000,
                random_state=42,
            ),
        ))
        return Pipeline(steps)

    @staticmethod
    def _rank01(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        n = values.size
        if n <= 1:
            return values.astype(np.float64)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        return ranks / max(n - 1, 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        if X.shape[1] != 2432:
            raise ValueError(f"Expected 2432 Track C features, got {X.shape[1]}")

        self.models = []
        for view_slice, pca_components in zip(self.VIEW_SLICES, self.PCA_COMPONENTS):
            X_view = X[:, view_slice]
            pipe = self._make_pipeline(
                n_features=X_view.shape[1],
                n_samples=X_view.shape[0],
                pca_components=pca_components,
            )
            pipe.fit(X_view, y)
            self.models.append(pipe)

        self._is_fitted = True
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in candidates:
            pred = (probs >= threshold).astype(int)
            score = f1_score(y_val, pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        self._threshold = best_threshold
        return self

    def _view_probabilities(self, X: np.ndarray) -> list[np.ndarray]:
        if not self._is_fitted or not self.models:
            raise RuntimeError("Call fit() before predict/predict_proba.")
        X = np.asarray(X, dtype=np.float32)
        return [
            model.predict_proba(X[:, view_slice])[:, 1]
            for model, view_slice in zip(self.models, self.VIEW_SLICES)
        ]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        view_probs = self._view_probabilities(X)
        if len(view_probs[0]) <= 1:
            fused = np.mean(np.vstack(view_probs), axis=0)
        else:
            fused = np.mean(np.vstack([self._rank01(p) for p in view_probs]), axis=0)
        fused = np.clip(fused, 0.0, 1.0)
        return np.column_stack([1.0 - fused, fused])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)
