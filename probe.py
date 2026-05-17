"""probe.py — final Track A probe.

The probe matches the best Track A experiment:
A__advanced_all__top1250__pca256 + LogisticRegression(C=0.003).
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class HallucinationProbe(nn.Module):
    """Linear probe with top-k feature selection and PCA compression."""

    def __init__(self) -> None:
        super().__init__()
        self.k = 1250
        self.pca_dim = 256
        self._threshold = 0.5
        self.model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("selector", SelectKBest(score_func=f_classif, k=self.k)),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=self.pca_dim, random_state=42)),
                (
                    "clf",
                    LogisticRegression(
                        C=0.003,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        )

    @staticmethod
    def _clean(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X = self._clean(X)
        y = np.asarray(y).astype(int)

        real_k = min(self.k, X.shape[1])
        real_pca = min(self.pca_dim, real_k, X.shape[0] - 1)

        self.model.set_params(
            selector__k=real_k,
            pca__n_components=real_pca,
        )
        self.model.fit(X, y)
        return self

    def fit_hyperparameters(self, X_val, y_val):
        del X_val, y_val
        self._threshold = 0.5
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._clean(X)
        return self.model.predict_proba(X)
