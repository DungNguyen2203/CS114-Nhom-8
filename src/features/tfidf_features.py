"""Sparse TF-IDF and TruncatedSVD text feature extractor fitted strictly on train."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.sparse import spmatrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfSvdExtractor:
    """Fit TF-IDF and optional TruncatedSVD strictly on train masked listing texts."""

    def __init__(
        self,
        max_features: int = 100_000,
        min_df: int = 5,
        ngram_range: tuple[int, int] = (1, 2),
        svd_components: int = 64,
        seed: int = 42,
    ) -> None:
        self.max_features = int(max_features)
        self.min_df = int(min_df)
        self.ngram_range = tuple(ngram_range)
        self.svd_components = int(svd_components)
        self.seed = int(seed)
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.svd: TruncatedSVD | None = None

    def fit(self, train_texts: Sequence[str], fit_svd: bool = True) -> "TfidfSvdExtractor":
        X_sp = self.vectorizer.fit_transform([str(t or "") for t in train_texts])
        if fit_svd:
            max_k = min(self.svd_components, max(1, X_sp.shape[1] - 1), max(1, X_sp.shape[0] - 1))
            self.svd = TruncatedSVD(n_components=max_k, random_state=self.seed)
            self.svd.fit(X_sp)
        return self

    def transform_sparse(self, texts: Sequence[str]) -> spmatrix:
        return self.vectorizer.transform([str(t or "") for t in texts])

    def transform_svd(self, texts: Sequence[str]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("SVD was not fitted. Call fit(..., fit_svd=True) first.")
        X_sp = self.transform_sparse(texts)
        return self.svd.transform(X_sp).astype(np.float32)

    def transform(self, texts: Sequence[str]) -> tuple[spmatrix, np.ndarray]:
        X_sp = self.transform_sparse(texts)
        if self.svd is None:
            raise RuntimeError("SVD was not fitted.")
        X_svd = self.svd.transform(X_sp).astype(np.float32)
        return X_sp, X_svd

    def fit_transform(self, train_texts: Sequence[str], fit_svd: bool = True) -> tuple[spmatrix, np.ndarray | None]:
        X_sp = self.vectorizer.fit_transform([str(t or "") for t in train_texts])
        if fit_svd:
            max_k = min(self.svd_components, max(1, X_sp.shape[1] - 1), max(1, X_sp.shape[0] - 1))
            self.svd = TruncatedSVD(n_components=max_k, random_state=self.seed)
            X_svd = self.svd.fit_transform(X_sp).astype(np.float32)
            return X_sp, X_svd
        return X_sp, None
