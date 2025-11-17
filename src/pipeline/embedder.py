from __future__ import annotations

from typing import Iterable, List, Optional
import numpy as np

_HAS_ST = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except Exception:
    TfidfVectorizer = None  # type: ignore


class TextEmbedder:
    """Sentence embedder wrapper: Sentence-Transformers first, TF-IDF fallback if unavailable."""

    def __init__(
        self,
        model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        use_fallback_tfidf: bool = True,
    ) -> None:
        self.use_fallback_tfidf = use_fallback_tfidf
        self._mode: str = "none"
        self._st_model = None
        self._tfidf = None
        self._tfidf_fitted: bool = False

        if _HAS_ST and model_name:
            try:
                self._st_model = SentenceTransformer(model_name, device=None if device == "auto" else device)
                self._mode = "st"
            except Exception:
                self._st_model = None
                self._mode = "none"

        if self._mode == "none" and use_fallback_tfidf:
            if TfidfVectorizer is None:
                raise RuntimeError("TF-IDF fallback not available. Install scikit-learn.")
            self._tfidf = TfidfVectorizer()
            self._mode = "tfidf"

        if self._mode == "none":
            raise RuntimeError("No available embedder. Install sentence-transformers or enable TF-IDF.")

    @property
    def mode(self) -> str:
        return self._mode

    def fit(self, corpus: Iterable[str]) -> None:
        if self._mode == "tfidf":
            assert self._tfidf is not None
            self._tfidf.fit(list(corpus))
            self._tfidf_fitted = True

    def encode(self, texts: Iterable[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        texts = list(texts)
        if self._mode == "st":
            assert self._st_model is not None
            vecs = self._st_model.encode(texts, batch_size=batch_size, normalize_embeddings=normalize, convert_to_numpy=True)
            return vecs.astype(np.float32)
        elif self._mode == "tfidf":
            if not self._tfidf_fitted:
                self.fit(texts)
            assert self._tfidf is not None
            vecs = self._tfidf.transform(texts).astype(np.float32)
            if normalize:
                norms = np.sqrt((vecs.multiply(vecs)).sum(axis=1)).A1 + 1e-12
                vecs = vecs.multiply(1.0 / norms[:, None])
            return vecs
        else:
            raise RuntimeError("Embedding mode is not set.")


