from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

try:
    from scipy.sparse import issparse as _scipy_issparse  # type: ignore
except Exception:
    _scipy_issparse = None  # type: ignore


def _is_sparse(x) -> bool:
    if _scipy_issparse is not None:
        try:
            return _scipy_issparse(x)
        except Exception:
            return False
    return hasattr(x, "tocsr") or hasattr(x, "tocoo") or hasattr(x, "format")


def cosine_sim_matrix(query_vecs, candidate_vecs) -> np.ndarray:
    if _is_sparse(query_vecs) or _is_sparse(candidate_vecs):
        scores = query_vecs @ candidate_vecs.T
        return scores.toarray() if hasattr(scores, "toarray") else scores
    return (query_vecs @ candidate_vecs.T).astype(np.float32)


def rank_candidates(query_vec: np.ndarray, candidate_vecs: np.ndarray, top_k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    if query_vec.ndim == 1:
        q = query_vec.reshape(1, -1)
    else:
        q = query_vec
    scores = cosine_sim_matrix(q, candidate_vecs)
    scores_1d = scores[0]
    top_k = min(top_k, scores_1d.shape[0])
    idx = np.argpartition(-scores_1d, kth=top_k - 1)[:top_k]
    idx = idx[np.argsort(-scores_1d[idx])]
    return idx, scores_1d[idx]


def combine_scores(feature_to_scores: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    combined = None
    for name, scores in feature_to_scores.items():
        w = weights.get(name, 0.0)
        part = scores * w
        combined = part if combined is None else (combined + part)
    if combined is None:
        raise ValueError("No scores to combine.")
    return combined


