from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import numpy as np

from src.utils.config import load_yaml_config
from src.pipeline.preprocessing import build_image_query_text, build_music_text
from src.pipeline.embedder import TextEmbedder
from src.pipeline.retrieval import rank_candidates, combine_scores, cosine_sim_matrix


class TextPivotModel:
    """Text-axis normalization based image ↔ music matching model."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        econf = config.get("embedder", {})
        self.embedder = TextEmbedder(
            model_name=econf.get("name"),
            device=econf.get("device", "auto"),
            use_fallback_tfidf=econf.get("use_fallback_tfidf", True),
        )

        pconf = config.get("preprocessing", {})
        self.lowercase: bool = pconf.get("lowercase", True)
        self.deduplicate_tokens: bool = pconf.get("deduplicate_tokens", True)
        self.min_token_length: int = pconf.get("min_token_length", 2)
        self.joiner: str = pconf.get("joiner", " ")
        self.fields_order: List[str] = list(pconf.get("fields_order", []))

        rconf = config.get("retrieval", {})
        self.default_top_k: int = rconf.get("top_k_default", 20)
        self.weights: Dict[str, float] = rconf.get("weights", {"embedding": 1.0})

        self._music_ids: List[str] = []
        self._music_texts: List[str] = []
        self._music_vectors = None
        self._music_meta: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def from_config(cls, path: str) -> "TextPivotModel":
        cfg = load_yaml_config(path)
        return cls(cfg)

    def _build_music_text(self, item: Dict[str, Any]) -> str:
        return build_music_text(
            item,
            fields_order=self.fields_order,
            lowercase=self.lowercase,
            deduplicate_tokens=self.deduplicate_tokens,
            joiner=self.joiner,
            min_token_length=self.min_token_length,
        )

    def _build_image_query_text(self, image_keywords: Iterable[str]) -> str:
        return build_image_query_text(
            image_keywords,
            lowercase=self.lowercase,
            deduplicate=self.deduplicate_tokens,
            joiner=self.joiner,
            min_token_length=self.min_token_length,
        )

    def prepare_music_index(self, music_items: List[Dict[str, Any]]) -> None:
        self._music_ids = []
        self._music_texts = []
        self._music_meta = {}

        for it in music_items:
            mid = str(it.get("id"))
            if not mid:
                continue
            text = self._build_music_text(it)
            if not text:
                continue
            self._music_ids.append(mid)
            self._music_texts.append(text)
            self._music_meta[mid] = it

        if self.embedder.mode == "tfidf":
            self.embedder.fit(self._music_texts)

        self._music_vectors = self.embedder.encode(self._music_texts)

    def _ensure_index(self) -> None:
        if self._music_vectors is None or len(self._music_ids) == 0:
            raise RuntimeError("Index is empty. Call prepare_music_index() first.")

    def encode_text_query(self, text_query: str) -> np.ndarray:
        return self.embedder.encode([text_query])

    def search_by_text(
        self,
        text_query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_index()
        qv = self.encode_text_query(text_query)
        top_k = top_k or self.default_top_k
        idxs, scores = rank_candidates(qv[0], self._music_vectors, top_k=top_k)
        return self._format_results(idxs, scores)

    def search_by_image_keywords(
        self,
        image_keywords: Iterable[str],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_index()
        qtext = self._build_image_query_text(image_keywords)
        qv = self.encode_text_query(qtext)
        top_k = top_k or self.default_top_k
        idxs, scores = rank_candidates(qv[0], self._music_vectors, top_k=top_k)
        return self._format_results(idxs, scores, query_text=qtext)

    def batch_search_by_text(
        self,
        text_queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        self._ensure_index()
        qv = self.embedder.encode(text_queries)
        top_k = top_k or self.default_top_k
        results: List[List[Dict[str, Any]]] = []
        sim = cosine_sim_matrix(qv, self._music_vectors)
        for i in range(sim.shape[0]):
            scores = sim[i]
            idx = np.argpartition(-scores, kth=min(top_k, scores.shape[0]-1))[:top_k]
            idx = idx[np.argsort(-scores[idx])]
            results.append(self._format_results(idx, scores[idx], query_text=text_queries[i]))
        return results

    def _format_results(
        self,
        idxs: np.ndarray,
        scores: np.ndarray,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, s in zip(idxs, scores):
            mid = self._music_ids[int(i)]
            out.append(
                {
                    "id": mid,
                    "score": float(s),
                    "text": self._music_texts[int(i)],
                    "meta": self._music_meta.get(mid, {}),
                    **({"query_text": query_text} if query_text is not None else {}),
                }
            )
        return out


