from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import numpy as np

from src.utils.config import load_yaml_config
from src.pipeline.preprocess import build_image_query_text, build_music_text
from src.pipeline.embedder import TextEmbedder, ClipImageTextEmbedder, ClapAudioTextEmbedder
from src.pipeline.retrieval import rank_candidates, combine_scores, cosine_sim_matrix


class ClipClapModel:
    """Image ↔ Music retrieval using CLIP (image-text) and CLAP (audio-text) with text fallback."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        tf_conf = config.get("text_fallback", {})
        self.text_enabled: bool = tf_conf.get("enable", True)
        self.text_embedder = TextEmbedder(
            model_name=tf_conf.get("sentence_transformer", "sentence-transformers/all-MiniLM-L6-v2"),
            use_tfidf_fallback=tf_conf.get("use_tfidf_fallback", True),
        ) if self.text_enabled else None

        c_conf = config.get("clip", {})
        self.clip_enabled: bool = c_conf.get("enable", True)
        self.clip = ClipImageTextEmbedder(
            model_name=c_conf.get("model_name", "ViT-B-32"),
            pretrained=c_conf.get("pretrained", "laion2b_s34b_b79k"),
            device=c_conf.get("device", "auto"),
            text_fallback=self.text_embedder if self.text_enabled else None,
        ) if self.clip_enabled else None

        a_conf = config.get("clap", {})
        self.clap_enabled: bool = a_conf.get("enable", True)
        self.clap = ClapAudioTextEmbedder(
            device=a_conf.get("device", "auto"),
            text_fallback=self.text_embedder if self.text_enabled else None,
        ) if self.clap_enabled else None

        pconf = config.get("preprocessing", {})
        self.lowercase: bool = pconf.get("lowercase", True)
        self.deduplicate_tokens: bool = pconf.get("deduplicate_tokens", True)
        self.min_token_length: int = pconf.get("min_token_length", 2)
        self.joiner: str = pconf.get("joiner", " ")
        self.fields_order: List[str] = list(pconf.get("fields_order", []))

        rconf = config.get("retrieval", {})
        self.default_top_k: int = rconf.get("top_k_default", 20)
        self.weights: Dict[str, float] = rconf.get("weights", {"clip": 1.0, "clap": 1.0, "text": 0.5})

        self._music_ids: List[str] = []
        self._music_texts: List[str] = []
        self._music_meta: Dict[str, Dict[str, Any]] = {}
        self._clip_text_vecs = None
        self._clap_audio_vecs = None
        self._text_vecs = None

    @classmethod
    def from_config(cls, path: str) -> "ClipClapModel":
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
        self._clip_text_vecs = None
        self._clap_audio_vecs = None
        self._text_vecs = None

        audio_paths: List[str] = []
        audio_id_idx: List[int] = []

        for it in music_items:
            mid = str(it.get("id"))
            if not mid:
                continue
            text = self._build_music_text(it)
            if not text:
                continue
            idx = len(self._music_ids)
            self._music_ids.append(mid)
            self._music_texts.append(text)
            self._music_meta[mid] = it

            ap = it.get("audio_path")
            if ap:
                audio_paths.append(ap)
                audio_id_idx.append(idx)

        if self.clip_enabled and self.clip is not None:
            self._clip_text_vecs = self.clip.encode_texts(self._music_texts)

        if self.clap_enabled and self.clap is not None and len(audio_paths) > 0:
            audio_vecs = self.clap.encode_audios(audio_paths)
            clap_dim = audio_vecs.shape[1]
            aligned = np.zeros((len(self._music_ids), clap_dim), dtype=np.float32)
            for i, gid in enumerate(audio_id_idx):
                aligned[gid] = audio_vecs[i]
            norms = np.linalg.norm(aligned, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            aligned = aligned / norms
            self._clap_audio_vecs = aligned

        if self.text_enabled and self.text_embedder is not None:
            self._text_vecs = self.text_embedder.encode(self._music_texts)

    def _ensure_any_index(self) -> None:
        if len(self._music_ids) == 0:
            raise RuntimeError("Index is empty. Call prepare_music_index() first.")

    def search_by_image_paths(self, image_paths: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        self._ensure_any_index()
        top_k = top_k or self.default_top_k
        feature_scores = {}

        if self.clip_enabled and self.clip is not None and self._clip_text_vecs is not None:
            q_img = self.clip.encode_images(image_paths)
            q_vec = q_img.mean(axis=0, keepdims=True)
            sim = cosine_sim_matrix(q_vec, self._clip_text_vecs)[0]
            feature_scores["clip"] = sim

        if not feature_scores:
            raise RuntimeError("No modality available for image-path query. Enable CLIP or add fallbacks.")
        combined = combine_scores(feature_scores, self.weights)
        idx = np.argpartition(-combined, kth=min(top_k, combined.shape[0]-1))[:top_k]
        idx = idx[np.argsort(-combined[idx])]
        scores = combined[idx]
        return self._format_results(idx, scores)

    def search_by_image_keywords(self, image_keywords: Iterable[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        self._ensure_any_index()
        top_k = top_k or self.default_top_k
        qtext = self._build_image_query_text(image_keywords)
        feature_scores = {}

        if self.clap_enabled and self.clap is not None and self._clap_audio_vecs is not None:
            q_clap = self.clap.encode_texts([qtext])
            sim_clap = cosine_sim_matrix(q_clap, self._clap_audio_vecs)[0]
            feature_scores["clap"] = sim_clap

        if self.text_enabled and self.text_embedder is not None and self._text_vecs is not None:
            q_text = self.text_embedder.encode([qtext])
            sim_text = cosine_sim_matrix(q_text, self._text_vecs)[0]
            feature_scores["text"] = sim_text

        if self.clip_enabled and self.clip is not None and self._clip_text_vecs is not None:
            q_clip_text = self.clip.encode_texts([qtext])
            sim_clip = cosine_sim_matrix(q_clip_text, self._clip_text_vecs)[0]
            feature_scores["clip"] = sim_clip

        if not feature_scores:
            raise RuntimeError("No modality available for keyword query. Enable at least one of CLAP/CLIP/Text.")

        combined = combine_scores(feature_scores, self.weights)
        idx = np.argpartition(-combined, kth=min(top_k, combined.shape[0]-1))[:top_k]
        idx = idx[np.argsort(-combined[idx])]
        scores = combined[idx]
        return self._format_results(idx, scores, query_text=qtext)

    def _format_results(self, idxs: np.ndarray, scores: np.ndarray, query_text: Optional[str] = None) -> List[Dict[str, Any]]:
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


