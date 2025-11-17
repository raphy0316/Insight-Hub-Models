from __future__ import annotations

from typing import Iterable, List, Optional
import numpy as np

_HAS_OPEN_CLIP = False
try:
    import open_clip  # type: ignore
    _HAS_OPEN_CLIP = True
except Exception:
    open_clip = None  # type: ignore

_HAS_LAION_CLAP = False
try:
    import laion_clap  # type: ignore
    _HAS_LAION_CLAP = True
except Exception:
    laion_clap = None  # type: ignore

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

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore


def _pick_device(device: str) -> str | None:
    if device == "auto":
        return None
    return device


class TextEmbedder:
    """Sentence-Transformers → TF-IDF text embedder fallback chain."""

    def __init__(
        self,
        model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        use_tfidf_fallback: bool = True,
    ) -> None:
        self._mode: str = "none"
        self._st_model = None
        self._tfidf = None
        self._tfidf_fitted = False
        self.use_tfidf_fallback = use_tfidf_fallback

        if _HAS_ST and model_name:
            try:
                self._st_model = SentenceTransformer(model_name)
                self._mode = "st"
            except Exception:
                self._st_model = None
                self._mode = "none"

        if self._mode == "none" and use_tfidf_fallback:
            if TfidfVectorizer is None:
                raise RuntimeError("TF-IDF fallback not available. Install scikit-learn.")
            self._tfidf = TfidfVectorizer()
            self._mode = "tfidf"

        if self._mode == "none":
            raise RuntimeError("No available text embedder. Install sentence-transformers or enable TF-IDF.")

    @property
    def mode(self) -> str:
        return self._mode

    def fit(self, corpus: Iterable[str]) -> None:
        if self._mode == "tfidf":
            assert self._tfidf is not None
            self._tfidf.fit(list(corpus))
            self._tfidf_fitted = True

    def encode(self, texts: Iterable[str], batch_size: int = 64, normalize: bool = True):
        texts = list(texts)
        if self._mode == "st":
            assert self._st_model is not None
            vecs = self._st_model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
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
            raise RuntimeError("Text embedder is not initialized.")


class ClipImageTextEmbedder:
    """CLIP image and text encoder via open_clip; falls back to RGB hist for image and text fallback for text."""

    def __init__(self, model_name: str, pretrained: str, device: str = "auto", text_fallback: Optional[TextEmbedder] = None) -> None:
        self._mode = "hist"
        self._text_mode = "fallback"
        self._text_fallback = text_fallback
        self._clip_model = None
        self._clip_preprocess = None
        self._device = _pick_device(device)

        if _HAS_OPEN_CLIP:
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self._device)
                tokenizer = open_clip.get_tokenizer(model_name)
                self._clip_model = (model, tokenizer)
                self._clip_preprocess = preprocess
                self._mode = "clip"
                self._text_mode = "clip"
            except Exception:
                pass

        if self._text_mode != "clip" and self._text_fallback is None:
            self._text_fallback = TextEmbedder()

    @property
    def mode(self) -> str:
        return self._mode

    def encode_images(self, image_paths: Iterable[str], normalize: bool = True):
        paths = list(image_paths)
        if self._mode == "clip":
            model, tokenizer = self._clip_model  # type: ignore
            assert self._clip_preprocess is not None
            model.eval()
            import torch  # type: ignore
            imgs = []
            for p in paths:
                if Image is None:
                    raise RuntimeError("Pillow is required for image loading.")
                img = Image.open(p).convert("RGB")
                imgs.append(self._clip_preprocess(img))
            batch = torch.stack(imgs)  # type: ignore
            if self._device:
                batch = batch.to(self._device)
            with torch.no_grad():
                feats = model.encode_image(batch)  # type: ignore
                if normalize:
                    feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype(np.float32)

        vecs = []
        for p in paths:
            if Image is None:
                raise RuntimeError("Pillow is required for image loading.")
            img = Image.open(p).convert("RGB")
            hist = []
            for ch in img.split():
                h = ch.histogram(bins=256)
                h = np.asarray(h, dtype=np.float32)
                h = h / (np.linalg.norm(h) + 1e-12)
                hist.append(h)
            vec = np.concatenate(hist, axis=0)
            if normalize:
                vec = vec / (np.linalg.norm(vec) + 1e-12)
            vecs.append(vec.astype(np.float32))
        return np.stack(vecs, axis=0)

    def encode_texts(self, texts: Iterable[str], normalize: bool = True):
        texts = list(texts)
        if self._text_mode == "clip":
            model, tokenizer = self._clip_model  # type: ignore
            import torch  # type: ignore
            model.eval()
            tokens = tokenizer(texts)
            if self._device:
                tokens = tokens.to(self._device)  # type: ignore
            with torch.no_grad():
                feats = model.encode_text(tokens)  # type: ignore
                if normalize:
                    feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype(np.float32)
        assert self._text_fallback is not None
        return self._text_fallback.encode(texts, normalize=normalize)


class ClapAudioTextEmbedder:
    """CLAP audio/text encoder via laion_clap; falls back to text embedder for text only."""

    def __init__(self, device: str = "auto", text_fallback: Optional[TextEmbedder] = None) -> None:
        self._mode = "none"
        self._device = _pick_device(device)
        self._clap_model = None
        self._text_fallback = text_fallback or TextEmbedder()

        if _HAS_LAION_CLAP:
            try:
                self._clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')  # type: ignore
                self._mode = "clap"
            except Exception:
                self._clap_model = None
                self._mode = "none"

    @property
    def mode(self) -> str:
        return self._mode

    def encode_audios(self, audio_paths: Iterable[str], normalize: bool = True):
        if self._mode != "clap":
            raise RuntimeError("CLAP audio encoder is not available.")
        paths = list(audio_paths)
        audio_emb = self._clap_model.get_audio_embedding_from_filelist(x=paths, use_tensor=False)  # type: ignore
        audio_emb = np.array(audio_emb, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(audio_emb, axis=1, keepdims=True) + 1e-12
            audio_emb = audio_emb / norms
        return audio_emb

    def encode_texts(self, texts: Iterable[str], normalize: bool = True):
        texts = list(texts)
        if self._mode == "clap":
            text_emb = self._clap_model.get_text_embedding(texts, use_tensor=False)  # type: ignore
            text_emb = np.array(text_emb, dtype=np.float32)
            if normalize:
                norms = np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-12
                text_emb = text_emb / norms
            return text_emb
        return self._text_fallback.encode(texts, normalize=normalize)


