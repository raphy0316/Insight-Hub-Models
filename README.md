# CLIP × CLAP: Cross-modal retrieval via image-text (CLIP) and audio-text (CLAP)

This package provides a minimal pipeline to match images and music by leveraging:
- CLIP for image-text alignment (image vs. music textual metadata)
- CLAP for audio-text alignment (audio vs. query text)
- A text fallback (Sentence-Transformers → TF-IDF) when CLIP/CLAP are unavailable

Key idea: Normalize both image and music into comparable textual or embedding spaces and aggregate scores.

## Structure
- `src/pipeline/preprocess.py`: Normalize/compose textual fields
- `src/pipeline/embedder.py`: Encoders for CLIP image/text, CLAP audio/text, and text fallback
- `src/pipeline/retrieval.py`: Cosine similarity and ranking
- `src/model/clip_clap.py`: End-to-end index build and search
- `config/default.yaml`: Runtime configuration
- `notebooks/clip_clap_experiments.ipynb`: Minimal demo notebook
- `results/`: Metrics, samples, plots (optional)

## Install
```bash
pip install -r requirements.txt
```

Or install individually:
- Recommended (optional):
```bash
pip install open-clip-torch laion_clap sentence-transformers pyyaml numpy scikit-learn pillow
```
- Fallback only (no CLIP/CLAP):
```bash
pip install sentence-transformers pyyaml numpy scikit-learn pillow
```

## Quickstart
```python
from src.model.clip_clap import ClipClapModel

model = ClipClapModel.from_config("config/default.yaml")

music_items = [
    {
        "id": "track_001",
        "title": "Midnight Drive",
        "artists": ["Luna Wave"],
        "genres": ["synthwave"],
        "editorial_tags": ["retro", "night", "neon"],
        "lyrics": "Riding through the empty roads under neon lights...",
        "summary": "Retro electronic vibe for night cityscapes.",
        # "audio_path": "path/to/audio/file.wav",  # optional for CLAP
    },
]
model.prepare_music_index(music_items)

# Search with image keywords (text pivot; works with CLAP-text and text fallback)
results = model.search_by_image_keywords(["neon", "night city", "retro"], top_k=10)
for r in results:
    print(r["id"], r["score"], r["text"][:60])

# Or search with image files directly (requires CLIP)
# results = model.search_by_image_paths(["path/to/image1.jpg", "path/to/image2.jpg"], top_k=10)
```

## Search Methods
- `search_by_image_keywords()`: Search using text keywords (works with CLAP-text, text fallback, and optionally CLIP-text)
- `search_by_image_paths()`: Search using image file paths (requires CLIP to be enabled)

## Notes
- If CLIP is available, image files can be compared against CLIP text embeddings derived from music metadata.
- If CLAP is available and `audio_path` exists, CLAP audio embeddings are used; otherwise, CLAP path is skipped.
- When neither CLIP nor CLAP is available, the pipeline falls back to text-only retrieval (Sentence-Transformers → TF-IDF).


