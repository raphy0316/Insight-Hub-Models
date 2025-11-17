# Text-Pivot: Text-axis normalization for image ↔ music matching

Normalize image keywords (user input) and music keywords (Spotify metadata/lyrics/editorial tags) to a common text axis, then compute similarity in text embedding space to generate rankings.

## Structure
- `src/pipeline/preprocessing.py`: Text normalization and metadata → sentence composition
- `src/pipeline/embedder.py`: Text embedder (default: Sentence-Transformers, fallback: TF-IDF)
- `src/pipeline/retrieval.py`: Cosine similarity / weighted sum scores / ranking
- `src/model/text_pivot.py`: Index building, query embedding, search/ranking end-to-end
- `config/default.yaml`: Pipeline configuration
- `notebooks/text_pivot_experiments.ipynb`: Experiment notebook

## Install
```bash
pip install -r requirements.txt
```

Or install individually:
- Recommended (optional): Sentence-Transformers
```bash
pip install sentence-transformers pyyaml numpy scikit-learn
```
- Fallback (TF-IDF) only:
```bash
pip install pyyaml numpy scikit-learn
```

## Quickstart
```python
from src.model.text_pivot import TextPivotModel

# Load config and create model
model = TextPivotModel.from_config("config/default.yaml")

# Build music metadata index (example)
music_items = [
    {
        "id": "track_001",
        "title": "Midnight Drive",
        "artists": ["Luna Wave"],
        "album": "Nocturne",
        "genres": ["synthwave", "electronic"],
        "editorial_tags": ["retro", "night city", "neon"],
        "lyrics": "Riding through the empty roads under neon lights...",
        "summary": "Retro electronic vibe suitable for night-time cityscapes."
    },
    # ... more tracks
]
model.prepare_music_index(music_items)

# Search with image keywords (user input)
image_keywords = ["neon", "night city", "retro", "car driving"]
results = model.search_by_image_keywords(image_keywords, top_k=10)
for r in results:
    print(r["id"], r["score"], r["text"][:80])

# Or search with pure text query
# results = model.search_by_text("synthwave night drive", top_k=10)

# Or batch search multiple queries
# batch_results = model.batch_search_by_text(["query1", "query2"], top_k=10)
```

## Search Methods
- `search_by_image_keywords()`: Search using image keywords (user input) normalized to text
- `search_by_text()`: Search using a pure text query string
- `batch_search_by_text()`: Batch search with multiple text queries

## Configuration (`config/default.yaml`)
- Defines embedder model, preprocessing options, ranking weights, etc.

## Fallback Strategy
- Automatically falls back to TF-IDF when Sentence-Transformers is unavailable.
- TF-IDF learns from the full corpus during index building, so index rebuild may be needed.

## Results
- `results/metrics.json`: Simple metrics (optional) logging
- `results/samples/`: Sample outputs (optional)
- `results/embedding_plots/`: Embedding visualizations (optional)
