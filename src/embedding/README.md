# Embedding

Interfaces for turning paper text into vector representations using SciBERT.

## Public API

- `SciBERTEmbedder` – wraps the `allenai/scibert_scivocab_uncased` model.
- `load_model()` – convenience factory that returns a ready embedder.
- `embed_text()` / `embed_text_async()` – synchronous or asynchronous embedding helpers.
- `to_binary()` / `from_binary()` / `translate()` – convert between NumPy arrays and raw bytes.
- `utils.clean_text()` – light text normalisation before embedding.

## Usage

```python
from src.embedding import load_model, embed_text, translate
from src.embedding.utils import clean_text

model = load_model()
text = clean_text(" Some\nabstract  text ")
vector = embed_text(model, text)
blob = translate(vector)  # bytes for storage
```

## Contributing

The embedder currently targets SciBERT. New models or pooling strategies can be
added by extending `SciBERTEmbedder` or creating parallel classes. Utilities
should remain framework-agnostic and accept/return NumPy arrays where possible.
