"""SciBERT-based embeddings for scientific papers."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.models import Paper


DEFAULT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
_CACHED_TOKENIZER: Optional[AutoTokenizer] = None
_CACHED_MODEL: Optional[AutoModel] = None


def _get_sci_bert(model_name: str = DEFAULT_MODEL_NAME) -> Tuple[AutoTokenizer, AutoModel]:
    """Lazily load and cache the SciBERT tokenizer and model."""

    global _CACHED_TOKENIZER, _CACHED_MODEL
    if _CACHED_TOKENIZER is None or _CACHED_MODEL is None:
        _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _CACHED_MODEL = AutoModel.from_pretrained(model_name)
        _CACHED_MODEL.eval()
    return _CACHED_TOKENIZER, _CACHED_MODEL


def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    masked_states = hidden_states * mask
    return masked_states.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def embed_papers(
    papers: Iterable[Paper],
    *,
    tokenizer: Optional[AutoTokenizer] = None,
    model: Optional[AutoModel] = None,
    max_length: int = 512,
) -> Tuple[np.ndarray, str]:
    """Embed papers with SciBERT.

    The tokenizer and model can be supplied (useful for lightweight testing);
    otherwise the default SciBERT weights are fetched and cached.
    """

    if (tokenizer is None) != (model is None):
        raise ValueError("`tokenizer` and `model` must be provided together or omitted together.")

    if tokenizer is None:
        tokenizer, model = _get_sci_bert()
    texts = [f"{paper.title}\n{paper.abstract}".strip() for paper in papers]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoded)
    embeddings = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
    return embeddings.cpu().numpy(), getattr(model.config, "name_or_path", DEFAULT_MODEL_NAME)
