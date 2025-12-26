import numpy as np
import torch

from src.embedding import embed_papers
from src.models import Paper


class DummyTokenizer:
    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        batch_size = len(texts)
        seq_length = 4
        return {
            "input_ids": torch.ones((batch_size, seq_length), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_length), dtype=torch.long),
        }


class DummyModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size, "name_or_path": "dummy"})

    def forward(self, input_ids=None, attention_mask=None):
        batch, seq_length = input_ids.shape
        hidden = torch.arange(batch * seq_length * self.config.hidden_size, dtype=torch.float32)
        hidden = hidden.view(batch, seq_length, self.config.hidden_size)
        return type("Output", (), {"last_hidden_state": hidden})


def test_embed_papers_creates_sci_bert_embeddings():
    papers = [
        Paper(title="Alpha", abstract="Study of alpha particles", references=[]),
        Paper(title="Beta", abstract="Beta decay in physics", references=[]),
    ]

    matrix, model_name = embed_papers(
        papers,
        tokenizer=DummyTokenizer(),
        model=DummyModel(hidden_size=12),
    )

    assert matrix.shape == (2, 12)
    assert model_name == "dummy"
    assert np.allclose(matrix[0], matrix[1]) is False
