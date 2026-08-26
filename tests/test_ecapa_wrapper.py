from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models import EcapaSpeakerEncoder


class _FeatureExtractor(nn.Module):
    def forward(self, waveforms: Tensor) -> Tensor:
        return waveforms.unsqueeze(-1)


class _IdentityNorm(nn.Module):
    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        del lengths
        return features


class _EmbeddingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 2, bias=False)
        nn.init.constant_(self.projection.weight, 1.0)

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        del lengths
        return self.projection(features.mean(dim=1)).unsqueeze(1)


def test_ecapa_wrapper_returns_l2_normalized_embeddings() -> None:
    embedding_model = _EmbeddingModel()
    model = EcapaSpeakerEncoder(_FeatureExtractor(), _IdentityNorm(), embedding_model)

    embeddings = model(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    assert embeddings.shape == (2, 2)
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(2))
    model.set_embedding_trainable(False)
    assert not any(
        parameter.requires_grad for parameter in embedding_model.parameters()
    )
