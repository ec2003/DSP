from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class EcapaSpeakerEncoder(nn.Module):
    """Gradient-enabled wrapper for SpeechBrain's pretrained ECAPA encoder."""

    def __init__(
        self,
        compute_features: nn.Module,
        mean_var_norm: nn.Module,
        embedding_model: nn.Module,
    ) -> None:
        super().__init__()
        self.compute_features = compute_features
        self.mean_var_norm = mean_var_norm
        self.embedding_model = embedding_model

    @classmethod
    def from_pretrained(
        cls,
        *,
        source: str,
        cache_dir: Path,
        device: torch.device,
    ) -> EcapaSpeakerEncoder:
        from speechbrain.inference.speaker import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir=str(cache_dir),
            run_opts={"device": str(device)},
        )
        return cls(
            compute_features=classifier.mods.compute_features,
            mean_var_norm=classifier.mods.mean_var_norm,
            embedding_model=classifier.mods.embedding_model,
        ).to(device)

    def forward(self, waveforms: Tensor) -> Tensor:
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim != 2:
            raise ValueError("ECAPA expects waveforms with shape [batch, time]")

        lengths = torch.ones(waveforms.shape[0], device=waveforms.device)
        features = self.compute_features(waveforms)
        features = self.mean_var_norm(features, lengths)
        embeddings = self.embedding_model(features, lengths)
        embeddings = embeddings.squeeze(1)
        return functional.normalize(embeddings, p=2, dim=-1)

    def set_embedding_trainable(self, trainable: bool) -> None:
        for parameter in self.embedding_model.parameters():
            parameter.requires_grad = trainable
