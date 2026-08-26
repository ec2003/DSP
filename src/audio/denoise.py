from __future__ import annotations

import numpy as np
import torch
from scipy.signal import wiener
from torch import Tensor

from src.data.vctk import ManifestRecord


class WienerDenoiser:
    """Apply SciPy's local Wiener filter while preserving the waveform contract."""

    def __init__(self, window_size: int) -> None:
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer of at least 3")
        self.window_size = window_size

    def __call__(
        self, waveform: Tensor, sample_rate: int, record: ManifestRecord
    ) -> Tensor:
        del sample_rate, record
        if waveform.ndim != 1:
            raise ValueError("WienerDenoiser expects a one-dimensional mono waveform")
        denoised = wiener(waveform.detach().cpu().numpy(), mysize=self.window_size)
        if not np.isfinite(denoised).all():
            raise ValueError("Wiener filter produced non-finite waveform values")
        return torch.as_tensor(denoised, dtype=waveform.dtype, device=waveform.device)
