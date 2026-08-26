from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch import Tensor

from src.data.vctk import ManifestRecord


def apply_sos_filter(
    waveform: Tensor,
    *,
    sample_rate: int,
    cutoff_hz: float | tuple[float, float],
    filter_type: str,
    order: int,
) -> Tensor:
    """Apply an offline zero-phase Butterworth filter to a mono waveform."""
    if waveform.ndim != 1:
        raise ValueError("Butterworth filters expect a one-dimensional mono waveform")
    if order < 1:
        raise ValueError("Butterworth filter order must be positive")

    nyquist_hz = sample_rate / 2
    normalized_cutoff = np.asarray(cutoff_hz, dtype=float) / nyquist_hz
    if np.any(normalized_cutoff <= 0) or np.any(normalized_cutoff >= 1):
        raise ValueError("Filter cutoffs must lie strictly between zero and Nyquist")
    if normalized_cutoff.size == 2 and normalized_cutoff[0] >= normalized_cutoff[1]:
        raise ValueError("Band-pass cutoffs must be ordered")

    sos = butter(order, normalized_cutoff, btype=filter_type, output="sos")
    try:
        filtered = sosfiltfilt(sos, waveform.detach().cpu().numpy())
    except ValueError as error:
        raise ValueError(
            "Waveform is too short for zero-phase Butterworth filtering"
        ) from error
    if not np.isfinite(filtered).all():
        raise ValueError("Butterworth filter produced non-finite waveform values")
    filtered_tensor = torch.as_tensor(
        np.ascontiguousarray(filtered), dtype=waveform.dtype, device=waveform.device
    )
    return filtered_tensor / filtered_tensor.abs().max().clamp_min(1.0)


class HighPassFilter:
    def __init__(self, cutoff_hz: float, order: int) -> None:
        self.cutoff_hz = cutoff_hz
        self.order = order

    def __call__(
        self, waveform: Tensor, sample_rate: int, record: ManifestRecord
    ) -> Tensor:
        del record
        return apply_sos_filter(
            waveform,
            sample_rate=sample_rate,
            cutoff_hz=self.cutoff_hz,
            filter_type="highpass",
            order=self.order,
        )


class LowPassFilter:
    def __init__(self, cutoff_hz: float, order: int) -> None:
        self.cutoff_hz = cutoff_hz
        self.order = order

    def __call__(
        self, waveform: Tensor, sample_rate: int, record: ManifestRecord
    ) -> Tensor:
        del record
        return apply_sos_filter(
            waveform,
            sample_rate=sample_rate,
            cutoff_hz=self.cutoff_hz,
            filter_type="lowpass",
            order=self.order,
        )


class DspTransformChain:
    """Apply named waveform transforms in their declared offline DSP order."""

    def __init__(
        self, *transforms: Callable[[Tensor, int, ManifestRecord], Tensor]
    ) -> None:
        if not transforms:
            raise ValueError("DspTransformChain requires at least one transform")
        self.transforms = transforms

    def __call__(
        self, waveform: Tensor, sample_rate: int, record: ManifestRecord
    ) -> Tensor:
        for transform in self.transforms:
            waveform = transform(waveform, sample_rate, record)
        return waveform
