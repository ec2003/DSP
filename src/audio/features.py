"""Inspectable DSP features used for analysis and the DSP-only reference."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F


def stft_magnitude(waveform: Tensor, *, n_fft: int = 512, hop_length: int = 160) -> Tensor:
    """Return a finite magnitude STFT with frequency x frame layout."""
    _mono(waveform)
    result = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype), return_complex=True).abs()
    if not torch.isfinite(result).all():
        raise ValueError("STFT contains non-finite values")
    return result


def welch_psd(waveform: Tensor, *, sample_rate: int, nperseg: int = 512) -> tuple[np.ndarray, np.ndarray]:
    _mono(waveform)
    frequencies, power = welch(waveform.detach().cpu().numpy(), fs=sample_rate, nperseg=min(nperseg, waveform.numel()))
    if not np.isfinite(power).all():
        raise ValueError("PSD contains non-finite values")
    return frequencies, power


def band_power(frequencies: np.ndarray, power: np.ndarray, low_hz: float, high_hz: float) -> float:
    if not 0 <= low_hz < high_hz or frequencies.shape != power.shape:
        raise ValueError("invalid band or PSD arrays")
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    return float(np.trapezoid(power[mask], frequencies[mask])) if mask.any() else 0.0


def mfcc_features(waveform: Tensor, *, sample_rate: int, n_mfcc: int = 40, n_mels: int = 80) -> Tensor:
    _mono(waveform)
    transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": n_mels})
    result = transform(waveform)
    if result.shape[0] != n_mfcc or not torch.isfinite(result).all():
        raise ValueError("invalid MFCC output")
    return result


def mfcc_summary_embedding(waveform: Tensor, *, sample_rate: int, n_mfcc: int = 40, n_mels: int = 80) -> Tensor:
    """Mean/std MFCC utterance vector, L2-normalised for cosine verification."""
    features = mfcc_features(waveform, sample_rate=sample_rate, n_mfcc=n_mfcc, n_mels=n_mels)
    vector = torch.cat((features.mean(dim=1), features.std(dim=1, correction=0)))
    return F.normalize(vector, p=2, dim=0)


def residual_snr_db(clean: Tensor, processed: Tensor, *, epsilon: float = 1e-8) -> float:
    _mono(clean); _mono(processed)
    if clean.shape != processed.shape:
        raise ValueError("clean and processed waveforms must have equal lengths")
    return float(10 * torch.log10(clean.square().mean().clamp_min(epsilon) / (processed - clean).square().mean().clamp_min(epsilon)))


def _mono(waveform: Tensor) -> None:
    if waveform.ndim != 1 or waveform.numel() < 2:
        raise ValueError("expected non-empty mono waveform")
