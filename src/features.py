"""Feature extraction: log-mel / MFCC (cepstral), PSD band power, and
statistical / entropy descriptors.

The log-mel front-end feeds the CNN embedding model and runs batched on the
accelerator; the handcrafted vector feeds the classical SVM / random-forest
baselines and is computed per clip on CPU.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torchaudio
from scipy import signal
from torch import nn

EPS = 1e-10

#: Octave-band edges (Hz) covering the 16 kHz telephone-to-wideband range.
OCTAVE_BAND_EDGES: tuple[tuple[float, float], ...] = (
    (20.0, 125.0),
    (125.0, 250.0),
    (250.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 4000.0),
    (4000.0, 6000.0),
    (6000.0, 7900.0),
)


# --------------------------------------------------------------------------- #
# Torch front-end used by the CNN
# --------------------------------------------------------------------------- #
class LogMelSpectrogram(nn.Module):
    """Batched log-mel front-end with per-utterance mean-variance normalisation."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 20.0,
        f_max: float = 7900.0,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = torch.log(self.mel(waveform) + EPS)
        mean = features.mean(dim=-1, keepdim=True)
        std = features.std(dim=-1, keepdim=True).clamp_min(1e-5)
        return (features - mean) / std


@lru_cache(maxsize=4)
def _mfcc_transform(
    sample_rate: int, n_mfcc: int, n_mels: int, n_fft: int, hop_length: int
):
    return torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "f_min": 20.0,
            "f_max": 7900.0,
            "power": 2.0,
        },
    )


def mfcc_frames(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    """MFCC frames with shape ``(n_mfcc, n_frames)``."""
    transform = _mfcc_transform(sample_rate, n_mfcc, n_mels, n_fft, hop_length)
    tensor = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
    return transform(tensor).numpy()


def mfcc_statistics(frames: np.ndarray) -> np.ndarray:
    """Concatenated per-coefficient mean, std, and mean delta."""
    delta = np.diff(frames, axis=1) if frames.shape[1] > 1 else np.zeros_like(frames)
    return np.concatenate(
        [frames.mean(axis=1), frames.std(axis=1), np.abs(delta).mean(axis=1)]
    )


# --------------------------------------------------------------------------- #
# Spectral / power descriptors
# --------------------------------------------------------------------------- #
def welch_psd(waveform: np.ndarray, sample_rate: int = 16000, nperseg: int = 512):
    """Welch power spectral density estimate."""
    return signal.welch(waveform, fs=sample_rate, nperseg=min(nperseg, waveform.size))


def band_power(
    freqs: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float
) -> float:
    """Integrate the PSD over ``[low_hz, high_hz)``."""
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not mask.any():
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def octave_band_powers(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    edges: tuple[tuple[float, float], ...] = OCTAVE_BAND_EDGES,
) -> np.ndarray:
    """Band power per octave band, in dB."""
    freqs, psd = welch_psd(waveform, sample_rate)
    powers = np.array([band_power(freqs, psd, low, high) for low, high in edges])
    return 10.0 * np.log10(powers + EPS)


def spectral_entropy(waveform: np.ndarray, sample_rate: int = 16000) -> float:
    """Normalised Shannon entropy of the power spectrum.

    Tonal/structured spectra score low; noise-like flat spectra approach 1.
    """
    _, psd = welch_psd(waveform, sample_rate)
    distribution = psd / (psd.sum() + EPS)
    entropy = -np.sum(distribution * np.log2(distribution + EPS))
    return float(entropy / np.log2(len(distribution)))


def spectral_shape_statistics(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    """Frame-wise centroid, bandwidth, roll-off, and flatness (mean and std)."""
    freqs, _, spectrum = signal.stft(
        waveform, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    magnitude = np.abs(spectrum) + EPS
    weights = magnitude / magnitude.sum(axis=0, keepdims=True)

    centroid = (freqs[:, None] * weights).sum(axis=0)
    bandwidth = np.sqrt((((freqs[:, None] - centroid) ** 2) * weights).sum(axis=0))
    cumulative = np.cumsum(weights, axis=0)
    rolloff = freqs[np.argmax(cumulative >= 0.85, axis=0)]
    flatness = np.exp(np.mean(np.log(magnitude), axis=0)) / np.mean(magnitude, axis=0)

    stacked = np.stack([centroid, bandwidth, rolloff, flatness])
    return np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)])


def time_domain_statistics(waveform: np.ndarray, frame_length: int = 400) -> np.ndarray:
    """Frame-wise RMS and zero-crossing rate (mean and std)."""
    n_frames = max(1, waveform.size // frame_length)
    frames = waveform[: n_frames * frame_length].reshape(n_frames, frame_length)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    return np.array([rms.mean(), rms.std(), zcr.mean(), zcr.std()])


def handcrafted_features(waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Feature vector for the classical baselines.

    Combines cepstral (MFCC), spectral-power (octave band power), and
    statistical/entropy descriptors, satisfying three DSP technique categories.
    """
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    return np.concatenate(
        [
            mfcc_statistics(mfcc_frames(waveform, sample_rate)),
            octave_band_powers(waveform, sample_rate),
            spectral_shape_statistics(waveform, sample_rate),
            time_domain_statistics(waveform),
            [spectral_entropy(waveform, sample_rate)],
        ]
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
# Signal-quality diagnostics
# --------------------------------------------------------------------------- #
def residual_snr_db(clean: np.ndarray, processed: np.ndarray) -> float:
    """SNR of ``processed`` treating ``clean`` as the reference signal."""
    length = min(clean.size, processed.size)
    reference = clean[:length]
    error = processed[:length] - reference
    return float(
        10.0 * np.log10((np.sum(reference**2) + EPS) / (np.sum(error**2) + EPS))
    )
