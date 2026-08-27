"""DSP front-end: IIR filtering, adaptive notch, and STFT-domain denoising.

All filters are zero-phase (``filtfilt``/``sosfiltfilt``), so the chain is an
offline front-end and is not claimed to be causal/streaming-deployable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

Waveform = np.ndarray
Stage = Callable[[Waveform], Waveform]

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Basic IIR filtering
# --------------------------------------------------------------------------- #
def _normalise_cutoff(
    cutoff_hz: float | Sequence[float], sample_rate: int
) -> float | list[float]:
    nyquist = sample_rate / 2.0
    cutoff = np.atleast_1d(np.asarray(cutoff_hz, dtype=float))
    if np.any(cutoff <= 0.0) or np.any(cutoff >= nyquist):
        raise ValueError(f"cutoff {cutoff_hz} Hz must lie in (0, {nyquist}) Hz")
    if cutoff.size == 2 and cutoff[0] >= cutoff[1]:
        raise ValueError(f"band edges must be increasing, got {cutoff_hz}")
    normalised = cutoff / nyquist
    return float(normalised[0]) if normalised.size == 1 else normalised.tolist()


def butter_sos(
    kind: str,
    cutoff_hz: float | Sequence[float],
    sample_rate: int,
    order: int = 4,
) -> np.ndarray:
    """Design a Butterworth filter as second-order sections."""
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    return signal.butter(
        order, _normalise_cutoff(cutoff_hz, sample_rate), btype=kind, output="sos"
    )


def apply_sos(waveform: Waveform, sos: np.ndarray) -> Waveform:
    """Zero-phase forward-backward filtering."""
    return np.asarray(signal.sosfiltfilt(sos, waveform), dtype=np.float32)


def highpass(
    waveform: Waveform, sample_rate: int, cutoff_hz: float, order: int = 4
) -> Waveform:
    return apply_sos(waveform, butter_sos("highpass", cutoff_hz, sample_rate, order))


def lowpass(
    waveform: Waveform, sample_rate: int, cutoff_hz: float, order: int = 4
) -> Waveform:
    return apply_sos(waveform, butter_sos("lowpass", cutoff_hz, sample_rate, order))


def bandpass(
    waveform: Waveform,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> Waveform:
    return apply_sos(
        waveform, butter_sos("bandpass", (low_hz, high_hz), sample_rate, order)
    )


def notch(
    waveform: Waveform, sample_rate: int, freq_hz: float, quality: float = 30.0
) -> Waveform:
    """Zero-phase IIR notch at ``freq_hz``."""
    nyquist = sample_rate / 2.0
    if not 0.0 < freq_hz < nyquist:
        raise ValueError(f"notch frequency {freq_hz} Hz must lie in (0, {nyquist}) Hz")
    b, a = signal.iirnotch(freq_hz / nyquist, quality)
    return np.asarray(signal.filtfilt(b, a, waveform), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Adaptive notch: find narrowband tonal interference from the Welch PSD
# --------------------------------------------------------------------------- #
def detect_tonal_peaks(
    waveform: Waveform,
    sample_rate: int,
    max_peaks: int = 2,
    prominence_db: float = 8.0,
    min_freq_hz: float = 40.0,
    max_freq_hz: float | None = None,
    nperseg: int = 1024,
) -> list[float]:
    """Return frequencies of narrowband peaks that stand out from the PSD trend.

    A peak is tonal only if it rises ``prominence_db`` above the local spectral
    envelope, which is estimated by median-filtering the PSD in dB.
    """
    max_freq_hz = max_freq_hz if max_freq_hz is not None else sample_rate / 2.0 * 0.95
    freqs, psd = signal.welch(
        waveform, fs=sample_rate, nperseg=min(nperseg, waveform.size)
    )
    psd_db = 10.0 * np.log10(psd + EPS)
    kernel = max(3, (len(psd_db) // 24) | 1)
    envelope_db = signal.medfilt(psd_db, kernel_size=kernel)
    excess_db = psd_db - envelope_db

    band = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
    candidates, properties = signal.find_peaks(
        np.where(band, excess_db, -np.inf), height=prominence_db
    )
    if candidates.size == 0:
        return []
    order = np.argsort(properties["peak_heights"])[::-1][:max_peaks]
    return [float(freqs[candidates[i]]) for i in np.sort(order)]


def adaptive_notch(
    waveform: Waveform,
    sample_rate: int,
    max_peaks: int = 2,
    prominence_db: float = 8.0,
    quality: float = 30.0,
) -> Waveform:
    """Notch out tonal interference detected in the signal itself."""
    out = waveform
    for freq_hz in detect_tonal_peaks(
        waveform, sample_rate, max_peaks=max_peaks, prominence_db=prominence_db
    ):
        out = notch(out, sample_rate, freq_hz, quality=quality)
    return out


# --------------------------------------------------------------------------- #
# STFT-domain denoising
# --------------------------------------------------------------------------- #
def _stft(waveform: Waveform, sample_rate: int, n_fft: int, hop_length: int):
    return signal.stft(
        waveform,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
        padded=True,
    )


def _istft(
    spectrum: np.ndarray, sample_rate: int, n_fft: int, hop_length: int, length: int
):
    _, waveform = signal.istft(
        spectrum,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=True,
    )
    out = np.zeros(length, dtype=np.float32)
    usable = min(length, waveform.size)
    out[:usable] = waveform[:usable]
    return out


def estimate_noise_psd(magnitude_sq: np.ndarray, quantile: float = 0.1) -> np.ndarray:
    """Average power of the lowest-energy frames, used as the noise floor estimate.

    Speech is sparse in time, so the quietest frames of a mixture are dominated
    by the stationary noise component.
    """
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile must lie in (0, 1], got {quantile}")
    frame_energy = magnitude_sq.sum(axis=0)
    n_frames = magnitude_sq.shape[1]
    keep = max(1, round(quantile * n_frames))
    quietest = np.argsort(frame_energy)[:keep]
    return magnitude_sq[:, quietest].mean(axis=1) + EPS


def stft_wiener(
    waveform: Waveform,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 160,
    noise_quantile: float = 0.1,
    alpha: float = 0.98,
    gain_floor_db: float = -18.0,
) -> Waveform:
    """Wiener filter with decision-directed a-priori SNR estimation.

    Gain is ``G = xi / (1 + xi)`` where ``xi`` is the a-priori SNR smoothed
    across frames (Ephraim-Malah decision-directed rule). The gain floor limits
    musical-noise artefacts caused by isolated spectral bins collapsing to zero.
    """
    _, _, spectrum = _stft(waveform, sample_rate, n_fft, hop_length)
    magnitude_sq = np.abs(spectrum) ** 2
    noise_psd = estimate_noise_psd(magnitude_sq, noise_quantile)[:, None]

    posterior_snr = magnitude_sq / noise_psd
    instantaneous = np.maximum(posterior_snr - 1.0, 0.0)
    gain_floor = 10.0 ** (gain_floor_db / 20.0)

    enhanced = np.empty_like(spectrum)
    previous_clean_psd = instantaneous[:, 0] * noise_psd[:, 0]
    for frame in range(spectrum.shape[1]):
        prior_snr = (
            alpha * (previous_clean_psd / noise_psd[:, 0])
            + (1.0 - alpha) * instantaneous[:, frame]
        )
        gain = np.maximum(prior_snr / (1.0 + prior_snr), gain_floor)
        enhanced[:, frame] = gain * spectrum[:, frame]
        previous_clean_psd = np.abs(enhanced[:, frame]) ** 2

    return _istft(enhanced, sample_rate, n_fft, hop_length, waveform.size)


def spectral_subtraction(
    waveform: Waveform,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 160,
    noise_quantile: float = 0.1,
    over_subtraction: float = 1.5,
    spectral_floor: float = 0.05,
) -> Waveform:
    """Magnitude spectral subtraction with an over-subtraction factor and floor."""
    _, _, spectrum = _stft(waveform, sample_rate, n_fft, hop_length)
    magnitude = np.abs(spectrum)
    noise_magnitude = np.sqrt(estimate_noise_psd(magnitude**2, noise_quantile))[:, None]

    subtracted = magnitude - over_subtraction * noise_magnitude
    cleaned = np.maximum(subtracted, spectral_floor * magnitude)
    enhanced = cleaned * np.exp(1j * np.angle(spectrum))
    return _istft(enhanced, sample_rate, n_fft, hop_length, waveform.size)


# --------------------------------------------------------------------------- #
# Composable chain
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DspChain:
    """Ordered DSP stages applied to a waveform, with the stage names retained."""

    names: tuple[str, ...] = ()
    stages: tuple[Stage, ...] = field(default=(), repr=False)

    def __call__(self, waveform: Waveform) -> Waveform:
        out = np.asarray(waveform, dtype=np.float32)
        for stage in self.stages:
            out = np.asarray(stage(out), dtype=np.float32)
        return out
