"""MUSAN noise pool and SNR-controlled mixing.

Noise segments are cached once into a fixed pool so that the noise a clip
receives is a deterministic function of its sample id, identical across every
condition. That keeps the DSP arms comparable: only the front-end differs.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

EPS = 1e-12


def discover_noise_files(musan_root: Path) -> list[Path]:
    """All MUSAN ``noise`` recordings, sorted for reproducibility."""
    noise_root = Path(musan_root) / "noise"
    if not noise_root.is_dir():
        raise FileNotFoundError(f"MUSAN noise directory not found: {noise_root}")
    files = sorted(noise_root.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files below {noise_root}")
    return files


def build_noise_pool(
    musan_root: Path,
    cache_root: Path,
    *,
    seed: int,
    sample_rate: int,
    segment_seconds: float,
    pool_size: int = 2000,
) -> Path:
    """Cache ``pool_size`` fixed-length noise segments drawn across MUSAN files."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    segment_samples = round(sample_rate * segment_seconds)
    files = discover_noise_files(musan_root)

    pool = np.zeros((pool_size, segment_samples), dtype=np.float32)
    sources: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for index in range(pool_size):
        path = files[index % len(files)]
        audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = audio.mean(axis=1)
        if source_rate != sample_rate:
            waveform = (
                torchaudio.functional.resample(
                    torch.from_numpy(waveform), source_rate, sample_rate
                )
                .numpy()
                .astype(np.float32)
            )

        if waveform.size < segment_samples:
            waveform = np.tile(
                waveform, int(np.ceil(segment_samples / max(1, waveform.size)))
            )
        start = int(rng.integers(0, waveform.size - segment_samples + 1))
        segment = waveform[start : start + segment_samples].astype(np.float32)

        pool[index] = segment
        sources.append({"index": index, "path": str(path), "start_sample": start})

    np.save(cache_root / "noise-pool.npy", pool)
    (cache_root / "noise-pool.json").write_text(
        json.dumps(sources, indent=2) + "\n", encoding="utf-8"
    )
    return cache_root / "noise-pool.npy"


def load_noise_pool(cache_root: Path) -> np.ndarray:
    pool_path = Path(cache_root) / "noise-pool.npy"
    if not pool_path.is_file():
        raise FileNotFoundError(
            f"Missing noise pool {pool_path}; run `run.py prepare` first."
        )
    return np.load(pool_path)


def deterministic_choice(namespace: str, size: int) -> int:
    digest = sha256(namespace.encode()).digest()
    return int.from_bytes(digest[:8], "big") % size


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add ``noise`` to ``speech`` at the requested RMS SNR, avoiding clipping."""
    if speech.shape != noise.shape:
        raise ValueError(
            f"shape mismatch: speech {speech.shape} vs noise {noise.shape}"
        )
    speech_power = float(np.mean(speech.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if speech_power <= EPS or noise_power <= EPS:
        return speech.astype(np.float32)

    target_noise_power = speech_power / (10.0 ** (snr_db / 10.0))
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)
    mixture = speech + scaled_noise

    peak = float(np.max(np.abs(mixture)))
    if peak > 1.0:
        mixture = mixture / peak
    return mixture.astype(np.float32)


def snr_for_sample(sample_id: str, seed: int, snr_choices: tuple[float, ...]) -> float:
    """Deterministic training SNR so every condition sees the same difficulty."""
    return float(
        snr_choices[deterministic_choice(f"{seed}:{sample_id}:snr", len(snr_choices))]
    )


def noise_for_sample(sample_id: str, seed: int, pool: np.ndarray) -> np.ndarray:
    """Deterministic noise segment, identical across conditions for a given clip."""
    return pool[deterministic_choice(f"{seed}:{sample_id}:noise", pool.shape[0])]
