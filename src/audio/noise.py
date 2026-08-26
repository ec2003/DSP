from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import torch
import torchaudio
from torch import Tensor

from src.audio.filters import apply_sos_filter
from src.data.vctk import ManifestRecord


def mix_at_snr(
    speech: Tensor, noise: Tensor, snr_db: float, *, epsilon: float = 1e-8
) -> Tensor:
    """Mix mono waveforms at a target RMS signal-to-noise ratio without clipping."""
    if speech.ndim != 1 or noise.ndim != 1:
        raise ValueError("speech and noise must be one-dimensional mono waveforms")
    if speech.numel() == 0 or noise.numel() == 0:
        raise ValueError("speech and noise must not be empty")

    aligned_noise = _match_length(noise, speech.numel())
    speech_rms = speech.pow(2).mean().sqrt().clamp_min(epsilon)
    noise_rms = aligned_noise.pow(2).mean().sqrt().clamp_min(epsilon)
    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    mixed = speech + aligned_noise * (target_noise_rms / noise_rms)
    peak = mixed.abs().max().clamp_min(1.0)
    return mixed / peak


class MusanNoiseMixer:
    def __init__(
        self, noise_files: Iterable[Path], *, seed: int, snr_db: tuple[int, ...]
    ) -> None:
        self.noise_files = sorted(Path(path) for path in noise_files)
        if not self.noise_files:
            raise ValueError("At least one MUSAN audio file is required")
        if not snr_db:
            raise ValueError("snr_db must not be empty")
        self.seed = seed
        self.snr_db = snr_db

    @classmethod
    def from_root(
        cls, musan_root: Path, *, seed: int, snr_db: tuple[int, ...]
    ) -> MusanNoiseMixer:
        audio_extensions = {".wav", ".flac", ".mp3"}
        noise_files = [
            path
            for path in musan_root.rglob("*")
            if path.suffix.lower() in audio_extensions
        ]
        return cls(noise_files, seed=seed, snr_db=snr_db)

    def __call__(
        self, waveform: Tensor, sample_rate: int, record: ManifestRecord
    ) -> Tensor:
        noise_path = self._select_noise_file(record.sample_id)
        noise, noise_sample_rate = torchaudio.load(noise_path)
        noise = noise.mean(dim=0)
        if noise_sample_rate != sample_rate:
            noise = torchaudio.functional.resample(
                noise, noise_sample_rate, sample_rate
            )
        return mix_at_snr(waveform, noise, self._select_snr(record.sample_id))

    def _select_noise_file(self, sample_id: str) -> Path:
        return self.noise_files[
            _deterministic_index(sample_id, self.seed, len(self.noise_files), "noise")
        ]

    def _select_snr(self, sample_id: str) -> int:
        return self.snr_db[
            _deterministic_index(sample_id, self.seed, len(self.snr_db), "snr")
        ]


@dataclass(frozen=True)
class CompositeNoiseSample:
    components: dict[str, Tensor]
    source_paths: dict[str, Path]
    composite: Tensor
    snr_db: int


class CompositeMusanNoiseMixer:
    """Create paired environmental, low-band, and high-band MUSAN noise."""

    component_names = ("environmental", "low_band", "high_band")

    def __init__(
        self,
        noise_files: Iterable[Path],
        *,
        seed: int,
        snr_db: tuple[int, ...],
        low_noise_band_hz: tuple[float, float],
        high_noise_band_hz: tuple[float, float],
        filter_order: int,
    ) -> None:
        self.noise_files = sorted(Path(path) for path in noise_files)
        if not self.noise_files:
            raise ValueError("At least one MUSAN audio file is required")
        if not snr_db:
            raise ValueError("snr_db must not be empty")
        self.seed = seed
        self.snr_db = snr_db
        self.low_noise_band_hz = low_noise_band_hz
        self.high_noise_band_hz = high_noise_band_hz
        self.filter_order = filter_order

    @classmethod
    def from_root(
        cls,
        musan_root: Path,
        *,
        seed: int,
        snr_db: tuple[int, ...],
        low_noise_band_hz: tuple[float, float],
        high_noise_band_hz: tuple[float, float],
        filter_order: int,
    ) -> CompositeMusanNoiseMixer:
        audio_extensions = {".wav", ".flac", ".mp3"}
        noise_files = [
            path
            for path in musan_root.rglob("*")
            if path.suffix.lower() in audio_extensions
        ]
        return cls(
            noise_files,
            seed=seed,
            snr_db=snr_db,
            low_noise_band_hz=low_noise_band_hz,
            high_noise_band_hz=high_noise_band_hz,
            filter_order=filter_order,
        )

    def __call__(
        self, waveform: Tensor, sample_rate: int, record: ManifestRecord
    ) -> Tensor:
        noise_sample = self.build_noise(
            record, sample_rate=sample_rate, target_length=waveform.numel()
        )
        return mix_at_snr(waveform, noise_sample.composite, noise_sample.snr_db)

    def build_noise(
        self,
        record: ManifestRecord,
        *,
        sample_rate: int,
        target_length: int,
        snr_db: int | None = None,
    ) -> CompositeNoiseSample:
        source_paths = self._select_component_paths(record.sample_id)
        components = {
            component_name: _normalize_rms(
                self._load_component(
                    component_name,
                    source_path,
                    sample_rate=sample_rate,
                    target_length=target_length,
                )
            )
            for component_name, source_path in source_paths.items()
        }
        composite = torch.stack(list(components.values())).sum(dim=0)
        return CompositeNoiseSample(
            components=components,
            source_paths=source_paths,
            composite=composite,
            snr_db=self._select_snr(record.sample_id) if snr_db is None else snr_db,
        )

    def _load_component(
        self,
        component_name: str,
        source_path: Path,
        *,
        sample_rate: int,
        target_length: int,
    ) -> Tensor:
        waveform, source_sample_rate = torchaudio.load(source_path)
        waveform = waveform.mean(dim=0)
        if source_sample_rate != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, source_sample_rate, sample_rate
            )
        waveform = _match_length(waveform, target_length)
        if component_name == "low_band":
            return apply_sos_filter(
                waveform,
                sample_rate=sample_rate,
                cutoff_hz=self.low_noise_band_hz,
                filter_type="bandpass",
                order=self.filter_order,
            )
        if component_name == "high_band":
            return apply_sos_filter(
                waveform,
                sample_rate=sample_rate,
                cutoff_hz=self.high_noise_band_hz,
                filter_type="bandpass",
                order=self.filter_order,
            )
        return waveform

    def _select_component_paths(self, sample_id: str) -> dict[str, Path]:
        selected_paths: dict[str, Path] = {}
        used_paths: set[Path] = set()
        for component_name in self.component_names:
            start_index = _deterministic_index(
                sample_id, self.seed, len(self.noise_files), component_name
            )
            source_path = self._next_available_path(start_index, used_paths)
            selected_paths[component_name] = source_path
            used_paths.add(source_path)
        return selected_paths

    def _next_available_path(self, start_index: int, used_paths: set[Path]) -> Path:
        if len(self.noise_files) < len(self.component_names):
            return self.noise_files[start_index]
        for offset in range(len(self.noise_files)):
            source_path = self.noise_files[
                (start_index + offset) % len(self.noise_files)
            ]
            if source_path not in used_paths:
                return source_path
        raise RuntimeError("Unable to select a distinct MUSAN source path")

    def _select_snr(self, sample_id: str) -> int:
        return self.snr_db[
            _deterministic_index(sample_id, self.seed, len(self.snr_db), "snr")
        ]


def _match_length(waveform: Tensor, target_length: int) -> Tensor:
    if waveform.numel() >= target_length:
        return waveform[:target_length]
    repeat_count = (target_length + waveform.numel() - 1) // waveform.numel()
    return waveform.repeat(repeat_count)[:target_length]


def _normalize_rms(waveform: Tensor, *, epsilon: float = 1e-8) -> Tensor:
    return waveform / waveform.square().mean().sqrt().clamp_min(epsilon)


def _deterministic_index(sample_id: str, seed: int, size: int, value_type: str) -> int:
    digest = sha256(f"{seed}:{value_type}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % size
