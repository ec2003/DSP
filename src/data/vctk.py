from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

WaveformTransform = Callable[[Tensor, int, "ManifestRecord"], Tensor]


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    speaker_id: str
    audio_path: str
    split: str
    duration_seconds: float
    crop_start_seconds: float


def build_manifests(
    vctk_root: Path,
    manifest_root: Path,
    *,
    seed: int,
    segment_seconds: float,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    clips_per_speaker: int = 50,
    dataset_root: Path | None = None,
) -> dict[str, Path]:
    """Create reproducible speaker-disjoint VCTK manifests."""
    if (
        not 0 < train_ratio < 1
        or not 0 < validation_ratio < 1
        or train_ratio + validation_ratio >= 1
    ):
        raise ValueError(
            "train_ratio and validation_ratio must be in (0, 1) and sum to less than 1"
        )
    if not vctk_root.is_dir():
        raise FileNotFoundError(f"VCTK wav48 directory not found: {vctk_root}")

    speakers = sorted(path for path in vctk_root.iterdir() if path.is_dir())
    if len(speakers) < 3:
        raise ValueError(
            "VCTK needs at least three speaker directories for train/validation/test splits"
        )

    shuffled_speakers = speakers[:]
    random.Random(seed).shuffle(shuffled_speakers)
    train_end = int(len(shuffled_speakers) * train_ratio)
    validation_end = train_end + int(len(shuffled_speakers) * validation_ratio)
    speaker_splits = {
        "train": shuffled_speakers[:train_end],
        "validation": shuffled_speakers[train_end:validation_end],
        "test": shuffled_speakers[validation_end:],
    }

    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_paths: dict[str, Path] = {}
    for split, split_speakers in speaker_splits.items():
        records = _records_for_speakers(
            split_speakers,
            split,
            seed,
            segment_seconds,
            clips_per_speaker=clips_per_speaker,
            dataset_root=dataset_root or Path.cwd(),
        )
        manifest_path = manifest_root / f"{split}.jsonl"
        manifest_path.write_text(
            "".join(
                f"{json.dumps(asdict(record), sort_keys=True)}\n" for record in records
            ),
            encoding="utf-8",
        )
        manifest_paths[split] = manifest_path
    return manifest_paths


def load_manifest(manifest_path: Path) -> list[ManifestRecord]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return [
        ManifestRecord(**json.loads(line))
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class VCTKWaveformDataset(Dataset[dict[str, str | Tensor]]):
    def __init__(
        self,
        records: Iterable[ManifestRecord],
        *,
        sample_rate: int,
        segment_seconds: float,
        waveform_transform: WaveformTransform | None = None,
        dataset_root: Path | None = None,
    ) -> None:
        self.records = list(records)
        self.sample_rate = sample_rate
        self.segment_samples = round(sample_rate * segment_seconds)
        self.waveform_transform = waveform_transform
        self.dataset_root = dataset_root or Path.cwd()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        record = self.records[index]
        audio_path = Path(record.audio_path)
        waveform, source_sample_rate = torchaudio.load(
            audio_path if audio_path.is_absolute() else self.dataset_root / audio_path
        )
        waveform = waveform.mean(dim=0)
        if source_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, source_sample_rate, self.sample_rate
            )

        start_sample = round(record.crop_start_seconds * self.sample_rate)
        waveform = waveform[start_sample : start_sample + self.segment_samples]
        waveform = torch.nn.functional.pad(
            waveform, (0, max(0, self.segment_samples - waveform.numel()))
        )
        if self.waveform_transform is not None:
            waveform = self.waveform_transform(waveform, self.sample_rate, record)

        return {
            "waveform": waveform,
            "speaker_id": record.speaker_id,
            "sample_id": record.sample_id,
            "audio_path": record.audio_path,
        }


def _records_for_speakers(
    speakers: Iterable[Path],
    split: str,
    seed: int,
    segment_seconds: float,
    *,
    clips_per_speaker: int,
    dataset_root: Path,
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for speaker_dir in speakers:
        # VCTK 0.92 supplies FLAC mic1 recordings; WAV remains supported for fixtures.
        source_files = sorted(
            path
            for path in speaker_dir.rglob("*")
            if path.suffix.lower() in {".flac", ".wav"}
            and (path.suffix.lower() != ".flac" or "mic1" in path.name)
        )
        if not source_files:
            continue
        for clip_index in range(clips_per_speaker):
            audio_path = source_files[
                _deterministic_file_index(
                    speaker_dir.name, clip_index, seed, len(source_files)
                )
            ]
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate <= 0:
                raise ValueError(f"Invalid sample rate for {audio_path}")
            duration_seconds = waveform.shape[-1] / sample_rate
            sample_id = f"{speaker_dir.name}/{audio_path.stem}/clip-{clip_index:03d}"
            crop_start_seconds = _deterministic_crop_start(
                sample_id,
                duration_seconds,
                segment_seconds,
                seed,
            )
            records.append(
                ManifestRecord(
                    sample_id=sample_id,
                    speaker_id=speaker_dir.name,
                    audio_path=str(
                        audio_path.relative_to(dataset_root)
                        if audio_path.is_relative_to(dataset_root)
                        else Path(
                            __import__("os").path.relpath(audio_path, dataset_root)
                        )
                    ),
                    split=split,
                    duration_seconds=duration_seconds,
                    crop_start_seconds=crop_start_seconds,
                )
            )
    return records


def _deterministic_file_index(
    speaker_id: str, clip_index: int, seed: int, size: int
) -> int:
    digest = sha256(f"{seed}:{speaker_id}:{clip_index}:file".encode()).digest()
    return int.from_bytes(digest[:8], "big") % size


def _deterministic_crop_start(
    sample_id: str, duration_seconds: float, segment_seconds: float, seed: int
) -> float:
    available_seconds = max(0.0, duration_seconds - segment_seconds)
    if available_seconds == 0:
        return 0.0
    digest = sha256(f"{seed}:{sample_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / (2**64 - 1)
    return available_seconds * fraction
