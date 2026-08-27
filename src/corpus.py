"""VCTK corpus handling: speaker-disjoint manifests and a 16 kHz clip cache.

The corpus ships untrimmed 48 kHz WAV files. Decoding, silence-trimming and
resampling them on every epoch of every condition would dominate runtime, so
``prepare`` materialises the deterministic crops once into one array per split.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

#: Frames quieter than this fraction of the clip peak RMS are treated as silence.
SILENCE_RMS_RATIO = 0.05


@dataclass(frozen=True)
class ClipRecord:
    sample_id: str
    speaker_id: str
    audio_path: str
    split: str
    duration_seconds: float
    crop_fraction: float


def _deterministic_index(namespace: str, size: int) -> int:
    digest = sha256(namespace.encode()).digest()
    return int.from_bytes(digest[:8], "big") % size


def _deterministic_fraction(namespace: str) -> float:
    digest = sha256(namespace.encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


def split_speakers(
    vctk_root: Path, seed: int, train_speakers: int, validation_speakers: int
) -> dict[str, list[Path]]:
    """Partition speaker directories into disjoint train/validation/test groups."""
    speaker_dirs = sorted(path for path in Path(vctk_root).iterdir() if path.is_dir())
    if len(speaker_dirs) < train_speakers + validation_speakers + 1:
        raise ValueError(
            f"{vctk_root} has {len(speaker_dirs)} speakers, need more than "
            f"{train_speakers + validation_speakers}"
        )
    order = np.random.default_rng(seed).permutation(len(speaker_dirs))
    shuffled = [speaker_dirs[index] for index in order]
    validation_end = train_speakers + validation_speakers
    return {
        "train": sorted(shuffled[:train_speakers], key=lambda path: path.name),
        "validation": sorted(
            shuffled[train_speakers:validation_end], key=lambda path: path.name
        ),
        "test": sorted(shuffled[validation_end:], key=lambda path: path.name),
    }


def trim_silence(waveform: np.ndarray, frame_length: int = 512) -> np.ndarray:
    """Drop leading/trailing low-energy frames; VCTK 0.80 is not pre-trimmed."""
    n_frames = waveform.size // frame_length
    if n_frames < 3:
        return waveform
    frames = waveform[: n_frames * frame_length].reshape(n_frames, frame_length)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    voiced = np.flatnonzero(rms > SILENCE_RMS_RATIO * rms.max())
    if voiced.size == 0:
        return waveform
    return waveform[voiced[0] * frame_length : (voiced[-1] + 1) * frame_length]


def _records_for_speaker(
    speaker_dir: Path,
    split: str,
    seed: int,
    clips_per_speaker: int,
    start_index: int = 0,
) -> list[ClipRecord]:
    source_files = sorted(speaker_dir.glob("*.wav"))
    if not source_files:
        return []
    if clips_per_speaker + start_index > len(source_files):
        raise ValueError(
            f"{speaker_dir.name} has {len(source_files)} utterances, cannot draw "
            f"{clips_per_speaker} disjoint clips from offset {start_index}"
        )

    # Deterministic rotation over distinct utterances, so a speaker is not
    # represented by repeated crops of the same sentence. ``start_index`` shifts
    # the window, which is how the held-out seen-speaker split stays
    # utterance-disjoint from the training clips.
    offset = _deterministic_index(
        f"{seed}:{speaker_dir.name}:offset", len(source_files)
    )
    records: list[ClipRecord] = []
    for clip_index in range(clips_per_speaker):
        candidate = source_files[
            (offset + start_index + clip_index) % len(source_files)
        ]
        # sample_id seeds the noise and SNR draw, so the base window keeps its
        # original form; only the shifted held-out window is marked.
        prefix = "clip" if start_index == 0 else "heldout-clip"
        sample_id = f"{speaker_dir.name}/{candidate.stem}/{prefix}-{clip_index:03d}"
        records.append(
            ClipRecord(
                sample_id=sample_id,
                speaker_id=speaker_dir.name,
                audio_path=str(candidate),
                split=split,
                duration_seconds=float(sf.info(str(candidate)).duration),
                crop_fraction=_deterministic_fraction(f"{seed}:{sample_id}:crop"),
            )
        )
    return records


def _load_clip(
    record: ClipRecord, sample_rate: int, segment_samples: int
) -> np.ndarray:
    audio, source_rate = sf.read(record.audio_path, dtype="float32", always_2d=True)
    waveform = audio.mean(axis=1)
    if source_rate != sample_rate:
        waveform = (
            torchaudio.functional.resample(
                torch.from_numpy(waveform), source_rate, sample_rate
            )
            .numpy()
            .astype(np.float32)
        )

    waveform = trim_silence(waveform)
    available = max(0, waveform.size - segment_samples)
    start = int(round(available * record.crop_fraction))
    clip = waveform[start : start + segment_samples]
    if clip.size < segment_samples:
        clip = np.pad(clip, (0, segment_samples - clip.size))

    peak = float(np.max(np.abs(clip)))
    if peak > 0:
        clip = clip / peak * 0.95
    return clip.astype(np.float32)


def build_corpus_cache(
    vctk_root: Path,
    cache_root: Path,
    *,
    seed: int,
    sample_rate: int,
    segment_seconds: float,
    clips_per_speaker: int,
    eval_clips_per_speaker: int,
    train_speakers: int,
    validation_speakers: int,
    closed_set_clips: int = 0,
) -> dict[str, Path]:
    """Write per-split manifests and cached 16 kHz clips; returns manifest paths."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    segment_samples = round(sample_rate * segment_seconds)
    speaker_splits = split_speakers(
        vctk_root, seed, train_speakers, validation_speakers
    )
    # Only training benefits from a larger clip budget; evaluation splits stay
    # small so the metrics stay cheap and comparable across runs.
    plan = [
        ("train", speaker_splits["train"], clips_per_speaker, 0),
        ("validation", speaker_splits["validation"], eval_clips_per_speaker, 0),
        ("test", speaker_splits["test"], eval_clips_per_speaker, 0),
    ]
    if closed_set_clips:
        plan.append(
            ("seen_test", speaker_splits["train"], closed_set_clips, clips_per_speaker)
        )

    manifest_paths: dict[str, Path] = {}
    for split, speaker_dirs, n_clips, start_index in plan:
        records: list[ClipRecord] = []
        for speaker_dir in speaker_dirs:
            records.extend(
                _records_for_speaker(speaker_dir, split, seed, n_clips, start_index)
            )

        clips = np.zeros((len(records), segment_samples), dtype=np.float32)
        for index, record in enumerate(records):
            clips[index] = _load_clip(record, sample_rate, segment_samples)

        manifest_path = cache_root / f"{split}.jsonl"
        manifest_path.write_text(
            "".join(
                f"{json.dumps(asdict(record), sort_keys=True)}\n" for record in records
            ),
            encoding="utf-8",
        )
        np.save(cache_root / f"{split}.npy", clips)
        manifest_paths[split] = manifest_path
        print(f"  {split}: {len(speaker_dirs)} speakers, {len(records)} clips")
    return manifest_paths


def load_manifest(cache_root: Path, split: str) -> list[ClipRecord]:
    manifest_path = Path(cache_root) / f"{split}.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing manifest {manifest_path}; run `run.py prepare` first."
        )
    return [
        ClipRecord(**json.loads(line))
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_clips(cache_root: Path, split: str) -> np.ndarray:
    clip_path = Path(cache_root) / f"{split}.npy"
    if not clip_path.is_file():
        raise FileNotFoundError(
            f"Missing clip cache {clip_path}; run `run.py prepare` first."
        )
    return np.load(clip_path)
