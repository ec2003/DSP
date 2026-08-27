"""MUSAN noise pools and SNR-controlled mixing.

Noise segments are cached once into fixed pools so that the noise a clip
receives is a deterministic function of its sample id, identical across every
condition. That keeps the DSP arms comparable: only the front-end differs.

The MUSAN recordings are partitioned into disjoint train / validation / test
groups first. Without that split the encoder can memorise the very noise
waveforms it is later tested on, which flatters the learned-robustness arm and
invalidates any comparison against a DSP front-end.
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

#: Noise partitions, in the order ``noise_split_fractions`` lists them.
NOISE_PARTITIONS = ("train", "validation", "test")

#: Corpus split -> noise partition. ``seen_test`` is an evaluation split (unseen
#: utterances of training speakers), so it draws from the test noise partition.
SPLIT_NOISE_PARTITION = {
    "train": "train",
    "validation": "validation",
    "test": "test",
    "seen_test": "test",
}


def discover_noise_files(musan_root: Path) -> list[Path]:
    """All MUSAN ``noise`` recordings, sorted for reproducibility."""
    noise_root = Path(musan_root) / "noise"
    if not noise_root.is_dir():
        raise FileNotFoundError(f"MUSAN noise directory not found: {noise_root}")
    files = sorted(noise_root.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files below {noise_root}")
    return files


def split_noise_files(
    files: list[Path], seed: int, fractions: tuple[float, ...]
) -> dict[str, list[Path]]:
    """Partition recordings so no source file appears in two partitions.

    The split is stratified by MUSAN sub-corpus (``free-sound`` /
    ``sound-bible``) so every partition keeps the same mix of provenance and the
    disjointness is about recordings, not about recording style.
    """
    if len(fractions) != len(NOISE_PARTITIONS):
        raise ValueError(
            f"need {len(NOISE_PARTITIONS)} fractions, got {len(fractions)}"
        )
    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(path.parent.name, []).append(path)

    rng = np.random.default_rng(seed)
    partitions: dict[str, list[Path]] = {name: [] for name in NOISE_PARTITIONS}
    for _, group in sorted(groups.items()):
        order = rng.permutation(len(group))
        shuffled = [group[index] for index in order]
        edges = np.cumsum([0.0, *fractions]) * len(shuffled)
        for position, name in enumerate(NOISE_PARTITIONS):
            start, stop = int(round(edges[position])), int(round(edges[position + 1]))
            partitions[name].extend(shuffled[start:stop])

    for name, selected in partitions.items():
        if not selected:
            raise ValueError(f"noise partition {name!r} is empty; widen its fraction")
        partitions[name] = sorted(selected)
    return partitions


def _segments_from_files(
    files: list[Path], *, seed: int, sample_rate: int, segment_samples: int, count: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    pool = np.zeros((count, segment_samples), dtype=np.float32)
    sources: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for index in range(count):
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
        pool[index] = waveform[start : start + segment_samples]
        sources.append({"index": index, "path": str(path), "start_sample": start})
    return pool, sources


def build_noise_pools(
    musan_root: Path,
    cache_root: Path,
    *,
    seed: int,
    sample_rate: int,
    segment_seconds: float,
    split_fractions: tuple[float, ...],
    pool_size: int = 2000,
) -> dict[str, Path]:
    """Cache one fixed-length segment pool per partition of disjoint recordings."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    segment_samples = round(sample_rate * segment_seconds)
    files = discover_noise_files(musan_root)
    partitions = split_noise_files(files, seed, split_fractions)

    pool_paths: dict[str, Path] = {}
    manifest: dict[str, object] = {}
    for offset, (name, partition_files) in enumerate(partitions.items()):
        count = max(1, round(pool_size * len(partition_files) / len(files)))
        pool, sources = _segments_from_files(
            partition_files,
            seed=seed + offset,
            sample_rate=sample_rate,
            segment_samples=segment_samples,
            count=count,
        )
        pool_path = cache_root / f"noise-pool-{name}.npy"
        np.save(pool_path, pool)
        pool_paths[name] = pool_path
        manifest[name] = {
            "n_files": len(partition_files),
            "n_segments": count,
            "files": [str(path) for path in partition_files],
            "segments": sources,
        }

    (cache_root / "noise-pool.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return pool_paths


def load_noise_pool(cache_root: Path, partition: str) -> np.ndarray:
    if partition not in NOISE_PARTITIONS:
        raise KeyError(f"unknown noise partition {partition!r}")
    pool_path = Path(cache_root) / f"noise-pool-{partition}.npy"
    if not pool_path.is_file():
        raise FileNotFoundError(
            f"Missing noise pool {pool_path}; run `run.py prepare` first."
        )
    return np.load(pool_path)


def load_noise_pool_for_split(cache_root: Path, split: str) -> np.ndarray:
    """The pool a corpus split is allowed to draw from."""
    if split not in SPLIT_NOISE_PARTITION:
        raise KeyError(f"no noise partition mapped for split {split!r}")
    return load_noise_pool(cache_root, SPLIT_NOISE_PARTITION[split])


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
