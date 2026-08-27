"""Recording-disjoint MUSAN pools and reproducible RMS-controlled mixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.config import MUSAN_FAMILIES

EPS = 1e-12
NOISE_PARTITIONS = ("train", "validation", "test")
SPLIT_NOISE_PARTITION = {"train": "train", "validation": "validation", "test": "test", "seen_test": "test"}


@dataclass(frozen=True)
class MixResult:
    mixture: np.ndarray
    clean_component: np.ndarray
    noise_component: np.ndarray
    metadata: dict[str, object]


def deterministic_choice(namespace: str, size: int) -> int:
    if size < 1:
        raise ValueError("cannot choose from an empty pool")
    return int.from_bytes(sha256(namespace.encode()).digest()[:8], "big") % size


def deterministic_uniform(namespace: str, low: float, high: float) -> float:
    if high < low:
        raise ValueError("uniform range must be increasing")
    value = int.from_bytes(sha256(namespace.encode()).digest()[:8], "big") / (2**64 - 1)
    return float(low + value * (high - low))


def rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(waveform, dtype=np.float64) ** 2)))


def measured_snr_db(clean_component: np.ndarray, noise_component: np.ndarray) -> float:
    return float(20 * np.log10((rms(clean_component) + EPS) / (rms(noise_component) + EPS)))


def discover_noise_files(musan_root: Path, family: str) -> list[Path]:
    if family not in {"noise", "music", "speech"}:
        raise ValueError(f"MUSAN source family must be noise, music, or speech; got {family!r}")
    root = Path(musan_root) / family
    files = sorted(root.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files below MUSAN {family!r} directory: {root}")
    return files


def split_noise_files(files: list[Path], seed: int, fractions: tuple[float, ...]) -> dict[str, list[Path]]:
    """Split source recordings before segmentation, stratified by parent corpus."""
    if len(fractions) != len(NOISE_PARTITIONS):
        raise ValueError("need train, validation, and test split fractions")
    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(path.parent.name, []).append(path)
    rng = np.random.default_rng(seed)
    out = {name: [] for name in NOISE_PARTITIONS}
    for group in groups.values():
        shuffled = [group[i] for i in rng.permutation(len(group))]
        boundaries = np.rint(np.cumsum((0.0, *fractions)) * len(shuffled)).astype(int)
        for i, partition in enumerate(NOISE_PARTITIONS):
            out[partition].extend(shuffled[boundaries[i] : boundaries[i + 1]])
    for partition, selected in out.items():
        if not selected:
            raise ValueError(f"MUSAN partition {partition!r} is empty")
        out[partition] = sorted(selected)
    return out


def _segment_pool(files: list[Path], *, seed: int, sample_rate: int, segment_samples: int, count: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    pool = np.zeros((count, segment_samples), dtype=np.float32)
    metadata: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for i in range(count):
        path = files[i % len(files)]
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        signal = audio.mean(axis=1)
        if rate != sample_rate:
            signal = torchaudio.functional.resample(torch.from_numpy(signal), rate, sample_rate).numpy().astype(np.float32)
        if signal.size < segment_samples:
            signal = np.tile(signal, int(np.ceil(segment_samples / max(1, signal.size))))
        offset = int(rng.integers(0, signal.size - segment_samples + 1))
        pool[i] = signal[offset : offset + segment_samples]
        metadata.append({"index": i, "source_recording": str(path), "offset_samples": offset})
    return pool, metadata


def build_noise_pools(musan_root: Path, cache_root: Path, *, seed: int, sample_rate: int, segment_seconds: float, split_fractions: tuple[float, ...], pool_size: int = 2000) -> dict[str, Path]:
    """Make a pool for each source family × recording-disjoint partition."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    n_samples = round(sample_rate * segment_seconds)
    manifest: dict[str, object] = {"families": {}, "split_policy": "recording-disjoint before segmentation"}
    paths: dict[str, Path] = {}
    for family_index, family in enumerate(("noise", "music", "speech")):
        files = discover_noise_files(musan_root, family)
        partitions = split_noise_files(files, seed + family_index, split_fractions)
        family_manifest: dict[str, object] = {}
        for partition_index, (partition, selected) in enumerate(partitions.items()):
            count = max(1, round(pool_size * len(selected) / len(files)))
            pool, segments = _segment_pool(selected, seed=seed + family_index * 31 + partition_index, sample_rate=sample_rate, segment_samples=n_samples, count=count)
            path = cache_root / f"musan-{family}-{partition}.npy"
            np.save(path, pool)
            paths[f"{family}:{partition}"] = path
            family_manifest[partition] = {"n_files": len(selected), "n_segments": count, "files": [str(p) for p in selected], "segments": segments}
        manifest["families"][family] = family_manifest
    (cache_root / "musan-pools.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return paths


def load_noise_pool(cache_root: Path, partition: str, family: str) -> np.ndarray:
    if partition not in NOISE_PARTITIONS or family not in ("noise", "music", "speech"):
        raise ValueError("invalid MUSAN family or partition")
    path = Path(cache_root) / f"musan-{family}-{partition}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing pool {path}; run `run.py prepare` first.")
    return np.load(path)


def load_noise_metadata(cache_root: Path, partition: str, family: str) -> list[dict[str, object]]:
    manifest = json.loads((Path(cache_root) / "musan-pools.json").read_text(encoding="utf-8"))
    return manifest["families"][family][partition]["segments"]


def load_noise_pool_for_split(cache_root: Path, split: str, family: str = "noise") -> np.ndarray:
    return load_noise_pool(cache_root, SPLIT_NOISE_PARTITION[split], family)


def load_noise_metadata_for_split(cache_root: Path, split: str, family: str = "noise") -> list[dict[str, object]]:
    """Use the same seen_test→test policy as waveform-pool loading."""
    return load_noise_metadata(cache_root, SPLIT_NOISE_PARTITION[split], family)


def _unit_rms(waveform: np.ndarray) -> np.ndarray:
    return np.asarray(waveform, dtype=np.float32) / max(rms(waveform), EPS)


def mix_at_snr_components(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RMS-normalise the selected noise, scale to SNR, then apply shared gain."""
    speech = np.asarray(speech, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    if speech.shape != noise.shape:
        raise ValueError(f"shape mismatch: speech {speech.shape} vs noise {noise.shape}")
    speech_rms = rms(speech)
    if speech_rms <= EPS or rms(noise) <= EPS:
        return speech.copy(), speech.copy(), np.zeros_like(speech)
    clean_component = speech.copy()
    noise_component = _unit_rms(noise) * (speech_rms / (10 ** (snr_db / 20)))
    peak = float(np.max(np.abs(clean_component + noise_component)))
    shared_gain = min(1.0, 0.99 / peak) if peak > EPS else 1.0
    clean_component *= shared_gain
    noise_component *= shared_gain
    return (clean_component + noise_component).astype(np.float32), clean_component.astype(np.float32), noise_component.astype(np.float32)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Compatibility helper returning only the controlled mixture."""
    return mix_at_snr_components(speech, noise, snr_db)[0]


def _selection(sample_id: str, seed: int, family: str, pool: np.ndarray, metadata: list[dict[str, object]], component: int = 0) -> tuple[np.ndarray, dict[str, object]]:
    index = deterministic_choice(f"{seed}:{sample_id}:{family}:{component}", len(pool))
    return pool[index], dict(metadata[index])


def make_mixture(speech: np.ndarray, sample_id: str, *, seed: int, family: str, snr_db: float, pools: dict[str, np.ndarray], pool_metadata: dict[str, list[dict[str, object]]], babble_sources_range: tuple[int, int] = (3, 7), namespace: str = "eval") -> MixResult:
    """Return one deterministic mixture and auditable source/SNR metadata."""
    if family not in MUSAN_FAMILIES:
        raise ValueError(f"unknown noise family {family!r}")
    key = f"{namespace}:{sample_id}"
    if family == "babble":
        speech_pool, speech_meta = pools["speech"], pool_metadata["speech"]
        count = deterministic_choice(f"{seed}:{key}:babble-count", babble_sources_range[1] - babble_sources_range[0] + 1) + babble_sources_range[0]
        components, sources = zip(*[_selection(key, seed, "babble", speech_pool, speech_meta, i) for i in range(count)], strict=True)
        noise = np.sum(np.stack(components), axis=0)
        source_metadata = list(sources)
    else:
        noise, source = _selection(key, seed, family, pools[family], pool_metadata[family])
        source_metadata = [source]
    mixture, clean_component, noise_component = mix_at_snr_components(speech, noise, snr_db)
    return MixResult(mixture, clean_component, noise_component, {
        "sample_id": sample_id,
        "family": family,
        "target_snr_db": float(snr_db),
        "measured_snr_db": measured_snr_db(clean_component, noise_component),
        "source_recordings": source_metadata,
        "seed": seed,
        "namespace": namespace,
    })


def training_augmentation_choice(sample_id: str, *, seed: int, epoch: int, p_noise: float, families: tuple[str, ...], snr_range_db: tuple[float, float]) -> tuple[bool, str | None, float | None]:
    key = f"{seed}:{epoch}:{sample_id}"
    noisy = deterministic_uniform(f"{key}:noisy", 0, 1) < p_noise
    if not noisy:
        return False, None, None
    family = families[deterministic_choice(f"{key}:family", len(families))]
    return True, family, deterministic_uniform(f"{key}:snr", *snr_range_db)


def gaussian_positive_control(clean: np.ndarray, snr_db: float, seed: int = 0) -> MixResult:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(np.asarray(clean).shape).astype(np.float32)
    mixture, clean_component, noise_component = mix_at_snr_components(clean, noise, snr_db)
    return MixResult(mixture, clean_component, noise_component, {"family": "synthetic_gaussian", "target_snr_db": float(snr_db), "measured_snr_db": measured_snr_db(clean_component, noise_component), "seed": seed})
