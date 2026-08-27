"""Offline invariants for the frozen-CNN 2×2 factorial study."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src import dsp
from src.analysis import config_hash
from src.config import Condition, FACTORIAL_CELLS, load_config
from src.corpus import ClipRecord
from src.noise import (
    NOISE_PARTITIONS, deterministic_uniform, gaussian_positive_control, make_mixture,
    measured_snr_db, mix_at_snr_components, split_noise_files, training_augmentation_choice,
)
from src.pipeline import build_chain, frozen_band_edges, process_split
from src.study import paired_cluster_bootstrap

RATE, N = 16000, 32000


def signal() -> np.ndarray:
    t = np.arange(N) / RATE
    return (.25 * np.sin(2 * np.pi * 300 * t) + .1 * np.sin(2 * np.pi * 1400 * t)).astype(np.float32)


def config_at(tmp_path: Path):
    return replace(load_config("configs/config.json"), output_root=str(tmp_path), run_id=1)


def freeze_design(config) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.dsp_design_path.write_text(json.dumps({"artifact": "frozen-dsp-band-design", "config_hash": config_hash(config), "selected_edges_hz": {"low": 80.0, "high": 6000.0}}))


def test_source_disjoint_musan_partitions():
    files = [Path(f"musan/speech/{source}/record-{i}.wav") for source in ("a", "b") for i in range(100)]
    split = split_noise_files(files, seed=7, fractions=(.6, .15, .25))
    assert set().union(*(set(split[x]) for x in NOISE_PARTITIONS)) == set(files)
    assert all(not set(split[a]) & set(split[b]) for a in NOISE_PARTITIONS for b in NOISE_PARTITIONS if a != b)
    assert split_noise_files(files, 7, (.6, .15, .25)) == split


def test_training_augmentation_is_deterministic_continuous_and_balanced_family():
    draws = [training_augmentation_choice(f"p1/u{i}", seed=11, epoch=3, p_noise=1, families=("noise", "music", "speech", "babble"), snr_range_db=(0, 20)) for i in range(200)]
    assert draws == [training_augmentation_choice(f"p1/u{i}", seed=11, epoch=3, p_noise=1, families=("noise", "music", "speech", "babble"), snr_range_db=(0, 20)) for i in range(200)]
    assert all(0 <= snr <= 20 for _, _, snr in draws)
    assert len({round(float(snr), 4) for _, _, snr in draws}) > 100
    counts = {family: sum(x[1] == family for x in draws) for family in ("noise", "music", "speech", "babble")}
    assert all(counts[family] > 25 for family in counts)
    assert deterministic_uniform("x", 0, 1) == deterministic_uniform("x", 0, 1)


def test_mixing_records_target_measured_snr_and_shared_gain():
    rng = np.random.default_rng(4)
    mixture, clean, noise = mix_at_snr_components(signal(), rng.normal(0, 1, N).astype(np.float32), 3.7)
    assert mixture.shape == clean.shape == noise.shape
    assert abs(measured_snr_db(clean, noise) - 3.7) < 1e-5
    assert np.max(np.abs(mixture)) <= .99001


def test_babble_is_sum_of_normalised_components_and_auditable():
    pools = {"noise": np.ones((10, N), np.float32), "music": np.ones((10, N), np.float32), "speech": np.stack([np.full(N, i + 1, np.float32) for i in range(10)])}
    metadata = {name: [{"index": i, "source_recording": f"{name}-{i}.wav", "offset_samples": 0} for i in range(10)] for name in pools}
    result = make_mixture(signal(), "p1/x", seed=1, family="babble", snr_db=5, pools=pools, pool_metadata=metadata, babble_sources_range=(3, 7))
    assert 3 <= len(result.metadata["source_recordings"]) <= 7
    assert abs(result.metadata["measured_snr_db"] - 5) < 1e-5
    assert np.isfinite(result.mixture).all()


def test_frozen_cutoff_artifact_is_required_and_hash_bound(tmp_path):
    config = config_at(tmp_path)
    with pytest.raises(FileNotFoundError):
        frozen_band_edges(config)
    freeze_design(config)
    assert frozen_band_edges(config) == (80.0, 6000.0)
    changed = replace(config, bandpass_order=2)
    with pytest.raises(ValueError):
        frozen_band_edges(changed)


def test_all_four_cells_receive_identical_pre_dsp_mixture(tmp_path):
    config = config_at(tmp_path); freeze_design(config)
    cache = config.cache_root; cache.mkdir(parents=True)
    record = ClipRecord("p1/u/clip", "p1", "x.wav", "test", 2.0, 0.0)
    (cache / "test.jsonl").write_text(json.dumps(record.__dict__) + "\n")
    np.save(cache / "test.npy", np.stack([signal()]))
    manifest = {"families": {family: {"test": {"segments": [{"index": i, "source_recording": f"{family}-{i}.wav", "offset_samples": 0} for i in range(4)]}} for family in ("noise", "music", "speech")}}
    (cache / "musan-pools.json").write_text(json.dumps(manifest))
    for family in ("noise", "music", "speech"):
        np.save(cache / f"musan-{family}-test.npy", np.random.default_rng(len(family)).normal(0, 1, (4, N)).astype(np.float32))
    mixtures = [process_split(np.stack([signal()]), [record], cell, config, seed=11, family="noise", snr_db=5).mixtures for cell in config.conditions]
    assert all(np.array_equal(mixtures[0], item) for item in mixtures[1:])


def test_clean_controls_are_available_for_each_cell(tmp_path):
    config = config_at(tmp_path); freeze_design(config)
    record = ClipRecord("p1/u/clip", "p1", "x.wav", "test", 2.0, 0.0)
    clips = np.stack([signal()])
    for cell in config.conditions:
        output = process_split(clips, [record], cell, config, seed=11, family=None, snr_db=None)
        assert output.mixing[0]["family"] == "clean"
        assert output.mixing[0]["measured_snr_db"] is None


def test_seen_test_uses_test_partition_for_pool_and_metadata(tmp_path):
    config = config_at(tmp_path); freeze_design(config)
    cache = config.cache_root; cache.mkdir(parents=True)
    record = ClipRecord("p1/seen/clip", "p1", "x.wav", "seen_test", 2.0, 0.0)
    manifest = {"families": {family: {"test": {"segments": [{"index": 0, "source_recording": f"{family}.wav", "offset_samples": 0}]}} for family in ("noise", "music", "speech")}}
    (cache / "musan-pools.json").write_text(json.dumps(manifest))
    for family in ("noise", "music", "speech"):
        np.save(cache / f"musan-{family}-test.npy", np.ones((1, N), np.float32))
    output = process_split(np.stack([signal()]), [record], config.condition("raw"), config, seed=11, family="noise", snr_db=5)
    assert output.mixing[0]["source_recordings"][0]["source_recording"] == "noise.wav"


def test_wiener_positive_control_has_low_snr_gain():
    clean = signal() * np.hanning(N)
    for snr in (0, 5):
        trial = gaussian_positive_control(clean, snr, seed=snr)
        before = 10 * np.log10(np.mean(trial.clean_component**2) / np.mean((trial.mixture - trial.clean_component) ** 2))
        after = 10 * np.log10(np.mean(trial.clean_component**2) / np.mean((dsp.stft_wiener(trial.mixture, RATE) - trial.clean_component) ** 2))
        assert after > before


def test_factorial_bootstrap_resamples_speakers_not_seed_snr_as_independent():
    rows = [
        {"speaker_id": speaker, "seed": seed, "snr": snr, "delta": delta}
        for speaker, delta in (("p1", .2), ("p2", -.1), ("p3", .1))
        for seed in (11, 22, 33) for snr in (0, 5)
    ]
    result = paired_cluster_bootstrap(rows, replicates=500, seed=9)
    assert result["n_speakers"] == 3
    assert result["n_paired_speaker_strata"] == 18
    assert result["resampling_unit"].startswith("test speaker")
    assert abs(result["mean_delta"] - (0.2 - .1 + .1) / 3) < 1e-9


def test_invalid_config_and_stage_are_rejected(tmp_path):
    payload = json.loads(Path("configs/config.json").read_text())
    payload["p_noise"] = 1.2
    path = tmp_path / "bad.json"; path.write_text(json.dumps(payload))
    with pytest.raises(ValueError): load_config(path)
    with pytest.raises(ValueError): Condition("bad", ("magic",))
    assert tuple(load_config("configs/config.json").condition(x).name for x in FACTORIAL_CELLS) == FACTORIAL_CELLS
