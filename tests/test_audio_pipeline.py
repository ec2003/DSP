from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from src.audio import (
    CompositeMusanNoiseMixer,
    DspTransformChain,
    HighPassFilter,
    LowPassFilter,
    WienerDenoiser,
    mix_at_snr,
)
from src.config.settings import ExperimentConfig
from src.data import ManifestRecord, VCTKWaveformDataset, build_manifests, load_manifest
from src.experiments.train import build_waveform_transform


def test_manifests_are_reproducible_and_speaker_disjoint(tmp_path: Path) -> None:
    vctk_root = tmp_path / "wav48"
    for speaker_index in range(10):
        speaker_dir = vctk_root / f"p{speaker_index:03d}"
        speaker_dir.mkdir(parents=True)
        torchaudio.save(speaker_dir / "utterance.wav", torch.ones(1, 800), 16_000)

    first_paths = build_manifests(
        vctk_root, tmp_path / "manifests-one", seed=7, segment_seconds=1.0
    )
    second_paths = build_manifests(
        vctk_root, tmp_path / "manifests-two", seed=7, segment_seconds=1.0
    )
    first_records = {split: load_manifest(path) for split, path in first_paths.items()}

    assert (tmp_path / "manifests-one" / "train.jsonl").read_bytes() == (
        tmp_path / "manifests-two" / "train.jsonl"
    ).read_bytes()
    split_speakers = [
        {record.speaker_id for record in records} for records in first_records.values()
    ]
    assert not split_speakers[0] & split_speakers[1]
    assert not split_speakers[0] & split_speakers[2]
    assert not split_speakers[1] & split_speakers[2]

    dataset = VCTKWaveformDataset(
        first_records["train"],
        sample_rate=16_000,
        segment_seconds=1.0,
    )
    assert dataset[0]["waveform"].shape == (16_000,)


def test_noise_and_wiener_preserve_waveform_contract() -> None:
    record = ManifestRecord("p001/utterance", "p001", "unused.wav", "train", 1.0, 0.0)
    speech = torch.full((160,), 0.1)
    noise = torch.linspace(-0.2, 0.2, 40)

    mixed = mix_at_snr(speech, noise, 10)
    denoised = WienerDenoiser(29)(mixed, 16_000, record)

    assert mixed.shape == speech.shape == denoised.shape
    assert torch.isfinite(denoised).all()
    assert mixed.abs().max() <= 1


def test_composite_noise_is_deterministic_and_uses_total_snr(tmp_path: Path) -> None:
    sample_rate = 16_000
    noise_root = tmp_path / "noise"
    noise_root.mkdir()
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    for index, frequency in enumerate((100.0, 1_000.0, 4_000.0)):
        torchaudio.save(
            noise_root / f"noise-{index}.wav",
            torch.sin(2 * torch.pi * frequency * time).unsqueeze(0),
            sample_rate,
        )

    mixer = CompositeMusanNoiseMixer.from_root(
        noise_root,
        seed=7,
        snr_db=(10,),
        low_noise_band_hz=(20.0, 300.0),
        high_noise_band_hz=(3_000.0, 7_500.0),
        filter_order=4,
    )
    record = ManifestRecord("p001/utterance", "p001", "unused.wav", "train", 1.0, 0.0)
    speech = 0.1 * torch.sin(2 * torch.pi * 500 * time)

    first_noise = mixer.build_noise(
        record, sample_rate=sample_rate, target_length=sample_rate
    )
    second_noise = mixer.build_noise(
        record, sample_rate=sample_rate, target_length=sample_rate
    )
    mixed = mixer(speech, sample_rate, record)
    measured_snr = 10 * torch.log10(
        speech.square().mean() / (mixed - speech).square().mean()
    )

    assert len(set(first_noise.source_paths.values())) == 3
    assert first_noise.source_paths == second_noise.source_paths
    assert torch.equal(first_noise.composite, second_noise.composite)
    assert abs(measured_snr.item() - first_noise.snr_db) < 0.1


def test_dsp_condition_applies_filters_before_wiener(tmp_path: Path) -> None:
    sample_rate = 16_000
    noise_root = tmp_path / "musan"
    noise_root.mkdir()
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    for index, frequency in enumerate((100.0, 1_000.0, 4_000.0)):
        torchaudio.save(
            noise_root / f"noise-{index}.wav",
            torch.sin(2 * torch.pi * frequency * time).unsqueeze(0),
            sample_rate,
        )

    record = ManifestRecord("p001/utterance", "p001", "unused.wav", "train", 1.0, 0.0)
    speech = 0.1 * torch.sin(2 * torch.pi * 500 * time)
    noisy_config = ExperimentConfig(
        condition="noisy", musan_root=noise_root, snr_db=(10,)
    )
    dsp_config = ExperimentConfig(
        condition="noisy_wiener", musan_root=noise_root, snr_db=(10,)
    )
    noisy_transform = build_waveform_transform(noisy_config)
    dsp_transform = build_waveform_transform(dsp_config)
    assert noisy_transform is not None and dsp_transform is not None

    raw_composite = noisy_transform(speech, sample_rate, record)
    expected_dsp = DspTransformChain(
        HighPassFilter(dsp_config.high_pass_hz, dsp_config.filter_order),
        LowPassFilter(dsp_config.low_pass_hz, dsp_config.filter_order),
        WienerDenoiser(dsp_config.wiener_window_size),
    )(raw_composite, sample_rate, record)

    assert torch.equal(dsp_transform(speech, sample_rate, record), expected_dsp)
    assert expected_dsp.shape == speech.shape
    assert torch.isfinite(expected_dsp).all()
