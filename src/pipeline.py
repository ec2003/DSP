"""Assembly of the per-condition front-end: noise mixing followed by DSP stages.

Pipeline A (``A_raw_noisy``) is noise with no DSP. Pipeline B (``B_full``) is
noise plus the full designed chain. The intermediate arms isolate the
contribution of each stage.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from src import dsp
from src.config import Condition, ExperimentConfig
from src.corpus import ClipRecord
from src.noise import mix_at_snr, noise_for_sample, snr_for_sample


def build_chain(condition: Condition, config: ExperimentConfig) -> dsp.DspChain:
    """Materialise the DSP stages named by ``condition``."""
    builders = {
        "preemph": lambda x: dsp.pre_emphasis(x, config.preemph_coefficient),
        "highpass": lambda x: dsp.highpass(x, config.sample_rate, config.highpass_hz),
        "lowpass": lambda x: dsp.lowpass(x, config.sample_rate, config.lowpass_hz),
        "telephone": lambda x: dsp.bandpass(
            x, config.sample_rate, *config.telephone_band_hz
        ),
        "notch": lambda x: dsp.adaptive_notch(
            x,
            config.sample_rate,
            max_peaks=config.notch_max_peaks,
            prominence_db=config.notch_prominence_db,
            quality=config.notch_quality,
        ),
        "wiener": lambda x: dsp.stft_wiener(
            x,
            config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            noise_quantile=config.wiener_noise_quantile,
            alpha=config.wiener_alpha,
            gain_floor_db=config.wiener_gain_floor_db,
        ),
        "specsub": lambda x: dsp.spectral_subtraction(
            x,
            config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            noise_quantile=config.wiener_noise_quantile,
            over_subtraction=config.specsub_over_subtraction,
            spectral_floor=config.specsub_spectral_floor,
        ),
    }
    return dsp.DspChain(
        names=condition.stages,
        stages=tuple(builders[name] for name in condition.stages),
    )


def apply_condition(
    clip: np.ndarray,
    sample_id: str,
    condition: Condition,
    config: ExperimentConfig,
    seed: int,
    noise_pool: np.ndarray | None,
    snr_db: float | None,
) -> np.ndarray:
    """Mix noise (if the arm requires it) then run the DSP chain."""
    signal = np.asarray(clip, dtype=np.float32)
    if condition.add_noise:
        if noise_pool is None:
            raise ValueError(f"condition {condition.name!r} needs a noise pool")
        level_db = (
            snr_db
            if snr_db is not None
            else snr_for_sample(sample_id, seed, config.train_snr_db)
        )
        signal = mix_at_snr(
            signal, noise_for_sample(sample_id, seed, noise_pool), level_db
        )
    return build_chain(condition, config)(signal)


def _process_one(
    item: tuple[np.ndarray, str],
    condition: Condition,
    config: ExperimentConfig,
    seed: int,
    noise_pool: np.ndarray | None,
    snr_db: float | None,
) -> np.ndarray:
    clip, sample_id = item
    return apply_condition(clip, sample_id, condition, config, seed, noise_pool, snr_db)


def process_split(
    clips: np.ndarray,
    records: list[ClipRecord],
    condition: Condition,
    config: ExperimentConfig,
    *,
    seed: int,
    noise_pool: np.ndarray | None,
    snr_db: float | None = None,
    workers: int = 8,
) -> np.ndarray:
    """Run the front-end over a whole split, in parallel across processes."""
    if len(clips) != len(records):
        raise ValueError(f"clip/record mismatch: {len(clips)} vs {len(records)}")

    worker = partial(
        _process_one,
        condition=condition,
        config=config,
        seed=seed,
        noise_pool=noise_pool,
        snr_db=snr_db,
    )
    items = [(clips[i], records[i].sample_id) for i in range(len(records))]

    if workers <= 1 or not condition.stages:
        processed = [worker(item) for item in items]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            processed = list(pool.map(worker, items, chunksize=32))
    return np.stack(processed).astype(np.float32)
