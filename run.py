"""DSP501 study CLI.

    uv run python run.py all --config configs/dsp501-v2.json

Stages: prepare -> analyse-bands -> tune -> train -> evaluate -> analyze.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from src.config import ExperimentConfig, load_config

STAGES = ("prepare", "analyse-dsp", "tune", "train", "evaluate", "analyze")


def stage_prepare(config: ExperimentConfig, args: argparse.Namespace) -> None:
    from src.corpus import build_corpus_cache
    from src.noise import build_noise_pool

    seed = config.seeds[0]
    manifests = build_corpus_cache(
        Path(config.vctk_root),
        config.cache_root,
        seed=seed,
        sample_rate=config.sample_rate,
        segment_seconds=config.segment_seconds,
        clips_per_speaker=config.clips_per_speaker,
        eval_clips_per_speaker=config.eval_clips_per_speaker,
        train_speakers=config.train_speakers,
        validation_speakers=config.validation_speakers,
        closed_set_clips=config.closed_set_clips,
    )
    pool_path = build_noise_pool(
        Path(config.musan_root),
        config.cache_root,
        seed=seed,
        sample_rate=config.sample_rate,
        segment_seconds=config.segment_seconds,
    )
    for split, path in manifests.items():
        print(f"  manifest {split}: {path}")
    print(f"  noise pool: {pool_path}")


def stage_analyse_dsp(config: ExperimentConfig, args: argparse.Namespace) -> None:
    from src.analysis import analyse_bands, analyse_dsp_effect

    bands = analyse_bands(config)
    print(json.dumps(bands["recommended_cutoffs"], indent=2))

    effect = analyse_dsp_effect(config)
    hardest = min(config.test_snr_db)
    print(f"  front-end characterisation at {hardest:.0f} dB input:")
    for row in effect["measurements"]:
        if row["input_snr_db"] == hardest:
            print(
                f"    {row['condition']:<18} SNR gain {row['snr_gain_db']:+.2f} dB, "
                f"clean-path distortion {row['clean_path_snr_db']:.2f} dB"
            )


def stage_tune(config: ExperimentConfig, args: argparse.Namespace) -> None:
    from src.study import tune_hyperparameters

    print(json.dumps(tune_hyperparameters(config, workers=args.workers), indent=2))


def stage_train(config: ExperimentConfig, args: argparse.Namespace) -> None:
    from src.study import train_study

    for result in train_study(config, workers=args.workers, only=args.condition):
        print(
            f"  {result.condition:<18} seed {result.seed}  "
            f"val-acc {result.best_validation_accuracy:.4f} @ epoch {result.best_epoch}"
        )


def stage_evaluate(config: ExperimentConfig, args: argparse.Namespace) -> None:
    from src.study import evaluate_study

    evaluate_study(config, workers=args.workers, only=args.condition)


def stage_analyze(config: ExperimentConfig, args: argparse.Namespace) -> None:
    from src.study import analyze_study

    analyze_study(config)


DISPATCH = {
    "prepare": stage_prepare,
    "analyse-dsp": stage_analyse_dsp,
    "tune": stage_tune,
    "train": stage_train,
    "evaluate": stage_evaluate,
    "analyze": stage_analyze,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "all"))
    parser.add_argument("--config", type=Path, default=Path("configs/dsp501-v2.json"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--workers", type=int, default=8, help="DSP front-end worker processes"
    )
    parser.add_argument(
        "--condition", default=None, help="restrict train/evaluate to one arm"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_root is not None:
        config = replace(config, output_root=str(args.output_root))
    config.study_root.mkdir(parents=True, exist_ok=True)

    for stage in STAGES if args.stage == "all" else (args.stage,):
        print(f"[{stage}]")
        DISPATCH[stage](config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
