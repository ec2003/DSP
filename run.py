#!/usr/bin/env python3
"""Canonical CLI for the DSP501 reproducible speaker-verification study."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.settings import PROJECT_ROOT, load_config
from src.data import download_datasets
from src.experiments.study import (
    analyze_study,
    evaluate_study,
    package_study,
    prepare_study,
    train_study,
    tune_raw_noisy,
    write_run_metadata,
)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "dsp501-v1.json"
    )
    cli.add_argument(
        "--output-root",
        type=Path,
        help="Override the project-relative output directory",
    )
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("download-data").add_argument(
        "--accept-data-licenses", action="store_true"
    )
    for name in ("prepare", "tune", "train", "evaluate", "analyze"):
        commands.add_parser(name)
    commands.add_parser("package").add_argument("--dry-run", action="store_true")
    commands.add_parser("all").add_argument("--skip-package", action="store_true")
    return cli


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config, output_root=args.output_root)
    if args.command == "download-data":
        download_datasets(
            config.vctk_root.parent.parent,
            accept_data_licenses=args.accept_data_licenses,
        )
    elif args.command == "prepare":
        prepare_study(config)
    elif args.command == "tune":
        tune_raw_noisy(config)
    elif args.command == "train":
        train_study(config)
    elif args.command == "evaluate":
        evaluate_study(config)
    elif args.command == "analyze":
        analyze_study(config)
    elif args.command == "package":
        package_study(config, dry_run=args.dry_run)
    else:
        prepare_study(config)
        tune_raw_noisy(config)
        train_study(config)
        evaluate_study(config)
        analyze_study(config)
        if not args.skip_package:
            package_study(config)
    write_run_metadata(config, args.command)


if __name__ == "__main__":
    main()
