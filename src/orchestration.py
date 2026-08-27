"""Notebook-first, resumable phase execution for immutable experiment runs."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from src.config import ExperimentConfig, load_config
from src.corpus import split_speakers
from src.noise import discover_noise_files, split_noise_files

PHASES = ("prepare", "eda", "train", "evaluate_raw", "evaluate_dsp", "signal_analysis", "statistics", "figures")
PHASE_DEPENDENCIES = {
    "prepare": (), "eda": ("prepare",), "train": ("prepare",),
    "evaluate_raw": ("train",), "evaluate_dsp": ("eda", "evaluate_raw"),
    "signal_analysis": ("eda",), "statistics": ("evaluate_raw", "evaluate_dsp"),
    "figures": ("statistics", "signal_analysis"),
}


class PreflightError(RuntimeError):
    """The local data tree cannot support the configured experimental split."""


class ConfigMismatchError(RuntimeError):
    """A working config differs from the immutable config copied into a run."""


@dataclass(frozen=True)
class RunContext:
    config: ExperimentConfig
    manifest: dict[str, Any]
    project_root: Path
    working_config_path: Path


@dataclass(frozen=True)
class PhaseResult:
    phase: str
    status: str
    data: dict[str, Any]
    manifest: dict[str, Any]


def find_project_root(start: Path | str | None = None) -> Path:
    here = (Path.cwd() if start is None else Path(start)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "run.py").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find project root above {here}.")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(path: Path) -> str:
    """Hash exact config bytes: each run is bound to its copied snapshot."""
    return sha256(path.read_bytes()).hexdigest()


def _git_revision(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def next_run_id(config: ExperimentConfig) -> int:
    runs = config.study_root / "runs"
    values = [int(path.name[4:]) for path in runs.glob("run-*") if path.is_dir() and path.name[4:].isdigit()]
    return max(values, default=0) + 1


def bind_new_run(config: ExperimentConfig) -> ExperimentConfig:
    return replace(config, run_id=next_run_id(config))


def bind_existing_run(config: ExperimentConfig, run_id: int) -> ExperimentConfig:
    if run_id < 1:
        raise ValueError("run ID must be a positive integer")
    bound = replace(config, run_id=run_id)
    if not bound.manifest_path.is_file() or not bound.config_snapshot_path.is_file():
        raise FileNotFoundError(f"{bound.run_tag} has no config snapshot and manifest.")
    return bound


def load_manifest(config: ExperimentConfig) -> dict[str, Any]:
    return json.loads(config.manifest_path.read_text(encoding="utf-8"))


def write_manifest(config: ExperimentConfig, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _timestamp()
    config.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _append_log(config: ExperimentConfig, message: str) -> None:
    with (config.run_dir / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{_timestamp()} {message}\n")


def create_run(working_config_path: Path | str, *, project_root: Path | str | None = None) -> RunContext:
    """Allocate `run-N`, copy its config verbatim, and create a durable manifest."""
    source, root = Path(working_config_path).resolve(), find_project_root(project_root)
    config = bind_new_run(load_config(source))
    config.run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, config.config_snapshot_path)
    manifest: dict[str, Any] = {
        "study_id": config.study_id, "run_id": config.run_id, "run_tag": config.run_tag,
        "created_at": _timestamp(), "updated_at": _timestamp(), "git_revision": _git_revision(root),
        "config_hash": config.config_hash, "source_config_hash": _json_hash(source), "data_cache_hash": config.data_cache_hash,
        "status": "running", "error": None,
        "phases": {name: {"status": "pending", "attempts": 0, "started_at": None, "completed_at": None, "error": None, "outputs": []} for name in PHASES},
        "checkpoints": [],
    }
    write_manifest(config, manifest)
    return RunContext(config, manifest, root, source)


def open_run(working_config_path: Path | str, run_id: int | None = None, *, project_root: Path | str | None = None) -> RunContext:
    """Allocate a fresh run, or safely open an unfinished matching run snapshot."""
    if run_id is None:
        return create_run(working_config_path, project_root=project_root)
    source, root = Path(working_config_path).resolve(), find_project_root(project_root)
    bound = bind_existing_run(load_config(source), run_id)
    if _json_hash(source) != _json_hash(bound.config_snapshot_path):
        raise ConfigMismatchError(f"{bound.run_tag} uses a different config snapshot; create a new run ID.")
    config = replace(load_config(bound.config_snapshot_path), run_id=run_id)
    return RunContext(config, load_manifest(config), root, source)


def _wav_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.wav") if path.is_file())


def preflight_dataset(config: ExperimentConfig) -> dict[str, Any]:
    vctk = Path(config.vctk_root)
    if not vctk.is_dir():
        raise PreflightError(f"VCTK wav48 directory is missing: {vctk}")
    speakers = sorted(path for path in vctk.iterdir() if path.is_dir())
    if len(speakers) < config.train_speakers + config.validation_speakers + 1:
        raise PreflightError(f"VCTK has {len(speakers)} speaker directories; need at least {config.train_speakers + config.validation_speakers + 1}.")
    no_wav = [path.name for path in speakers if not _wav_files(path)]
    if no_wav:
        raise PreflightError(f"VCTK speakers without WAV recordings: {', '.join(no_wav[:8])}")
    speaker_splits = split_speakers(vctk, config.seeds[0], config.train_speakers, config.validation_speakers)
    short = []
    for split, selected in speaker_splits.items():
        minimum = config.clips_per_speaker if split == "train" else config.eval_clips_per_speaker
        if split == "train" and config.closed_set_clips:
            minimum += config.closed_set_clips
        short.extend(f"{p.name} ({len(_wav_files(p))} < {minimum})" for p in selected if len(_wav_files(p)) < minimum)
    if short:
        raise PreflightError("VCTK lacks configured clips: " + ", ".join(short[:8]))
    families = {}
    for index, family in enumerate(("noise", "music", "speech")):
        try:
            files = discover_noise_files(Path(config.musan_root), family)
            parts = split_noise_files(files, config.seeds[0] + index, config.noise_split_fractions)
        except (FileNotFoundError, ValueError) as error:
            raise PreflightError(f"MUSAN {family!r} cannot make recording-disjoint pools: {error}") from error
        families[family] = {"recordings": len(files), **{name: len(items) for name, items in parts.items()}}
    estimated = (config.clips_per_speaker * config.train_speakers + config.eval_clips_per_speaker * (len(speakers) - config.train_speakers) + config.noise_pool_size) * config.segment_samples * 4
    return {"vctk_wav48": str(vctk), "vctk_speakers": len(speakers), "vctk_speaker_ids": [p.name for p in speakers], "musan_root": str(config.musan_root), "musan": families, "cache_root": str(config.cache_root), "estimated_cache_bytes": estimated, "estimated_cache_gib": round(estimated / 1024**3, 2)}


def _phase_prepare(config: ExperimentConfig, workers: int) -> dict[str, Any]:
    from src.corpus import build_corpus_cache
    from src.noise import build_noise_pools
    inventory = preflight_dataset(config)
    seed = config.seeds[0]
    build_corpus_cache(Path(config.vctk_root), config.cache_root, seed=seed, sample_rate=config.sample_rate, segment_seconds=config.segment_seconds, clips_per_speaker=config.clips_per_speaker, eval_clips_per_speaker=config.eval_clips_per_speaker, train_speakers=config.train_speakers, validation_speakers=config.validation_speakers, closed_set_clips=config.closed_set_clips)
    pools = build_noise_pools(Path(config.musan_root), config.cache_root, seed=seed, sample_rate=config.sample_rate, segment_seconds=config.segment_seconds, split_fractions=config.noise_split_fractions, pool_size=config.noise_pool_size)
    return {"inventory": inventory, "cache_root": str(config.cache_root), "noise_pools": [str(p) for p in pools.values()]}


def _phase_eda(config: ExperimentConfig, workers: int) -> dict[str, Any]:
    from src.analysis import derive_band_design
    from src.plots import plot_dsp_design
    design = derive_band_design(config)
    return {"design": design, "design_path": str(config.dsp_design_path), "figure": plot_dsp_design(config, design)}


def _phase_train(config: ExperimentConfig, workers: int) -> dict[str, Any]:
    from src.study import train_study
    from src.plots import plot_training_histories
    results = train_study(config, workers=workers, missing_only=True)
    return {"training": [result.__dict__ for result in results], "checkpoints": checkpoint_paths(config), "figures": plot_training_histories(config)}


def _phase_evaluate_raw(config: ExperimentConfig, workers: int) -> dict[str, Any]:
    from src.study import evaluate_study, evaluation_summary
    reports = evaluate_study(config, workers=workers, cells=("raw",))
    return {"reports": reports, "summary": evaluation_summary(config, ("raw",), "raw")}


def _phase_evaluate_dsp(config: ExperimentConfig, workers: int) -> dict[str, Any]:
    from src.study import evaluate_study, evaluation_summary
    cells = ("bandpass", "wiener", "bandpass_wiener")
    reports = evaluate_study(config, workers=workers, cells=cells)
    return {"reports": reports, "summary": evaluation_summary(config, ("raw", *cells), "dsp-matched")}


def _phase_signal_analysis(config: ExperimentConfig, workers: int) -> dict[str, Any]:
    from src.analysis import characterize_frontends, positive_control_wiener
    control = positive_control_wiener(config)
    if any(row["snr_gain_db"] <= 0 for row in control["measurements"]):
        raise RuntimeError("Wiener positive control did not produce positive SNR gain")
    return {"positive_control": control, "characterisation": characterize_frontends(config)}


def _phase_statistics(config: ExperimentConfig, workers: int) -> dict[str, Any]:
    from src.study import statistical_analysis
    return statistical_analysis(config)


def _phase_figures(config: ExperimentConfig, workers: int) -> dict[str, Any]:
    from src.study import render_figures
    return render_figures(config)


PHASE_HANDLERS: dict[str, Callable[[ExperimentConfig, int], dict[str, Any]]] = {
    "prepare": _phase_prepare, "eda": _phase_eda, "train": _phase_train,
    "evaluate_raw": _phase_evaluate_raw, "evaluate_dsp": _phase_evaluate_dsp,
    "signal_analysis": _phase_signal_analysis, "statistics": _phase_statistics, "figures": _phase_figures,
}


def _outputs_since(config: ExperimentConfig, before: set[Path]) -> list[str]:
    return sorted(str(path.relative_to(config.run_dir)) for path in config.run_dir.rglob("*") if path.is_file() and path not in before and path.name != "run-manifest.json")


def checkpoint_paths(config: ExperimentConfig) -> list[str]:
    paths = [config.seed_dir(seed) / "encoder.pt" for seed in config.seeds]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Expected robust_cnn checkpoints are missing: " + ", ".join(missing))
    return [str(path) for path in paths]


def execute_phase(context: RunContext, phase: str, *, workers: int = 0) -> PhaseResult:
    """Execute exactly one phase; completed dependencies are never rerun."""
    if phase not in PHASE_HANDLERS:
        raise ValueError(f"unknown phase {phase!r}")
    config, manifest = context.config, load_manifest(context.config)
    if manifest.get("status") == "completed":
        raise RuntimeError(f"{config.run_tag} is completed and immutable; allocate a new run.")
    unmet = [item for item in PHASE_DEPENDENCIES[phase] if manifest["phases"][item]["status"] != "completed"]
    if unmet:
        raise RuntimeError(f"{phase} requires completed phase(s): {', '.join(unmet)}")
    state = manifest["phases"][phase]
    if state["status"] == "completed":
        return PhaseResult(phase, "completed", {"skipped": True, "outputs": state["outputs"]}, manifest)
    state.update({"status": "running", "attempts": int(state["attempts"]) + 1, "started_at": _timestamp(), "error": None})
    manifest.update({"status": "running", "error": None})
    _append_log(config, f"{phase}: running (attempt {state['attempts']})")
    write_manifest(config, manifest)
    before = set(config.run_dir.rglob("*"))
    try:
        data = PHASE_HANDLERS[phase](config, workers)
        if phase == "train":
            manifest["checkpoints"] = checkpoint_paths(config)
        state.update({"status": "completed", "completed_at": _timestamp(), "outputs": _outputs_since(config, before)})
        _append_log(config, f"{phase}: completed")
        if all(manifest["phases"][item]["status"] == "completed" for item in PHASES):
            manifest["status"] = "completed"
        write_manifest(config, manifest)
        return PhaseResult(phase, "completed", data, manifest)
    except Exception as error:
        state.update({"status": "failed", "completed_at": _timestamp(), "error": str(error)})
        manifest.update({"status": "failed", "error": {"phase": phase, "message": str(error)}})
        _append_log(config, f"{phase}: failed — {error}")
        write_manifest(config, manifest)
        raise


run_phase = execute_phase
allocate_or_open_run = open_run
