"""Run identity, phase dependency, and notebook wiring regressions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.orchestration as orchestration
from src.config import load_config
from src.orchestration import (
    ConfigMismatchError,
    PHASE_HANDLERS,
    PreflightError,
    create_run,
    execute_phase,
    next_run_id,
    open_run,
    preflight_dataset,
)


def tiny_config_path(tmp_path: Path) -> Path:
    payload = json.loads(Path("configs/config.json").read_text())
    payload.update(
        {
            "output_root": str(tmp_path / "outputs"),
            "vctk_root": str(tmp_path / "wav48"),
            "musan_root": str(tmp_path / "musan"),
            "train_speakers": 1,
            "validation_speakers": 1,
            "clips_per_speaker": 1,
            "eval_clips_per_speaker": 2,
            "enrollment_clips": 1,
            "closed_set_clips": 0,
            "noise_pool_size": 12,
        }
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def dataset_layout(config) -> None:
    for speaker in ("p001", "p002", "p003"):
        for index in range(2):
            path = Path(config.vctk_root) / speaker / f"recording-{index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    for family in ("noise", "music", "speech"):
        for index in range(12):
            path = (
                Path(config.musan_root)
                / family
                / "source-set"
                / f"recording-{index}.wav"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()


def test_preflight_accepts_full_layout_and_rejects_missing_family(tmp_path):
    path = tiny_config_path(tmp_path)
    config = load_config(path)
    dataset_layout(config)
    assert preflight_dataset(config)["musan"]["speech"]["test"] > 0
    for item in Path(config.musan_root).joinpath("music").rglob("*.wav"):
        item.unlink()
    with pytest.raises(PreflightError, match="music"):
        preflight_dataset(config)


def test_run_allocation_snapshot_and_source_mismatch(tmp_path):
    path = tiny_config_path(tmp_path)
    first = create_run(path, project_root=Path.cwd())
    assert first.config.run_tag == "run-1"
    assert first.config.config_snapshot_path.read_text() == path.read_text()
    first.config.run_dir.mkdir(exist_ok=True)
    assert next_run_id(load_config(path)) == 2
    payload = json.loads(path.read_text())
    payload["epochs"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigMismatchError, match="new run ID"):
        open_run(path, 1, project_root=Path.cwd())


def test_failure_resume_and_completed_immutability(tmp_path, monkeypatch):
    context = create_run(tiny_config_path(tmp_path), project_root=Path.cwd())
    calls: list[str] = []

    def fail(config, workers):
        calls.append("prepare")
        raise RuntimeError("stop")

    monkeypatch.setitem(PHASE_HANDLERS, "prepare", fail)
    with pytest.raises(RuntimeError, match="stop"):
        execute_phase(context, "prepare")
    manifest = json.loads(context.config.manifest_path.read_text())
    assert manifest["phases"]["prepare"]["status"] == "failed"

    def finish(config, workers):
        calls.append("prepare")
        return {"ok": True}

    monkeypatch.setitem(PHASE_HANDLERS, "prepare", finish)
    assert execute_phase(context, "prepare").data == {"ok": True}
    assert (
        json.loads(context.config.manifest_path.read_text())["phases"]["prepare"][
            "attempts"
        ]
        == 2
    )
    with pytest.raises(RuntimeError, match="requires completed"):
        execute_phase(context, "evaluate_dsp")
    manifest = json.loads(context.config.manifest_path.read_text())
    manifest["status"] = "completed"
    context.config.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="immutable"):
        execute_phase(context, "prepare")


def test_evaluation_phase_routes_only_its_cells_and_notebook_is_direct_api():
    notebook = json.loads(Path("main.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "subprocess.run" not in source
    assert "execute_phase(context, 'evaluate_raw'" in source
    assert "execute_phase(context, 'evaluate_dsp'" in source
    assert "display(" in source


def test_evaluation_handlers_route_only_their_intended_cells(tmp_path, monkeypatch):
    context = create_run(tiny_config_path(tmp_path), project_root=Path.cwd())
    routed = []
    monkeypatch.setattr(
        "src.study.evaluate_study",
        lambda config, *, workers, cells: routed.append(cells) or [],
    )
    monkeypatch.setattr(
        "src.study.evaluation_summary",
        lambda config, cells, label: {"rows": [], "label": label},
    )
    orchestration._phase_evaluate_raw(context.config, 0)
    orchestration._phase_evaluate_dsp(context.config, 0)
    assert routed == [("raw",), ("bandpass", "wiener", "bandpass_wiener")]


def test_dsp_phase_resumes_without_rerunning_completed_dependencies(
    tmp_path, monkeypatch
):
    context = create_run(tiny_config_path(tmp_path), project_root=Path.cwd())
    manifest = json.loads(context.config.manifest_path.read_text())
    for phase in ("prepare", "eda", "tune", "train", "evaluate_raw"):
        manifest["phases"][phase]["status"] = "completed"
    context.config.manifest_path.write_text(json.dumps(manifest))
    calls = []
    monkeypatch.setitem(
        PHASE_HANDLERS,
        "evaluate_dsp",
        lambda config, workers: calls.append("evaluate_dsp") or {"reports": []},
    )
    assert execute_phase(context, "evaluate_dsp").status == "completed"
    assert calls == ["evaluate_dsp"]


def test_notebook_phase_cells_execute_independently_with_mock_backend(tmp_path):
    notebook = json.loads(Path("main.ipynb").read_text())
    calls, displayed = [], []

    def fake_phase(context, phase, *, workers):
        calls.append(phase)
        marker = tmp_path / f"{phase}.json"
        marker.write_text("persisted")
        payloads = {
            "prepare": {"inventory": {}},
            "eda": {
                "design": {
                    "selected_edges_hz": {"low": 1.0, "high": 2.0},
                    "retained_energy_pct": {"speech": 1.0, "noise": 2.0},
                },
                "figures": ["eda.png", "gallery.png"],
            },
            "tune": {
                "selected": {"learning_rate": 1e-3, "arcface_margin": 0.2},
                "trials": [],
                "figure": "tuning.png",
            },
            "train": {
                "hyperparameters": {},
                "training": [],
                "checkpoints": [],
                "figures": [],
            },
            "evaluate_raw": {
                "reports": [],
                "summary": {"rows": [], "error_samples": []},
            },
            "evaluate_dsp": {
                "reports": [],
                "summary": {"rows": [], "error_samples": []},
            },
            "signal_analysis": {
                "positive_control": {"measurements": []},
                "characterisation": {"measurements": []},
            },
            "statistics": {"factorial": {}},
            "figures": {"figures": [], "metric_rows": 0},
        }
        return SimpleNamespace(data=payloads[phase])

    namespace = {
        "context": object(),
        "WORKERS": 0,
        "execute_phase": fake_phase,
        "display": displayed.append,
        "Image": lambda **kwargs: kwargs,
        "Markdown": str,
        "Path": Path,
        "json": json,
        "table": lambda headers, rows: displayed.append(rows),
        "show": displayed.append,
        "config": SimpleNamespace(run_dir=tmp_path, report_root=tmp_path),
    }
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "code" and "execute_phase(context" in source:
            exec(source, namespace)
    assert calls == [
        "prepare",
        "eda",
        "tune",
        "train",
        "evaluate_raw",
        "evaluate_dsp",
        "signal_analysis",
        "statistics",
        "figures",
    ]
    assert all((tmp_path / f"{phase}.json").is_file() for phase in calls)
    assert displayed
