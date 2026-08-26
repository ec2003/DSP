from __future__ import annotations

import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "main.ipynb"
REPORT_PATH = PROJECT_ROOT / "report.qmd"


def _notebook_source() -> tuple[dict[str, object], str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return notebook, "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebook_uses_current_five_condition_protocol_without_legacy_names() -> None:
    _, source = _notebook_source()
    for condition in ("clean_reference", "raw_noisy", "high_pass", "high_pass_low_pass", "full_dsp"):
        assert condition in source
    for legacy_condition in ("clean_baseline", "noisy_wiener", "CONDITIONS = ('clean", "Fine-Tune Three Conditions"):
        assert legacy_condition not in source


def test_notebook_cli_cells_only_use_supported_run_commands() -> None:
    notebook, source = _notebook_source()
    parser_source = (PROJECT_ROOT / "run.py").read_text(encoding="utf-8")
    supported = set(re.findall(r'commands\.add_parser\("([a-z-]+)"\)', parser_source))
    loop = re.search(r"for name in \((.*?)\):\s+commands\.add_parser\(name\)", parser_source, flags=re.DOTALL)
    assert loop is not None
    supported.update(re.findall(r'"([a-z-]+)"', loop.group(1)))
    commands = re.findall(r"run\.py(?:\s+--config\s+\S+)?\s+([a-z-]+)", source)
    assert commands and set(commands) <= supported
    assert "run.py all" in source
    assert {"download-data", "prepare", "tune", "train", "evaluate", "analyze", "package", "all"} <= set(commands)
    assert any("full-study" in cell.get("metadata", {}).get("tags", []) for cell in notebook["cells"])


def test_notebook_dashboard_reads_saved_artifacts_without_reimplementing_study_logic() -> None:
    notebook, source = _notebook_source()
    dashboard_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "dashboard" in cell.get("metadata", {}).get("tags", [])
    )
    assert dashboard_source
    for artifact in ("tuning.json", "metric-summary.csv", "paired-bootstrap-ci.json", "evaluation-test-snr-", "waveforms.png", "filter-response.png"):
        assert artifact in dashboard_source
    for forbidden in ("train_condition", "evaluate_checkpoint", "prepare_manifests", "verification_metrics", "CompositeMusanNoiseMixer", "matplotlib", "src.audio", "src.experiments"):
        assert forbidden not in source
    assert "read_text" in dashboard_source and "Image(" in dashboard_source


def test_report_has_render_targets_sections_citations_and_saved_artifact_paths() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    assert report.startswith("---\n") and "format:\n  html:" in report and "  pdf:" in report
    assert "embed-resources: true" in report and "bibliography: references.bib" in report
    for heading in (
        "# Research problem and questions",
        "# DSP methodology",
        "# Models and comparison",
        "# Experimental controls",
        "# Results",
        "# Error analysis",
        "# Limitations",
        "# Ethics and AI declaration",
        "# Reproducibility",
    ):
        assert heading in report
    for citation in ("@vctk", "@musan", "@ecapa", "@speechbrain"):
        assert citation in report
    for artifact in ("metric-summary.csv", "paired-bootstrap-ci.json", "snr-curves.png", "ablation-f1.png", "confusion-matrices.png", "evaluation-mfcc-cosine.json"):
        assert artifact in report
    assert "#fig-snr-curves" in report and "FileNotFoundError" in report
    assert (PROJECT_ROOT / "_quarto.yml").is_file() and (PROJECT_ROOT / "references.bib").is_file()
