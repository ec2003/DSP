"""Optional thin CLI around the notebook-first experiment phase API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.orchestration import PHASES, execute_phase, find_project_root, open_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=(*PHASES, "all"))
    parser.add_argument("--config", type=Path, default=Path("configs/config.json"))
    parser.add_argument("--run-id", type=int, help="existing incomplete experiment run ID")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    root = find_project_root(Path(__file__).resolve().parent)
    if Path.cwd().resolve() != root:
        parser.error(f"run.py must be invoked from {root}")
    context = open_run(root / args.config, args.run_id, project_root=root)
    if args.output_root is not None:
        # A destination override changes the experiment identity; it must be in config.json.
        parser.error("put output_root in config.json so it is snapshot and hash-bound")
    phases = PHASES if args.phase == "all" else (args.phase,)
    for phase in phases:
        result = execute_phase(context, phase, workers=args.workers)
        print(json.dumps({"phase": result.phase, "status": result.status, "data": result.data}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
