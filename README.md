# DSP501 — Frozen-CNN inference-time front-end factorial ablation

This package trains exactly one `robust_cnn` recipe per seed: VCTK speech with
on-the-fly MUSAN augmentation. The checkpoint is then frozen and evaluated with a
2×2 inference-time DSP front-end, not retrained per DSP condition.

| Band-pass | Wiener | Cell |
|---:|---:|---|
| no | no | `raw` |
| yes | no | `bandpass` |
| no | yes | `wiener` |
| yes | yes | `bandpass_wiener` |

The same cells are evaluated on clean controls and matched mixtures from MUSAN
`noise`, `music`, `speech`, and 3–7-source `babble`. Source recordings are split
train/validation/test before segmentation. Training uses 80% noisy clips, equal
family sampling, and a deterministic continuous Uniform(0, 20) dB SNR keyed by
`(seed, epoch, sample_id)`.

## Reproduce

```bash
cd /home/truong51972/projects/dsp
uv sync
uv run pytest -q
uv run python run.py all --config configs/config.json
QUARTO_PYTHON="$PWD/.venv/bin/python" quarto render report.qmd
```

Download the VCTK distribution used here from
[Kaggle: pratt3000/vctk-corpus](https://www.kaggle.com/datasets/pratt3000/vctk-corpus)
and extract its `wav48` tree to `dataset/VCTK-Corpus-0.80/wav48`. Download MUSAN
from [OpenSLR 17](https://www.openslr.org/17/) and extract it to `dataset/musan`.
MUSAN `noise`, `music`, and `speech` recordings are all required; `babble` is
constructed from 3–7 source-disjoint `speech` recordings.

`all` without `--run-id` allocates the next immutable experiment (`run-1`,
`run-2`, ...). A completed run cannot be overwritten; use `--run-id 1` only
to resume its incomplete phase with the unchanged snapshot configuration.

Stages:

| Stage | Output |
|---|---|
| `prepare` | speaker-disjoint VCTK clips and recording-disjoint MUSAN pools in `cache/<data-config-hash>/` |
| `eda` | frozen run-local `dsp-design.json` and retained-energy design summary |
| `train` | only `runs/run-N/seed-*/robust_cnn/encoder.pt` checkpoints |
| `evaluate_raw` | raw-only clean/noisy, unseen/seen reports |
| `evaluate_dsp` | bandpass, Wiener, and combined reports matched to raw |
| `signal_analysis` | Wiener positive control and front-end waveform characterization |
| `statistics` | metric CSV and paired speaker-cluster bootstrap factorial effects |
| `figures` | report-ready PNG charts from persisted results |

The frozen band-pass is derived only from training speech and training MUSAN pools.
Its artifact contains the selected edges, inputs, rule and config hash; front-end
code rejects missing or mismatched artifacts. Inference reports retain target and
measured SNR, family, source recordings and offsets. Bootstrap resamples test
speakers while retaining all of their seed, family, SNR and query observations;
it does not treat seed×SNR cells as independent experiments.

`runs/run-N/config.json` is the exact copied input configuration and its
`run-manifest.json` records hashes, git revision, phase attempts, outputs,
checkpoints, and failures. `main.ipynb` creates a new run when `RUN_ID = None`;
an integer resumes only its failed or pending phase. Quarto only reads a selected
run's snapshot and artifacts.
