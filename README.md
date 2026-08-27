# DSP501 — Noise-Robust Speaker Embedding with a Designed DSP Front-End

Reproducible research package for the DSP501 final assignment.

**Research problem.** For a speaker embedding trained on noisy VCTK speech,
which DSP front-end stages actually recover speaker-discriminative information,
and how much?

The study compares the two systems the assignment requires:

* **Pipeline A** (`A_raw_noisy`) — noisy audio with minimal preprocessing.
* **Pipeline B** (`B_full`) — the full designed chain: high-pass, low-pass,
  adaptive notch, and an STFT-domain Wiener filter.

Cut-off frequencies are **derived from measured data**, not assumed. `run.py
analyse-bands` measures the long-term average spectra of the training speech and
of the MUSAN noise pool and selects the band that discards at most 1% of speech
energy at each edge.

## Setup

```bash
uv sync
```

### Data

Both corpora are public and must be downloaded manually under their own
licences. Place them as follows:

| Corpus | Link | Expected path |
|---|---|---|
| VCTK 0.80 | <https://datashare.ed.ac.uk/handle/10283/2651> | `dataset/VCTK-Corpus-0.80/wav48/pXXX/*.wav` |
| MUSAN (SLR17) | <https://www.openslr.org/17/> | `dataset/musan/noise/**/*.wav` |

Only the VCTK `wav48` audio and the MUSAN `noise` subset are used.

## Run

```bash
uv run python run.py all --config configs/dsp501-v2.json
```

Individual stages, in dependency order:

| Stage | What it does |
|---|---|
| `prepare` | Speaker-disjoint splits, deterministic silence-trimmed 16 kHz clip cache, MUSAN noise pool. |
| `analyse-bands` | Measures speech/noise band power and reports the derived cut-offs. |
| `tune` | Grid search on Pipeline A with the first seed; the winner is locked for every arm. |
| `train` | Trains the CNN encoder for each arm and seed. |
| `evaluate` | Identification and clustering metrics on unseen test speakers across the SNR grid. |
| `analyze` | Metric summary CSV, paired significance tests, and all report figures. |

Useful flags: `--condition <arm>` restricts train/evaluate to one arm,
`--workers N` sets the DSP worker-process count, `--output-root DIR` redirects
outputs.

## Experimental arms

| Arm | Noise | DSP stages | Role |
|---|---|---|---|
| `clean` | no | — | ceiling reference |
| `A_raw_noisy` | yes | — | **Pipeline A** |
| `B1_hpf` | yes | high-pass | ablation |
| `B2_hpf_lpf` | yes | + low-pass | ablation |
| `B3_hpf_lpf_notch` | yes | + notch | ablation |
| `B_full` | yes | + Wiener | **Pipeline B** |
| `C_specsub` | yes | spectral subtraction instead of Wiener | denoiser comparison |
| `X_telephone` | yes | 300–3400 Hz band-pass | over-filtering control |

Arms named in `primary_conditions` run on all three seeds; the remaining
ablation arms run on the first seed.

## Layout

| Path | Role |
|---|---|
| `run.py` | Canonical CLI; the entry point for reproduction. |
| `configs/dsp501-v2.json` | Every parameter that affects a result. |
| `src/dsp.py` | IIR filters, tonal-peak detection, STFT Wiener, spectral subtraction. |
| `src/features.py` | Log-mel, MFCC, Welch PSD, band power, entropy and statistical features. |
| `src/analysis.py` | Data-driven filter design. |
| `src/corpus.py` | VCTK splits, silence trimming, clip cache. |
| `src/noise.py` | MUSAN pool and SNR-controlled mixing. |
| `src/pipeline.py` | Per-condition front-end assembly. |
| `src/models.py` | CNN speaker encoder and ArcFace head. |
| `src/train.py` | Training loop. |
| `src/eval.py` | Identification, clustering, error analysis, significance tests. |
| `src/study.py` | Stage orchestration. |
| `src/plots.py` | Report figures. |
| `main.ipynb` | Read-only analysis dashboard over the saved artifacts. |
| `report.qmd` | Final report; renders from saved artifacts only. |

## Reproducibility

Noise segment, SNR, crop position, and enrolment partition are deterministic
functions of the clip identity and the seed, so every arm sees identical
difficulty and reruns are bit-identical. Hyperparameters are tuned once on
Pipeline A and then locked, so no arm receives a tuning advantage.

```bash
uv run pytest -q          # offline DSP and metric tests, no corpus needed
quarto render report.qmd  # after a completed run
```

## Citation

Cite VCTK and MUSAN per their published terms. This repository supplies the
experiment; the literature review, ethics statement, and AI declaration required
by the assignment are authored separately in `report.qmd`.
