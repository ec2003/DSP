# DSP501: Noise-Robust Speaker Verification

This is a reproducible study package for text-independent speaker verification with a SpeechBrain ECAPA-TDNN encoder. It asks whether an offline DSP front end improves verification of VCTK utterances mixed with deterministic composite MUSAN noise.

The primary comparison uses the same source clips, noise provenance, and balanced verification pairs for every condition and seed. No test-set threshold is selected: each pipeline/seed threshold is calibrated on validation pairs and then fixed for all test SNRs.

## Study protocol

The versioned study definition is [configs/dsp501-v1.json](configs/dsp501-v1.json). It fixes 3-second clips, 50 deterministic clips per speaker, speaker-disjoint 80/10/10 splits, seeds `11/22/33`, and test SNRs `5/10/15/20 dB`.

| Condition | Input transform |
| --- | --- |
| `clean_reference` | Clean 16 kHz speech ceiling/reference |
| `raw_noisy` | Composite MUSAN noise only |
| `high_pass` | Noise + 80 Hz high-pass |
| `high_pass_low_pass` | Noise + 80 Hz high-pass + 7.5 kHz low-pass |
| `full_dsp` | Noise + both filters + Wiener filter (window 29) |

The filters are zero-phase Butterworth filters, so this is an offline method; it is not presented as deployable streaming enhancement. ECAPA receives mono 16 kHz waveform input, with the model-observable band ending at 8 kHz.

DSP inspection utilities expose 512-point STFTs (160-sample hop), Welch PSD/band power, 40-coefficient MFCCs over 80 mel bands, residual SNR, and ECAPA’s internal filterbank frames. The DSP-only reference summarizes MFCC frames by coefficient mean and standard deviation, L2-normalizes the vector, and scores pairs by cosine similarity.

## Setup and data

```bash
uv sync
uv run python run.py download-data --accept-data-licenses
```

The download command is intentionally opt-in. It checks archive MD5s, rejects unsafe archive paths, extracts only VCTK 0.92 `wav48_silence_trimmed` `mic1` FLAC and MUSAN `noise` content, and validates the expected layout. Read the licenses and citations first: [VCTK 0.92](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full), [MUSAN / SLR17](https://www.openslr.org/17/).

## Run

`run.py` is the canonical workflow. The related delivery artifacts have distinct roles:

| Artifact | Role |
| --- | --- |
| `run.py` | Canonical, versioned study execution and artifact generation. |
| `main.ipynb` | Technical dashboard and CLI orchestrator; it only reads saved artifacts for reporting. |
| `report.qmd` | Final Quarto delivery document; it embeds the completed saved figures and fails clearly if they are absent. |

```bash
uv run python run.py prepare
uv run python run.py tune
uv run python run.py train
uv run python run.py evaluate
uv run python run.py analyze
uv run python run.py package
```

`tune` runs the required `raw_noisy` grid of learning rate `{3e-5, 1e-4}` and encoder-freeze epochs `{0, 1}`, selects lowest validation EER, and records the locked values in `outputs/<study-id>/tuning.json`. All seeds and ablations use that result. `all` runs the complete sequence.

Evaluation reports ROC-AUC, EER, accuracy, precision, recall, F1, and `tn/fp/fn/tp`; error entries retain score, threshold margin, source IDs, SNR, residual-SNR, and MFCC diagnostics. Analysis emits machine-readable JSON/CSV, SNR curves, and paired stratified-bootstrap 95% CIs for full-DSP minus raw-noisy F1. The release command creates an ignored `release/<study-id>/` bundle with source/config/lock files, checkpoints and reports when present, optional ECAPA cache, and a SHA-256 manifest; raw datasets are deliberately excluded.

Run the offline tests with:

```bash
uv run pytest -q
```

After a completed run, render the final report (HTML or PDF) with Quarto:

```bash
quarto render report.qmd
```

The HTML target uses embedded resources for a portable delivery file. Before submission, replace the ethics and AI-use declaration placeholders in `report.qmd` with the required course-specific statements.

Research-protocol, ethics/AI-declaration, and literature-matrix inputs should accompany a submitted report. This repository supplies the reproducible experiment rather than a completed paper or literature review.
