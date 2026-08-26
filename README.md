# DSP Speaker Embedding Experiment

This project fine-tunes the same pretrained ECAPA-TDNN speaker encoder under three controlled conditions:

1. `clean_baseline`: clean VCTK audio.
2. `noisy`: VCTK mixed with deterministic composite MUSAN noise.
3. `noisy_wiener`: the same composite noisy VCTK segments processed by high-pass, low-pass, then SciPy Wiener filtering.

The encoder returns L2-normalized speaker embeddings. Two utterances are compared with cosine similarity: larger values mean more similar voices. The product-facing score may display `(cosine + 1) / 2 * 100` as a similarity index; it is not a calibrated probability.

## Signal Chain

VCTK `wav48` is the clean source at 48 kHz. The ECAPA checkpoint expects mono 16 kHz, so the dataset loader resamples before model input. The source and model Nyquist frequencies are therefore $24\,\text{kHz}$ and $8\,\text{kHz}$ respectively. All experiment noise, analysis, DSP cutoffs, and model claims after resampling are limited to the model-observable $0$-$8\,\text{kHz}$ band.

The familiar $20$-$20{,}000\,\text{Hz}$ range describes typical human hearing, not the required input bandwidth for speaker embeddings. This experiment does not claim a particular microphone response; it distinguishes the VCTK source bandwidth from the model's 16 kHz input bandwidth.

Each noisy segment is a deterministic composite of three separately selected MUSAN `noise` files:

- `environmental`: raw MUSAN noise.
- `low_band`: MUSAN noise band-limited to $20$-$300\,\text{Hz}$.
- `high_band`: MUSAN noise band-limited to $3000$-$7500\,\text{Hz}$.

The configured SNR ($5$, $10$, $15$, or $20\,\text{dB}$) applies to the **total composite noise**, not to every component. Components receive equal nominal power and the summed waveform is RMS-corrected to the selected total SNR.

For `noisy_wiener`, DSP is applied offline in this fixed order:

$$
	ext{high-pass }80\,\text{Hz}
\rightarrow
	ext{low-pass }7500\,\text{Hz}
\rightarrow
	ext{local Wiener filter}.
$$

The $7.5$-$8\,\text{kHz}$ interval is a guard band below model Nyquist. The Butterworth filters use zero-phase SciPy SOS filtering; phase 2 and phase 3 start with identical raw composite noisy waveforms.

## Setup

```bash
uv sync
uv run jupyter lab main.ipynb
```

Place the datasets in this layout:

```text
dataset/
	VCTK-Corpus/
		wav48/
			p225/
				p225_001.wav
	musan/
		noise/
			free-sound/
				... audio files ...
```

VCTK can be downloaded from the link in [dataset/README.md](dataset/README.md). Download MUSAN separately and keep the `noise` subset under `dataset/musan/noise`. Use only data that is permitted by its respective license.

## Run Experiment

Open [main.ipynb](main.ipynb) after the datasets are in place. It is the primary workflow and includes cells to:

1. Explain VCTK source sampling, model sampling, Nyquist, and anti-aliasing before processing data.
2. Check the VCTK and MUSAN layout, then set seed, paths, SNR, bands, and DSP cutoffs in one place.
3. Create reusable speaker-disjoint manifests.
4. Analyze one held-out segment from 48 kHz source through 16 kHz model input, three noise components, composite mixture, high-pass, low-pass, and Wiener stages.
5. Plot waveform, Welch PSD, STFT, band-energy, filter-response, component provenance, and stage-by-stage residual-SNR evidence before training.
6. Fine-tune the clean, noisy, and noisy-Wiener conditions from the same ECAPA checkpoint.
7. Evaluate all checkpoints on shared composite-noisy test audio, then evaluate the clean baseline on clean test as a separate reference.
8. Plot verification and HDBSCAN metrics across the three noisy-test conditions.

The first training cell downloads the SpeechBrain checkpoint into `pretrained_models/spkrec-ecapa-voxceleb`. Checkpoints and reports are written under `outputs/<condition>/`.

## Results

Each evaluation writes `outputs/<condition>/evaluation-*.json` with:

- `roc_auc`, `eer`, and threshold-based verification `accuracy` for same/different speaker pairs.
- HDBSCAN `ari`, `nmi`, `v_measure`, `clustered_coverage`, and `outlier_rate` against held-out speaker labels.

Lower EER and higher ROC-AUC, ARI, NMI, V-measure, and coverage are better. HDBSCAN is supporting evidence only; use the verification metrics for the primary claim that Wiener training/inference improves over the raw-noisy condition and approaches the separate clean reference.

The notebook's signal-level charts report residual SNR relative to the paired clean 16 kHz waveform after each stage. A positive final-DSP minus composite-noisy SNR delta indicates that the complete DSP chain is closer to clean for that inspected segment. Inspect the PSD and spectrogram too: band filtering or Wiener smoothing can suppress speaker-relevant speech detail along with noise.

## Validation

```bash
uv run pytest -q
```

The tests cover deterministic speaker-disjoint manifests, waveform length/finiteness, ECAPA embedding normalization, cosine-pair evaluation, and HDBSCAN result fields.