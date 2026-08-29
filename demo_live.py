#!/usr/bin/env python3
"""DSP501 — Gradio Demo: Live Voice Similarity & DSP Front-End Verification.

Allows users to record 2 audio samples (or upload audio files), runs speaker
verification through two pipelines (Without DSP vs With DSP front-end), and
compares their cosine similarity, audio waveforms, spectrograms, and PSDs.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy import signal

from src.config import Condition, ExperimentConfig, load_config
from src.corpus import trim_silence
from src.features import LogMelSpectrogram
from src.models import SpeakerCNN
from src.pipeline import build_chain, frozen_band_edges
from src.train import load_encoder, resolve_device


# --------------------------------------------------------------------------- #
# Global Model Manager & Cache
# --------------------------------------------------------------------------- #
class DemoModelManager:
    """Manages lazy loading and caching of SpeakerCNN models and DSP chains."""

    def __init__(self, config_path: str = "configs/config.json", run_id: int = 2):
        self.config_path = Path(config_path)
        self.run_id = run_id
        self.device = resolve_device()
        self.config: ExperimentConfig | None = None
        self.encoders: dict[int, SpeakerCNN] = {}
        self.dsp_chains: dict[str, Any] = {}
        self.band_edges: tuple[float, float] = (31.25, 3500.0)
        self.mel_transform: LogMelSpectrogram | None = None
        self._initialize()

    def _initialize(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        raw_cfg = load_config(self.config_path)
        # Check available runs in output_root
        study_runs_dir = Path(raw_cfg.output_root) / raw_cfg.study_id / "runs"
        available_runs = sorted([
            int(p.name.replace("run-", ""))
            for p in study_runs_dir.glob("run-*")
            if (p / "run-manifest.json").exists()
        ])

        if available_runs:
            self.run_id = self.run_id if self.run_id in available_runs else available_runs[-1]
        
        self.config = replace(raw_cfg, run_id=self.run_id)
        
        try:
            self.band_edges = frozen_band_edges(self.config)
        except Exception:
            self.band_edges = (31.25, 3500.0)

        self.mel_transform = LogMelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
        )

        # Build DSP chains for available conditions
        for cond_name in ("raw", "bandpass", "wiener", "bandpass_wiener"):
            try:
                cond = self.config.condition(cond_name)
                self.dsp_chains[cond_name] = build_chain(cond, self.config)
            except Exception as e:
                print(f"Warning building chain for {cond_name}: {e}")

    def get_encoder(self, seed: int = 11) -> SpeakerCNN:
        if seed in self.encoders:
            return self.encoders[seed]

        ckpt_path = self.config.seed_dir(seed) / "encoder.pt"
        if not ckpt_path.exists():
            # Fallback to any available seed checkpoint
            checkpoints = list(self.config.run_dir.glob("seed-*/robust_cnn/encoder.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"No trained encoder checkpoint found in {self.config.run_dir}")
            ckpt_path = checkpoints[0]

        print(f"Loading encoder from {ckpt_path} to {self.device}...")
        encoder = load_encoder(self.config, ckpt_path, self.device)
        self.encoders[seed] = encoder
        return encoder


# Initialize Manager Singleton
MANAGER = DemoModelManager()


# --------------------------------------------------------------------------- #
# Audio Preprocessing & Synthetic Noise Helpers
# --------------------------------------------------------------------------- #
def preprocess_audio(
    audio_input: tuple[int, np.ndarray] | str | Path | None,
    target_sr: int = 16000,
    segment_seconds: float = 2.0,
) -> np.ndarray | None:
    """Standardizes audio input from Gradio (microphone or upload) to 16 kHz float32 mono."""
    if audio_input is None:
        return None

    if isinstance(audio_input, (str, Path)):
        audio, sr = sf.read(str(audio_input), dtype="float32", always_2d=True)
        waveform = audio.mean(axis=1)
    elif isinstance(audio_input, tuple):
        sr, data = audio_input
        if data is None or len(data) == 0:
            return None
        data = np.asarray(data)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)

        if data.ndim > 1:
            # Multi-channel: convert to mono
            waveform = data.mean(axis=1 if data.shape[1] < data.shape[0] else 0)
        else:
            waveform = data
    else:
        return None

    # Resample to target_sr if needed
    if sr != target_sr:
        tensor_w = torch.from_numpy(waveform).unsqueeze(0)
        resampled = torchaudio.functional.resample(tensor_w, sr, target_sr)
        waveform = resampled.squeeze(0).numpy().astype(np.float32)

    # Trim silence
    waveform = trim_silence(waveform)
    if waveform.size == 0:
        return np.zeros(int(target_sr * segment_seconds), dtype=np.float32)

    # Segment / Pad to segment_seconds
    target_samples = int(target_sr * segment_seconds)
    if waveform.size < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.size))
    elif waveform.size > target_samples:
        # Take central slice or first target_samples
        waveform = waveform[:target_samples]

    # Peak normalization
    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform = (waveform / peak) * 0.95

    return waveform.astype(np.float32)


def add_synthetic_noise(
    waveform: np.ndarray,
    noise_type: str,
    snr_db: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Adds synthetic or simulated noise at a specified target SNR (dB)."""
    if noise_type == "None" or waveform.size == 0:
        return waveform

    speech_power = float(np.mean(waveform**2) + 1e-12)
    target_noise_power = speech_power / (10.0 ** (snr_db / 10.0))

    if noise_type == "White Noise (Gaussian)":
        noise = np.random.randn(*waveform.shape).astype(np.float32)
    elif noise_type == "Pink Noise (1/f)":
        # Approximate pink noise via IIR filtering of white noise
        white = np.random.randn(*waveform.shape).astype(np.float32)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        noise = signal.lfilter(b, a, white).astype(np.float32)
    elif noise_type == "Tonal Hum (50 Hz + harmonics)":
        t = np.arange(waveform.size) / sample_rate
        noise = (
            np.sin(2 * np.pi * 50 * t)
            + 0.5 * np.sin(2 * np.pi * 100 * t)
            + 0.3 * np.sin(2 * np.pi * 150 * t)
        ).astype(np.float32)
    else:
        noise = np.random.randn(*waveform.shape).astype(np.float32)

    current_noise_power = float(np.mean(noise**2) + 1e-12)
    scaled_noise = noise * np.sqrt(target_noise_power / current_noise_power)
    noisy_waveform = waveform + scaled_noise

    # Normalize peak to avoid clipping
    peak = float(np.max(np.abs(noisy_waveform)))
    if peak > 0:
        noisy_waveform = (noisy_waveform / peak) * 0.95

    return noisy_waveform.astype(np.float32)


# --------------------------------------------------------------------------- #
# Embedding & Similarity Extraction
# --------------------------------------------------------------------------- #
def extract_embedding(
    waveform: np.ndarray,
    encoder: SpeakerCNN,
    device: torch.device,
) -> np.ndarray:
    """Extracts a 192-d L2-normalized speaker embedding."""
    with torch.no_grad():
        tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)
        embedding = encoder(tensor).cpu().numpy()[0]
    return embedding


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Computes cosine similarity between two unit vectors."""
    dot = float(np.dot(emb1, emb2))
    return float(np.clip(dot, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# Plotting & Visualization
# --------------------------------------------------------------------------- #
def generate_comparison_plot(
    w1_raw: np.ndarray,
    w1_dsp: np.ndarray,
    w2_raw: np.ndarray,
    w2_dsp: np.ndarray,
    sample_rate: int = 16000,
    band_edges: tuple[float, float] = (31.25, 3500.0),
) -> plt.Figure:
    """Creates a 4-row visualization: Waveform, STFT Spectrogram, Log-Mel, and PSD."""
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), dpi=100)
    time = np.arange(len(w1_raw)) / sample_rate

    signals = [
        ("Audio 1 (Gốc / Raw)", w1_raw, "Audio 1 (Sau khi qua DSP)", w1_dsp),
        ("Audio 2 (Gốc / Raw)", w2_raw, "Audio 2 (Sau khi qua DSP)", w2_dsp),
    ]

    # Row 0: Waveforms
    for col, (title_raw, w_raw, title_dsp, w_dsp) in enumerate(signals):
        ax = axes[0, col]
        ax.plot(time, w_raw, color="#3498db", alpha=0.6, linewidth=0.8, label="Raw")
        ax.plot(time, w_dsp, color="#e74c3c", alpha=0.85, linewidth=0.8, label="DSP Enhanced")
        ax.set_title(f"Waveform: {title_raw.split('(')[0].strip()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Thời gian (giây)", fontsize=9)
        ax.set_ylabel("Biên độ", fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)

    # Row 1: STFT Spectrogram (Raw)
    for col, (title_raw, w_raw, _, _) in enumerate(signals):
        ax = axes[1, col]
        _, _, stft_raw = signal.stft(w_raw, fs=sample_rate, nperseg=512, noverlap=352)
        stft_db = 20 * np.log10(np.abs(stft_raw) + 1e-6)
        im = ax.imshow(
            stft_db,
            origin="lower",
            aspect="auto",
            cmap="inferno",
            extent=[0, time[-1], 0, sample_rate / 2],
            vmin=-80,
            vmax=0,
        )
        ax.set_title(f"STFT Spectrogram: {title_raw}", fontsize=10)
        ax.set_xlabel("Thời gian (giây)", fontsize=9)
        ax.set_ylabel("Tần số (Hz)", fontsize=9)

    # Row 2: STFT Spectrogram (DSP Processed)
    for col, (_, _, title_dsp, w_dsp) in enumerate(signals):
        ax = axes[2, col]
        _, _, stft_dsp = signal.stft(w_dsp, fs=sample_rate, nperseg=512, noverlap=352)
        stft_db = 20 * np.log10(np.abs(stft_dsp) + 1e-6)
        im = ax.imshow(
            stft_db,
            origin="lower",
            aspect="auto",
            cmap="inferno",
            extent=[0, time[-1], 0, sample_rate / 2],
            vmin=-80,
            vmax=0,
        )
        ax.axhline(band_edges[0], color="#2ecc71", linestyle="--", alpha=0.7, label=f"Low: {band_edges[0]:.1f} Hz")
        ax.axhline(band_edges[1], color="#2ecc71", linestyle="--", alpha=0.7, label=f"High: {band_edges[1]:.1f} Hz")
        ax.set_title(f"STFT Spectrogram: {title_dsp}", fontsize=10)
        ax.set_xlabel("Thời gian (giây)", fontsize=9)
        ax.set_ylabel("Tần số (Hz)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)

    # Row 3: Welch Power Spectral Density (PSD)
    for col, (title_raw, w_raw, title_dsp, w_dsp) in enumerate(signals):
        ax = axes[3, col]
        f_raw, psd_raw = signal.welch(w_raw, fs=sample_rate, nperseg=512)
        f_dsp, psd_dsp = signal.welch(w_dsp, fs=sample_rate, nperseg=512)
        ax.semilogy(f_raw, psd_raw + 1e-12, color="#3498db", label="PSD Gốc (Raw)", linewidth=1.2)
        ax.semilogy(f_dsp, psd_dsp + 1e-12, color="#e74c3c", label="PSD sau DSP", linewidth=1.2)
        ax.axvspan(band_edges[0], band_edges[1], color="#2ecc71", alpha=0.15, label="DSP Passband")
        ax.set_title(f"Mật độ Phổ Công suất (PSD): {title_raw.split('(')[0].strip()}", fontsize=10)
        ax.set_xlabel("Tần số (Hz)", fontsize=9)
        ax.set_ylabel("Công suất / Tần số", fontsize=9)
        ax.set_xlim(0, sample_rate / 2)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Main Gradio Prediction Handler
# --------------------------------------------------------------------------- #
def process_and_compare(
    audio1_input: tuple[int, np.ndarray] | str | None,
    audio2_input: tuple[int, np.ndarray] | str | None,
    dsp_condition: str = "bandpass_wiener",
    sim_threshold: float = 0.45,
    noise_type: str = "None",
    snr_db: float = 10.0,
    seed: int = 11,
) -> tuple[
    str,          # Raw Result Markdown
    str,          # DSP Result Markdown
    str,          # Comparison Summary Markdown
    tuple[int, np.ndarray] | None,  # Processed Audio 1
    tuple[int, np.ndarray] | None,  # Processed Audio 2
    plt.Figure | None,              # Analysis Plot
]:
    """Runs dual-audio speaker verification with and without DSP front-end."""
    if audio1_input is None or audio2_input is None:
        empty_md = (
            "<div style='text-align: center; padding: 20px; color: #7f8c8d;'>"
            "⚠️ Vui lòng ghi âm hoặc tải lên cả 2 đoạn âm thanh (Audio 1 và Audio 2) để so sánh."
            "</div>"
        )
        return empty_md, empty_md, empty_md, None, None, None

    sr = MANAGER.config.sample_rate if MANAGER.config else 16000
    seg_sec = MANAGER.config.segment_seconds if MANAGER.config else 2.0

    # 1. Preprocess both audio inputs
    w1 = preprocess_audio(audio1_input, target_sr=sr, segment_seconds=seg_sec)
    w2 = preprocess_audio(audio2_input, target_sr=sr, segment_seconds=seg_sec)

    if w1 is None or w2 is None:
        error_md = "<div style='color: red;'>Lỗi khi nạp file âm thanh. Vui lòng thử lại.</div>"
        return error_md, error_md, error_md, None, None, None

    # 2. Add synthetic noise if requested
    w1_noisy = add_synthetic_noise(w1, noise_type, snr_db, sample_rate=sr)
    w2_noisy = add_synthetic_noise(w2, noise_type, snr_db, sample_rate=sr)

    # 3. Apply DSP Chains
    chain_raw = MANAGER.dsp_chains.get("raw", lambda x: x)
    chain_dsp = MANAGER.dsp_chains.get(dsp_condition, MANAGER.dsp_chains.get("bandpass_wiener", lambda x: x))

    w1_raw = chain_raw(w1_noisy)
    w2_raw = chain_raw(w2_noisy)
    w1_dsp = chain_dsp(w1_noisy)
    w2_dsp = chain_dsp(w2_noisy)

    # 4. Extract Embeddings & Compute Cosine Similarities
    encoder = MANAGER.get_encoder(seed=seed)
    
    emb1_raw = extract_embedding(w1_raw, encoder, MANAGER.device)
    emb2_raw = extract_embedding(w2_raw, encoder, MANAGER.device)
    sim_raw = compute_cosine_similarity(emb1_raw, emb2_raw)

    emb1_dsp = extract_embedding(w1_dsp, encoder, MANAGER.device)
    emb2_dsp = extract_embedding(w2_dsp, encoder, MANAGER.device)
    sim_dsp = compute_cosine_similarity(emb1_dsp, emb2_dsp)

    # 5. Format Decision Cards
    is_same_raw = sim_raw >= sim_threshold
    is_same_dsp = sim_dsp >= sim_threshold
    delta_sim = sim_dsp - sim_raw

    # Colors
    color_raw = "#27ae60" if is_same_raw else "#e74c3c"
    badge_raw = "✅ CÙNG NGƯỜI NÓI" if is_same_raw else "❌ KHÁC NGƯỜI NÓI"
    
    color_dsp = "#27ae60" if is_same_dsp else "#e74c3c"
    badge_dsp = "✅ CÙNG NGƯỜI NÓI" if is_same_dsp else "❌ KHÁC NGƯỜI NÓI"

    raw_html = f"""
    <div style="background: #fdfefe; border: 2px solid #bdc3c7; border-radius: 10px; padding: 18px; text-align: center;">
        <h3 style="margin-top:0; color:#2c3e50;">🔴 Model Không có DSP (Raw)</h3>
        <div style="font-size: 32px; font-weight: bold; color: {color_raw}; margin: 10px 0;">
            {sim_raw:.4f}
        </div>
        <div style="font-size: 14px; color: #7f8c8d; margin-bottom: 12px;">
            Cosine Similarity: <b>{sim_raw * 100:.1f}%</b>
        </div>
        <div style="display: inline-block; background: {color_raw}; color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; font-size: 14px;">
            {badge_raw}
        </div>
    </div>
    """

    dsp_html = f"""
    <div style="background: #fdfefe; border: 2px solid #3498db; border-radius: 10px; padding: 18px; text-align: center;">
        <h3 style="margin-top:0; color:#2980b9;">🔵 Model Có DSP Front-end ({dsp_condition})</h3>
        <div style="font-size: 32px; font-weight: bold; color: {color_dsp}; margin: 10px 0;">
            {sim_dsp:.4f}
        </div>
        <div style="font-size: 14px; color: #7f8c8d; margin-bottom: 12px;">
            Cosine Similarity: <b>{sim_dsp * 100:.1f}%</b>
        </div>
        <div style="display: inline-block; background: {color_dsp}; color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; font-size: 14px;">
            {badge_dsp}
        </div>
    </div>
    """

    delta_color = "#27ae60" if delta_sim > 0 else ("#e67e22" if delta_sim < 0 else "#7f8c8d")
    delta_sign = "+" if delta_sim > 0 else ""

    summary_html = f"""
    <div style="background: #f4f6f7; border-radius: 10px; padding: 15px; margin-top: 10px; border-left: 5px solid #2980b9;">
        <h4 style="margin: 0 0 8px 0; color: #2c3e50;">📊 Đánh giá tác động của DSP Front-end</h4>
        <ul style="margin: 0; padding-left: 20px; font-size: 14px; line-height: 1.6;">
            <li><b>Độ lệch Similarity (&Delta;):</b> <span style="font-weight: bold; color: {delta_color};">{delta_sign}{delta_sim:.4f}</span> ({delta_sign}{delta_sim*100:.2f}%)</li>
            <li><b>Ngưỡng xác thực (Threshold):</b> <code>{sim_threshold:.2f}</code></li>
            <li><b>Cấu hình lọc DSP:</b> <code>{dsp_condition}</code> (Dải thông: <code>{MANAGER.band_edges[0]:.1f} Hz – {MANAGER.band_edges[1]:.1f} Hz</code>)</li>
            <li><b>Trạng thái nhiễu đầu vào:</b> <code>{noise_type}</code> {f"với SNR = {snr_db} dB" if noise_type != "None" else "(Âm thanh nguyên bản)"}</li>
        </ul>
    </div>
    """

    # 6. Audio Playback Outputs (Normalized 16-bit PCM for browser playback)
    out_audio1 = (sr, (w1_dsp * 32767).astype(np.int16))
    out_audio2 = (sr, (w2_dsp * 32767).astype(np.int16))

    # 7. Generate Visual Comparison Plot
    fig = generate_comparison_plot(
        w1_raw, w1_dsp, w2_raw, w2_dsp, sample_rate=sr, band_edges=MANAGER.band_edges
    )

    return raw_html, dsp_html, summary_html, out_audio1, out_audio2, fig


# --------------------------------------------------------------------------- #
# Preset Samples Loader
# --------------------------------------------------------------------------- #
def get_sample_preset_paths() -> list[list[Any]]:
    """Finds sample audio files in the dataset for quick 1-click testing."""
    vctk_root = Path("dataset/VCTK-Corpus-0.80/wav48")
    presets = []
    
    if vctk_root.exists():
        # Speaker p225
        p225_1 = vctk_root / "p225" / "p225_001.wav"
        p225_2 = vctk_root / "p225" / "p225_002.wav"
        p225_3 = vctk_root / "p225" / "p225_003.wav"
        # Speaker p226
        p226_1 = vctk_root / "p226" / "p226_001.wav"
        # Speaker p227
        p227_1 = vctk_root / "p227" / "p227_001.wav"

        if p225_1.exists() and p225_2.exists():
            presets.append([str(p225_1), str(p225_2), "bandpass_wiener", 0.45, "None", 10.0])
        if p225_1.exists() and p225_3.exists():
            presets.append([str(p225_1), str(p225_3), "bandpass_wiener", 0.45, "White Noise (Gaussian)", 5.0])
        if p225_1.exists() and p226_1.exists():
            presets.append([str(p225_1), str(p226_1), "bandpass_wiener", 0.45, "None", 10.0])
        if p225_1.exists() and p227_1.exists():
            presets.append([str(p225_1), str(p227_1), "bandpass_wiener", 0.45, "Pink Noise (1/f)", 10.0])

    return presets


# --------------------------------------------------------------------------- #
# Gradio UI Construction
# --------------------------------------------------------------------------- #
def build_gradio_app() -> gr.Blocks:
    """Builds the comprehensive Gradio UI Blocks."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
    )

    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: auto !important; }
    .header-box { text-align: center; margin-bottom: 20px; padding: 20px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; border-radius: 12px; }
    .header-box h1 { color: #ffffff !important; margin-bottom: 8px; font-weight: 800; }
    .header-box p { color: #e0e6ed; font-size: 15px; margin: 0; }
    .footer-text { text-align: center; color: #7f8c8d; font-size: 13px; margin-top: 30px; }
    """

    demo = gr.Blocks(title="DSP501 — Speaker Verification Demo")
    demo.theme = theme
    demo.css = custom_css

    with demo:
        # Header Banner
        gr.HTML("""
        <div class="header-box">
            <h1>🎙️ DSP501: So sánh Độ Tương Đồng Giọng Nói (Speaker Verification)</h1>
            <p>Khảo sát & Đánh giá hiệu quả của tiền xử lý DSP Front-end (Bandpass + Wiener Denoising) kết hợp mạng Frozen-CNN Speaker Encoder</p>
        </div>
        """)

        with gr.Row():
            # Left Column: Inputs & Settings
            with gr.Column(scale=5):
                gr.Markdown("### 1️⃣ Nhập 2 Mẫu Âm Thanh Cần So Sánh")
                with gr.Row():
                    with gr.Column():
                        audio1 = gr.Audio(
                            sources=["microphone", "upload"],
                            type="numpy",
                            label="Mẫu Âm Thanh 1 (Audio 1)",
                        )
                    with gr.Column():
                        audio2 = gr.Audio(
                            sources=["microphone", "upload"],
                            type="numpy",
                            label="Mẫu Âm Thanh 2 (Audio 2)",
                        )

                with gr.Accordion("⚙️ Tùy chọn Nâng cao & Thử nghiệm Khử Nhiễu", open=True):
                    with gr.Row():
                        dsp_cond = gr.Dropdown(
                            choices=["bandpass_wiener", "bandpass", "wiener"],
                            value="bandpass_wiener",
                            label="Chuỗi xử lý DSP (Front-end)",
                            info="Bandpass (31.25Hz-3.5kHz) + STFT Wiener Denoising",
                        )
                        threshold_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.45,
                            step=0.01,
                            label="Ngưỡng Nhận Diện (Threshold)",
                            info="Điểm Cosine Similarity tối thiểu để xem là cùng 1 người",
                        )

                    with gr.Row():
                        noise_selector = gr.Dropdown(
                            choices=[
                                "None",
                                "White Noise (Gaussian)",
                                "Pink Noise (1/f)",
                                "Tonal Hum (50 Hz + harmonics)",
                            ],
                            value="None",
                            label="Mô phỏng Thêm Nhiễu (Noise Simulation)",
                            info="Kiểm tra khả năng chống chịu nhiễu của DSP",
                        )
                        snr_slider = gr.Slider(
                            minimum=0.0,
                            maximum=20.0,
                            value=10.0,
                            step=1.0,
                            label="Mức SNR mục tiêu (dB)",
                            info="SNR càng thấp thì nhiễu càng lớn (0 dB: rất ồn, 20 dB: ít ồn)",
                        )

                with gr.Row():
                    btn_compare = gr.Button("🚀 Phân Tích & So Sánh Độ Tương Đồng", variant="primary", scale=2)
                    btn_clear = gr.ClearButton([audio1, audio2], value="🗑️ Xóa Input", scale=1)

            # Right Column: Comparison Results
            with gr.Column(scale=7):
                gr.Markdown("### 2️⃣ Kết Quả So Sánh Độ Tương Đồng")
                with gr.Row():
                    out_raw_card = gr.HTML(
                        "<div style='text-align:center; padding:30px; border:1px dashed #ccc; border-radius:8px; color:#888;'>"
                        "Chưa có dữ liệu. Nhấn 'Phân Tích & So Sánh' để xem kết quả."
                        "</div>"
                    )
                    out_dsp_card = gr.HTML(
                        "<div style='text-align:center; padding:30px; border:1px dashed #ccc; border-radius:8px; color:#888;'>"
                        "Chưa có dữ liệu. Nhấn 'Phân Tích & So Sánh' để xem kết quả."
                        "</div>"
                    )

                out_summary = gr.HTML()

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("🎧 **Audio 1 sau lọc DSP:**")
                        out_audio1_dsp = gr.Audio(label="Audio 1 (Processed)", type="numpy", interactive=False)
                    with gr.Column():
                        gr.Markdown("🎧 **Audio 2 sau lọc DSP:**")
                        out_audio2_dsp = gr.Audio(label="Audio 2 (Processed)", type="numpy", interactive=False)

        # Visual Analytics Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 3️⃣ Trực Quan Hóa Tín Hiệu: Waveform, STFT Spectrogram & PSD")
                out_plot = gr.Plot(label="Biểu đồ phân tích phổ âm thanh")

        # Preset Examples Table
        preset_examples = get_sample_preset_paths()
        if preset_examples:
            gr.Markdown("### 💡 Mẫu Kiểm Thử Có Sẵn (Preset Examples)")
            gr.Examples(
                examples=preset_examples,
                inputs=[audio1, audio2, dsp_cond, threshold_slider, noise_selector, snr_slider],
                label="Chọn mẫu thử nghiệm nhanh:",
            )

        # Wire click event
        btn_compare.click(
            fn=process_and_compare,
            inputs=[audio1, audio2, dsp_cond, threshold_slider, noise_selector, snr_slider],
            outputs=[out_raw_card, out_dsp_card, out_summary, out_audio1_dsp, out_audio2_dsp, out_plot],
        )

        gr.HTML("""
        <div class="footer-text">
            DSP501 Project • Noise-Robust Speaker Verification • Frozen-CNN Factorial Study • Gradio Interactive Interface
        </div>
        """)

    return demo


# --------------------------------------------------------------------------- #
# CLI Entry Point
# --------------------------------------------------------------------------- #
def main():
    import argparse
    parser = argparse.ArgumentParser(description="DSP501 Gradio Speaker Similarity Demo")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    app = build_gradio_app()
    print(f"Starting Gradio Demo on http://{args.host}:{args.port}...")
    try:
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            theme=getattr(app, "theme", None),
            css=getattr(app, "css", None),
        )
    except TypeError:
        app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
