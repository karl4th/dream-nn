"""
Shared utilities for NNAI-S benchmarks.

Audio loading, data preprocessing, and common helpers.
"""

import torch
import librosa
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    test_name: str
    passed: bool
    metrics: dict
    details: str = ""


def load_audio_files(
    audio_dir: str,
    target_sr: int = 16000,
    n_mels: int = 80,
    hop_length: int = 256,
    n_fft: int = 1024
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Load audio files and extract mel spectrograms.

    Parameters
    ----------
    audio_dir : str
        Directory containing .wav files
    target_sr : int
        Target sample rate for resampling
    n_mels : int
        Number of mel bins
    hop_length : int
        Hop length for STFT
    n_fft : int
        FFT window size

    Returns
    -------
    features : List[torch.Tensor]
        List of mel spectrogram tensors (time, n_mels)
    names : List[str]
        List of filenames
    """
    audio_path = Path(audio_dir)
    files = sorted(audio_path.glob("*.wav"))

    features = []
    names = []

    for f in files:
        y, sr = librosa.load(str(f), sr=target_sr)
        melspec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
        )
        log_mels = librosa.power_to_db(melspec, ref=np.max)
        feat = torch.tensor(log_mels.T, dtype=torch.float32)
        # Normalize per file
        feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        features.append(feat)
        names.append(f.name)

    return features, names


def pad_sequences(sequences: List[torch.Tensor]) -> torch.Tensor:
    """
    Pad sequences to maximum length.

    Parameters
    ----------
    sequences : List[torch.Tensor]
        List of tensors with shape (time, features)

    Returns
    -------
    torch.Tensor
        Padded batch tensor (batch, max_time, features)
    """
    max_len = max(s.shape[0] for s in sequences)
    n_features = sequences[0].shape[1]
    batch = torch.zeros(len(sequences), max_len, n_features)
    for i, seq in enumerate(sequences):
        batch[i, :seq.shape[0], :] = seq
    return batch


def add_noise(
    signal: torch.Tensor,
    snr_db: float = 10.0
) -> torch.Tensor:
    """
    Add white noise at specified SNR level.

    Parameters
    ----------
    signal : torch.Tensor
        Input signal tensor
    snr_db : float
        Signal-to-noise ratio in decibels

    Returns
    -------
    torch.Tensor
        Noisy signal
    """
    signal_power = signal.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(signal) * noise_power.sqrt()
    return signal + noise


def create_model_comparison(
    model_names: List[str],
    results: List[dict]
) -> str:
    """
    Create formatted comparison table.

    Parameters
    ----------
    model_names : List[str]
        Names of models compared
    results : List[dict]
        List of result dicts with metrics

    Returns
    -------
    str
        Formatted markdown table
    """
    if not results:
        return "No results"

    # Get all metric keys
    all_metrics = set()
    for r in results:
        all_metrics.update(r.get('metrics', {}).keys())

    # Build table
    lines = []
    lines.append("| Model | " + " | ".join(sorted(all_metrics)) + " |")
    lines.append("|" + "---|" * (len(all_metrics) + 1))

    for name, result in zip(model_names, results):
        metrics = result.get('metrics', {})
        values = [f"{metrics.get(m, 'N/A'):.4f}" if isinstance(metrics.get(m), float) else str(metrics.get(m, 'N/A'))
                  for m in sorted(all_metrics)]
        lines.append(f"| {name} | " + " | ".join(values) + " |")

    return "\n".join(lines)
