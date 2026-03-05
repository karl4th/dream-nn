"""
Overfitting test for DREAM cell on mel spectrogram data.

This test verifies that DREAM can memorize audio patterns by:
1. Training on 9 audio files (reconstruction task)
2. Testing on 1 held-out file
3. Monitoring loss decrease and U norm growth

Uses mel spectrograms (80 bins) for audio representation.

Run in Google Colab:
    !pip install dreamnn librosa
    !python tests/test_001_overfit.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple, Optional
import time

from dream import DREAMConfig, DREAMCell, DREAM, DREAMState


def generate_synthetic_melspec(
    n_files: int = 10,
    seq_len: int = 500,
    n_mels: int = 80,
    seed: int = 42
) -> torch.Tensor:
    """
    Generate synthetic mel spectrogram-like data for testing.

    Each file has unique patterns to simulate different speakers.

    Parameters
    ----------
    n_files : int
        Number of audio files to generate
    seq_len : int
        Sequence length (time frames)
    n_mels : int
        Number of mel bins (default 80)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    torch.Tensor
        Mel spectrogram data of shape (n_files, seq_len, n_mels)
    """
    torch.manual_seed(seed)
    data = torch.zeros(n_files, seq_len, n_mels)

    for i in range(n_files):
        # Each file has unique base pattern (different "speaker")
        base_freq = 0.5 + i * 0.1
        t = torch.linspace(0, 10 * (i + 1), seq_len)

        # Create structured patterns across mel bins
        # Lower bins = lower frequencies = more energy
        for j in range(n_mels):
            freq = base_freq * (1 + j * 0.02)
            # Log-scale energy distribution (like real speech)
            energy_decay = 1.0 / (1 + j * 0.05)
            data[i, :, j] = (
                torch.sin(freq * t) * energy_decay * 0.5 +
                torch.sin(2 * freq * t) * energy_decay * 0.3 +
                torch.randn(seq_len) * 0.1 * energy_decay
            )

        # Convert to log scale (like real mel spectrogram)
        data[i] = torch.log1p(torch.abs(data[i]))

        # Normalize per file
        data[i] = (data[i] - data[i].mean()) / (data[i].std() + 1e-6)

    return data


def load_real_audio_melspec(
    file_paths: list,
    target_sr: int = 16000,
    n_mels: int = 80,
    hop_length: int = 256,
    n_fft: int = 1024
) -> torch.Tensor:
    """
    Load real audio files and extract mel spectrogram features.

    Parameters
    ----------
    file_paths : list
        List of paths to .wav files
    target_sr : int
        Target sample rate
    n_mels : int
        Number of mel bins
    hop_length : int
        Hop length for STFT
    n_fft : int
        FFT window size

    Returns
    -------
    torch.Tensor
        Mel spectrogram features (n_files, max_seq_len, n_mels)
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        raise ImportError("Install librosa: pip install librosa numpy")

    features_list = []

    for path in file_paths:
        # Load audio
        y, sr = librosa.load(path, sr=target_sr)

        # Extract mel spectrogram
        melspec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
        )

        # Convert to log scale
        log_melspec = librosa.power_to_db(melspec, ref=np.max)

        # Transpose to (time, n_mels)
        features = torch.tensor(log_melspec.T, dtype=torch.float32)

        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-6)
        features_list.append(features)

    # Pad to max length
    max_len = max(f.shape[0] for f in features_list)
    padded = torch.zeros(len(features_list), max_len, features_list[0].shape[1])

    for i, f in enumerate(features_list):
        padded[i, :f.shape[0], :] = f

    return padded


class DREAMReconstructor(nn.Module):
    """
    Simple reconstruction model using DREAM.

    Takes input sequence and tries to reconstruct it.
    """

    def __init__(
        self,
        input_dim: int = 39,
        hidden_dim: int = 256,
        rank: int = 16,
        **kwargs
    ):
        super().__init__()
        self.config = DREAMConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            **kwargs
        )
        self.cell = DREAMCell(self.config)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[DREAMState] = None
    ) -> Tuple[torch.Tensor, DREAMState]:
        """
        Process sequence and reconstruct input.

        Parameters
        ----------
        x : torch.Tensor
            Input (batch, time, input_dim)
        state : DREAMState, optional
            Initial state

        Returns
        -------
        reconstruction : torch.Tensor
            Reconstructed input (batch, time, input_dim)
        state : DREAMState
            Final state
        """
        batch_size, time_steps, _ = x.shape

        if state is None:
            state = self.cell.init_state(batch_size, device=x.device, dtype=x.dtype)

        outputs = []

        for t in range(time_steps):
            x_t = x[:, t, :]  # (batch, input_dim)
            h, state = self.cell(x_t, state)
            recon_t = self.decoder(h)  # (batch, input_dim)
            outputs.append(recon_t.unsqueeze(1))

        reconstruction = torch.cat(outputs, dim=1)
        return reconstruction, state

    def init_state(self, batch_size: int, **kwargs) -> DREAMState:
        """Initialize model state."""
        return self.cell.init_state(batch_size, **kwargs)


def train_epoch(
    model: DREAMReconstructor,
    train_data: torch.Tensor,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    segment_size: int = 100,
    device: str = 'cpu'
) -> Tuple[float, float, float, float]:
    """
    Train model for one epoch using truncated BPTT.

    Parameters
    ----------
    model : DREAMReconstructor
        Model to train
    train_data : torch.Tensor
        Training data (n_files, seq_len, input_dim)
    criterion : nn.Module
        Loss function
    optimizer : optim.Optimizer
        Optimizer
    segment_size : int
        Segment size for truncated BPTT
    device : str
        Device to use

    Returns
    -------
    avg_loss : float
        Average reconstruction loss
    avg_u_norm : float
        Average U matrix norm
    avg_u_target_norm : float
        Average U_target norm
    avg_surprise : float
        Average surprise value
    """
    model.train()
    total_loss = 0.0
    n_segments = 0

    # Initialize state ONCE (preserve between segments/epochs)
    batch_size = train_data.shape[0]
    state = model.init_state(batch_size, device=device)

    seq_len = train_data.shape[1]

    # Process in segments
    for start in range(0, seq_len, segment_size):
        end = min(start + segment_size, seq_len)
        segment = train_data[:, start:end, :].to(device)

        optimizer.zero_grad()

        # Forward through segment
        reconstruction, state = model(segment, state)

        # Compute loss
        loss = criterion(reconstruction, segment)

        # Backward
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_segments += 1

        # Detach state between segments (truncated BPTT)
        state = state.detach()

    # Compute metrics
    avg_loss = total_loss / n_segments

    # Get U norms from state
    u_norm = state.U.norm(dim=(1, 2)).mean().item()
    u_target_norm = state.U_target.norm(dim=(1, 2)).mean().item()
    avg_surprise_val = state.avg_surprise.mean().item() if state.avg_surprise.dim() > 0 else state.avg_surprise.item()

    return avg_loss, u_norm, u_target_norm, avg_surprise_val


def evaluate(
    model: DREAMReconstructor,
    test_data: torch.Tensor,
    criterion: nn.Module,
    device: str = 'cpu'
) -> float:
    """
    Evaluate model on test data.

    Parameters
    ----------
    model : DREAMReconstructor
        Model to evaluate
    test_data : torch.Tensor
        Test data (1, seq_len, input_dim)
    criterion : nn.Module
        Loss function
    device : str
        Device to use

    Returns
    -------
    test_loss : float
        Test reconstruction loss
    """
    model.eval()
    batch_size = test_data.shape[0]
    state = model.init_state(batch_size, device=device)

    with torch.no_grad():
        reconstruction, _ = model(test_data.to(device), state)
        loss = criterion(reconstruction, test_data.to(device))

    return loss.item()


def run_overfit_test(
    n_train_files: int = 9,
    n_test_files: int = 1,
    seq_len: int = 500,
    n_mels: int = 80,
    hidden_dim: int = 256,
    rank: int = 16,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    segment_size: int = 100,
    use_synthetic: bool = True,
    audio_paths: Optional[list] = None,
    device: Optional[str] = None
) -> dict:
    """
    Run overfitting test for DREAM.

    Parameters
    ----------
    n_train_files : int
        Number of training files
    n_test_files : int
        Number of test files
    seq_len : int
        Sequence length for synthetic data
    n_mels : int
        Number of mel bins (default 80)
    hidden_dim : int
        Hidden dimension
    rank : int
        Fast weights rank
    n_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate
    segment_size : int
        Segment size for truncated BPTT
    use_synthetic : bool
        Use synthetic data if True, real audio if False
    audio_paths : list, optional
        Paths to real audio files (if use_synthetic=False)
    device : str, optional
        Device to use (default: cuda if available)

    Returns
    -------
    results : dict
        Training history and metrics
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Generate/load data
    print(f"\nGenerating {'synthetic' if use_synthetic else 'real'} mel spectrogram data...")

    if use_synthetic:
        all_data = generate_synthetic_melspec(
            n_files=n_train_files + n_test_files,
            seq_len=seq_len,
            n_mels=n_mels
        )
    else:
        if audio_paths is None or len(audio_paths) < n_train_files + n_test_files:
            raise ValueError(f"Need at least {n_train_files + n_test_files} audio files")
        all_data = load_real_audio_melspec(audio_paths[:n_train_files + n_test_files])

    # Set input_dim based on data
    input_dim = n_mels

    # Split train/test
    train_data = all_data[:n_train_files]
    test_data = all_data[n_train_files:n_train_files + n_test_files]

    print(f"Train: {train_data.shape}, Test: {test_data.shape}")

    # Create model
    model = DREAMReconstructor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        rank=rank,
        forgetting_rate=0.005,
        base_plasticity=0.5,
        base_threshold=0.3,
        entropy_influence=0.1,
        surprise_temperature=0.05,
        error_smoothing=0.05,
        surprise_smoothing=0.05,
        ltc_enabled=True,
        ltc_tau_sys=5.0,
        ltc_surprise_scale=5.0,
        kappa=0.5,
        sleep_rate=0.01,
        min_surprise_for_sleep=0.15,
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 80)

    history = {
        'train_loss': [],
        'test_loss': [],
        'u_norm': [],
        'u_target_norm': [],
        'surprise': [],
    }

    # State persistence: initialize once
    train_state = model.init_state(n_train_files, device=device)

    start_time = time.time()

    for epoch in range(n_epochs):
        # Train
        train_loss, u_norm, u_target_norm, surprise = train_epoch(
            model, train_data, criterion, optimizer, segment_size, device
        )

        # Evaluate
        test_loss = evaluate(model, test_data, criterion, device)

        # Update scheduler
        scheduler.step(train_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['u_norm'].append(u_norm)
        history['u_target_norm'].append(u_target_norm)
        history['surprise'].append(surprise)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:4d}/{n_epochs}: "
                f"Train Loss={train_loss:8.4f}, "
                f"Test Loss={test_loss:8.4f}, "
                f"U={u_norm:6.3f}±{u_target_norm:6.3f}, "
                f"τ={surprise:6.3f}"
            )

    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"Training completed in {elapsed:.1f}s")

    # Final metrics
    initial_loss = history['train_loss'][0]
    final_loss = history['train_loss'][-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100

    print(f"\nResults:")
    print(f"  Initial Loss: {initial_loss:.4f}")
    print(f"  Final Loss:   {final_loss:.4f}")
    print(f"  Improvement:  {improvement:.1f}%")
    print(f"  Final U norm: {history['u_norm'][-1]:.4f}")

    # Success criteria
    success = final_loss < initial_loss * 0.5  # 50% improvement
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: "
          f"Model {'can' if success else 'cannot'} memorize patterns")

    return {
        'history': history,
        'final_train_loss': final_loss,
        'final_test_loss': history['test_loss'][-1],
        'improvement_pct': improvement,
        'success': success,
        'model': model,
    }


if __name__ == '__main__':
    # Run test with synthetic mel spectrogram data (no external dependencies)
    results = run_overfit_test(
        n_train_files=9,
        n_test_files=1,
        seq_len=500,
        n_mels=80,
        hidden_dim=256,
        rank=16,
        n_epochs=200,
        learning_rate=5e-3,
        segment_size=100,
        use_synthetic=True,
    )

    # For real audio, uncomment and provide paths:
    # audio_files = [
    #     'audio/file1.wav',
    #     'audio/file2.wav',
    #     ...
    # ]
    # results = run_overfit_test(
    #     n_train_files=9,
    #     n_test_files=1,
    #     audio_paths=audio_files,
    #     use_synthetic=False,
    #     n_epochs=100,
    # )
