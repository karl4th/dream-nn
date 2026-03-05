"""
Test 1: Basic ASR Reconstruction.

Based on DREAM Architecture Specification Section 7.2.1.

Tests the model's ability to memorize and reconstruct audio patterns.

Expected Results (Spec 7.5):
- DREAM: >50% loss improvement, fast convergence
- LSTM: Good reconstruction but slower adaptation
- Transformer: Best reconstruction with enough data, but no online adaptation

Run:
    uv run python tests/benchmarks/test_01_basic_asr.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
from typing import Optional
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import load_audio_files, pad_sequences, BenchmarkResult
from benchmarks.models import create_model, count_parameters


def train_model(
    model: nn.Module,
    model_name: str,
    train_data: torch.Tensor,
    n_epochs: int = 100,
    segment_size: int = 100,
    lr: float = 5e-3,
    device: str = 'cpu'
) -> dict:
    """
    Train model for reconstruction task.

    Uses truncated BPTT for memory efficiency.
    """
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    history = {'loss': []}

    print(f"\nTraining {model_name} on {train_data.shape}...")

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        n_segments = 0

        seq_len = train_data.shape[1]
        
        # Initialize state at start of epoch
        if model_name == 'dream':
            state = model.init_state(train_data.shape[0], device=device)
        elif model_name == 'lstm':
            state = model.init_state(train_data.shape[0], device=device)
        else:
            state = None

        for start in range(0, seq_len, segment_size):
            end = min(start + segment_size, seq_len)
            segment = train_data[:, start:end, :].to(device)

            if model_name == 'dream':
                recon, state = model(segment, state)
                # Detach state for truncated BPTT
                if isinstance(state, tuple):
                    state = tuple(s.detach() for s in state)
                else:
                    state = state.detach()
            elif model_name == 'lstm':
                recon, state = model(segment, state)
                # Detach hidden state for truncated BPTT
                if state is not None:
                    state = tuple(s.detach() for s in state)
            else:  # transformer
                recon, state = model(segment, state)

            loss = criterion(recon, segment)
            loss.backward()

            total_loss += loss.item()
            n_segments += 1

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_loss = total_loss / n_segments
        scheduler.step(avg_loss)
        history['loss'].append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.6f}")

    return history


def run_basic_asr_test(
    audio_dir: str = 'audio_test',
    hidden_dim: int = 256,
    n_epochs: int = 100,
    device: Optional[str] = None
) -> dict:
    """
    Run basic ASR reconstruction test for all models.

    Compares DREAM, LSTM, and Transformer on reconstruction quality.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("DREAM Benchmark - Test 1: Basic ASR Reconstruction")
    print("=" * 70)

    # Load data
    print("\nLoading audio files...")
    features, names = load_audio_files(audio_dir)
    print(f"Loaded {len(features)} files")

    train_data = pad_sequences(features[:9])  # Use 9 files for training
    print(f"Training data: {train_data.shape}")

    results = {}
    models_to_test = ['dream', 'lstm', 'transformer']

    for model_name in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing: {model_name.upper()}")
        print('='*50)

        # Create model
        if model_name == 'dream':
            model = create_model(model_name, input_dim=80, hidden_dim=hidden_dim, rank=16)
        elif model_name == 'lstm':
            model = create_model(model_name, input_dim=80, hidden_dim=hidden_dim, num_layers=2)
        else:  # transformer
            model = create_model(model_name, input_dim=80, d_model=128, nhead=4, num_layers=4)

        model = model.to(device)

        # Count parameters
        n_params = count_parameters(model)
        print(f"Parameters: {n_params:,}")

        # Train
        start_time = time.time()
        history = train_model(
            model, model_name, train_data,
            n_epochs=n_epochs,
            segment_size=100,
            lr=5e-3 if model_name == 'dream' else 1e-3,
            device=device
        )
        train_time = time.time() - start_time

        # Compute metrics
        initial_loss = history['loss'][0]
        final_loss = history['loss'][-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100

        # Success criteria
        passed = final_loss < initial_loss * 0.5  # >50% improvement

        results[model_name] = {
            'passed': passed,
            'metrics': {
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'improvement_pct': improvement,
                'train_time_s': train_time,
                'n_params': n_params,
            },
            'history': history,
        }

        print(f"\nResults:")
        print(f"  Initial Loss: {initial_loss:.6f}")
        print(f"  Final Loss:   {final_loss:.6f}")
        print(f"  Improvement:  {improvement:.1f}%")
        print(f"  Train Time:   {train_time:.1f}s")
        print(f"  {'✅ PASSED' if passed else '❌ FAILED'}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n| Model | Initial Loss | Final Loss | Improvement | Time |")
    print(f"|-------|--------------|------------|-------------|------|")

    for name in models_to_test:
        m = results[name]['metrics']
        status = '✅' if results[name]['passed'] else '❌'
        print(f"| {name.upper():7} | {m['initial_loss']:12.4f} | {m['final_loss']:10.4f} | "
              f"{m['improvement_pct']:10.1f}% | {m['train_time_s']:4.0f}s | {status}")

    # Save results
    output_file = Path(__file__).parent / 'results' / 'results_basic_asr.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_results = {
        name: {
            'passed': data['passed'],
            'metrics': data['metrics']
        }
        for name, data in results.items()
    }
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Basic ASR Reconstruction Test')
    parser.add_argument('--audio-dir', type=str, default='audio_test',
                       help='Directory with audio files')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    results = run_basic_asr_test(
        audio_dir=args.audio_dir,
        hidden_dim=args.hidden_dim,
        n_epochs=args.epochs,
        device=args.device,
    )
