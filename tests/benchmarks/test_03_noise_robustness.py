"""
Test 3: Noise Robustness.

Based on DREAM Architecture Specification Section 7.2.3.

Tests the model's robustness to additive noise at different SNR levels.

Expected Results (Spec 7.5):
- DREAM: Surprise gate filters constant noise, stable performance
- LSTM: Degrades gracefully with noise
- Transformer: May overfit to clean data, worse on noisy

Run:
    uv run python tests/benchmarks/test_03_noise_robustness.py
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, List
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import load_audio_files, pad_sequences, add_noise, BenchmarkResult
from benchmarks.models import create_model


def test_noise_robustness(
    model: nn.Module,
    model_name: str,
    test_data: torch.Tensor,
    snr_levels: List[float] = [20, 10, 5, 0],
    device: str = 'cpu'
) -> dict:
    """
    Test model robustness to additive noise.

    Measures reconstruction loss and surprise response at each SNR level.
    """
    model.eval()
    criterion = nn.MSELoss()

    results = {}

    with torch.no_grad():
        for snr in snr_levels:
            noisy_data = add_noise(test_data, snr)
            noisy_data = noisy_data.to(device)

            batch_size = noisy_data.shape[0]

            # Initialize state
            if model_name == 'dream':
                state = model.init_state(batch_size, device=device)
            elif model_name == 'lstm':
                state = model.init_state(batch_size, device=device)
            else:
                state = None

            # Track metrics during processing
            surprises_per_step = []
            losses_per_step = []

            for t in range(noisy_data.shape[1]):
                x_t = noisy_data[:, t:t+1, :]

                if model_name == 'dream':
                    recon, state = model(x_t, state, return_all=False)
                    surprise = state.avg_surprise.mean().item()
                    surprises_per_step.append(surprise)
                elif model_name == 'lstm':
                    recon, state = model(x_t, state, return_all=False)
                else:
                    recon, state = model(x_t, state, return_all=False)

                # Per-step loss
                loss = (recon - x_t.squeeze(1)).pow(2).mean(dim=-1).sqrt()
                losses_per_step.append(loss.item())

            # Compute full sequence reconstruction loss
            if model_name == 'dream':
                state = model.init_state(batch_size, device=device)
                recon, _ = model(noisy_data, state)
            elif model_name == 'lstm':
                state = model.init_state(batch_size, device=device)
                recon, _ = model(noisy_data, state)
            else:
                recon, _ = model(noisy_data)

            full_loss = criterion(recon, noisy_data).item()

            # Aggregate metrics
            if surprises_per_step:
                max_surprise = max(surprises_per_step)
                avg_surprise = np.mean(surprises_per_step)
            else:
                max_surprise = 0.0
                avg_surprise = 0.0

            results[snr] = {
                'full_loss': full_loss,
                'avg_step_loss': np.mean(losses_per_step),
                'max_step_loss': max(losses_per_step),
                'max_surprise': max_surprise,
                'avg_surprise': avg_surprise,
            }

    return results


def run_noise_robustness_test(
    audio_dir: str = 'audio_test',
    hidden_dim: int = 256,
    device: Optional[str] = None
) -> dict:
    """
    Run noise robustness test for all models.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("DREAM Benchmark - Test 3: Noise Robustness")
    print("=" * 70)

    # Load data
    print("\nLoading audio files...")
    features, names = load_audio_files(audio_dir)
    print(f"Loaded {len(features)} files")

    # Use last file for testing
    test_data = pad_sequences(features[9:10])
    print(f"Test data: {test_data.shape}")

    results = {}
    models_to_test = ['dream', 'lstm', 'transformer']
    snr_levels = [20, 10, 5, 0]

    for model_name in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing: {model_name.upper()}")
        print('='*50)

        # Create model
        if model_name == 'dream':
            model = create_model(model_name, input_dim=80, hidden_dim=hidden_dim, rank=16)
        elif model_name == 'lstm':
            model = create_model(model_name, input_dim=80, hidden_dim=hidden_dim, num_layers=2)
        else:
            model = create_model(model_name, input_dim=80, d_model=128, nhead=4, num_layers=4)

        model = model.to(device)

        # Test noise robustness
        noise_results = test_noise_robustness(
            model, model_name, test_data, snr_levels, device
        )

        # Print results
        print(f"\nResults by SNR:")
        print(f"  SNR | Loss    | Max Loss | Max Surprise")
        print(f"  ----|---------|----------|-------------")

        for snr in snr_levels:
            r = noise_results[snr]
            surprise_str = f"{r['max_surprise']:.3f}" if r['max_surprise'] > 0 else "N/A"
            print(f"  {snr:3d} | {r['full_loss']:7.4f} | {r['max_step_loss']:8.4f} | {surprise_str:>12}")

        # Success criteria
        clean_loss = noise_results[20]['full_loss']
        noisy_loss = noise_results[10]['full_loss']

        # Check if surprise responds to noise (DREAM only)
        surprise_responds = False
        if model_name == 'dream':
            # Check if surprise increases with noise level
            clean_surprise = noise_results[20]['max_surprise']
            noisy_surprise = noise_results[0]['max_surprise']  # Compare with highest noise
            # Surprise should increase as noise increases
            surprise_responds = noisy_surprise > clean_surprise * 1.05  # At least 5% increase

        # Loss should not explode at moderate noise
        loss_stable = noisy_loss < clean_loss * 3.0

        passed = loss_stable or surprise_responds

        results[model_name] = {
            'passed': passed,
            'metrics': {
                'clean_loss': clean_loss,
                'noisy_loss_10db': noisy_loss,
                'loss_ratio': noisy_loss / (clean_loss + 1e-6),
                'surprise_responds': surprise_responds,
            },
            'full_results': noise_results,
        }

        print(f"\n  Clean Loss:     {clean_loss:.4f}")
        print(f"  Noisy Loss:     {noisy_loss:.4f}")
        print(f"  Loss Ratio:     {noisy_loss / (clean_loss + 1e-6):.2f}x")
        if model_name == 'dream':
            print(f"  Surprise Response: {'✅ Yes' if surprise_responds else '❌ No'}")
        print(f"  {'✅ PASSED' if passed else '❌ FAILED'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n| Model | Clean Loss | 10dB Loss | Ratio | Surprise |")
    print(f"|-------|------------|-----------|-------|----------|")

    for name in models_to_test:
        m = results[name]['metrics']
        status = '✅' if results[name]['passed'] else '❌'
        surprise_str = '✅' if m.get('surprise_responds', False) else 'N/A'
        print(f"| {name.upper():7} | {m['clean_loss']:10.4f} | {m['noisy_loss_10db']:9.4f} | "
              f"{m['loss_ratio']:5.2f}x | {surprise_str:>8} | {status}")

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)

    dream_responds = results['dream']['metrics'].get('surprise_responds', False)
    dream_ratio = results['dream']['metrics']['loss_ratio']

    print(f"DREAM noise response:")
    print(f"  - Loss ratio (10dB/clean): {dream_ratio:.2f}x")
    print(f"  - Surprise detects noise:  {'✅ Yes' if dream_responds else '❌ No'}")

    if dream_responds:
        print("\n✅ Surprise gate successfully detects and responds to noise!")
    else:
        print("\n⚠️ Surprise gate could be more sensitive to noise")

    # Save results
    output_file = Path(__file__).parent / 'results' / 'results_noise_robustness.json'
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

    parser = argparse.ArgumentParser(description='Run Noise Robustness Test')
    parser.add_argument('--audio-dir', type=str, default='audio_test',
                       help='Directory with audio files')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    results = run_noise_robustness_test(
        audio_dir=args.audio_dir,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
