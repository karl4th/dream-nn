"""
Test 2: Speaker Adaptation.

Based on DREAM Architecture Specification Section 7.2.2.

Tests the model's ability to adapt to speaker change mid-sequence.

Expected Results (Spec 7.5):
- DREAM: Adapts within <50 steps due to fast weights
- LSTM: Requires retraining or fine-tuning
- Transformer: No online adaptation capability

Run:
    uv run python tests/benchmarks/test_02_speaker_adaptation.py
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, List
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import load_audio_files, pad_sequences, BenchmarkResult
from benchmarks.models import create_model


def test_speaker_adaptation(
    model: nn.Module,
    model_name: str,
    speaker1_data: torch.Tensor,
    speaker2_data: torch.Tensor,
    device: str = 'cpu'
) -> dict:
    """
    Test speaker adaptation by switching speakers mid-sequence.

    For DREAM: Uses persistent state to track adaptation.
    For LSTM/Transformer: Tests reconstruction quality on each speaker.
    """
    model.eval()

    # Create combined sequence: speaker1 -> speaker2
    seq1 = speaker1_data[0:1, :200, :]  # (1, 200, 80)
    seq2 = speaker2_data[0:1, :200, :]  # (1, 200, 80)
    combined = torch.cat([seq1, seq2], dim=1).to(device)  # (1, 400, 80)

    switch_point = 200

    # Process sequence
    batch_size = 1
    if model_name == 'dream':
        state = model.init_state(batch_size, device=device)
    elif model_name == 'lstm':
        state = model.init_state(batch_size, device=device)
    else:
        state = None

    losses = []
    surprises = []  # Only for DREAM

    with torch.no_grad():
        for t in range(combined.shape[1]):
            x_t = combined[:, t:t+1, :]

            if model_name == 'dream':
                recon, state = model(x_t, state, return_all=False)
                surprise = state.avg_surprise.mean().item()
                surprises.append(surprise)
            elif model_name == 'lstm':
                recon, state = model(x_t, state, return_all=False)
            else:  # transformer
                recon, state = model(x_t, state, return_all=False)

            # Reconstruction error
            loss = (recon - x_t.squeeze(1)).pow(2).mean(dim=-1).sqrt()
            losses.append(loss.item())

    # Analyze adaptation
    pre_switch_losses = losses[:switch_point]
    post_switch_losses = losses[switch_point:]

    baseline_loss = np.mean(pre_switch_losses)
    max_post_loss = max(post_switch_losses)

    # Find adaptation point (when loss returns to baseline)
    adapted = False
    adaptation_steps = 0

    for i, loss in enumerate(post_switch_losses):
        if loss < baseline_loss * 1.5:  # Within 50% of baseline
            adapted = True
            adaptation_steps = i
            break

    # Surprise analysis (DREAM only)
    surprise_spike = 0.0
    if surprises:
        pre_surprise = np.mean(surprises[:switch_point])
        post_surprises = surprises[switch_point:switch_point+20]
        if post_surprises:
            surprise_spike = max(post_surprises) - pre_surprise

    return {
        'baseline_loss': baseline_loss,
        'max_post_switch_loss': max_post_loss,
        'adapted': adapted,
        'adaptation_steps': adaptation_steps,
        'surprise_spike': surprise_spike,
        'losses': losses,
        'surprises': surprises if surprises else None,
    }


def run_speaker_adaptation_test(
    audio_dir: str = 'audio_test',
    hidden_dim: int = 256,
    device: Optional[str] = None
) -> dict:
    """
    Run speaker adaptation test for all models.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("DREAM Benchmark - Test 2: Speaker Adaptation")
    print("=" * 70)

    # Load data
    print("\nLoading audio files...")
    features, names = load_audio_files(audio_dir)
    print(f"Loaded {len(features)} files: {names[:3]}...")

    train_data = pad_sequences(features[:9])
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
        else:
            model = create_model(model_name, input_dim=80, d_model=128, nhead=4, num_layers=4)

        model = model.to(device)

        # Test adaptation
        speaker1 = train_data[0:1]
        speaker2 = train_data[8:9]

        adapt_results = test_speaker_adaptation(
            model, model_name, speaker1, speaker2, device
        )

        # Success criteria
        if model_name == 'dream':
            # DREAM should adapt within 50 steps
            passed = adapt_results['adapted'] and adapt_results['adaptation_steps'] < 50
        else:
            # LSTM/Transformer: just check they can process both speakers
            passed = adapt_results['adapted']

        results[model_name] = {
            'passed': passed,
            'metrics': {
                'baseline_loss': adapt_results['baseline_loss'],
                'max_post_switch': adapt_results['max_post_switch_loss'],
                'adaptation_steps': adapt_results['adaptation_steps'],
                'surprise_spike': adapt_results['surprise_spike'],
            },
            'details': adapt_results,
        }

        print(f"\nResults:")
        print(f"  Baseline Loss:      {adapt_results['baseline_loss']:.4f}")
        print(f"  Max Post-Switch:    {adapt_results['max_post_switch_loss']:.4f}")
        print(f"  Adapted:            {adapt_results['adapted']}")
        print(f"  Adaptation Steps:   {adapt_results['adaptation_steps']}")
        if model_name == 'dream':
            print(f"  Surprise Spike:     {adapt_results['surprise_spike']:.3f}")
        print(f"  {'✅ PASSED' if passed else '❌ FAILED'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n| Model | Baseline | Max Post | Adapt Steps | Surprise |")
    print(f"|-------|----------|----------|-------------|----------|")

    for name in models_to_test:
        m = results[name]['metrics']
        status = '✅' if results[name]['passed'] else '❌'
        surprise_str = f"{m['surprise_spike']:.3f}" if m['surprise_spike'] else "N/A"
        print(f"| {name.upper():7} | {m['baseline_loss']:8.4f} | {m['max_post_switch']:8.4f} | "
              f"{m['adaptation_steps']:11d} | {surprise_str:>8} | {status}")

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    dream_steps = results['dream']['metrics']['adaptation_steps']
    print(f"DREAM adapts to speaker change in {dream_steps} steps")
    print("Expected: <50 steps (Spec 7.5)")
    print(f"Result: {'✅ MEETS SPEC' if dream_steps < 50 else '❌ EXCEEDS SPEC'}")

    # Save results
    output_file = Path(__file__).parent / 'results' / 'results_speaker_adaptation.json'
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

    parser = argparse.ArgumentParser(description='Run Speaker Adaptation Test')
    parser.add_argument('--audio-dir', type=str, default='audio_test',
                       help='Directory with audio files')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    results = run_speaker_adaptation_test(
        audio_dir=args.audio_dir,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
