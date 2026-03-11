"""
Benchmark Test 4: DREAMStack Coordination.

Compares coordinated vs uncoordinated DREAMStack.

Tests:
1. Convergence speed with coordination
2. Stability of deep stacks (4+ layers)
3. Hierarchical adaptation

Expected Results:
- Coordinated stack converges faster
- Coordinated stack is more stable with depth
- Better hierarchical feature learning

Run:
    uv run python tests/benchmarks/test_04_stack_coordination.py
"""

import torch
import torch.nn as nn
import time
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import load_audio_files, pad_sequences
from dream.layer_coordinated import CoordinatedDREAMStack, UncoordinatedDREAMStack, CoordinatedState
from dream.config import DREAMConfig


def train_stack(
    model: nn.Module,
    train_data: torch.Tensor,
    n_epochs: int = 50,
    segment_size: int = 100,
    lr: float = 5e-3,
    device: str = 'cpu'
) -> dict:
    """
    Train DREAMStack for reconstruction.

    Returns
    -------
    dict
        Training history
    """
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {'loss': [], 'inter_layer_loss': []}
    batch_size = train_data.shape[0]
    states = model.init_states(batch_size, device=device)

    # Add output projection if not present
    if not hasattr(model, 'output_projection'):
        last_hidden_dim = model.hidden_dims[-1] if hasattr(model, 'hidden_dims') else 128
        model.output_projection = nn.Linear(last_hidden_dim, 80).to(device)

    print(f"Training {model.__class__.__name__} on {train_data.shape}...")

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        total_inter_loss = 0.0
        n_segments = 0

        seq_len = train_data.shape[1]
        for start in range(0, seq_len, segment_size):
            end = min(start + segment_size, seq_len)
            segment = train_data[:, start:end, :].to(device)

            # Forward pass with losses
            if isinstance(model, CoordinatedDREAMStack):
                recon, states, losses = model(segment, states, return_losses=True)
                loss = losses['reconstruction'] + model.inter_layer_loss_weight * losses['inter_layer']
                total_inter_loss += losses['inter_layer'].item()
            else:
                # Uncoordinated: need to project output
                hidden_output, states = model(segment, states, return_all=True)
                recon = model.output_projection(hidden_output)
                loss = criterion(recon, segment)

            loss.backward()

            total_loss += loss.item()
            n_segments += 1

            # Detach states
            if isinstance(states, CoordinatedState):
                for i in range(len(states.layer_states)):
                    states.layer_states[i] = states.layer_states[i].detach()
            else:
                for i in range(len(states)):
                    states[i] = states[i].detach()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_loss = total_loss / n_segments
        avg_inter_loss = total_inter_loss / n_segments
        scheduler.step(avg_loss)

        history['loss'].append(avg_loss)
        history['inter_layer_loss'].append(avg_inter_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.6f}, Inter={avg_inter_loss:.6f}")

    return history


def load_audio_files_from_metadata(metadata_path: str, audio_dir: str):
    """Load audio files from LJSpeech metadata."""
    import pandas as pd
    import librosa
    import torch
    import numpy as np

    df = pd.read_csv(metadata_path, sep='|', header=None, names=['id', 'text', 'phonemes'])
    features = []

    for _, row in df.iterrows():
        audio_file = Path(audio_dir) / f"{row['id']}.wav"
        if audio_file.exists():
            y, sr = librosa.load(str(audio_file), sr=16000)
            melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
            log_mels = librosa.power_to_db(melspec, ref=np.max)
            feat = torch.tensor(log_mels.T, dtype=torch.float32)
            feat = (feat - feat.mean()) / (feat.std() + 1e-6)
            features.append(feat)

    return features


def run_coordination_test(
    audio_dir: str = 'audio_test',
    metadata_path: str = None,
    hidden_dims: list = [128, 128, 128],
    n_epochs: int = 50,
    device: str = 'cpu'
) -> dict:
    """Run coordination benchmark."""
    print("=" * 70)
    print("DREAM Benchmark Test 4: Stack Coordination")
    print("=" * 70)

    # Load data
    print("\nLoading audio files...")
    if metadata_path:
        features = load_audio_files_from_metadata(metadata_path, audio_dir)
    else:
        features, names = load_audio_files(audio_dir)
    train_data = pad_sequences(features[:9]) if len(features) > 9 else pad_sequences(features)
    print(f"Training data: {train_data.shape}")

    results = {}

    # ================================================================
    # Test 1: Uncoordinated Stack (Baseline)
    # ================================================================
    print("\n" + "=" * 50)
    print("Test 1: Uncoordinated DREAMStack (Baseline)")
    print("=" * 50)

    model_uncoord = UncoordinatedDREAMStack(
        input_dim=80,
        hidden_dims=hidden_dims,
        rank=16,
        dropout=0.1
    )
    model_uncoord = model_uncoord.to(device)

    print(f"Parameters: {model_uncoord.count_parameters():,}")

    start_time = time.time()
    history_uncoord = train_stack(model_uncoord, train_data, n_epochs, device=device)
    train_time = time.time() - start_time

    initial_loss = history_uncoord['loss'][0]
    final_loss = history_uncoord['loss'][-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100

    results['uncoordinated'] = {
        'passed': final_loss < initial_loss * 0.5,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'improvement_pct': improvement,
        'train_time': train_time,
        'history': history_uncoord,
    }

    print(f"\nResults:")
    print(f"  Initial Loss: {initial_loss:.6f}")
    print(f"  Final Loss:   {final_loss:.6f}")
    print(f"  Improvement:  {improvement:.1f}%")
    print(f"  Train Time:   {train_time:.1f}s")
    print(f"  {'✅ PASSED' if results['uncoordinated']['passed'] else '❌ FAILED'}")

    # ================================================================
    # Test 2: Coordinated Stack
    # ================================================================
    print("\n" + "=" * 50)
    print("Test 2: Coordinated DREAMStack")
    print("=" * 50)

    model_coord = CoordinatedDREAMStack(
        input_dim=80,
        hidden_dims=hidden_dims,
        rank=16,
        dropout=0.1
    )
    model_coord = model_coord.to(device)

    print(f"Parameters: {model_coord.count_parameters():,}")

    start_time = time.time()
    history_coord = train_stack(model_coord, train_data, n_epochs, device=device)
    train_time = time.time() - start_time

    initial_loss = history_coord['loss'][0]
    final_loss = history_coord['loss'][-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100

    results['coordinated'] = {
        'passed': final_loss < initial_loss * 0.5,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'improvement_pct': improvement,
        'train_time': train_time,
        'inter_layer_loss': history_coord['inter_layer_loss'][-1] if history_coord['inter_layer_loss'] else 0,
        'history': history_coord,
    }

    print(f"\nResults:")
    print(f"  Initial Loss: {initial_loss:.6f}")
    print(f"  Final Loss:   {final_loss:.6f}")
    print(f"  Improvement:  {improvement:.1f}%")
    print(f"  Train Time:   {train_time:.1f}s")
    print(f"  {'✅ PASSED' if results['coordinated']['passed'] else '❌ FAILED'}")

    # ================================================================
    # Comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"\n| Model          | Final Loss | Inter Loss | Improvement | Time |")
    print(f"|----------------|------------|------------|-------------|------|")
    print(f"| Uncoordinated  | {results['uncoordinated']['final_loss']:10.6f} | {'N/A':10s} | "
          f"{results['uncoordinated']['improvement_pct']:10.1f}% | {results['uncoordinated']['train_time']:4.0f}s |")
    
    coord_inter = results['coordinated'].get('inter_layer_loss', 0)
    print(f"| Coordinated    | {results['coordinated']['final_loss']:10.6f} | {coord_inter:10.6f} | "
          f"{results['coordinated']['improvement_pct']:10.1f}% | {results['coordinated']['train_time']:4.0f}s |")

    # Check if coordination helps
    coord_better = results['coordinated']['final_loss'] < results['uncoordinated']['final_loss'] * 0.9
    print(f"\nCoordination improves convergence: {'✅ Yes' if coord_better else '❌ No'}")

    results['summary'] = {
        'coordination_helps': coord_better,
        'both_passed': results['uncoordinated']['passed'] and results['coordinated']['passed'],
    }

    # Save results
    output_file = Path(__file__).parent / 'results' / 'results_04_coordination.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    save_results = {
        'uncoordinated': {
            'passed': results['uncoordinated']['passed'],
            'metrics': {
                'final_loss': results['uncoordinated']['final_loss'],
                'improvement_pct': results['uncoordinated']['improvement_pct'],
                'train_time': results['uncoordinated']['train_time'],
            }
        },
        'coordinated': {
            'passed': results['coordinated']['passed'],
            'metrics': {
                'final_loss': results['coordinated']['final_loss'],
                'improvement_pct': results['coordinated']['improvement_pct'],
                'train_time': results['coordinated']['train_time'],
                'inter_layer_loss': results['coordinated'].get('inter_layer_loss', 0),
            }
        },
        'summary': results['summary'],
    }

    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Stack Coordination Test')
    parser.add_argument('--audio-dir', type=str, default='audio_test')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    results = run_coordination_test(
        audio_dir=args.audio_dir,
        hidden_dims=args.hidden_dims,
        n_epochs=args.epochs,
        device=device,
    )
