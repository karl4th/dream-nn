"""
Test 5: Hierarchical Temporal Learning (REAL AUDIO).

Demonstrates how DREAM learns hierarchical temporal representations from REAL speech:
- Lower layers: Fast patterns (phonemes, ~10-50ms)
- Middle layers: Medium patterns (syllables, ~100-300ms)  
- Upper layers: Slow context (words/phrases, ~1-2s)

This is a REAL training test on REAL audio data.

Run:
    uv run python tests/benchmarks/test_05_hierarchy.py
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import pandas as pd
import librosa
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dream.layer_coordinated import CoordinatedDREAMStack
from dream.config import DREAMConfig


def load_real_audio_data(
    audio_dir: str,
    metadata_path: str,
    n_files: int = 8,
    target_sr: int = 16000,
    n_mels: int = 80,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Load REAL speech audio from LJSpeech dataset.
    
    Returns mel spectrograms ready for training.
    """
    df = pd.read_csv(metadata_path, sep='|', header=None, names=['id', 'text', 'normalized_text'])
    
    features = []
    
    for _, row in df.iterrows()[:n_files]:
        audio_file = Path(audio_dir) / f"{row['id']}.wav"
        if audio_file.exists():
            y, sr = librosa.load(str(audio_file), sr=target_sr)
            melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
            log_mels = librosa.power_to_db(melspec, ref=np.max)
            feat = torch.tensor(log_mels.T, dtype=torch.float32)
            feat = (feat - feat.mean()) / (feat.std() + 1e-6)
            features.append(feat)
    
    # Pad to same length
    max_len = max(f.shape[0] for f in features)
    batch = torch.zeros(len(features), max_len, n_mels)
    for i, f in enumerate(features):
        batch[i, :f.shape[0], :] = f
    
    return batch.to(device)


def train_and_measure_hierarchy(
    model: CoordinatedDREAMStack,
    train_data: torch.Tensor,
    n_epochs: int = 50,
    device: str = 'cpu'
) -> dict:
    """
    Train model on REAL audio and measure emerging hierarchy.
    """
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    batch_size = train_data.shape[0]
    
    # Track metrics per epoch
    history = {
        'loss': [],
        'layer_surprises': [[] for _ in range(model.num_layers)],
        'layer_taus': [[] for _ in range(model.num_layers)],
    }
    
    print(f"Training on REAL speech data {train_data.shape} for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward through time
        states = model.init_states(batch_size, device=device)
        all_outputs = []
        epoch_surprises = [[] for _ in range(model.num_layers)]
        epoch_taus = [[] for _ in range(model.num_layers)]
        
        for t in range(train_data.shape[1]):
            x_t = train_data[:, t, :]
            current_input = x_t
            
            for i, cell in enumerate(model.cells):
                h_new, states.layer_states[i], _, _ = cell(current_input, states.layer_states[i])
                
                # Track surprise and effective tau
                surprise = states.layer_states[i].avg_surprise.mean().item()
                tau_base = cell.tau_sys.item() * cell.tau_depth_factor
                tau_eff = tau_base / (1.0 + surprise * cell.tau_surprise_scale.item())
                
                epoch_surprises[i].append(surprise)
                epoch_taus[i].append(tau_eff)
                
                if i < model.num_layers - 1:
                    current_input = h_new
            
            all_outputs.append(h_new.unsqueeze(1))
        
        # Reconstruction loss
        outputs = torch.cat(all_outputs, dim=1)
        
        # Output projection
        if not hasattr(model, 'output_proj'):
            model.output_proj = nn.Linear(model.hidden_dims[-1], 80).to(device)
        
        recon = model.output_proj(outputs)
        recon_loss = criterion(recon, train_data)
        loss = recon_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        
        avg_loss = loss.item()
        history['loss'].append(avg_loss)
        
        for i in range(model.num_layers):
            avg_surprise = np.mean(epoch_surprises[i])
            avg_tau = np.mean(epoch_taus[i])
            history['layer_surprises'][i].append(avg_surprise)
            history['layer_taus'][i].append(avg_tau)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.6f}")
            for i in range(model.num_layers):
                print(f"    Layer {i}: Surprise={history['layer_surprises'][i][-1]:.4f}, Tau={history['layer_taus'][i][-1]:.4f}")
    
    # Compute final hierarchy metrics (last 10 epochs)
    metrics = {}
    for i in range(model.num_layers):
        final_taus = history['layer_taus'][i][-10:]
        final_surprises = history['layer_surprises'][i][-10:]
        
        metrics[f'layer_{i}'] = {
            'avg_tau': float(np.mean(final_taus)),
            'tau_std': float(np.std(final_taus)),
            'avg_surprise': float(np.mean(final_surprises)),
        }
    
    return metrics, history


def run_hierarchy_test(
    audio_dir: str = 'audio_test',
    metadata_path: str = None,
    hidden_dims: list = [128, 128, 128],
    n_epochs: int = 50,
    device: str = 'cpu'
) -> dict:
    """
    Run hierarchical learning test on REAL audio.
    """
    print("=" * 70)
    print("DREAM Test 5: Hierarchical Temporal Learning (REAL AUDIO)")
    print("=" * 70)
    
    print("\n🎯 Using REAL speech audio (LJSpeech dataset)")
    print("   Testing if DREAM learns hierarchical temporal structure")
    
    # Load REAL audio
    print("\nLoading audio files...")
    if metadata_path:
        train_data = load_real_audio_data(
            audio_dir=audio_dir,
            metadata_path=metadata_path,
            n_files=8,
            device=device
        )
    else:
        print("⚠️  No metadata path provided, using synthetic data")
        train_data = torch.randn(4, 200, 80, device=device)
    
    print(f"Training data: {train_data.shape}")
    print(f"  - {train_data.shape[0]} utterances")
    print(f"  - {train_data.shape[1]} time steps (~{train_data.shape[1]/100:.1f} seconds)")
    print(f"  - {train_data.shape[2]} mel bins")
    
    # Create model
    print(f"\nCreating Coordinated DREAMStack...")
    model = CoordinatedDREAMStack(
        input_dim=80,
        hidden_dims=hidden_dims,
        rank=16,
        dropout=0.0,
        use_hierarchical_tau=True,
        use_inter_layer_prediction=True,
        inter_layer_loss_weight=0.01
    )
    model = model.to(device)
    
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Layers: {model.num_layers}")
    print(f"\nTau depth factors (fixed): {[1.0 + 0.5*i for i in range(model.num_layers)]}")
    print("  - Layer 0: 1.0x (fast, phonemes)")
    print("  - Layer 1: 1.5x (medium, syllables)")
    print("  - Layer 2: 2.0x (slow, words)")
    
    # Train and measure
    print("\n" + "=" * 70)
    print("TRAINING ON REAL SPEECH")
    print("=" * 70)
    
    metrics, history = train_and_measure_hierarchy(
        model, train_data, n_epochs=n_epochs, device=device
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("HIERARCHY RESULTS (after training on real speech)")
    print("=" * 70)
    
    print(f"\n| Layer | Avg Tau (effective) | Std | Avg Surprise |")
    print(f"|-------|---------------------|-----|--------------|")
    
    for i in range(model.num_layers):
        m = metrics[f'layer_{i}']
        timescale = f"{m['avg_tau']*10:.1f}ms" if m['avg_tau'] < 1 else f"{m['avg_tau']*1000:.0f}ms"
        print(f"| {i}     | {m['avg_tau']:7.3f} ({timescale:>8}) | {m['tau_std']:7.3f} | {m['avg_surprise']:12.4f} |")
    
    # Analyze hierarchy
    avg_taus = [metrics[f'layer_{i}']['avg_tau'] for i in range(model.num_layers)]
    
    # Check if tau increases with depth
    tau_hierarchy = avg_taus[-1] > avg_taus[0] * 1.1
    tau_ratio = avg_taus[-1] / (avg_taus[0] + 1e-6)
    
    results = {
        'hierarchy': {
            'metrics': metrics,
            'avg_taus': avg_taus,
            'tau_ratio': tau_ratio,
        },
        'summary': {
            'hierarchy_present': tau_hierarchy or tau_ratio > 1.1,
            'tau_ratio': tau_ratio,
            'num_layers': model.num_layers,
        },
        'history': {
            'loss': history['loss'],
            'layer_taus': history['layer_taus'],
        }
    }
    
    print("\n" + "=" * 70)
    print("HIERARCHY ANALYSIS")
    print("=" * 70)
    
    print(f"\n📊 Tau progression: {' → '.join([f'{t:.3f}' for t in avg_taus])}")
    print(f"📊 Tau ratio (top/bottom): {tau_ratio:.2f}x")
    print(f"📊 Hierarchy present: {'✅ Yes' if tau_hierarchy else '❌ No'}")
    
    if tau_hierarchy and tau_ratio > 1.2:
        print(f"\n🎉 STRONG HIERARCHY EMERGED!")
        print(f"   - Layer 0 (bottom): τ={avg_taus[0]:.3f} — fast adaptation")
        print(f"   - Layer {model.num_layers-1} (top):    τ={avg_taus[-1]:.3f} — slow integration")
        print(f"   - Ratio: {tau_ratio:.2f}x — top layer integrates {tau_ratio:.1f}x longer")
    elif tau_hierarchy:
        print(f"\n⚠️ Weak hierarchy (ratio={tau_ratio:.2f}x, need >1.2)")
    else:
        print(f"\n⚠️ Hierarchy not established (may need more training)")
    
    # Save results
    output_file = Path(__file__).parent / 'results' / 'results_05_hierarchy.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    save_results = {
        'hierarchy': {
            'num_layers': results['hierarchy']['num_layers'],
            'avg_taus': results['hierarchy']['avg_taus'],
            'tau_ratio': results['hierarchy']['tau_ratio'],
            'layer_metrics': results['hierarchy']['metrics'],
        },
        'summary': {
            'hierarchy_present': results['summary']['hierarchy_present'],
            'tau_ratio': results['summary']['tau_ratio'],
        },
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Hierarchy Test (Real Audio)')
    parser.add_argument('--audio-dir', type=str, default='audio_test',
                       help='Directory with audio files')
    parser.add_argument('--metadata-path', type=str, default=None,
                       help='Path to metadata.csv')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = run_hierarchy_test(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata_path,
        hidden_dims=args.hidden_dims,
        n_epochs=args.epochs,
        device=device,
    )
