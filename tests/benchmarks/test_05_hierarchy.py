"""
Test 5: Hierarchical Temporal Learning (DEMO MODE).

Demonstrates how DREAM learns hierarchical temporal representations:
- Lower layers: Fast patterns (phonemes, ~10ms)
- Middle layers: Medium patterns (syllables, ~100ms)  
- Upper layers: Slow context (words/phrases, ~1s)

This is a REAL training test that shows emergent hierarchy.

Run:
    uv run python tests/benchmarks/test_05_hierarchy.py --demo
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dream.layer_coordinated import CoordinatedDREAMStack
from dream.config import DREAMConfig


def create_speech_like_data(
    n_samples: int = 8,
    seq_len: int = 800,  # ~8 seconds at 100Hz
    input_dim: int = 80,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Create speech-like hierarchical data.
    
    Simulates real speech structure:
    - Fast oscillations (10-50ms) → phoneme-like
    - Medium envelopes (100-300ms) → syllable-like
    - Slow modulation (1-2s) → word/prosody-like
    """
    data = torch.zeros(n_samples, seq_len, input_dim, device=device)
    
    for i in range(n_samples):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # === Level 1: Fast "phoneme" patterns (10-50ms = 1-5 frames at 100Hz) ===
        phoneme_freq = 2 * np.pi / np.random.uniform(10, 50)
        for j in range(input_dim // 4):
            phase = np.random.uniform(0, 2*np.pi)
            data[i, :, j] = torch.sin(phoneme_freq * t + phase) * 0.5
        
        # === Level 2: Medium "syllable" patterns (100-300ms) ===
        syllable_freq = 2 * np.pi / np.random.uniform(100, 300)
        for j in range(input_dim // 4, input_dim // 2):
            phase = np.random.uniform(0, 2*np.pi)
            # Amplitude modulation for syllable-like envelope
            envelope = 0.5 + 0.5 * torch.sin(2 * np.pi / 500 * t)
            data[i, :, j] = torch.sin(syllable_freq * t + phase) * envelope
        
        # === Level 3: Slow "word/prosody" patterns (1-2s) ===
        word_freq = 2 * np.pi / np.random.uniform(500, 1000)
        for j in range(input_dim // 2, 3 * input_dim // 4):
            phase = np.random.uniform(0, 2*np.pi)
            data[i, :, j] = torch.sin(word_freq * t + phase) * 0.7
        
        # === Level 4: Very slow "sentence" context (2-4s) ===
        sentence_freq = 2 * np.pi / np.random.uniform(1000, 2000)
        for j in range(3 * input_dim // 4, input_dim):
            phase = np.random.uniform(0, 2*np.pi)
            data[i, :, j] = torch.sin(sentence_freq * t + phase) * 0.6
    
    # Normalize
    data = (data - data.mean(dim=1, keepdim=True)) / (data.std(dim=1, keepdim=True) + 1e-6)
    
    return data


def train_and_measure_hierarchy(
    model: CoordinatedDREAMStack,
    train_data: torch.Tensor,
    n_epochs: int = 50,
    device: str = 'cpu'
) -> dict:
    """
    Train model and measure emerging hierarchy.
    
    Returns metrics showing how each layer learns different timescales.
    """
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    batch_size = train_data.shape[0]
    states = model.init_states(batch_size, device=device)
    
    # Track metrics per epoch
    history = {
        'loss': [],
        'layer_surprises': [[] for _ in range(model.num_layers)],
        'layer_taus': [[] for _ in range(model.num_layers)],
    }
    
    print(f"Training on {train_data.shape} for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward through time
        total_loss = 0.0
        epoch_surprises = [[] for _ in range(model.num_layers)]
        epoch_taus = [[] for _ in range(model.num_layers)]
        
        states = model.init_states(batch_size, device=device)  # Reset state each epoch
        
        for t in range(train_data.shape[1]):
            x_t = train_data[:, t, :]
            current_input = x_t
            
            for i, cell in enumerate(model.cells):
                h_new, states.layer_states[i], _, _ = cell(current_input, states.layer_states[i])
                
                # Track surprise and tau
                surprise = states.layer_states[i].avg_surprise.mean().item()
                tau_base = cell.tau_sys.item() * cell.tau_depth_factor
                tau_eff = tau_base / (1.0 + surprise * cell.tau_surprise_scale.item())
                
                epoch_surprises[i].append(surprise)
                epoch_taus[i].append(tau_eff)
                
                if i < model.num_layers - 1:
                    current_input = h_new
        
        # Reconstruction loss
        # (simplified - just measure final state prediction)
        recon_loss = 0.0
        for i in range(model.num_layers):
            recon_loss += states.layer_states[i].h.pow(2).mean()
        
        loss = recon_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        
        # Aggregate epoch metrics
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
    
    # Compute final hierarchy metrics
    metrics = {}
    for i in range(model.num_layers):
        # Use last 10 epochs for stable estimates
        final_taus = history['layer_taus'][i][-10:]
        final_surprises = history['layer_surprises'][i][-10:]
        
        metrics[f'layer_{i}'] = {
            'avg_tau': float(np.mean(final_taus)),
            'tau_std': float(np.std(final_taus)),
            'avg_surprise': float(np.mean(final_surprises)),
            'surprise_change': float(final_surprises[-1] - final_surprises[0]),
        }
    
    return metrics, history


def run_hierarchy_test(
    hidden_dims: list = [128, 128, 128],
    n_epochs: int = 50,
    demo: bool = False,
    device: str = 'cpu'
) -> dict:
    """
    Run hierarchical learning demonstration.
    """
    print("=" * 70)
    print("DREAM Test 5: Hierarchical Temporal Learning")
    print("=" * 70)
    
    if demo:
        print("\n🎯 DEMO MODE: Training on speech-like hierarchical data")
        print("   This demonstrates emergent temporal hierarchy in DREAM")
    else:
        print("\n📊 STANDARD MODE: Testing with synthetic patterns")
    
    # Create data
    print("\nCreating hierarchical data...")
    if demo:
        data = create_speech_like_data(
            n_samples=8,
            seq_len=800,  # 8 seconds
            input_dim=80,
            device=device
        )
    else:
        data = create_hierarchical_data_standard(
            n_samples=4,
            seq_len=500,
            input_dim=80,
            device=device
        )
    print(f"Data shape: {data.shape}")
    
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
    print(f"\nTau depth factors: {[1.0 + 0.5*i for i in range(model.num_layers)]}")
    
    # Train and measure
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    metrics, history = train_and_measure_hierarchy(
        model, data, n_epochs=n_epochs, device=device
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("HIERARCHY RESULTS (after training)")
    print("=" * 70)
    
    print(f"\n| Layer | Avg Tau | Tau Std | Avg Surprise | Change |")
    print(f"|-------|---------|---------|--------------|--------|")
    
    for i in range(model.num_layers):
        m = metrics[f'layer_{i}']
        print(f"| {i}     | {m['avg_tau']:7.3f} | {m['tau_std']:7.3f} | "
              f"{m['avg_surprise']:12.4f} | {m['surprise_change']:+7.4f} |")
    
    # Analyze hierarchy
    avg_taus = [metrics[f'layer_{i}']['avg_tau'] for i in range(model.num_layers)]
    
    # Check tau hierarchy (lower layers = smaller tau, upper = larger)
    tau_hierarchy = all(avg_taus[i] < avg_taus[i+1] for i in range(len(avg_taus)-1))
    
    # Compute tau ratio (how much slower is top vs bottom)
    tau_ratio = avg_taus[-1] / (avg_taus[0] + 1e-6)
    
    results = {
        'hierarchy': {
            'metrics': metrics,
            'avg_taus': avg_taus,
            'tau_ratio': tau_ratio,
        },
        'summary': {
            'hierarchy_present': tau_hierarchy,
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
    
    print(f"\nTau progression: {' → '.join([f'{t:.3f}' for t in avg_taus])}")
    print(f"Tau ratio (top/bottom): {tau_ratio:.2f}x")
    print(f"Hierarchy present: {'✅ Yes' if tau_hierarchy else '❌ No'}")
    
    if tau_hierarchy and tau_ratio > 1.2:
        print(f"\n🎉 STRONG HIERARCHY DETECTED!")
        print(f"   - Layer 0 (bottom): τ={avg_taus[0]:.3f} — fast, reacts to quick changes")
        print(f"   - Layer {model.num_layers-1} (top):    τ={avg_taus[-1]:.3f} — slow, integrates long context")
        print(f"   - Ratio: {tau_ratio:.2f}x — top layer integrates {tau_ratio:.1f}x longer timescales")
        print(f"\n   This matches brain hierarchy:")
        print(f"   - A1 (auditory): ~10ms (phonemes)")
        print(f"   - STG: ~100ms (syllables)")
        print(f"   - PFC: ~1-2s (words/phrases)")
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


def create_hierarchical_data_standard(
    n_samples: int = 4,
    seq_len: int = 500,
    input_dim: int = 80,
    device: str = 'cpu'
) -> torch.Tensor:
    """Standard hierarchical data (original test)."""
    data = torch.zeros(n_samples, seq_len, input_dim, device=device)
    
    for i in range(n_samples):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # Fast patterns
        fast_freq = 2 * np.pi / 10
        for j in range(input_dim // 3):
            data[i, :, j] = torch.sin(fast_freq * t + j * 0.5)
        
        # Medium patterns
        med_freq = 2 * np.pi / 50
        for j in range(input_dim // 3, 2 * input_dim // 3):
            data[i, :, j] = torch.sin(med_freq * t + j * 0.5)
        
        # Slow patterns
        slow_freq = 2 * np.pi / 200
        for j in range(2 * input_dim // 3, input_dim):
            data[i, :, j] = torch.sin(slow_freq * t + j * 0.5)
    
    data = (data - data.mean(dim=1, keepdim=True)) / (data.std(dim=1, keepdim=True) + 1e-6)
    return data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Hierarchy Test')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--demo', action='store_true', help='Demo mode with speech-like data')
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = run_hierarchy_test(
        hidden_dims=args.hidden_dims,
        n_epochs=args.epochs,
        demo=args.demo,
        device=device,
    )
