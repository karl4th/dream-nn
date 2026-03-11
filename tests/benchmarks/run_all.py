#!/usr/bin/env python3
"""
DREAM Benchmark Runner - Run All 5 Tests

Usage:
    uv run python tests/benchmarks/run_all.py --audio-dir /path/to/LJSpeech-1.1 --device cuda
"""

# Check dependencies first
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import dream
try:
    import dream
except ImportError:
    print("ERROR: dream module not found. Install with: pip install -e .")
    sys.exit(1)

# Now import other dependencies
import argparse
import json
import time
import random
import shutil
from datetime import datetime
from typing import Dict, Any

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_ljspeech_metadata(audio_dir: str, n_files: int = 10, seed: int = 42):
    """Load LJSpeech metadata and create temporary files."""
    audio_path = Path(audio_dir)
    wav_dir = audio_path / 'wavs'
    metadata_path = audio_path / 'metadata.csv'

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path, sep='|', header=None, names=['id', 'text', 'normalized_text'])
    df['phonemes'] = df['normalized_text']

    if n_files and n_files < len(df):
        random.seed(seed)
        indices = random.sample(range(len(df)), n_files)
        df = df.iloc[indices].reset_index(drop=True)

    print(f"Loaded {len(df)} files from LJSpeech")

    # Create temporary metadata.csv
    temp_dir = Path('temp_benchmark')
    temp_dir.mkdir(exist_ok=True)
    temp_metadata = temp_dir / 'metadata.csv'
    df[['id', 'text', 'phonemes']].to_csv(temp_metadata, sep='|', index=False, header=False)

    return str(wav_dir), str(temp_metadata)


def run_test_1(audio_dir: str, metadata_path: str, hidden_dim: int, epochs: int, device: str) -> Dict[str, Any]:
    """Test 1: Basic ASR Reconstruction."""
    from test_01_basic_asr import run_basic_asr_test
    return run_basic_asr_test(
        audio_dir=audio_dir,
        metadata_path=metadata_path,
        hidden_dim=hidden_dim,
        n_epochs=epochs,
        device=device,
    )


def run_test_2(audio_dir: str, metadata_path: str, hidden_dim: int, device: str) -> Dict[str, Any]:
    """Test 2: Speaker Adaptation."""
    from test_02_speaker_adaptation import run_speaker_adaptation_test
    return run_speaker_adaptation_test(
        audio_dir=audio_dir,
        metadata_path=metadata_path,
        hidden_dim=hidden_dim,
        device=device,
    )


def run_test_3(audio_dir: str, metadata_path: str, hidden_dim: int, device: str) -> Dict[str, Any]:
    """Test 3: Noise Robustness."""
    from test_03_noise_robustness import run_noise_robustness_test
    return run_noise_robustness_test(
        audio_dir=audio_dir,
        metadata_path=metadata_path,
        hidden_dim=hidden_dim,
        device=device,
    )


def run_test_4(audio_dir: str, metadata_path: str, hidden_dims: list, epochs: int, device: str) -> Dict[str, Any]:
    """Test 4: Stack Coordination."""
    from test_04_stack_coordination import run_coordination_test
    return run_coordination_test(
        audio_dir=audio_dir,
        metadata_path=metadata_path,
        hidden_dims=hidden_dims,
        n_epochs=epochs,
        device=device,
    )


def run_test_5(audio_dir: str, metadata_path: str, hidden_dims: list, device: str) -> Dict[str, Any]:
    """Test 5: Hierarchical Processing (Real Audio)."""
    from test_05_hierarchy import run_hierarchy_test
    return run_hierarchy_test(
        audio_dir=audio_dir,
        metadata_path=metadata_path,
        hidden_dims=hidden_dims,
        n_epochs=50,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description='Run DREAM Benchmark Suite')
    parser.add_argument('--audio-dir', type=str, required=True, help='Path to LJSpeech-1.1')
    parser.add_argument('--n-files', type=int, default=10, help='Number of files (default: 10)')
    parser.add_argument('--tests', type=str, default='1,2,3,4,5', help='Tests to run (default: 1,2,3,4,5)')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dim for tests 1-3')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128, 128], help='Hidden dims for tests 4-5')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs for tests 1,4')
    parser.add_argument('--seq-len', type=int, default=500, help='Seq len for test 5')
    parser.add_argument('--device', type=str, default=None, help='cuda/cpu')
    parser.add_argument('--output-dir', type=str, default='tests/benchmarks/results', help='Output directory')

    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")

    # Load LJSpeech data
    audio_dir, metadata_path = load_ljspeech_metadata(args.audio_dir, args.n_files)

    test_numbers = [int(x.strip()) for x in args.tests.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    total_start = time.time()

    for test_num in test_numbers:
        if test_num < 1 or test_num > 5:
            print(f"Skipping invalid test: {test_num}")
            continue

        try:
            print("\n" + "=" * 70)
            print(f"RUNNING TEST {test_num}")
            print("=" * 70)

            start_time = time.time()

            if test_num == 1:
                results = run_test_1(audio_dir, metadata_path, args.hidden_dim, args.epochs, args.device)
            elif test_num == 2:
                results = run_test_2(audio_dir, metadata_path, args.hidden_dim, args.device)
            elif test_num == 3:
                results = run_test_3(audio_dir, metadata_path, args.hidden_dim, args.device)
            elif test_num == 4:
                results = run_test_4(audio_dir, metadata_path, args.hidden_dims, args.epochs, args.device)
            elif test_num == 5:
                results = run_test_5(audio_dir, metadata_path, args.hidden_dims, args.device)

            elapsed = time.time() - start_time
            print(f"\nTest {test_num} completed in {elapsed:.1f}s")

            if results:
                all_results[test_num] = results

        except Exception as e:
            print(f"\n❌ Test {test_num} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[test_num] = {'error': str(e)}

    # Cleanup
    temp_dir = Path('temp_benchmark')
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"TOTAL TIME: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("=" * 70)

    # Generate summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'tests_run': list(all_results.keys()),
        'results': {}
    }

    if 1 in all_results:
        r1 = all_results[1]
        # Test 1 returns dict by model name, check if DREAM passed
        passed = r1.get('dream', {}).get('passed', False)
        print(f"\n✓ Test 1 (ASR): {'PASS' if passed else 'FAIL'}")
        summary['results']['test_1_asr'] = {'passed': passed}

    if 2 in all_results:
        r2 = all_results[2]
        # Test 2 returns dict by model name, check if DREAM passed
        passed = r2.get('dream', {}).get('passed', False)
        print(f"✓ Test 2 (Adaptation): {'PASS' if passed else 'FAIL'}")
        summary['results']['test_2_adaptation'] = {'passed': passed}

    if 3 in all_results:
        r3 = all_results[3]
        # Test 3 returns dict by model name, check if DREAM passed
        passed = r3.get('dream', {}).get('passed', False)
        print(f"✓ Test 3 (Noise): {'PASS' if passed else 'FAIL'}")
        summary['results']['test_3_noise'] = {'passed': passed}

    if 4 in all_results:
        r4 = all_results[4]
        passed = r4.get('summary', {}).get('coordination_helps', False)
        print(f"✓ Test 4 (Coordination): {'PASS' if passed else 'FAIL'}")
        summary['results']['test_4_coordination'] = {'passed': passed}

    if 5 in all_results:
        r5 = all_results[5]
        # Check hierarchy from new test format
        passed = r5.get('summary', {}).get('hierarchy_present', False)
        tau_ratio = r5.get('summary', {}).get('tau_ratio', 0)
        print(f"✓ Test 5 (Hierarchy): {'PASS' if passed else 'FAIL'} (tau_ratio={tau_ratio:.2f}x)")
        summary['results']['test_5_hierarchy'] = {'passed': passed, 'tau_ratio': tau_ratio}

    all_passed = all(r.get('passed', False) for r in summary['results'].values())

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 70)

    summary['all_passed'] = all_passed

    summary_file = output_dir / 'benchmark_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_file}")

    # Generate visualization
    generate_visualization(all_results, output_dir)

    return 0 if all_passed else 1


def generate_visualization(all_results: Dict, output_dir: Path) -> None:
    """
    Generate comprehensive visualization for all benchmark tests.
    
    Creates:
    - 5 subplots (one per test)
    - Summary comparison
    """
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)
    
    # Use scientific style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('DREAM Benchmark Results\nDynamic Recall and Elastic Adaptive Memory', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # ================================================================
    # Plot 1: Test 1 - ASR Reconstruction
    # ================================================================
    ax1 = fig.add_subplot(2, 3, 1)
    if 1 in all_results:
        r1 = all_results[1]
        models = ['DREAM', 'LSTM', 'Transformer']
        improvements = [
            r1.get('dream', {}).get('metrics', {}).get('improvement_pct', 0),
            r1.get('lstm', {}).get('metrics', {}).get('improvement_pct', 0),
            r1.get('transformer', {}).get('metrics', {}).get('improvement_pct', 0),
        ]
        colors = ['#2ecc71', '#3498db', '#9b59b6']
        
        bars = ax1.bar(models, improvements, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Improvement (%)', fontsize=11)
        ax1.set_title('Test 1: ASR Reconstruction\nHigher is better', fontsize=12, fontweight='bold')
        ax1.axhline(y=50, color='red', linestyle='--', label='Spec threshold (50%)', linewidth=2)
        ax1.legend(loc='lower right')
        
        # Add value labels
        for bar, val in zip(bars, improvements):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ================================================================
    # Plot 2: Test 2 - Speaker Adaptation
    # ================================================================
    ax2 = fig.add_subplot(2, 3, 2)
    if 2 in all_results:
        r2 = all_results[2]
        models = ['DREAM', 'LSTM', 'Transformer']
        steps = [
            r2.get('dream', {}).get('metrics', {}).get('adaptation_steps', 0),
            r2.get('lstm', {}).get('metrics', {}).get('adaptation_steps', 0),
            r2.get('transformer', {}).get('metrics', {}).get('adaptation_steps', 0),
        ]
        surprise = [
            r2.get('dream', {}).get('metrics', {}).get('surprise_spike', 0),
            0, 0
        ]
        
        x = np.arange(len(models))
        bars = ax2.bar(x, steps, color=['#e74c3c', '#3498db', '#9b59b6'], edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Adaptation Steps', fontsize=11)
        ax2.set_title('Test 2: Speaker Adaptation\nLower is better', fontsize=12, fontweight='bold')
        ax2.axhline(y=50, color='green', linestyle='--', label='Spec threshold (50 steps)', linewidth=2)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend(loc='upper right')
        
        for bar, val in zip(bars, steps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ================================================================
    # Plot 3: Test 3 - Noise Robustness
    # ================================================================
    ax3 = fig.add_subplot(2, 3, 3)
    if 3 in all_results:
        r3 = all_results[3]
        models = ['DREAM', 'LSTM', 'Transformer']
        ratios = [
            r3.get('dream', {}).get('metrics', {}).get('loss_ratio_10db', 0),
            r3.get('lstm', {}).get('metrics', {}).get('loss_ratio_10db', 0),
            r3.get('transformer', {}).get('metrics', {}).get('loss_ratio_10db', 0),
        ]
        
        bars = ax3.bar(models, ratios, color=['#f39c12', '#3498db', '#9b59b6'], edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Loss Ratio (10dB / Clean)', fontsize=11)
        ax3.set_title('Test 3: Noise Robustness\nLower is better (closer to 1.0)', fontsize=12, fontweight='bold')
        ax3.axhline(y=1.0, color='green', linestyle='-', label='Perfect (1.0x)', linewidth=2)
        ax3.axhline(y=2.0, color='red', linestyle='--', label='Threshold (2.0x)', linewidth=2)
        ax3.legend(loc='upper right')
        
        for bar, val in zip(bars, ratios):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ================================================================
    # Plot 4: Test 4 - Coordination
    # ================================================================
    ax4 = fig.add_subplot(2, 3, 4)
    if 4 in all_results:
        r4 = all_results[4]
        models = ['Uncoordinated', 'Coordinated']
        final_loss = [
            r4.get('uncoordinated', {}).get('metrics', {}).get('final_loss', 0),
            r4.get('coordinated', {}).get('metrics', {}).get('final_loss', 0),
        ]
        train_time = [
            r4.get('uncoordinated', {}).get('metrics', {}).get('train_time', 0),
            r4.get('coordinated', {}).get('metrics', {}).get('train_time', 0),
        ]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, final_loss, width, label='Final Loss', color='#27ae60', edgecolor='black', linewidth=1.5)
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x + width/2, train_time, width, label='Train Time (s)', color='#34495e', edgecolor='black', linewidth=1.5)
        
        ax4.set_ylabel('Final Loss', fontsize=11, color='#27ae60')
        ax4_twin.set_ylabel('Train Time (s)', fontsize=11, color='#34495e')
        ax4.set_title('Test 4: Stack Coordination\nLower is better', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models)
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # ================================================================
    # Plot 5: Test 5 - Hierarchy
    # ================================================================
    ax5 = fig.add_subplot(2, 3, 5)
    if 5 in all_results:
        r5 = all_results[5]
        avg_taus = r5.get('hierarchy', {}).get('avg_taus', [])
        tau_ratio = r5.get('hierarchy', {}).get('tau_ratio', 0)
        
        if avg_taus:
            layers = [f'Layer {i}' for i in range(len(avg_taus))]
            timescales = [f'{t*10:.1f}ms' if t < 1 else f'{t*1000:.0f}ms' for t in avg_taus]
            
            bars = ax5.bar(layers, avg_taus, color=['#1abc9c', '#3498db', '#9b59b6'], edgecolor='black', linewidth=1.5)
            ax5.set_ylabel('Effective Tau (τ)', fontsize=11)
            ax5.set_title(f'Test 5: Temporal Hierarchy\nTau Ratio: {tau_ratio:.2f}x', fontsize=12, fontweight='bold')
            ax5.axhline(y=avg_taus[0] * 2, color='red', linestyle='--', label=f'2x baseline', linewidth=2)
            ax5.legend(loc='upper right')
            
            for bar, ts in zip(bars, timescales):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                        f'{ts}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ================================================================
    # Plot 6: Summary
    # ================================================================
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Summary text
    summary_text = "BENCHMARK SUMMARY\n\n"
    
    tests_passed = 0
    total_tests = len([k for k in all_results.keys() if k in [1, 2, 3, 4, 5]])
    
    if 1 in all_results:
        passed = all_results[1].get('dream', {}).get('passed', False)
        if passed: tests_passed += 1
        imp = all_results[1].get('dream', {}).get('metrics', {}).get('improvement_pct', 0)
        summary_text += f"✓ Test 1 (ASR):        {'PASS' if passed else 'FAIL'} ({imp:.1f}% improvement)\n"
    
    if 2 in all_results:
        passed = all_results[2].get('dream', {}).get('passed', False)
        if passed: tests_passed += 1
        steps = all_results[2].get('dream', {}).get('metrics', {}).get('adaptation_steps', 0)
        summary_text += f"✓ Test 2 (Adaptation): {'PASS' if passed else 'FAIL'} ({steps:.0f} steps)\n"
    
    if 3 in all_results:
        passed = all_results[3].get('dream', {}).get('passed', False)
        if passed: tests_passed += 1
        ratio = all_results[3].get('dream', {}).get('metrics', {}).get('loss_ratio_10db', 0)
        summary_text += f"✓ Test 3 (Noise):      {'PASS' if passed else 'FAIL'} ({ratio:.2f}x ratio)\n"
    
    if 4 in all_results:
        passed = all_results[4].get('summary', {}).get('coordination_helps', False)
        if passed: tests_passed += 1
        summary_text += f"✓ Test 4 (Coordination): {'PASS' if passed else 'FAIL'}\n"
    
    if 5 in all_results:
        passed = all_results[5].get('summary', {}).get('hierarchy_present', False)
        if passed: tests_passed += 1
        tau_ratio = all_results[5].get('summary', {}).get('tau_ratio', 0)
        summary_text += f"✓ Test 5 (Hierarchy):  {'PASS' if passed else 'FAIL'} ({tau_ratio:.2f}x ratio)\n"
    
    summary_text += f"\n{'='*40}\n"
    summary_text += f"TOTAL: {tests_passed}/{total_tests} tests passed\n"
    
    if tests_passed == total_tests:
        summary_text += "\n🎉 ALL TESTS PASSED!"
        ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, fontsize=14, 
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
    else:
        summary_text += f"\n⚠️  {total_tests - tests_passed} tests failed"
        ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, fontsize=14, 
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))
    
    # Save figure
    plt.tight_layout()
    output_file = output_dir / 'benchmark_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✅ Visualization saved to: {output_file}")
    print(f"   Resolution: 300 DPI (publication quality)")


if __name__ == '__main__':
    sys.exit(main())
