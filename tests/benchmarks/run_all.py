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


def run_test_5(hidden_dims: list, seq_len: int, device: str) -> Dict[str, Any]:
    """Test 5: Hierarchical Processing."""
    from test_05_hierarchy import run_hierarchy_test
    return run_hierarchy_test(
        hidden_dims=hidden_dims,
        n_epochs=50,  # Train for 50 epochs
        demo=False,  # Standard mode for benchmark
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
                results = run_test_5(args.hidden_dims, args.seq_len, args.device)

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

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
