#!/usr/bin/env python3
"""
DREAM Benchmark Runner - Run All 5 Tests

Single command to run all benchmarks:
    uv run python -m dream.benchmarks.run_all

Usage:
    # Run all 5 tests
    uv run python -m dream.benchmarks.run_all --audio-dir /path/to/LJSpeech-1.1 --device cuda

    # Run specific tests
    uv run python -m dream.benchmarks.run_all --tests 1,3,5 --device cuda

    # Run with custom epochs
    uv run python -m dream.benchmarks.run_all --epochs 100 --device cuda
"""

import argparse
import json
import time
import sys
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import torch
import pandas as pd


def load_ljspeech_metadata(
    audio_dir: str,
    n_files: int = None,
    seed: int = 42
) -> Tuple[Path, pd.DataFrame]:
    """
    Load LJSpeech dataset metadata.

    Parameters
    ----------
    audio_dir : str
        Path to LJSpeech-1.1 directory
    n_files : int, optional
        Number of files to use (None = all)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    Tuple[Path, pd.DataFrame]
        Audio directory path and metadata DataFrame
    """
    audio_path = Path(audio_dir)
    wav_dir = audio_path / 'wavs'

    # Load metadata.csv
    metadata_path = audio_path / 'metadata.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    # LJSpeech metadata format: ID|text|normalized_text
    df = pd.read_csv(
        metadata_path,
        sep='|',
        header=None,
        names=['id', 'text', 'normalized_text']
    )

    # Add phonemes placeholder (will be computed later if needed)
    df['phonemes'] = df['normalized_text']

    # Select subset if requested
    if n_files and n_files < len(df):
        random.seed(seed)
        indices = random.sample(range(len(df)), n_files)
        df = df.iloc[indices].reset_index(drop=True)

    print(f"Loaded {len(df)} files from LJSpeech")
    print(f"Audio directory: {wav_dir}")

    return wav_dir, df


def run_test(test_num: int, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run a single benchmark test.

    Parameters
    ----------
    test_num : int
        Test number (1-5)
    args : argparse.Namespace
        Command line arguments

    Returns
    -------
    dict
        Test results
    """
    print("\n" + "=" * 70)
    print(f"RUNNING TEST {test_num}")
    print("=" * 70)

    start_time = time.time()

    # Load LJSpeech data for tests 1-4
    if test_num in [1, 2, 3]:
        n_files = args.n_files or 10  # Default 10 files for tests 1-3
        audio_dir, metadata = load_ljspeech_metadata(
            args.audio_dir,
            n_files=n_files
        )

        # Create temporary metadata.csv for compatibility
        temp_dir = Path('temp_benchmark')
        temp_dir.mkdir(exist_ok=True)
        temp_metadata = temp_dir / 'metadata.csv'
        metadata[['id', 'text', 'phonemes']].to_csv(
            temp_metadata,
            sep='|',
            index=False,
            header=False
        )

        args.temp_audio_dir = str(audio_dir)
        args.temp_metadata = str(temp_metadata)

    if test_num == 1:
        # Test 1: Basic ASR Reconstruction
        from benchmarks.test_01_basic_asr import run_basic_asr_test
        results = run_basic_asr_test(
            audio_dir=args.temp_audio_dir,
            metadata_path=args.temp_metadata,
            hidden_dim=args.hidden_dim,
            n_epochs=args.epochs,
            device=args.device,
        )

    elif test_num == 2:
        # Test 2: Speaker Adaptation
        from benchmarks.test_02_speaker_adaptation import run_speaker_adaptation_test
        results = run_speaker_adaptation_test(
            audio_dir=args.temp_audio_dir,
            metadata_path=args.temp_metadata,
            hidden_dim=args.hidden_dim,
            device=args.device,
        )

    elif test_num == 3:
        # Test 3: Noise Robustness
        from benchmarks.test_03_noise_robustness import run_noise_robustness_test
        results = run_noise_robustness_test(
            audio_dir=args.temp_audio_dir,
            metadata_path=args.temp_metadata,
            hidden_dim=args.hidden_dim,
            device=args.device,
        )

    elif test_num == 4:
        # Test 4: Stack Coordination
        from benchmarks.test_04_stack_coordination import run_coordination_test
        results = run_coordination_test(
            audio_dir=args.temp_audio_dir,
            metadata_path=args.temp_metadata,
            hidden_dims=args.hidden_dims,
            n_epochs=args.epochs,
            device=args.device,
        )

    elif test_num == 5:
        # Test 5: Hierarchical Processing (doesn't need audio)
        from benchmarks.test_05_hierarchy import run_hierarchy_test
        results = run_hierarchy_test(
            hidden_dims=args.hidden_dims,
            seq_len=args.seq_len,
            device=args.device,
        )

    else:
        print(f"Unknown test number: {test_num}")
        return None

    elapsed = time.time() - start_time
    print(f"\nTest {test_num} completed in {elapsed:.1f}s")

    # Cleanup temp files
    if test_num in [1, 2, 3, 4]:
        import shutil
        temp_dir = Path('temp_benchmark')
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    return results


def generate_summary(all_results: Dict[int, Dict], output_dir: Path) -> None:
    """Generate summary report."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'tests_run': list(all_results.keys()),
        'results': {}
    }

    # Collect results from each test
    if 1 in all_results:
        r1 = all_results[1]
        passed = r1.get('summary', {}).get('all_passed', False)
        print(f"\n✓ Test 1 (ASR): {'PASS' if passed else 'FAIL'}")
        summary['results']['test_1_asr'] = {'passed': passed}

    if 2 in all_results:
        r2 = all_results[2]
        passed = r2.get('summary', {}).get('all_passed', False)
        print(f"✓ Test 2 (Adaptation): {'PASS' if passed else 'FAIL'}")
        summary['results']['test_2_adaptation'] = {'passed': passed}

    if 3 in all_results:
        r3 = all_results[3]
        passed = r3.get('summary', {}).get('all_passed', False)
        print(f"✓ Test 3 (Noise): {'PASS' if passed else 'FAIL'}")
        summary['results']['test_3_noise'] = {'passed': passed}

    if 4 in all_results:
        r4 = all_results[4]
        passed = r4.get('summary', {}).get('coordination_helps', False)
        print(f"✓ Test 4 (Coordination): {'PASS' if passed else 'FAIL'}")
        summary['results']['test_4_coordination'] = {'passed': passed}

    if 5 in all_results:
        r5 = all_results[5]
        passed = r5.get('summary', {}).get('hierarchy_present', False)
        print(f"✓ Test 5 (Hierarchy): {'PASS' if passed else 'FAIL'}")
        summary['results']['test_5_hierarchy'] = {'passed': passed}

    # Overall
    all_passed = all(
        r.get('passed', False)
        for r in summary['results'].values()
    )

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 70)

    summary['all_passed'] = all_passed

    # Save summary
    summary_file = output_dir / 'benchmark_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSummary saved to: {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run DREAM Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 5 tests with LJSpeech
  uv run python -m dream.benchmarks.run_all \\
      --audio-dir /root/.cache/kagglehub/datasets/dromosys/ljspeech/versions/1/LJSpeech-1.1 \\
      --device cuda

  # Run specific tests
  uv run python -m dream.benchmarks.run_all --tests 1,3,5 --device cuda

  # Run with custom epochs and files
  uv run python -m dream.benchmarks.run_all --epochs 100 --n-files 20 --device cuda
        """
    )

    # Data
    parser.add_argument(
        '--audio-dir',
        type=str,
        required=True,
        help='Path to LJSpeech-1.1 directory'
    )
    parser.add_argument(
        '--n-files',
        type=int,
        default=10,
        help='Number of audio files to use for tests 1-4 (default: 10)'
    )

    # Tests
    parser.add_argument(
        '--tests',
        type=str,
        default='1,2,3,4,5',
        help='Comma-separated test numbers to run (default: 1,2,3,4,5)'
    )

    # Model
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='Hidden dimension for tests 1-3 (default: 256)'
    )
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        default=[128, 128, 128],
        help='Hidden dimensions for tests 4-5 (default: 128 128 128)'
    )

    # Training
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs for tests 1,4 (default: 50)'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=500,
        help='Sequence length for test 5 (default: 500)'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='tests/benchmarks/results',
        help='Directory to save results (default: tests/benchmarks/results)'
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")

    # Parse test numbers
    test_numbers = [int(x.strip()) for x in args.tests.split(',')]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    all_results = {}
    total_start = time.time()

    for test_num in test_numbers:
        if test_num < 1 or test_num > 5:
            print(f"Skipping invalid test number: {test_num}")
            continue

        try:
            results = run_test(test_num, args)
            if results:
                all_results[test_num] = results
        except Exception as e:
            print(f"\n❌ Test {test_num} failed with error: {e}")
            import traceback
            traceback.print_exc()
            all_results[test_num] = {'error': str(e)}

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print(f"TOTAL TIME: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("=" * 70)

    # Generate summary
    generate_summary(all_results, output_dir)

    # Return exit code
    summary_file = output_dir / 'benchmark_summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        return 0 if summary.get('all_passed', False) else 1
    return 1


if __name__ == '__main__':
    sys.exit(main())
