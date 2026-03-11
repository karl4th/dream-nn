#!/usr/bin/env python3
"""
DREAM Benchmark Runner - Run All 5 Tests

Usage:
    # Run all benchmarks
    uv run python -m dream.benchmarks.run_all_benchmarks \
        --audio-dir audio_test \
        --device cuda \
        --epochs 50

    # Run specific tests
    uv run python -m dream.benchmarks.run_all_benchmarks \
        --audio-dir audio_test \
        --tests 1,2,3 \
        --device cuda
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import torch


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

    if test_num == 1:
        # Test 1: Basic ASR Reconstruction
        from benchmarks.test_01_basic_asr import run_basic_asr_test
        results = run_basic_asr_test(
            audio_dir=args.audio_dir,
            hidden_dim=args.hidden_dim,
            n_epochs=args.epochs,
            device=args.device,
        )

    elif test_num == 2:
        # Test 2: Speaker Adaptation
        from benchmarks.test_02_speaker_adaptation import run_speaker_adaptation_test
        results = run_speaker_adaptation_test(
            audio_dir=args.audio_dir,
            hidden_dim=args.hidden_dim,
            device=args.device,
        )

    elif test_num == 3:
        # Test 3: Noise Robustness
        from benchmarks.test_03_noise_robustness import run_noise_robustness_test
        results = run_noise_robustness_test(
            audio_dir=args.audio_dir,
            hidden_dim=args.hidden_dim,
            device=args.device,
        )

    elif test_num == 4:
        # Test 4: Stack Coordination
        from benchmarks.test_04_stack_coordination import run_coordination_test
        results = run_coordination_test(
            audio_dir=args.audio_dir,
            hidden_dims=args.hidden_dims,
            n_epochs=args.epochs,
            device=args.device,
        )

    elif test_num == 5:
        # Test 5: Hierarchical Processing
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

    return results


def generate_summary(all_results: Dict[int, Dict], output_dir: Path) -> None:
    """
    Generate summary report.

    Parameters
    ----------
    all_results : dict
        Results from all tests
    output_dir : Path
        Directory to save summary
    """
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'tests_run': list(all_results.keys()),
        'results': {}
    }

    # Test 1: ASR
    if 1 in all_results:
        r1 = all_results[1]
        print(f"\n✓ Test 1 (ASR): {r1.get('summary', {}).get('all_passed', False)}")
        summary['results']['test_1_asr'] = {
            'passed': r1.get('summary', {}).get('all_passed', False),
            'dream_improvement': r1.get('basic_asr', {}).get('improvement_pct', 0),
        }

    # Test 2: Adaptation
    if 2 in all_results:
        r2 = all_results[2]
        print(f"✓ Test 2 (Adaptation): {r2.get('summary', {}).get('all_passed', False)}")
        summary['results']['test_2_adaptation'] = {
            'passed': r2.get('summary', {}).get('all_passed', False),
            'dream_adapt_steps': r2.get('speaker_adaptation', {}).get('adaptation_steps', 0),
        }

    # Test 3: Noise
    if 3 in all_results:
        r3 = all_results[3]
        print(f"✓ Test 3 (Noise): {r3.get('summary', {}).get('all_passed', False)}")
        summary['results']['test_3_noise'] = {
            'passed': r3.get('summary', {}).get('all_passed', False),
            'dream_noise_ratio': r3.get('noise_robustness', {}).get('loss_ratio', 0),
        }

    # Test 4: Coordination
    if 4 in all_results:
        r4 = all_results[4]
        print(f"✓ Test 4 (Coordination): {r4.get('summary', {}).get('coordination_helps', False)}")
        summary['results']['test_4_coordination'] = {
            'passed': r4.get('summary', {}).get('coordination_helps', False),
        }

    # Test 5: Hierarchy
    if 5 in all_results:
        r5 = all_results[5]
        print(f"✓ Test 5 (Hierarchy): {r5.get('summary', {}).get('hierarchy_present', False)}")
        summary['results']['test_5_hierarchy'] = {
            'passed': r5.get('summary', {}).get('hierarchy_present', False),
        }

    # Overall
    all_passed = all(
        r.get('passed', False)
        for test_result in summary['results'].values()
        for r in [test_result] if isinstance(r, dict)
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

    # Save markdown report
    report_file = output_dir / 'BENCHMARK_REPORT.md'
    with open(report_file, 'w') as f:
        f.write("# DREAM Benchmark Report\n\n")
        f.write(f"**Generated:** {summary['timestamp']}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Tests Run:** {', '.join(map(str, summary['tests_run']))}\n")
        f.write(f"- **All Passed:** {'✅ Yes' if all_passed else '❌ No'}\n\n")
        f.write("## Results\n\n")
        for test_name, result in summary['results'].items():
            f.write(f"### {test_name}\n")
            for metric, value in result.items():
                f.write(f"- {metric}: {value}\n")
            f.write("\n")

    print(f"Report saved to: {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run DREAM Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  uv run python -m dream.benchmarks.run_all_benchmarks --audio-dir audio_test --device cuda

  # Run specific tests
  uv run python -m dream.benchmarks.run_all_benchmarks --tests 1,3,5 --device cuda

  # Run with custom epochs
  uv run python -m dream.benchmarks.run_all_benchmarks --epochs 100 --device cuda
        """
    )

    # Data
    parser.add_argument(
        '--audio-dir',
        type=str,
        default='audio_test',
        help='Directory with audio files (default: audio_test)'
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
    print(f"TOTAL TIME: {total_elapsed:.1f}s")
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
