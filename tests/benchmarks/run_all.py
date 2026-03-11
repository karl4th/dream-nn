#!/usr/bin/env python3
"""
Legacy Benchmark Runner - Redirects to new dream.benchmarks module.

For new usage, use:
    uv run python -m dream.benchmarks.run_all_benchmarks
"""

import sys
import subprocess
from pathlib import Path

# Redirect to new benchmark runner
if __name__ == '__main__':
    print("=" * 70)
    print("DEPRECATED: This script is deprecated.")
    print("Please use:")
    print("  uv run python -m dream.benchmarks.run_all_benchmarks")
    print("=" * 70)
    
    # Run new benchmark runner
    cmd = [
        sys.executable,
        "-m",
        "dream.benchmarks.run_all_benchmarks"
    ] + sys.argv[1:]
    
    subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
