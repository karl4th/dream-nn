#!/usr/bin/env python3
"""
Wrapper script to run SANDAY training.

Usage:
    python run_train.py --audio-dir /path/to/LJSpeech-1.1/wavs
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import and run
from sanday.train import main

if __name__ == '__main__':
    main()
