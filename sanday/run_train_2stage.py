#!/usr/bin/env python3
"""
2-Stage Training for SANDAY.

Stage 1: Pre-training (MSE reconstruction, fast weights frozen)
Stage 2: Adaptation (CTC loss, fast weights active)

Usage:
    python run_train_2stage.py --audio-dir /path/to/LJSpeech-1.1/wavs --subset 20
"""

import sys
import os

# Download NLTK data FIRST
print("[SANDAY] Downloading required NLTK data...")
import nltk
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('cmudict', quiet=True)
print("[SANDAY] NLTK data ready!")

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from sanday.train_2stage import main

if __name__ == '__main__':
    main()
