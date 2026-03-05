"""
DREAM: Dynamic Recall and Elastic Adaptive Memory

A PyTorch implementation of continuous-time RNN cells with:
- Surprise-driven plasticity
- Liquid Time-Constants (LTC)
- Fast weights with Hebbian learning
- Sleep consolidation

Example
-------
>>> from dream import DREAM, DREAMConfig, DREAMCell
>>> model = DREAM(input_dim=64, hidden_dim=128, rank=8)
>>> x = torch.randn(4, 50, 64)  # (batch, time, features)
>>> output, state = model(x)

ASR Module
----------
>>> from dream.asr import DREAMASR, DREAMASRTrainer
>>> model = DREAMASR(input_dim=80, num_phonemes=72)
>>> probs, log_probs, _ = model(mel_spec, lengths)
"""

from .config import DREAMConfig
from .state import DREAMState
from .cell import DREAMCell
from .statistics import RunningStatistics
from .layer import DREAM, DREAMStack



__version__ = "0.1.2"
__all__ = [
    # Config & State
    "DREAMConfig",
    "DREAMState",

    # Core
    "DREAMCell",
    "RunningStatistics",

    # High-level API
    "DREAM",
    "DREAMStack",
]
