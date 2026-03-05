# DREAM

**Dynamic Recall and Elastic Adaptive Memory**

A PyTorch implementation of continuous-time RNN cells with surprise-driven plasticity and liquid time-constants for adaptive neural dynamics.

## Features

- 🧠 **Surprise-Driven Plasticity** — Hebbian learning modulated by prediction error surprise
- ⏱️ **Liquid Time-Constants (LTC)** — Adaptive integration speeds based on novelty
- 🔁 **Fast Weights** — Low-rank decomposition for efficient meta-learning
- 😴 **Sleep Consolidation** — Memory stabilization during low-surprise periods
- 📦 **Batch Support** — Efficient processing of sequences with proper state management

## Installation

```bash
pip install dreamnn
```

## Quick Start

### Basic Usage

```python
import torch
from dream import DREAMConfig, DREAMCell

# Configure the cell
config = DREAMConfig(
    input_dim=128,     # Input feature dimension
    hidden_dim=256,    # Hidden state size
    rank=16,           # Fast weights rank
    ltc_enabled=True,  # Enable liquid time-constants
)

# Create cell
cell = DREAMCell(config)

# Initialize state
batch_size = 4
state = cell.init_state(batch_size)

# Process single timestep
x = torch.randn(batch_size, config.input_dim)
h_new, state_new = cell(x, state)

# Process full sequence
sequence = torch.randn(batch_size, 100, config.input_dim)  # (batch, time, features)
output, final_state = cell.forward_sequence(sequence, return_all=True)
print(f"Output shape: {output.shape}")  # (batch, time, hidden_dim)
```

### Stateful Processing (Memory Retention)

```python
# Process multiple sequences while preserving memory
state = cell.init_state(batch_size)

for seq in sequences:
    # State (U, h, adaptive_tau) is preserved between sequences
    output, state = cell.forward_sequence(seq, state)
    # Model adapts and remembers!
```

## Architecture

DREAM cell combines several mechanisms for adaptive processing:

```
┌─────────────────────────────────────────────────────────────┐
│                    DREAM Cell                               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐  │
│  │  Predictive  │     │   Surprise   │     │    Fast     │  │
│  │   Coding     │────▶│    Gate      │────▶│   Weights   │  │
│  │  (C, W, B)   │     │  (τ + habit) │     │   (U, V)    │  │
│  └──────────────┘     └──────────────┘     └─────────────┘  │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          Liquid Time-Constant (LTC)                 │    │
│  │   τ_eff = τ_sys / (1 + surprise × scale)            │    │
│  │   h_new = (1-α)·h_prev + α·tanh(input_effect)       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Predictive Coding** | Matrices C, W, B for input/output projection |
| **Surprise Gate** | Adaptive threshold with habituation |
| **Fast Weights** | Low-rank (U, V) decomposition for Hebbian learning |
| **LTC** | Surprise-modulated time constants |
| **Sleep Consolidation** | U_target stabilization |

## Configuration

### DREAMConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | 39 | Input feature dimension (39 for ASR MFCC) |
| `hidden_dim` | 256 | Hidden state size |
| `rank` | 16 | Fast weights rank |
| `time_step` | 0.1 | Integration time step (dt) |
| `forgetting_rate` | 0.005 | Fast weights decay (λ) |
| `base_plasticity` | 0.5 | Hebbian learning rate (η) |
| `base_threshold` | 0.3 | Surprise threshold (τ₀) |
| `entropy_influence` | 0.1 | Entropy effect on threshold (α) |
| `surprise_temperature` | 0.05 | Surprise scaling (γ) |
| `error_smoothing` | 0.05 | Error EMA (β) |
| `surprise_smoothing` | 0.05 | Surprise EMA (β_s) |
| `target_norm` | 2.0 | Fast weights norm constraint |
| `kappa` | 0.5 | Gain modulation coefficient (κ) |
| `ltc_enabled` | True | Enable liquid time-constants |
| `ltc_tau_sys` | 5.0 | Base LTC time constant |
| `ltc_surprise_scale` | 5.0 | Surprise modulation strength |
| `sleep_rate` | 0.005 | Sleep consolidation rate (ζ_sleep) |
| `min_surprise_for_sleep` | 0.2 | Minimum surprise for sleep (S_min) |

## Advanced Usage

### Custom State Management

```python
from dream import DREAMState

# Initialize with specific device
state = cell.init_state(batch_size, device='cuda')

# Access state components
print(state.h.shape)        # (batch, hidden_dim)
print(state.U.shape)        # (hidden_dim, rank)
print(state.adaptive_tau)   # (batch,)

# Detach state for truncated BPTT
state = state.detach()

# Manual state modification
state.U *= 0.5  # Scale fast weights
```

### Sequence Classification

```python
import torch
from dream import DREAMConfig, DREAMCell

config = DREAMConfig(input_dim=64, hidden_dim=128)
cell = DREAMCell(config)
classifier = torch.nn.Linear(128, 10)  # 10 classes

# Process sequence
batch_size = 32
seq_len = 50
x = torch.randn(batch_size, seq_len, 64)

state = cell.init_state(batch_size)
output, final_state = cell.forward_sequence(x)

# Classify using final hidden state
logits = classifier(final_state.h)
predictions = logits.argmax(dim=-1)
```

### Memory Retention Test

```python
# Test if model remembers across multiple presentations
state = cell.init_state(1)

for pass_idx in range(5):
    output, state = cell.forward_sequence(same_sequence, state)
    # Surprise should decrease as model adapts!
```

## API Reference

### DREAMCell

```python
class DREAMCell(nn.Module):
    def __init__(self, config: DREAMConfig) -> None: ...
    
    def init_state(self, batch_size: int = 1, 
                   device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None) -> DREAMState: ...
    
    def forward(self, x: torch.Tensor, 
                state: DREAMState) -> Tuple[torch.Tensor, DREAMState]: ...
    
    def forward_sequence(self, x_seq: torch.Tensor,
                        state: Optional[DREAMState] = None,
                        return_all: bool = False) -> Tuple[torch.Tensor, DREAMState]: ...
```

### DREAMState

```python
@dataclass
class DREAMState:
    h: torch.Tensor           # Hidden state
    U: torch.Tensor           # Fast weights
    U_target: torch.Tensor    # Target fast weights
    adaptive_tau: torch.Tensor  # Adaptive surprise threshold
    error_mean: torch.Tensor  # Error statistics
    error_var: torch.Tensor   # Error variance
    avg_surprise: torch.Tensor  # Average surprise
    
    def detach(self) -> DREAMState: ...
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Benchmarks

DREAM превосходит LSTM и Transformer на задачах обработки аудио:

| Модель | Параметры | ASR Improvement | Adaptation Steps | Noise Ratio |
|--------|-----------|-----------------|------------------|-------------|
| **DREAM** | 82K | **99.9%** | **0** | **1.09×** |
| LSTM | 893K | 93.9% | 0 | 1.09× |
| Transformer | 551K | 92.6% | 0 | 1.07× |

**Результаты:**
- ✅ 99.9% улучшение reconstruction loss (100 эпох)
- ✅ Мгновенная адаптация к смене диктора (<50 шагов)
- ✅ Стабильность к шуму (1.09× при 10dB SNR)

Запуск бенчмарков:
```bash
uv run python tests/benchmarks/run_all.py
```

См. `second.md` и `TECHNICAL_REPORT.md` для деталей.

## Citation

```bibtex
@software{dream2026,
  title = {DREAM: Dynamic Recall and Elastic Adaptive Memory},
  author = {Manifestro Team},
  year = {2026},
  url = {https://github.com/karl4th/dream-nn},
  version = "0.1.2"
}
```
