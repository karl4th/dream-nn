# DREAM ASR Training Scripts

Training scripts for Automatic Speech Recognition using DREAM (Dynamic Recall and Elastic Adaptive Memory).

## Quick Start

### Basic Training (Standard DREAM)

```bash
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --log-dir /content/drive/MyDrive/dream/experiments/dream_base \
    --epochs 50 \
    --batch-size 16
```

### Coordinated DREAM (with top-down modulation)

```bash
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --model coordinated \
    --log-dir /content/drive/MyDrive/dream/experiments/dream_coordinated \
    --epochs 50
```

### Quick Experiment (Subset Mode)

```bash
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --subset 100 \
    --log-dir /content/drive/MyDrive/dream/experiments/dream_quick \
    --epochs 10
```

## Parallel Training (Compare Models)

Run both models simultaneously to compare:

```bash
# Terminal 1: Standard DREAM
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --model dream \
    --run-name dream_base \
    --log-dir /content/drive/MyDrive/dream/experiments &

# Terminal 2: Coordinated DREAM
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --model coordinated \
    --run-name dream_coordinated \
    --log-dir /content/drive/MyDrive/dream/experiments &
```

## Ablation Studies

### Disable Fast Weights

```bash
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --no-fast-weights \
    --log-dir /content/drive/MyDrive/dream/experiments/dream_no_fw
```

### Disable LTC (Liquid Time-Constants)

```bash
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --no-ltc \
    --log-dir /content/drive/MyDrive/dream/experiments/dream_no_ltc
```

### Disable Sleep Consolidation

```bash
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --no-sleep \
    --log-dir /content/drive/MyDrive/dream/experiments/dream_no_sleep
```

### Freeze Fast Weights (Static Base Training)

```bash
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --freeze-fast-weights \
    --log-dir /content/drive/MyDrive/dream/experiments/dream_static
```

## All Command-Line Arguments

### Data Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--root` | (required) | Path to LJSpeech directory |
| `--subset` | None | Limit to N samples (quick experiments) |
| `--batch-size` | 16 | Batch size |
| `--num-workers` | 4 | Data loading workers |
| `--val-split` | 0.1 | Validation split fraction |

### Model Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | dream | `dream` or `coordinated` |
| `--hidden-dims` | [256, 256, 256] | Hidden dimensions per layer |
| `--rank` | 16 | Fast weights rank |
| `--dropout` | 0.1 | Dropout rate |

### Block Control Flags

| Flag | Effect |
|------|--------|
| `--no-fast-weights` | Disable fast weights |
| `--no-ltc` | Disable Liquid Time-Constants |
| `--no-sleep` | Disable sleep consolidation |
| `--freeze-fast-weights` | Freeze fast weights during training |
| `--no-hierarchical-tau` | Disable hierarchical tau (coordinated only) |
| `--no-inter-layer-prediction` | Disable inter-layer loss (coordinated only) |

### Training Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Number of epochs |
| `--lr` | 1e-3 | Learning rate |
| `--weight-decay` | 1e-5 | Weight decay |
| `--grad-clip` | 5.0 | Gradient clipping |
| `--use-amp` | False | Use automatic mixed precision |

### Logging

| Argument | Default | Description |
|----------|---------|-------------|
| `--log-dir` | ./logs | Directory for logs/checkpoints |
| `--run-name` | auto | Name for this run |
| `--log-interval` | 10 | Log every N batches |
| `--example-interval` | 5 | Show prediction examples every N epochs |

### Reproducibility

| Argument | Default | Description |
|----------|---------|-------------|
| `--seed` | 42 | Random seed |

## Output Structure

```
/content/drive/MyDrive/dream/experiments/
└── dream_base_20260312_143022/
    ├── best.pt              # Best model (lowest val loss)
    ├── epoch_5.pt           # Checkpoint every 5 epochs
    ├── epoch_10.pt
    ├── ...
    ├── final.pt             # Final model
    └── metrics.json         # Training metrics (loss, time, LR)
```

## Metrics Format (metrics.json)

```json
{
  "train_loss": [2.5, 2.1, 1.8, ...],
  "val_loss": [2.6, 2.2, 1.9, ...],
  "train_ctc_loss": [2.3, 1.9, 1.6, ...],
  "val_ctc_loss": [2.4, 2.0, 1.7, ...],
  "train_aux_loss": [0.02, 0.01, ...],
  "learning_rate": [0.001, 0.001, 0.0005, ...],
  "epoch_time": [120.5, 118.2, ...]
}
```

## Model Architecture

### Standard DREAM ASR

```
Audio (16kHz) → Mel Spectrogram (80 bins) → DREAM Stack (3×256) → Linear(27) → CTC Loss
```

### Coordinated DREAM ASR

```
Audio (16kHz) → Mel Spectrogram (80 bins) → Coordinated DREAM Stack (3×256) → Linear(27) → CTC Loss
                                        ↓
                            Top-down modulation + Inter-layer prediction loss
```

## Output Classes

27 classes: 26 letters (a-z) + space

- CTC blank token: index 0 (automatic)
- Space: index 1
- Letters a-z: indices 2-27

## Audio Preprocessing

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sample_rate` | 16000 Hz | Resampled from 22050 Hz |
| `n_mels` | 80 | Mel frequency bins |
| `hop_length` | 160 | 10ms at 16kHz |
| `n_fft` | 512 | 32ms window |
| `window` | hann | Hann window |

## Typical Training Times

| Dataset Size | Epochs | T4 GPU | A100 GPU |
|--------------|--------|--------|----------|
| Full (13100) | 50 | ~8 hours | ~3 hours |
| Subset (100) | 10 | ~5 minutes | ~2 minutes |
| Subset (1000) | 20 | ~30 minutes | ~10 minutes |

## Debugging Tips

### Enable Debug Mode

```bash
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --subset 10 \
    --debug \
    --log-interval 1
```

### Check Data Loading

```bash
python examples/dataset.py --root /content/drive/MyDrive/dream/dataset/ljspeech --subset 5
```

### Test Model Forward Pass

```bash
python examples/model.py --model dream
python examples/model.py --model coordinated
```

## Common Issues

### CUDA Out of Memory

Reduce batch size:
```bash
--batch-size 8  # or 4
```

### Slow Training

- Use `--use-amp` for mixed precision (2-3x faster on T4)
- Increase `--num-workers` if CPU is bottleneck
- Use `--subset` for quick prototyping

### NaN Loss

- Reduce learning rate: `--lr 1e-4`
- Enable gradient clipping: `--grad-clip 1.0`
- Check for corrupted audio files in dataset

## Citation

If you use this code, please cite:

```bibtex
@software{dream2026,
  title = {DREAM: Dynamic Recall and Elastic Adaptive Memory},
  author = {Your Name},
  year = {2026},
  version = {0.2.1}
}
```
