# DREAM Benchmark Suite

Comprehensive benchmarks for comparing DREAM, LSTM, and Transformer models on audio tasks.

Based on **DREAM Architecture Specification Section 7**.

---

## Quick Start

### Run All Benchmarks

```bash
# Full benchmark suite (10-30 minutes)
uv run python tests/benchmarks/run_all.py
```

### Run Individual Tests

```bash
# Test 1: Basic ASR Reconstruction
uv run python tests/benchmarks/test_01_basic_asr.py

# Test 2: Speaker Adaptation
uv run python tests/benchmarks/test_02_speaker_adaptation.py

# Test 3: Noise Robustness
uv run python tests/benchmarks/test_03_noise_robustness.py
```

---

### Run All Benchmarks

```bash
# Full benchmark suite (10-30 minutes)
uv run python tests/benchmarks/run_all.py
```

### Run Individual Tests

```bash
# Test 1: Basic ASR Reconstruction
uv run python tests/benchmarks/test_01_basic_asr.py

# Test 2: Speaker Adaptation
uv run python tests/benchmarks/test_02_speaker_adaptation.py

# Test 3: Noise Robustness
uv run python tests/benchmarks/test_03_noise_robustness.py
```

---

## Test Descriptions

### Test 1: Basic ASR Reconstruction

**Goal:** Evaluate model's ability to memorize and reconstruct audio patterns.

**Methodology:**
- Train on 9 audio files (mel spectrograms, 80 bins)
- Reconstruction task (autoencoding)
- Monitor loss convergence over 100 epochs

**Success Criteria (Spec 7.5):**
- >50% loss improvement
- Stable convergence

**Expected Results:**
| Model | Expected Performance |
|-------|---------------------|
| DREAM | Fast convergence, online adaptation |
| LSTM | Good reconstruction, slower |
| Transformer | Best with data, no online learning |

---

### Test 2: Speaker Adaptation

**Goal:** Measure adaptation speed when speaker changes mid-sequence.

**Methodology:**
- Concatenate two different speakers
- Process with persistent state
- Measure steps to recover baseline loss

**Success Criteria (Spec 7.5):**
- DREAM: Adapts within <50 steps
- LSTM/Transformer: No online adaptation

**Expected Results:**
| Model | Adaptation Steps |
|-------|-----------------|
| DREAM | <50 (fast weights) |
| LSTM | N/A (requires retraining) |
| Transformer | N/A (no recurrence) |

---

### Test 3: Noise Robustness

**Goal:** Evaluate robustness to additive noise at different SNR levels.

**Methodology:**
- Add white noise at SNR: 20dB, 10dB, 5dB, 0dB
- Measure reconstruction loss
- Track surprise response (DREAM only)

**Success Criteria (Spec 7.5):**
- Loss doesn't explode at moderate noise (<3x at 10dB)
- Surprise gate detects noise (DREAM)

**Expected Results:**
| Model | Noise Robustness | Surprise Detection |
|-------|-----------------|-------------------|
| DREAM | Stable, filters noise | ✅ Yes |
| LSTM | Degrades gracefully | N/A |
| Transformer | May overfit to clean | N/A |

---

## Models Compared

### DREAM (Ours)
- **Architecture:** Predictive coding + STDP + LTC
- **Parameters:** ~30K (hidden=256, rank=16)
- **Key Feature:** Online adaptation via fast weights

### LSTM (Baseline)
- **Architecture:** 2-layer LSTM
- **Parameters:** ~600K (hidden=256)
- **Key Feature:** Standard sequence modeling

### Transformer (Baseline)
- **Architecture:** 4-layer Transformer Encoder
- **Parameters:** ~200K (d_model=128)
- **Key Feature:** Self-attention, no recurrence

---

## Output Files

After running benchmarks:

```
tests/benchmarks/
├── results/
│   ├── results_basic_asr.json           # Test 1 results
│   ├── results_speaker_adaptation.json  # Test 2 results
│   ├── results_noise_robustness.json    # Test 3 results
│   ├── figures/
│   │   ├── fig1_training_curves.pdf     # Training curves (for paper)
│   │   ├── fig2_speaker_adaptation.pdf  # Adaptation results
│   │   ├── fig3_noise_robustness.pdf    # Noise robustness
│   │   ├── table_summary.pdf            # Summary table
│   │   └── benchmark_table.tex          # LaTeX table
│   └── BENCHMARK_REPORT.md              # Full report
```

### For arxiv.org Submission

1. **Figures**: Use PDF files from `results/figures/`
   - Fig. 1: Training convergence (`fig1_training_curves.pdf`)
   - Fig. 2: Speaker adaptation (`fig2_speaker_adaptation.pdf`)
   - Fig. 3: Noise robustness (`fig3_noise_robustness.pdf`)

2. **Tables**: Copy `benchmark_table.tex` into your LaTeX paper

3. **Citation**:
```bibtex
@software{dream-benchmarks,
  title = {DREAM Benchmark Suite},
  author = {Manifestro Team},
  year = {2026},
  url = {https://github.com/karl4th/dream-nn}
}
```

---

## Configuration

### Command Line Arguments

All tests accept:

```bash
--audio-dir    # Directory with .wav files (default: audio_test)
--hidden-dim   # Hidden dimension (default: 256)
--device       # cuda/cpu (default: auto-detect)
--epochs       # Training epochs (Test 1 only, default: 100)
```

### Example

```bash
uv run python tests/benchmarks/test_01_basic_asr.py \
    --audio-dir /path/to/audio \
    --hidden-dim 512 \
    --epochs 200 \
    --device cuda
```

---

## Hardware Requirements

**Minimum:**
- 8GB RAM
- CPU (slower)

**Recommended:**
- 16GB RAM
- GPU with 4GB+ VRAM
- SSD for fast audio loading

**Estimated Runtime:**
- Test 1 (ASR): ~5-10 min per model
- Test 2 (Adaptation): ~1 min per model
- Test 3 (Noise): ~2 min per model
- **Total:** ~15-30 minutes

---

## Interpreting Results

### Key Metrics

1. **Improvement %** (Test 1)
   - Higher = better learning
   - DREAM expected: >90%

2. **Adaptation Steps** (Test 2)
   - Lower = faster adaptation
   - DREAM expected: <50

3. **Loss Ratio** (Test 3)
   - Lower = more robust
   - DREAM expected: <3x at 10dB

### Success Indicators

✅ **DREAM Advantages:**
- Fast online adaptation (Test 1)
- Instant speaker adaptation (Test 2)
- Noise detection via surprise (Test 3)

❌ **Potential Issues:**
- Slow convergence → Check learning rate
- No adaptation → Verify state persistence
- Surprise saturation → Adjust gamma

---

## Citation

If you use these benchmarks:

```bibtex
@software{dream-benchmarks,
  title = {DREAM Benchmark Suite},
  author = {Manifestro Team},
  year = {2026},
  url = {https://github.com/karl4th/dream-nn}
}
```
