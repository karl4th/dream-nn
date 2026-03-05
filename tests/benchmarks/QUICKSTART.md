# DREAM Benchmark - Quick Start Guide

## Для запуска на сервере

### 1. Запуск всех тестов

```bash
cd /path/to/dream-nn
uv run python tests/benchmarks/run_all.py
```

**Время выполнения:** 15-30 минут (зависит от GPU)

### 2. Генерация графиков

Автоматически запускается после тестов, или вручную:

```bash
uv run python tests/benchmarks/visualize.py
```

### 3. Результаты

Все файлы сохраняются в `tests/benchmarks/results/`:

```
results/
├── results_basic_asr.json           # Test 1
├── results_speaker_adaptation.json  # Test 2
├── results_noise_robustness.json    # Test 3
├── figures/
│   ├── fig1_training_curves.pdf     ← Для статьи
│   ├── fig2_speaker_adaptation.pdf  ← Для статьи
│   ├── fig3_noise_robustness.pdf    ← Для статьи
│   ├── table_summary.pdf
│   └── benchmark_table.tex          ← LaTeX таблица
└── BENCHMARK_REPORT.md
```

---

## Для arxiv.org

### Файлы для включения в submission:

1. **Figures** (из `results/figures/`):
   - `fig1_training_curves.pdf`
   - `fig2_speaker_adaptation.pdf`
   - `fig3_noise_robustness.pdf`

2. **Tables**:
   - Скопировать `benchmark_table.tex` в LaTeX

3. **Шаблон статьи**:
   - См. `PAPER_TEMPLATE.md`

---

## Ожидаемые результаты (Spec 7.5)

| Метрика | DREAM | LSTM | Transformer |
|---------|-------|------|-------------|
| **ASR Improvement** | >90% | ~70% | ~80% |
| **Adaptation Steps** | <50 | N/A | N/A |
| **Noise Ratio (10dB)** | <1.5x | ~1.1x | ~1.1x |
| **Surprise Detection** | ✅ | ❌ | ❌ |

---

## Команды для отдельных тестов

```bash
# Test 1: ASR Reconstruction
uv run python tests/benchmarks/test_01_basic_asr.py --epochs 100 --device cuda

# Test 2: Speaker Adaptation
uv run python tests/benchmarks/test_02_speaker_adaptation.py --device cuda

# Test 3: Noise Robustness
uv run python tests/benchmarks/test_03_noise_robustness.py --device cuda
```

---

## Аргументы командной строки

```bash
--audio-dir    # Папка с аудио (default: audio_test)
--hidden-dim   # Размер hidden state (default: 256)
--epochs       # Количество эпох (default: 100, только Test 1)
--device       # cuda/cpu (default: auto)
```

---

## Требования к железу

**Минимум:**
- 8GB RAM
- CPU (медленно)

**Рекомендуется:**
- GPU с 4GB+ VRAM
- 16GB RAM
- SSD

---

## Пример для Colab

```python
# Setup
!pip install dreamnn librosa matplotlib

# Run benchmarks
!python tests/benchmarks/run_all.py

# Visualize
!python tests/benchmarks/visualize.py

# Download results
from google.colab import files
files.download('tests/benchmarks/results/figures/fig1_training_curves.pdf')
files.download('tests/benchmarks/results/figures/fig2_speaker_adaptation.pdf')
files.download('tests/benchmarks/results/figures/fig3_noise_robustness.pdf')
files.download('tests/benchmarks/results/benchmark_table.tex')
```

---

## Контакты

Вопросы → GitHub Issues: https://github.com/karl4th/dream-nn/issues
