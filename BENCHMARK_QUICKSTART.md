# DREAM Benchmark - Quick Start

## Запуск всех 5 тестов одной командой

```bash
uv run python -m dream.benchmarks.run_all --audio-dir /root/.cache/kagglehub/datasets/dromosys/ljspeech/versions/1/LJSpeech-1.1 --device cuda
```

## Параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--audio-dir` | Путь к LJSpeech-1.1 | **Обязательно** |
| `--tests` | Какие тесты запустить | `1,2,3,4,5` |
| `--n-files` | Сколько аудио файлов использовать | `10` |
| `--epochs` | Количество эпох для тестов 1,4 | `50` |
| `--device` | cuda или cpu | auto-detect |

## Примеры

### Запустить все 5 тестов
```bash
uv run python -m dream.benchmarks.run_all \
    --audio-dir /root/.cache/kagglehub/datasets/dromosys/ljspeech/versions/1/LJSpeech-1.1 \
    --device cuda
```

### Запустить только тесты 1, 3, 5
```bash
uv run python -m dream.benchmarks.run_all \
    --audio-dir /root/.cache/kagglehub/datasets/dromosys/ljspeech/versions/1/LJSpeech-1.1 \
    --tests 1,3,5 \
    --device cuda
```

### Запустить с 20 файлами и 100 эпохами
```bash
uv run python -m dream.benchmarks.run_all \
    --audio-dir /root/.cache/kagglehub/datasets/dromosys/ljspeech/versions/1/LJSpeech-1.1 \
    --n-files 20 \
    --epochs 100 \
    --device cuda
```

## Что делает каждый тест

| Тест | Файлов | Описание | Время |
|------|--------|----------|-------|
| 1 | 10 | ASR Reconstruction | ~10 мин |
| 2 | 10 | Speaker Adaptation | ~2 мин |
| 3 | 1 | Noise Robustness | ~2 мин |
| 4 | 10 | Stack Coordination | ~10 мин |
| 5 | 0 | Hierarchy (синтетика) | ~2 мин |

**Общее время:** ~25-30 минут на T4 GPU

## Результаты

После запуска в `tests/benchmarks/results/`:
- `benchmark_summary.json` — краткие результаты
- `BENCHMARK_REPORT.md` — подробный отчёт
- `results_*.json` — детали по каждому тесту

## Если что-то пошло не так

### Нет CUDA
```bash
uv run python -m dream.benchmarks.run_all \
    --audio-dir /path/to/LJSpeech-1.1 \
    --device cpu
```

### Мало памяти
```bash
uv run python -m dream.benchmarks.run_all \
    --audio-dir /path/to/LJSpeech-1.1 \
    --n-files 5 \
    --hidden-dim 128 \
    --device cuda
```
