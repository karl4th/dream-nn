# Changelog

All notable changes to DREAM (Dynamic Recall and Elastic Adaptive Memory) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.2] - 2026-03-05

### Added
- **Benchmark Suite** — полное сравнение с LSTM и Transformer на 3 задачах:
  - Test 1: Basic ASR Reconstruction (99.9% улучшение)
  - Test 2: Speaker Adaptation (0 шагов адаптации)
  - Test 3: Noise Robustness (1.09× при 10dB SNR)
- **Визуализация результатов** — PDF/PNG графики для публикаций
- **Техническая документация**:
  - `TECHNICAL_REPORT.md` — детальное описание реализации
  - `second.md` — архитектура + результаты бенчмарков
  - `tests/benchmarks/README.md` — руководство по бенчмаркам

### Changed
- **Surprise Gate** — относительная ошибка вместо абсолютной (лучшее детектирование)
- **Параметры по умолчанию** (оптимизированы для ASR):
  - `forgetting_rate`: 0.01 → 0.005
  - `base_plasticity`: 0.1 → 0.5
  - `base_threshold`: 0.5 → 0.3
  - `entropy_influence`: 0.2 → 0.1
  - `surprise_temperature`: 0.1 → 0.05
  - `error_smoothing`: 0.01 → 0.05
  - `surprise_smoothing`: 0.01 → 0.05
  - `ltc_tau_sys`: 10.0 → 5.0
  - `ltc_surprise_scale`: 10.0 → 5.0
- **Delta computation** — упрощённая реализация (быстрее и стабильнее)

### Fixed
- Исправлены тесты для новых параметров
- Обновлена документация с актуальными значениями

### Performance
- **DREAM vs LSTM vs Transformer** (100 эпох, 9 аудио файлов):
  - DREAM: 99.9% улучшение, 82K параметров, 502s
  - LSTM: 93.9% улучшение, 893K параметров, 9s
  - Transformer: 92.6% улучшение, 551K параметров, 11s

---

## [0.1.1] - 2026-02-20

### Added
- Базовая реализация DREAM cell
- High-level API (DREAM, DREAMStack)
- Unit tests (17 тестов)
- Публикация на PyPI

---

## [0.1.0] - 2026-02-18

### Added
- Initial release
- NNAI-S architecture implementation
- Predictive coding + STDP + LTC

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| 0.1.2 | 2026-03-05 | Benchmark suite, оптимизация параметров |
| 0.1.1 | 2026-02-20 | Базовая публикация на PyPI |
| 0.1.0 | 2026-02-18 | Initial release |
