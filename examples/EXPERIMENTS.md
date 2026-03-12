# Эксперименты: DREAM ASR на LJSpeech

## Быстрый старт

### 1. Проверка данных

Убедись что датасет существует:
```bash
ls /content/drive/MyDrive/dream/dataset/ljspeech/wavs | head
ls /content/drive/MyDrive/dream/dataset/ljspeech/metadata.csv | head
```

### 2. Запуск базового эксперимента

**Стандартный DREAM (полный датасет):**
```bash
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --log-dir /content/drive/MyDrive/dream/experiments/dream_base \
    --epochs 50 \
    --batch-size 16 \
    --use-amp
```

**Coordinated DREAM:**
```bash
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --model coordinated \
    --log-dir /content/drive/MyDrive/dream/experiments/dream_coordinated \
    --epochs 50 \
    --batch-size 16 \
    --use-amp
```

### 3. Параллельный запуск (сравнение моделей)

```bash
# Терминал 1
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --model dream \
    --run-name dream_base \
    --log-dir /content/drive/MyDrive/dream/experiments \
    --epochs 50 \
    --use-amp &

# Терминал 2
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --model coordinated \
    --run-name dream_coordinated \
    --log-dir /content/drive/MyDrive/dream/experiments \
    --epochs 50 \
    --use-amp &
```

### 4. Subset режим (быстрый тест)

```bash
# 100 файлов для быстрой проверки
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --subset 100 \
    --log-dir /content/drive/MyDrive/dream/experiments/dream_subset \
    --epochs 10
```

## Ablation Studies

### Отключение быстрых весов

```bash
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --no-fast-weights \
    --log-dir /content/drive/MyDrive/dream/experiments/dream_no_fw \
    --epochs 50
```

### Отключение LTC

```bash
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --no-ltc \
    --log-dir /content/drive/MyDrive/dream/experiments/dream_no_ltc \
    --epochs 50
```

### Отключение Sleep

```bash
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --no-sleep \
    --log-dir /content/drive/MyDrive/dream/experiments/dream_no_sleep \
    --epochs 50
```

### Freeze Fast Weights (статичная база)

```bash
python examples/train.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --freeze-fast-weights \
    --log-dir /content/drive/MyDrive/dream/experiments/dream_static \
    --epochs 50
```

## Мониторинг обучения

### Просмотр логов

```bash
# Последние строки логов
tail -f /content/drive/MyDrive/dream/experiments/dream_base_*/logs.txt

# Метрики в JSON
cat /content/drive/MyDrive/dream/experiments/dream_base_*/metrics.json | python -m json.tool
```

### Загрузка чекпоинта

```python
import torch
from examples.model import create_model
from examples.trainer import ASRTrainer

# Загрузить модель
checkpoint = torch.load('/content/drive/MyDrive/dream/experiments/dream_base/best.pt')
model = create_model('dream')
model.load_state_dict(checkpoint['model_state_dict'])

# Продолжить обучение
trainer = ASRTrainer(model, learning_rate=1e-3)
trainer.load_checkpoint('/content/drive/MyDrive/dream/experiments/dream_base/best.pt')
trainer.train(train_loader, val_loader, num_epochs=50)
```

## Ожидаемые результаты

### Время обучения (T4 GPU)

| Dataset | Epochs | Batch Size | Time |
|---------|--------|------------|------|
| Full (13100) | 50 | 16 | ~8 часов |
| Subset (1000) | 20 | 16 | ~30 минут |
| Subset (100) | 10 | 16 | ~5 минут |

### CTC Loss (ожидаемый)

| Epoch | Train CTC | Val CTC |
|-------|-----------|---------|
| 10 | ~2.5 | ~2.6 |
| 20 | ~1.5 | ~1.6 |
| 30 | ~0.8 | ~0.9 |
| 50 | ~0.3 | ~0.4 |

## Диагностика проблем

### CUDA Out of Memory

Уменьши batch size:
```bash
--batch-size 8  # или 4
```

### NaN Loss

Уменьши learning rate и включи clip:
```bash
--lr 1e-4 --grad-clip 1.0
```

### Медленная загрузка данных

Увеличь num workers:
```bash
--num-workers 8
```

### Проверка данных

```bash
# Тест датасета
python examples/dataset.py \
    --root /content/drive/MyDrive/dream/dataset/ljspeech \
    --subset 5
```

## Структура экспериментов

```
/content/drive/MyDrive/dream/experiments/
├── dream_base_20260312_143022/
│   ├── best.pt           # Лучшая модель
│   ├── epoch_5.pt        # Чекпоинты каждые 5 эпох
│   ├── epoch_10.pt
│   ├── ...
│   ├── final.pt          # Финальная модель
│   └── metrics.json      # Метрики обучения
│
├── dream_coordinated_20260312_143022/
│   └── ...
│
├── dream_no_fw_20260312_143022/
│   └── ...
│
└── dream_no_ltc_20260312_143022/
    └── ...
```

## Сравнение моделей

После обучения сравни метрики:

```bash
python -c "
import json
import glob

for exp in glob.glob('/content/drive/MyDrive/dream/experiments/dream_*/metrics.json'):
    with open(exp) as f:
        data = json.load(f)
    
    name = exp.split('/')[-2]
    best_val = min(data['val_ctc_loss'])
    best_epoch = data['val_ctc_loss'].index(best_val) + 1
    
    print(f'{name:30} | Best Val CTC: {best_val:.4f} (epoch {best_epoch})')
"
```

## Ключевые вопросы для исследования

1. **Влияют ли быстрые веса на скорость обучения?**
   - Сравни `dream_base` vs `dream_no_fw`
   - Ожидаем: без быстрых весов медленнее сходимость

2. **Нужен ли LTC для ASR?**
   - Сравни `dream_base` vs `dream_no_ltc`
   - Ожидаем: LTC помогает с длинными последовательностями

3. **Работает ли координация?**
   - Сравни `dream_base` vs `dream_coordinated`
   - Ожидаем: координация улучшает accuracy но медленнее

4. **Влияет ли sleep на стабильность?**
   - Сравни `dream_base` vs `dream_no_sleep`
   - Ожидаем: без sleep модель менее стабильна

## Следующие шаги

1. Запустить 4 эксперимента параллельно:
   - `dream_base`
   - `dream_no_fw`
   - `dream_no_ltc`
   - `dream_coordinated`

2. Сравнить CTC loss после 50 эпох

3. Выбрать лучшую конфигурацию

4. Провести hyperparameter tuning (lr, hidden_dims, rank)
