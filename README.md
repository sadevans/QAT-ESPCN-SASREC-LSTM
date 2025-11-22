# QAT-ESPCN-SASREC: Quantization-Aware Training для ESPCN и SASRec

Проект для обучения и квантизации моделей ESPCN (Efficient Sub-Pixel Convolutional Neural Network) для супер-разрешения изображений и SASRec (Self-Attentive Sequential Recommendation) для рекомендательных систем.

Проект поддерживает полный пайплайн квантизации: от обучения FP32 моделей до Post-Training Quantization (PTQ) и Quantization-Aware Training (QAT), включая экспорт в ONNX и динамическое INT8 квантование

## Возможности

### ESPCN (Super-Resolution)
- FP32 обучение базовой модели
- QAT: LSQ, APoT, QDrop
- PTQ: AdaRound
- Экспорт в ONNX и динамическое INT8 квантование
- Бенчмарк качества (PSNR, SSIM) и производительности (latency, throughput)
- Визуализация trade-offs и распределений весов/активаций

### SASRec (Sequential Recommendation)
- FP32 обучение базовой модели
- QAT: LSQ, APoT, QDrop
- PTQ: AdaRound
- Экспорт в ONNX и динамическое INT8 квантование
- Бенчмарк качества (NDCG@10, Hit@10) и производительности
- Визуализация метрик и распределений

## Структура проекта

```
QAT-ESPCN-SASREC/
├── espcn/                      # Модуль для супер-разрешения
│   ├── train.py               # Обучение FP32, QAT и AdaRound
│   ├── benchmark_quant.py     # Бенчмарк PyTorch-чекпоинтов
│   ├── benchmark_onnx_int8.py # Бенчмарк ONNX моделей (FP32/INT8)
│   ├── convert_to_int8_onnx.py # Экспорт в ONNX + динамическое INT8
│   ├── plot_quant_analysis.py # Построение графиков анализа
│   ├── aggregate_benchmark_md.py # Агрегация результатов в Markdown
│   ├── download_datasets.py   # Загрузка датасетов (DIV2K, Set5, Set14)
│   ├── data/                  # Датасеты и загрузчики
│   │   ├── datasets.py
│   │   └── dataloaders.py
│   ├── model/                 # Архитектура ESPCN
│   │   ├── base.py
│   │   └── quant.py
│   └── README.md              # Детальная документация ESPCN
│
├── sasrec/                     # Модуль для рекомендаций
│   ├── train.py               # Обучение FP32, QAT и AdaRound
│   ├── benchmark_quant.py     # Бенчмарк качества и производительности
│   ├── onnx_int8_benchmark.py # Бенчмарк ONNX моделей
│   ├── convert_to_int8_onnx.py # Экспорт в ONNX + динамическое INT8
│   ├── plot_quant_analysis.py # Построение графиков анализа
│   ├── aggregate_benchmark_md.py # Агрегация результатов в Markdown
│   ├── download_data.py       # Загрузка MovieLens-1M
│   ├── data/                  # Датасеты и загрузчики
│   │   ├── dataset.py
│   │   └── dataloaders.py
│   ├── model/                 # Архитектура SASRec
│   │   ├── base.py
│   │   └── quant.py
│   └── README.md              # Детальная документация SASRec
│
├── quant/                      # Стратегии квантизации
│   ├── base.py                # Базовые классы и утилиты
│   ├── lsq.py                 # LSQ (Learned Step Size Quantization)
│   ├── apot.py                # APoT (Additive Powers-of-Two)
│   ├── qdrop.py               # QDrop (Stochastic Quantization Drop)
│   └── adaround.py            # AdaRound (PTQ)
│
├── configs/                    # Конфигурационные файлы
│   ├── espcn/                 # Конфиги для ESPCN
│   │   ├── base.yaml
│   │   ├── espcn_fp32.yaml
│   │   ├── espcn_lsq.yaml
│   │   ├── espcn_apot.yaml
│   │   ├── espcn_qdrop.yaml
│   │   └── espcn_adaround.yaml
│   └── sasrec/                # Конфиги для SASRec
│       ├── base.yaml
│       ├── sasrec_fp32.yaml
│       ├── sasrec_lsq.yaml
│       ├── sasrec_apot.yaml
│       ├── sasrec_qdrop.yaml
│       └── sasrec_adaround.yaml
│
├── scripts/                    # Bash-скрипты для автоматизации
│   ├── run_espcn_train.sh     # Обучение всех ESPCN моделей
│   ├── run_espcn_benchmark.sh # Бенчмарк ESPCN моделей
│   ├── run_espcn_plots.sh     # Построение графиков для ESPCN
│   ├── run_sasrec_train.sh    # Обучение всех SASRec моделей
│   ├── run_sasrec_benchmark.sh # Бенчмарк SASRec моделей
│   └── run_sasrec_plots.sh    # Построение графиков для SASRec
│
├── checkpoints/                # Сохранённые чекпоинты моделей
│   ├── espcn_run/             # ESPCN чекпоинты
│   └── sasrec_runs/           # SASRec чекпоинты
│
├── results/                    # Результаты экспериментов
│   ├── espcn_results/         # Результаты ESPCN
│   ├── plots_espcn/           # Графики для ESPCN
│   └── sasrec_*.json          # Результаты SASRec
│
├── utils.py                    # Общие утилиты
├── requirements.txt            # Зависимости проекта
└── README.md                   # Этот файл
```

## Установка

### Требования

- Python 3.8+
- CUDA-capable GPU (рекомендуется)
- PyTorch 2.0+

### Установка зависимостей

```bash
# Создание виртуального окружения
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### Настройка PYTHONPATH

Добавьте корень проекта в PYTHONPATH:

```bash
# Для текущей сессии
export PYTHONPATH="./:$PYTHONPATH"

# Или для всех сессий (добавить в ~/.bashrc)
echo "export PYTHONPATH=./:\$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc
```

## Подготовка данных

### ESPCN (Super-Resolution)

Для обучения ESPCN требуется датасет DIV2K и бенчмарк-наборы Set5/Set14:

```bash
python espcn/download_datasets.py --data-root data
```

Датасеты будут сохранены в:
- `data/DIV2K_train_HR/` — обучающие изображения
- `data/Set5/` — валидационный набор
- `data/Set14/` — тестовый набор

### SASRec (Sequential Recommendation)

Для обучения SASRec требуется датасет MovieLens-1M:

```bash
python sasrec/download_data.py
```

Данные будут сохранены в `data/ml-1m/` с автоматической распаковкой.

## Методы квантизации

Проект поддерживает несколько стратегий квантизации:

### Quantization-Aware Training (QAT)

1. **LSQ (Learned Step Size Quantization)**
   - Обучаемый размер шага квантизации
   - Поддерживает per-channel и per-tensor квантование
   - Метод: `quantization.method: lsq`

2. **APoT (Additive Powers-of-Two)**
   - Квантирование с аддитивными степенями двойки
   - Параметры: `apot_m` (количество групп), `apot_k` (биты на группу)
   - Метод: `quantization.method: apot`

3. **QDrop (Stochastic Quantization Drop)**
   - Стохастический обход квантизации во время обучения
   - Параметр: `qdrop_p` (вероятность обхода)
   - Метод: `quantization.method: qdrop`

### Post-Training Quantization (PTQ)

4. **AdaRound**
   - Оптимизация округления весов для минимизации ошибки
   - Требует калибровочный датасет
   - Метод: `quantization.method: adaround`

## Быстрый старт

### ESPCN

Подробная документация: [`espcn/README.md`](espcn/README.md)

**Обучение FP32 модели:**
```bash
CUDA_VISIBLE_DEVICES=0 python espcn/train.py --config configs/espcn/espcn_fp32.yaml
```

**Обучение с квантизацией (пример LSQ):**
```bash
CUDA_VISIBLE_DEVICES=0 python espcn/train.py --config configs/espcn/espcn_lsq.yaml
```

**Автоматизированный запуск всех моделей:**
```bash
bash scripts/run_espcn_train.sh
```

**Бенчмарк:**
```bash
bash scripts/run_espcn_benchmark.sh
```

**Построение графиков:**
```bash
bash scripts/run_espcn_plots.sh
```

### SASRec

Подробная документация: [`sasrec/README.md`](sasrec/README.md)

**Обучение FP32 модели:**
```bash
CUDA_VISIBLE_DEVICES=0 python sasrec/train.py --config configs/sasrec/sasrec_fp32.yaml
```

**Обучение с квантизацией (пример LSQ):**
```bash
CUDA_VISIBLE_DEVICES=0 python sasrec/train.py --config configs/sasrec/sasrec_lsq.yaml
```

**Автоматизированный запуск всех моделей:**
```bash
bash scripts/run_sasrec_train.sh
```

**Бенчмарк:**
```bash
bash scripts/run_sasrec_benchmark.sh
```

**Построение графиков:**
```bash
bash scripts/run_sasrec_plots.sh
```

## Логирование в ClearML

Проект поддерживает автоматическое логирование экспериментов в ClearML.

### Настройка

Логирование настраивается через секцию `logging` в конфигурационных файлах:

```yaml
logging:
  backend: clearml
  clearml:
    project_name: "QAT-ESPCN-SASREC"
    task_name: null  # Автоматически генерируется из имени конфига
    tags: ["super-resolution", "espcn"]  # или ["recommendation", "sasrec"]
    log_images: true
    log_image_interval: 5
```

### Логируемые метрики

**ESPCN:**
- Train loss, Train PSNR
- Val loss, Val PSNR, Val SSIM
- Примеры изображений (LR, SR, HR)

**SASRec:**
- Train loss
- Val loss, Val NDCG@10, Val Hit@10
- Test NDCG@10, Test Hit@10

**AdaRound калибровка:**
- MSE loss, Regularization, Total loss (по итерациям)

### Отключение логирования

Для отключения ClearML во время тестирования:

```bash
CLEARML_DISABLE=1 python espcn/train.py --config configs/espcn/espcn_fp32.yaml
```

## Экспорт в ONNX и INT8 квантование

Проект поддерживает экспорт квантизованных моделей в ONNX формат с динамическим INT8 квантированием.

### ESPCN

**Экспорт в ONNX:**
```bash
PYTHONPATH=. python espcn/convert_to_int8_onnx.py \
  --inputs results/espcn_quant_benchmark.json \
  --base-config configs/espcn/base.yaml \
  --onnx-dir onnx_exports \
  --opset 18
```

**Бенчмарк ONNX моделей:**
```bash
PYTHONPATH=. python espcn/benchmark_onnx_int8.py \
  --base-config configs/espcn/base.yaml \
  --onnx-fp32 onnx_exports/espcn_fp32.onnx \
  --onnx-int8 onnx_exports/espcn_int8.onnx \
  --results-out results/espcn_onnx_benchmark.json
```

### SASRec

**Экспорт в ONNX:**
```bash
PYTHONPATH=. python sasrec/convert_to_int8_onnx.py \
  --inputs results/sasrec_quant_benchmark.json \
  --base-config configs/sasrec/base.yaml \
  --onnx-dir onnx_exports \
  --opset 18
```

**Бенчмарк ONNX моделей:**
```bash
PYTHONPATH=. python sasrec/onnx_int8_benchmark.py \
  --base-config configs/sasrec/base.yaml \
  --onnx-fp32 onnx_exports/sasrec_fp32.onnx \
  --onnx-int8 onnx_exports/sasrec_int8.onnx \
  --results-out results/sasrec_onnx_benchmark.json
```

## Результаты и чекпоинты

### Чекпоинты

Чекпоинты сохраняются в следующие директории:

- **ESPCN:** `checkpoints/espcn_run/`
  - `espcn_fp32.pth`
  - `espcn_lsq.pth`
  - `espcn_apot.pth`
  - `espcn_qdrop.pth`
  - `espcn_adaround.pth`

- **SASRec:** `checkpoints/sasrec_runs/<method_name>/`
  - `sasrec_fp32.pth`
  - `sasrec_lsq.pth`
  - `sasrec_apot.pth`
  - `sasrec_qdrop.pth`
  - `sasrec_adaround.pth`

### Результаты

Результаты экспериментов сохраняются в `results/`:

- **ESPCN:**
  - `results/espcn_quant_benchmark.json` — бенчмарк PyTorch моделей
  - `results/espcn_onnx_benchmark.json` — бенчмарк ONNX моделей
  - `results/espcn_quant_benchmark.md` — агрегированные таблицы
  - `results/plots_espcn/` — графики анализа

- **SASRec:**
  - `results/sasrec_quant_benchmark.json` — бенчмарк моделей
  - `results/sasrec_onnx_benchmark.json` — бенчмарк ONNX моделей
  - `results/sasrec_quant_benchmark.md` — агрегированные таблицы
  - `results/plots_sasrec/` — графики анализа

## Детальная документация

- **[ESPCN README](espcn/README.md)** — подробная документация для модуля супер-разрешения
- **[SASRec README](sasrec/README.md)** — подробная документация для модуля рекомендаций

