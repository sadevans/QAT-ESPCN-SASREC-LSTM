SASRec: квантование рекомендательной модели
===========================================

Этот модуль содержит реализацию SASRec (Self-Attentive Sequential Recommendation) с поддержкой:

- FP32 обучения
- QAT: LSQ, APoT, QDrop
- PTQ: AdaRound

Структура
---------

```
sasrec/
  train.py                 # обучение FP32, QAT и AdaRound
  benchmark_quant.py       # бенчмарк качества и производительности
  plot_quant_analysis.py   # построение графиков
  compute_metrics_cpu.py   # агрегация CPU-метрик из json логов
  data/
    dataloaders.py
    dataset.py
  model/
    base.py
    quant.py
  download_data.py         # скачивание MovieLens-1M
```

Датасет
-------

Используется MovieLens‑1M (рейтинг фильмов):

- исходные данные: `ratings.dat` формата `UserID::MovieID::Rating::Timestamp`;
- разбиение на train/val/test выполняется по схеме leave‑one‑out  
  (последний просмотр пользователя — test, предпоследний — val, остальные — train).

Скачивание:

```bash
python sasrec/download_data.py
```

Конкретный путь к данным задаётся в `configs/sasrec/base.yaml` (`data.path`),  
по умолчанию предполагается распаковка `ml-1m` в указанную директорию.

Конфиги
-------

Конфигурации SASRec находятся в `configs/sasrec/`:

- `base.yaml` — базовые настройки модели, данных, путей и логирования  
- `sasrec_fp32.yaml` — обучение FP32 модели  
- `sasrec_lsq.yaml` — QAT с LSQ  
- `sasrec_apot.yaml` — QAT с APoT  
- `sasrec_qdrop.yaml` — QAT с QDrop  
- `sasrec_adaround.yaml` — PTQ с AdaRound (использует предварительно обученный FP32 чекпоинт)

Ключевые блоки:

- `model`: `hidden_units`, `num_blocks`, `num_heads`, `maxlen`, `dropout_rate`
- `training`: `epochs`, `batch_size`, `eval_interval`
- `quantization`: `method`, `weight_bits`, `act_bits`, дополнительные параметры стратегий
- `paths`: `checkpoints_dir` (по умолчанию `./checkpoints/sasrec_runs`), `results_dir`

Запуск обучения
---------------

Перед запуском убедитесь, что PYTHONPATH указывает на корень проекта:

```bash
export PYTHONPATH="./:$PYTHONPATH"
```

FP32:

```bash
CUDA_VISIBLE_DEVICES=0 python sasrec/train.py --config configs/sasrec/sasrec_fp32.yaml
```

QAT (пример LSQ):

```bash
CUDA_VISIBLE_DEVICES=0 python sasrec/train.py --config configs/sasrec/sasrec_lsq.yaml
```

AdaRound (PTQ):

- В `configs/sasrec/sasrec_adaround.yaml` должен быть указан валидный `base_checkpoint` на FP32 чекпоинт.

```bash
CUDA_VISIBLE_DEVICES=0 python sasrec/train.py --config configs/sasrec/sasrec_adaround.yaml
```

Скрипт для запуска всех обучений:

```bash
scripts/run_sasrec_train.sh 0
```

Чекпоинты и результаты
----------------------

Чекпоинты сохраняются в подкаталоги `paths.checkpoints_dir` (по умолчанию `./checkpoints/sasrec_runs`):

- `./checkpoints/sasrec_runs/sasrec_fp32/`
- `./checkpoints/sasrec_runs/sasrec_lsq/`
- `./checkpoints/sasrec_runs/sasrec_apot/`
- `./checkpoints/sasrec_runs/sasrec_qdrop/`
- `./checkpoints/sasrec_runs/sasrec_adaround/`

Имена файлов:

- `sasrec_fp32.pth`
- `sasrec_lsq.pth`
- `sasrec_apot.pth`
- `sasrec_qdrop.pth`
- `sasrec_adaround.pth`

Логирование в ClearML
---------------------

Логирование настраивается через `logging` в `configs/sasrec/base.yaml`:

```yaml
logging:
  backend: clearml
  clearml:
    project_name: "SASREC-QAT"
    task_name: none          # имя задачи берется из имени YAML-конфига
    tags: ["recommendation", "sasrec"]
```

В `train.py` логируются:

- train loss
- val NDCG@10, val Hit@10
- при PTQ AdaRound: финальные NDCG/Hit + метрики калибровки (если включен логер)

Для локальных тестов ClearML можно отключить:

```bash
export CLEARML_DISABLE=1
```

Бенчмарк производительности и качества
--------------------------------------

`sasrec/benchmark_quant.py` сравнивает FP32 и все квантованные модели.

Для каждого метода считает:

- `ndcg`, `hit` (NDCG@10 и Hit@10) на CPU
- throughput (батчей в секунду) и среднюю/медианную задержку (мс на батч) на CPU
- размер чекпоинта в мегабайтах

Запуск:

```bash
scripts/run_sasrec_benchmark.sh
```

Результаты сохраняются в `results/sasrec_quant_benchmark.json`.

Скрипт `sasrec/aggregate_benchmark_md.py` конвертирует JSON в Markdown‑таблицу:

```bash
PYTHONPATH=. python sasrec/aggregate_benchmark_md.py \
  --input results/sasrec_quant_benchmark.json \
  --output results/sasrec_quant_benchmark.md
```

Построение графиков
-------------------

`sasrec/plot_quant_analysis.py` строит:

- NDCG@10 vs latency
- NDCG@10 vs размер модели
- Hit@10 vs latency
- распределения активаций FP32 и квантованных моделей (по Linear-слоям)

Запуск:

```bash
scripts/run_sasrec_plots.sh
```

Графики сохраняются в `results/plots_sasrec/`.
