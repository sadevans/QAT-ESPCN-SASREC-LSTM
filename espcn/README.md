ESPCN: супер-разрешение с квантованием
=====================================

Этот модуль содержит реализацию ESPCN для супер‑разрешения изображений и полный пайплайн квантования:

- FP32 обучение (базовая модель)
- QAT: LSQ, APoT, QDrop
- PTQ: AdaRound
- бенчмаркинг качества и производительности
- визуализация влияния квантования на веса и активации

Структура
---------

```
espcn/
  train.py               # обучение FP32, QAT и AdaRound
  benchmark_quant.py     # бенчмарк всех методов квантизации
  plot_quant_analysis.py # построение графиков и анализ весов/активаций
  data/
    dataloaders.py
    datasets.py
  model/
    base.py              # базовая архитектура ESPCN
    quant.py             # оболочка QuantESPCN с поддержкой стратегий quant/
```

Данные
------

Для обучения и оценки ESPCN используются стандартные датасеты для супер‑разрешения:

- **Обучение**: `DIV2K_train_HR` (путь задаётся в `configs/espcn/base.yaml` как `data.train_dir`),  
  из полноразмерных HR‑изображений генерируются LR‑патчи с бикубической деградацией (см. `DIV2KTrainDataset`).
- **Валидация и тест**: наборы эталонных изображений `Set5` и `Set14` (пути в `data.val_dirs`),  
  для них также строятся LR‑версии (`SRBenchmarkDataset`), а качество измеряется по PSNR/SSIM на HR.

Конфиги
-------

Конфигурации для ESPCN находятся в `configs/espcn/`:

- `base.yaml` — базовые настройки модели, данных, путей и логирования
- `espcn_fp32.yaml` — обучение FP32 модели
- `espcn_lsq.yaml` — QAT с LSQ
- `espcn_apot.yaml` — QAT с APoT
- `espcn_qdrop.yaml` — QAT с QDrop
- `espcn_adaround.yaml` — PTQ с AdaRound (использует заранее обученный FP32 чекпоинт)

Запуск обучения
---------------

Перед запуском убедитесь, что PYTHONPATH указывает на корень проекта:

```bash
export PYTHONPATH="./:$PYTHONPATH"
```

FP32:

```bash
CUDA_VISIBLE_DEVICES=0 python espcn/train.py --config configs/espcn/espcn_fp32.yaml
```

QAT (пример для LSQ):

```bash
CUDA_VISIBLE_DEVICES=0 python espcn/train.py --config configs/espcn/espcn_lsq.yaml
```

AdaRound (PTQ):

- В `configs/espcn/espcn_adaround.yaml` должен быть указан путь к FP32 чекпоинту `base_checkpoint`.

```bash
CUDA_VISIBLE_DEVICES=0 python espcn/train.py --config configs/espcn/espcn_adaround.yaml
```

Чекпоинты и результаты
----------------------

- Чекпоинты: `checkpoints/espcn_run/`
  - `espcn_fp32.pth`
  - `espcn_lsq.pth`
  - `espcn_apot.pth`
  - `espcn_qdrop.pth`
  - `espcn_adaround.pth`
- Финальные метрики теста: `results/espcn_run_results.json`

Логирование в ClearML
---------------------

Логирование настраивается через секцию `logging` в `configs/espcn/base.yaml`:

```yaml
logging:
  backend: clearml
  clearml:
    project_name: "QAT"
    task_name: none             # имя задачи берется из имени конфигурационного файла
    tags: ["super-resolution", "espcn"]
    log_images: true
    log_image_interval: 5
```

`train.py` логирует:

- train loss, train PSNR
- val loss, val PSNR, val SSIM
- примеры LR/HR/SR изображений
- во время калибровки AdaRound: mse_loss, regularization и total_loss по итерациям

Бенчмарк квантизации
--------------------

Скрипт `espcn/benchmark_quant.py` сравнивает FP32 и все методы квантизации по качеству и производительности.

Он вычисляет для каждого метода:

- PSNR и SSIM на валидационном датасете
- throughput (изображений в секунду) и latency (мс на изображение) на CPU
- размер чекпоинта в мегабайтах

Пример запуска:

```bash
PYTHONPATH=. python espcn/benchmark_quant.py \
  --base-config configs/espcn/base.yaml \
  --checkpoint-dir checkpoints/espcn_run \
  --results-out results/espcn_quant_benchmark.json
```

Построение графиков
-------------------

Скрипт `espcn/plot_quant_analysis.py` использует результаты бенчмарка и строит:

- PSNR vs latency
- PSNR vs размер модели
- SSIM vs latency
- распределения весов FP32 и квантованных моделей
- распределения активаций FP32 и квантованных моделей для одной выборки

Пример запуска:

```bash
PYTHONPATH=. python espcn/plot_quant_analysis.py \
  --benchmark-json results/espcn_quant_benchmark.json \
  --base-config configs/espcn/base.yaml \
  --checkpoint-dir checkpoints/espcn_run \
  --out-dir results/plots_espcn
```

После этого в `results/plots_espcn/` будут лежать PNG‑графики с трейдоффами по качеству/скорости и распределениями весов/активаций.


