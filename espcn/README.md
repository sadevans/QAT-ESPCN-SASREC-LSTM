ESPCN: супер-разрешение с квантованием
=====================================

Этот модуль содержит реализацию ESPCN для супер‑разрешения изображений и полный пайплайн квантования:

- FP32 обучение (базовая модель)
- QAT: LSQ, APoT, QDrop
- PTQ: AdaRound
- бенчмарк качества и производительности

Структура
---------

```
espcn/
  train.py                   # обучение FP32, QAT и AdaRound
  benchmark_quant.py         # бенчмарк PyTorch-чекпоинтов (PSNR/latency/size)
  benchmark_onnx_int8.py     # бенчмарк готовых ONNX моделей (FP32/INT8)
  convert_to_int8_onnx.py    # экспорт PyTorch чекпоинта в ONNX + динамическое INT8
  aggregate_benchmark_md.py  # агрегация результатов в Markdown-таблицы
  download_datasets.py       # загрузка датасетов (DIV2K, Set5, Set14)
  data/
    dataloaders.py           # фабрики DataLoader для обучения и валидации
    datasets.py              # классы датасетов (DIV2KTrainDataset, SRBenchmarkDataset)
  model/
    base.py                  # базовая архитектура ESPCN
    quant.py                 # QuantESPCN + интеграция стратегий quant/

scripts/
  run_espcn_train.sh         # пакетное обучение FP32 + всех квантованных версий
  run_espcn_benchmark.sh     # последовательный запуск PyTorch-бенчмарков
  run_espcn_plots.sh         # построение трейдофф-графиков и распределений
```

Данные
------

Для обучения и оценки ESPCN используются стандартные датасеты для супер‑разрешения:

- **Обучение**: `DIV2K_train_HR` (путь задаётся в `configs/espcn/base.yaml` как `data.train_dir`),  
  из полноразмерных HR‑изображений генерируются LR‑патчи с бикубической деградацией (см. `DIV2KTrainDataset`).
- **Валидация и тест**: наборы эталонных изображений `Set5` и `Set14` (пути в `data.val_dirs`),  качество измеряется по PSNR/SSIM на HR.

**Загрузка датасетов:**

Перед началом работы необходимо загрузить датасеты:

```bash
python espcn/download_datasets.py --data-root data
```

Скрипт автоматически загрузит:
- `DIV2K_train_HR` из официального источника ETH Zurich
- `Set5` и `Set14` из figshare

Все датасеты будут сохранены в указанную директорию (`data` по умолчанию).

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

Автоматизированный запуск:

```bash
# Обучение FP32 + все методы квантования
bash scripts/run_espcn_train.sh

# Бенчмарк качества/производительности
bash scripts/run_espcn_benchmark.sh

# Построение графиков и распределений
bash scripts/run_espcn_plots.sh
```

Чекпоинты и результаты
----------------------

- Чекпоинты: `checkpoints/espcn_run/`
  - `espcn_fp32.pth`
  - `espcn_lsq.pth`
  - `espcn_apot.pth`
  - `espcn_qdrop.pth`
  - `espcn_adaround.pth`
- Тестовые метрики и полный конфиг каждого запуска: `results/espcn_results/<config>_results.json`

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




Экспорт в ONNX и INT8 квантование
----------------------------------

Скрипт `espcn/convert_to_int8_onnx.py` экспортирует PyTorch чекпоинты в ONNX формат и создаёт динамически квантованные INT8 версии:

```bash
PYTHONPATH=. python espcn/convert_to_int8_onnx.py \
  --inputs results/espcn_quant_benchmark.json \
  --base-config configs/espcn/base.yaml \
  --onnx-dir onnx_exports \
  --opset 18
```

Этот скрипт читает записи из JSON файлов бенчмарка, загружает соответствующие чекпоинты, конвертирует квантованные веса в статические и экспортирует в ONNX.

**Бенчмарк ONNX моделей:**

Скрипт `espcn/benchmark_onnx_int8.py` сравнивает качество и производительность FP32 и INT8 ONNX моделей:

```bash
PYTHONPATH=. python espcn/benchmark_onnx_int8.py \
  --base-config configs/espcn/base.yaml \
  --onnx-fp32 onnx_exports/espcn_fp32.onnx \
  --onnx-int8 onnx_exports/espcn_int8.onnx \
  --results-out results/espcn_onnx_benchmark.json
```

Скрипт выполняет последовательность операций: загружает ONNX модели, оценивает качество (PSNR) и производительность (latency, throughput) на CPU, и сохраняет отчёт с метриками.

**Известные ограничения экспорта:** PyTorch ≥2.1 по умолчанию использует новый экспорт через `torch.export`. Если во время экспорта возникает ошибка `GuardOnDataDependentSymNode` (обычно в `quant/adaround.py` при проверке `bool(alpha_init)`), скрипт автоматически переключается на классический экспортатор (`TORCH_ONNX_EXPERIMENTAL_EXPORTER=0`).


Итоги бенчмарка
---------------

| Метод    | PSNR (dB) | SSIM   | Throughput (img/s, CPU) | Avg latency (ms) | Размер (MB) |
|----------|-----------|--------|-------------------------|------------------|-------------|
| FP32     | 24.84     | 0.9541 | 65.95                   | 15.16            | 0.29        |
| LSQ      | 21.27     | 0.9090 | 41.41                   | 24.15            | 0.30        |
| APoT     | 22.48     | 0.92 | 35.98                   | 27.79            | 0.29        |
| QDrop    | 24.45     | 0.9502 | 35.33                   | 28.30            | 0.30        |
| AdaRound | 24.83     | 0.9541 | 64.99                   | 15.39            | 0.20        |

**Лучший баланс даёт AdaRound**: он сохраняет PSNR/SSIM на уровне FP32 (разница <0.01 дБ / <0.0001), немного уменьшает размер чекпоинта (~0.20 MB против 0.29 MB) и практически не проигрывает в latency на CPU. QDrop близок по качеству, но почти вдвое медленнее; LSQ и APoT требуют дополнительного тюнинга или лучшей настройки параметров, поскольку снижают PSNR. Поэтому для продакшен‑инференса на CPU рекомендуется связка FP32→AdaRound→ONNX INT8.

`results/espcn_quant_benchmark.md` содержит агрегированную Markdown‑таблицу, сформированную скриптом:

```bash
PYTHONPATH=. python espcn/aggregate_benchmark_md.py \
  --input results/espcn_quant_benchmark.json \
  --output results/espcn_quant_benchmark.md
```

ONNX INT8 результаты
--------------------

| Модель | PSNR (dB) | Avg latency (ms) | Throughput (fps) | Размер (MB) |
|--------|-----------|------------------|------------------|-------------|
| FP32   | 24.8445   | 8.68             | 115.16           | 0.0117      |
| INT8   | 24.8428   | 8.72             | 114.67           | 0.0118      |

Выводы
------

- **FP32** остаётся эталоном: 24.84 dB / 0.9541 SSIM при ~15 мс на CPU и служит базой для всех PTQ/QAT сценариев.
- **LSQ** демонстрирует устойчивую скорость, но на текущих настройках теряет ≈3.6 dB и 0.045 SSIM, поэтому нуждается в лучшем подборе параметров (регуляризация, контроль scale_param).
- **APoT** в приведённой конфигурации деградирует сильнее остальных (PSNR ≈22.5 dB);
- **QDrop** почти не уступает FP32 по качеству (24.45 dB / 0.9502 SSIM), однако платит за это увеличением латентности (≈28 мс на CPU). Подходит, если приоритет — качество, а время работы не критично.
- **AdaRound** обеспечивает лучший баланс (24.83 dB / 0.9541 SSIM при 15.39 мс и размере чекпоинта ~0.20 MB) и выбран в качестве основного PTQ-варианта для CPU-инференса.
- Экспорт AdaRound в ONNX и динамическое INT8-квантование практически не меняют поведение модели: FP32 ONNX даёт 24.844 dB при 8.68 мс, INT8 ONNX — 24.843 dB при 8.72 мс, throughput отличаются менее чем на 0.5 fps.

