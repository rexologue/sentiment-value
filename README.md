# Sentiment Value: полнофункциональный пайплайн тональной классификации

## Описание проекта
Репозиторий объединяет полный цикл работы с моделью анализа тональности текста:
- классическое обучение классификатора (`classifier_training`);
- кластеризацию эмбеддингов и псевдоразметку для расширения данных (`clustering`);
- совместное обучение классификации и metric learning (`metric_classifier_training`);
- сервис инференса на FastAPI для единичных текстов и CSV (`runtime`).

Базовая задача — классификация тональности (например, негатив/нейтраль/позитив). Пайплайн опирается на Hugging Face модели (например, `jhu-clsp/mmBERT-base`) и дополняется metric learning и кластеризацией для повышения качества на «трудных» данных.

## Архитектура и взаимосвязь модулей
- **`classifier_training`** — базовое обучение модели по меткам из Parquet. Выдаёт чекпоинты и метрики качества.
- **`clustering`** — прогон модели по сырым текстам → сбор CLS-векторов → нормализация/PCA → MiniBatchKMeans (CPU или GPU) → кластерные метки, оценка purity и обновление датасетов псевдоразметкой.
- **`metric_classifier_training`** — дообучение модели, совмещающее CrossEntropy по размеченной части и SupCon для metric learning по выбранным примерам (например, после кластеризации и фильтрации).
- **`runtime`** — обёртка FastAPI над обученной моделью для онлайн-инференса: одиночные запросы, пакетная обработка CSV, сервисы лемматизации/NER и health-check.

## Основные возможности
- Обучение классификатора текста с поддержкой mixed precision, gradient accumulation, scheduler, label smoothing и Neptune-логированием.
- Генерация эмбеддингов, кластеризация (MiniBatchKMeans, опционально GPU через CuML), расчёт purity и обновление датасета псевдоразметкой.
- Совместное обучение классификации и metric learning (Supervised Contrastive Loss) с Recall@K и k-NN метриками по эмбеддингам.
- HTTP-сервис `runtime` для инференса единичных текстов и CSV-файлов.

## Структура репозитория
- `sentiment_value/classifier_training/` — код обучения классификатора ([README](sentiment_value/classifier_training/README.md)).
- `sentiment_value/clustering/` — кластеризация и вспомогательные скрипты ([README](sentiment_value/clustering/README.md)).
- `sentiment_value/metric_classifier_training/` — совместное обучение CE + SupCon ([README](sentiment_value/metric_classifier_training/README.md)).
- `runtime/` — сервис инференса, Dockerfile’ы, конфиг и Swagger ([README](runtime/README.md)).
- `*.example.yaml` в корне — минимальные шаблоны конфигов для каждого этапа.
- Скрипты верхнего уровня (`train_classifier.py`, `cluster_*`, `train_metric_classifier.py`) — CLI-обёртки над пакетами.

## Требования к окружению
- Python 3.10+.
- PyTorch с CUDA (рекомендуется для обучения); возможен CPU-режим, но медленнее.
- Hugging Face `transformers`, `pandas`, `pyarrow`, `scikit-learn`, `matplotlib`, `neptune` для обучения.
- Опционально:
  - `flash-attn` для ускоренной внимания (`attention_implementation: flash_attention_2` в конфигах).
  - `cuml` для GPU-варианта MiniBatchKMeans в кластеризации.
  - `ninja` (для сборки flash-attn).

Установить базовые зависимости: `pip install -r requirements.txt`. Для flash-attn:
```bash
python -m pip install ninja
python -m pip install --no-build-isolation -v flash-attn
```

## Установка
1. Клонируйте репозиторий и установите зависимости: `pip install -r requirements.txt`.
2. Скопируйте пример конфига под задачу: `cp classifier_training.example.yaml config.yaml` (аналогично для кластеризации и metric learning).
3. Отредактируйте пути к данным и гиперпараметры.

## Быстрый старт
1. **Подготовка данных.** Создайте `data/dataset.parquet` со столбцами `text` и `label`.
2. **Запуск обучения классификатора:**
   ```bash
   python train_classifier.py --config config.yaml
   ```
3. **Инференс обученной моделью в runtime:**
   - Скопируйте полученный чекпоинт/папку модели в директорию, указанную в `runtime/config.yaml`.
   - Запустите сервис (см. раздел "Runtime").
   - Отправьте запросы (доступны `/predict`, `/lemmatize`, `/ner`, `/csv`, `/health`):
     ```bash
     curl -X POST "http://localhost:8000/predict" \
       -H "Content-Type: application/json" \
       -d '{"text": "Пример текста"}'

     curl -X POST "http://localhost:8000/lemmatize" \
       -H "Content-Type: application/json" \
       -d '{"text": "Пример текста"}'
     ```

Более детальные примеры — в README каждого подпакета.

## Запуск из Docker
В каталоге `runtime/` есть Dockerfile для CPU (`Dockerfile.cpu`) и GPU (`Dockerfile.gpu`). Пример сборки:
```bash
# CPU
docker build -t sentiment-runtime:cpu -f runtime/Dockerfile.cpu .
# GPU
docker build -t sentiment-runtime:gpu -f runtime/Dockerfile.gpu .
```
Запуск с прокидыванием модели и конфига:
```bash
docker run --rm -p 8000:8000 \
  -v /path/to/model:/app/model \
  -v /path/to/config.yaml:/app/config.yaml \
  sentiment-runtime:cpu
```
Для GPU добавьте `--gpus all` и убедитесь, что внутри конфига `prefer_cuda: true`.

## Конфигурация через YAML
Каждый этап управляется отдельным `.yaml`:
- `classifier_training.example.yaml` — обучение классификатора.
- `clustering.example.yaml` — инференс эмбеддингов, PCA, KMeans, расчёт purity и обновление датасета.
- `metric_classifier_training.example.yaml` — совместное обучение CE + SupCon.
- `runtime/config.example.yaml` — параметры сервиса инференса.

Скопируйте нужный пример в рабочий `.yaml`, укажите пути к данным/модели и гиперпараметры.

## Логирование и контроль экспериментов
Neptune поддерживается во всех тренировочных скриптах: укажите `project`, `api_token`, `experiment_name` в соответствующем разделе конфига. Метрики (loss, accuracy, F1, Recall@K), изображения confusion matrix и гиперпараметры отправляются в Neptune. Локальные логи и чекпоинты сохраняются в директории `checkpoints`/выходных путях, заданных в YAML.

## Как расширять проект
- **Новый модуль/скрипт:** следуйте шаблону `sentiment_value/<module>`: выделяйте конфиги в YAML, добавляйте утилиты логирования и CLI-обёртку в корне.
- **Новый способ тренировки:** создайте отдельный конфиг и тренер, переиспользуя даталоадеры и `CheckpointManager` из `classifier_training`.
- **Новый endpoint в runtime:** добавьте маршрут в `runtime/app.py`, опишите схему в `runtime/swagger.yaml` и при необходимости расширьте `ModelWrapper` в `runtime/model.py`.

