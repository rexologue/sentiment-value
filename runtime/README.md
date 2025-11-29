# Модуль `runtime`

## Назначение
FastAPI-сервис для инференса модели тональной классификации. Поддерживает единичные запросы (`/predict`) и пакетную обработку CSV (`/csv`).

## Архитектура
- Использует `ModelWrapper` из `model.py`, оборачивающий `AutoModelForSequenceClassification` с оптимизациями (CUDA при `prefer_cuda: true`, опциональный `torch.compile`, выбор реализации внимания).
- Конфигурация читается из YAML (`config.yaml`) при старте, путь задаётся переменной окружения `SENTIMENT_CONFIG_PATH` (по умолчанию `/app/config.yaml`).
- Эндпоинты:
  - `POST /predict` — принимает JSON `{ "text": "..." }`, возвращает предсказанный класс и вероятности (`negative_score`, `neutral_score`, `positive_score`).
  - `POST /csv` — принимает CSV-файл со столбцами `ID` и `text`, возвращает CSV `ID,label`.
  - `GET /health` — статус сервиса и информация о устройстве/компиляции/внимании.

## Конфигурация
Пример — `runtime/config.example.yaml`:
```yaml
model_path: /app/model
prefer_cuda: true
enable_compile: false
host: 0.0.0.0
port: 8000
csv_batch_size: 64
log_level: INFO
log_format: plain
```
Основные параметры:
- `model_path` — путь к папке модели (чекпоинт в формате Hugging Face или конвертированный safetensors).
- `prefer_cuda` — попытка загрузить на GPU, при отсутствии падает на CPU.
- `enable_compile` — включает `torch.compile` при доступности.
- `csv_batch_size` — размер батча для обработки CSV.
- `log_level`, `log_format` — настройка логов.
- `host`, `port` — адрес для `uvicorn`.

## Локальный запуск
```bash
export SENTIMENT_CONFIG_PATH=/path/to/config.yaml
python -m runtime.app
# или
uvicorn runtime.app:app --host 0.0.0.0 --port 8000
```

## Запуск через Docker
В каталоге `runtime/` есть `Dockerfile.cpu` и `Dockerfile.gpu`.
```bash
# Сборка
docker build -t sentiment-runtime:cpu -f runtime/Dockerfile.cpu .
# Запуск CPU
docker run --rm -p 8000:8000 \
  -v /path/to/model:/app/model \
  -v /path/to/config.yaml:/app/config.yaml \
  sentiment-runtime:cpu
# Запуск GPU
docker run --rm -p 8000:8000 --gpus all \
  -e SENTIMENT_CONFIG_PATH=/app/config.yaml \
  -v /path/to/model:/app/model \
  -v /path/to/config.yaml:/app/config.yaml \
  sentiment-runtime:gpu
```
Убедитесь, что в `config.yaml` указан путь к модели внутри контейнера (`/app/model`).

## Примеры запросов
```bash
# Одиночный текст
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Мне понравился сервис"}'

# CSV (файл содержит колонки ID,text)
curl -X POST "http://localhost:8000/csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@samples.csv" \
  -o predictions.csv
```

## Swagger/OpenAPI
В файле `swagger.yaml` описаны схемы запросов/ответов и можно сгенерировать клиентов. Обновляйте его при добавлении новых эндпоинтов или полей.

