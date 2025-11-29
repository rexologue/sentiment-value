# Модуль `classifier_training`

## Назначение
Отвечает за классическое обучение модели классификации текста по размеченному Parquet-файлу. Поддерживает ускорение на GPU, mixed precision, Gradient Accumulation, label smoothing, выбор scheduler и логирование в Neptune.

## Формат входных данных
- Входной файл: `.parquet`.
- Обязательные столбцы: `text` (строка) и `label` (класс в числовом или строковом виде — строковые метки автоматически кодируются).
- Допустимая предобработка: очистка текста, приведение к Unicode. Пустые строки/NaN желательно фильтровать заранее.

## Конфигурация (YAML)
Используйте `classifier_training.example.yaml` как шаблон. Ключевые параметры:
- `model_name` — путь или имя модели Hugging Face.
- `data.parquet_path` — путь к Parquet-датасету; `val_ratio` — доля валидации; `num_workers` — воркеры даталоадеров; `upsample`/`downsample` — балансировка классов.
- `training` — `batch_size`, `num_epochs`, `gradient_accumulation_steps`, `max_seq_length`, `mixed_precision`, `attention_implementation` (например, `flash_attention_2`), `gradient_clip_val`, `label_smoothing`, `seed`, `resume_from` (путь к чекпоинту).
- `optimizer` — `learning_rate`, `weight_decay` (AdamW).
- `scheduler` — тип (`linear`/`cosine`), `warmup_steps`, `num_training_steps` (если не указан — вычисляется из эпох), `num_cycles` для косинуса.
- `checkpointing` — `checkpoints_dir`, `save_every_n_bathces` (период сохранения), `save_best_by` (`loss`/`accuracy`/`f1`).
- `neptune` — `project`, `api_token`, `experiment_name`, `run_id`, `tags`, `dependencies_path`, `env_path`.

## Запуск обучения
```bash
python train_classifier.py --config path/to/config.yaml
```
- `--config` — путь к YAML-конфигу.
- Перед запуском убедитесь, что `transformers` и `torch` установлены, а модель доступна локально или через интернет.

## Логирование и метрики
- Метрики на валидации: `loss`, `accuracy`, `precision`, `recall`, `f1`, матрица ошибок.
- Логи отправляются в Neptune при наличии `api_token`; дополнительно сохраняются локально (confusion matrix в PNG, чекпоинты в `checkpoints_dir`).
- Гиперпараметры автоматически логируются при старте.

## Чекпоинты
- Формат: директории в `checkpoints_dir` с `model.pt`, `optimizer.pt`, `scheduler.pt`, `state.json`.
- Периодическое сохранение управляется `save_every_n_bathces`; лучший чекпоинт определяется `save_best_by`.
- Возобновление обучения: укажите `training.resume_from` на нужный чекпоинт/директорию.

## Минимальный пример
```bash
cp classifier_training.example.yaml config.yaml
# отредактируйте путь к data.parquet
python train_classifier.py --config config.yaml
```
После обучения возьмите лучший чекпоинт из `checkpoints/` для последующего инференса или кластеризации.

