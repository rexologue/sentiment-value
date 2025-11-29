# Модуль `metric_classifier_training`

## Назначение
Совместное обучение модели по двум задачам:
- **Классификация** — CrossEntropy считается по объектам, где `classifier_mask = 1`.
- **Metric learning** — Supervised Contrastive Loss (SupCon) по объектам, где `metric_mask = 1`, с L2-нормализацией эмбеддингов перед расчётом SupCon.

Общий loss: `total = classification_loss_weight * CE + metric_loss_weight * SupCon`. Логиты для CE берутся из последнего скрытого состояния/пулера без нормализации, для SupCon используются нормализованные векторы.

## Формат данных
Ожидаемый Parquet/DataFrame содержит колонки:
- `text` — строка.
- `label` — класс (int или str, кодируется внутри пайплайна).
- `metric_mask` — 0/1-флаг участия в SupCon.
- `classifier_mask` — 0/1-флаг участия в CrossEntropy.

Маски позволяют, например, обучать SupCon на чистых кластерах, а CE — на размеченной выборке.

## Метрики
- Классические: loss, accuracy, macro-F1 по классификации.
- Metric learning: Recall@K (k задаются в конфиге), k-NN классификация по эмбеддингам (macro-F1), дополнительные статистики банка эмбеддингов.
- Банк эмбеддингов собирается из train за последние `keep_last_n_emb_steps` шагов и используется на валидации.

Лучший чекпоинт выбирается по `save_best_by` (`loss`/`acc`/`f1`).

## Конфигурация (YAML)
См. `metric_classifier_training.example.yaml`. Ключевые параметры:
- `model_name` — базовая модель Hugging Face.
- `data` — `parquet_path`, `val_ratio`, `num_workers`, `upsample`/`downsample`.
- `training` — `batch_size`, `num_epochs`, `gradient_accumulation_steps`, `max_seq_length`, `mixed_precision`, `attention_implementation`, `gradient_clip_val`, `seed`, `resume_from`, `supcon_temperature`, `classification_loss_weight`, `metric_loss_weight`.
- `optimizer` — `learning_rate`, `weight_decay`.
- `scheduler` — `warmup_steps`, `num_training_steps`, `name`, `num_cycles`.
- `checkpointing` — `checkpoints_dir`, `save_every_n_bathces`, `save_best_by`.
- `metric_validation` — `recall_at_k` (список), `distance` (`cos`/`l2`), `knn_k`, `keep_last_n_emb_steps`.
- `neptune` — параметры логирования.

## Запуск
```bash
python train_metric_classifier.py --config metric_classifier_training.example.yaml
```
- Укажите путь к своему YAML через `--config`.
- Перед запуском убедитесь, что в датасете присутствуют маски.

## Сценарии использования
- После кластеризации: отфильтруйте кластеры с высокой purity, выставьте `metric_mask=1` для надёжных примеров, `classifier_mask=1` для размеченной части; обучите SupCon + CE на объединённых данных.
- При наличии ограниченной ручной разметки: пометьте непроверенные примеры `classifier_mask=0`, но `metric_mask=1`, чтобы использовать их только в metric learning.

