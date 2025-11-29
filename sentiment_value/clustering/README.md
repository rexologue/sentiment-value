# Модуль `clustering`

## Назначение
Кластеризация эмбеддингов позволяет получить псевдоразметку, оценить структуру данных и отфильтровать шум. Пайплайн: инференс модели → сбор CLS-векторов и вероятностей → нормализация + PCA → обучение кластеризатора (MiniBatchKMeans на CPU/GPU) → оценка purity → обновление датасета псевдометками/фильтрами.

## Основные скрипты
- `infer.py` (alias `supervise.py`): прогон модели по текстам, сохранение шардов `.parquet` со столбцами `text`, `probs`, `cls`, `label` (если есть), поддержка шардинга и батчей.
- `normalize_and_pca.py`: L2-нормализация CLS-векторов и PCA до заданного числа компонент, сохранение в новый каталог.
- `train.py`: обучение кластеризатора MiniBatchKMeans. При `gpu: true` используется CuML (если установлена), иначе scikit-learn. Также умеет режим `predict` для назначения кластеров на готовые PCA-вектора.
- `purity_counter.py`: расчёт purity по колонкам с истинными/учительскими метками и кластерными id.
- `update_dataset.py`: обновление Parquet-датасета по порогам purity/уверенности — добавление/фильтрация меток.

Все скрипты используют общую конфигурацию из `clustering.example.yaml`.

## Формат входных/выходных данных
- **Шарды инференса (`supervise.py`)**: `.parquet` с колонками `text`, `cls` (список float), `probs` (вероятности класса), опционально `label`/`teacher_label`, `prediction` и др. Шардируются по файлам `part-*.parquet`.
- **После PCA**: добавляется колонка `pca` (список float длиной `n_components`), исходные `cls`/`probs` сохраняются.
- **После кластеризации (`train.py` в режиме predict)**: появляется колонка `cluster_id` (или иное имя из `centroid_column`).
- **Центроиды**: Parquet с колонками `centroid_id` (int) и `center` (список float) и joblib-модель `kmeans.joblib`.

## Конфигурация через YAML
Файл `clustering.example.yaml` содержит разделы:
- `supervise`: `model_name_or_path`, `input_parquet`, `output_dir`, `batch_size`, `device`, `num_workers`, `num_shards`, `dtype`, `resume`, `progress`.
- `normalize_and_pca`: `input_dir`, `output_dir`, `n_components`, `batch_size`, `device`, `overwrite`, `use_gpu_pca`.
- `train`: `input_dir`, `centroids_out`, `model_out`, `run_mode` (`train`/`predict`), `pca_column`, `centroid_column`, `n_clusters`, `batch_size`, `max_iter`, `init_size`, `random_state`, `gpu`, `num_workers`, `sample_limit`, `progress`, `log_level`.
- `purity_counter`: `shards_dir`, `centroids_path`, `teacher_label_column`, `cluster_column`, `progress`, `log_level`.
- `update_dataset`: пороги purity/уверенности, имена столбцов текста/меток/кластеров/вероятностей, пути входа/выхода, `progress`, `log_level`.

## Запуск
Примеры CLI (от корня репозитория):
```bash
# 1) Инференс и генерация шардов
python cluster_supervise.py --config clustering.example.yaml

# 2) Нормализация + PCA
python cluster_normalize_and_pca.py --config clustering.example.yaml

# 3) Обучение MiniBatchKMeans
python cluster_train_clusters.py --config clustering.example.yaml

# 4) Назначение кластеров (run_mode=predict в YAML)
python cluster_train_clusters.py --config clustering.example.yaml

# 5) Подсчёт purity
python cluster_purity_counter.py --config clustering.example.yaml

# 6) Обновление датасета псевдоразметкой
python cluster_update_dataset.py --config clustering.example.yaml
```

## Метрики и анализ кластеров
`purity_counter.py` вычисляет share совпадений метки и кластера (purity) для оценки качества разбиений. Можно использовать для подбора `n_clusters` или фильтрации кластеров с низкой чистотой.

## Пример end-to-end
1. Подготовить `data/raw_texts.parquet` (колонки `text`, опционально `label`/`teacher_label`).
2. Запустить `cluster_supervise.py` → получить `data/clustering/supervised/part-*.parquet` с `cls` и `probs`.
3. Запустить `cluster_normalize_and_pca.py` → получить PCA-вектора `pca` в `data/clustering/pca/`.
4. Запустить `cluster_train_clusters.py` в режиме `train` → сохранить `centroids.parquet` и `kmeans.joblib`.
5. Перевести `run_mode` в `predict` и снова вызвать `cluster_train_clusters.py` → записать `cluster_id` в шардовые файлы.
6. Посчитать purity (`cluster_purity_counter.py`) и при необходимости обновить датасет (`cluster_update_dataset.py`).

