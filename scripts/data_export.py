import os
from typing import Optional, Any
import pandas as pd
import joblib
import logging

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
OUTPUT_DIR = "output"

def save_dataframe(df: pd.DataFrame, filename: str) -> bool:
    """Сохраняет DataFrame в CSV-файл."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(filepath, index=False, encoding="utf-8")
        logging.info(f"[Сохранение] Файл '{filename}' сохранён в {OUTPUT_DIR}")
        return True
    except Exception as e:
        logging.error(f"[Ошибка сохранения DataFrame] {e}")
        return False

def save_model(model: Any, filename: str) -> bool:
    """Сохраняет модель машинного обучения в .pkl файл."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, filename)
        joblib.dump(model, filepath)
        logging.info(f"[Сохранение модели] Модель сохранена в {filepath}")
        return True
    except Exception as e:
        logging.error(f"[Ошибка сохранения модели] {e}")
        return False

def load_model(filename: str) -> Optional[Any]:
    """Загружает модель из .pkl файла."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        logging.error(f"[Ошибка загрузки] Файл '{filename}' не найден в {OUTPUT_DIR}.")
        return None

    try:
        model = joblib.load(filepath)
        logging.info(f"[Загрузка модели] '{filename}' успешно загружен.")
        return model
    except Exception as e:
        logging.error(f"[Ошибка при загрузке модели] {e}")
        return None

def save_metrics_to_csv(metrics, output_path):
    """Сохраняет метрики кластеризации в CSV файл."""
    if not metrics:
        logging.warning("Метрики кластеризации пусты. Файл не будет создан.")
        return

    metrics_df = pd.DataFrame(metrics, columns=["Model", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"])

    # Сохраняем DataFrame в CSV
    metrics_df.to_csv(output_path, index=False)
    logging.info(f"Результаты кластеризации сохранены в файл: {output_path}")

def save_results(
    movies_df: pd.DataFrame,
    best_model: Any,
    best_cluster_column: str = "cluster_kmeans",
    metrics: Optional[list] = None,
    movie_features: Optional[pd.DataFrame] = None,
    top_movies: Optional[pd.DataFrame] = None
):
    """Сохраняет результаты кластеризации, модели и сопутствующих артефактов."""
    try:
        ensure_output_directory()

        # Проверка на наличие столбца кластеров от лучшей модели
        if best_cluster_column not in movies_df.columns:
            logger.error(f"Столбец '{best_cluster_column}' не найден в DataFrame. Результаты не будут сохранены.")
            return

        # Сохраняем кластер лучшей модели как универсальный столбец "cluster"
        movies_df["cluster"] = movies_df[best_cluster_column]

        # Оставляем только нужные столбцы для экспорта
        columns_to_export = ["movieId", "title", "genres", "avg_rating", "rating_count", "cluster"]
        export_df = movies_df[columns_to_export]

        # Сохраняем итоговый файл
        export_df.to_csv("output/movies_with_clusters.csv", index=False, encoding="utf-8")
        logger.info("Фильмы с кластерами сохранены в 'output/movies_with_clusters.csv'.")

        # Сохраняем модель
        if best_model:
            joblib.dump(best_model, "output/best_model.pkl")
            logger.info("Лучшая модель сохранена в 'output/best_model.pkl'.")
        else:
            logger.warning("Лучшая модель не была сохранена (отсутствует).")

        # Сохраняем метрики
        if metrics:
            save_metrics_to_csv(metrics, "output/clustering_models_metrics.csv")
            logger.info("Метрики кластеризации сохранены.")

        # Сохраняем признаки фильмов
        if movie_features is not None:
            movie_features.to_csv("output/movie_features.csv", index=False)
            logger.info("Признаки фильмов сохранены.")

        # Сохраняем топ-фильмы
        if top_movies is not None:
            top_movies.to_csv("output/top_movies_by_cluster.csv", index=False)
            logger.info("Топ-фильмы по кластерам сохранены.")

    except Exception as e:
        logger.error(f"Ошибка при сохранении результатов: {e}")
