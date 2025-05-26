import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import logging
import gc
from scipy.sparse import csr_matrix
import os
from scripts.data_cleaning import clean_movies, clean_tags, clean_ratings, standardize_ratings
from scripts.data_processing import preprocess_categorical_data, standardize_data
def get_memory_usage():
    """Get current memory usage in MB"""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def optimize_memory():
    """Clear memory and run garbage collection"""
    try:
        gc.collect()
        if hasattr(pd, '_libs'):
            try:
                pd._libs.hashtable.PyObjectHashTable.clear_cache()
            except AttributeError:
                logger.debug("Pandas cache clearing method not available")
        logger.debug(f"Memory usage after optimization: {get_memory_usage():.2f} MB")
    except Exception as e:
        logger.warning(f"Memory optimization failed: {e}")


def prepare_ratings_agg(ratings_df_chunks, optimize_every: int = 5) -> pd.DataFrame:
    """Формирует агрегированные данные по рейтингам из чанков."""
    logger.debug("Начало работы prepare_ratings_agg")
    logger.info("Формируем агрегированные данные по рейтингам...")

    ratings_agg_chunks = []
    for i, chunk in enumerate(ratings_df_chunks):
        if 'movieId' not in chunk.columns or 'rating' not in chunk.columns:
            raise ValueError("В каждом чанке должны быть столбцы 'movieId' и 'rating'.")

        chunk = chunk[['movieId', 'rating']]
        chunk_agg = chunk.groupby('movieId', as_index=False).agg(avg_rating=('rating', 'mean'))

        ratings_agg_chunks.append(chunk_agg)

        if (i + 1) % optimize_every == 0:
            logger.debug(f"Оптимизация памяти после чанка {i + 1}")
            optimize_memory()

    ratings_agg = pd.concat(ratings_agg_chunks, ignore_index=True)
    ratings_agg = ratings_agg.groupby('movieId', as_index=False).agg(avg_rating=('avg_rating', 'mean'))

    logger.info(f"Агрегация завершена. Размер: {ratings_agg.shape}")
    return ratings_agg


def one_hot_encode_chunk(chunk: pd.DataFrame, top_tags: list, top_genres: list) -> pd.DataFrame:
    chunk.loc[:, 'tag'] = chunk['tag'].where(
        (chunk['tag'].isin(top_tags)) & (chunk['tag'].str.len() > 2) & (
            ~chunk['tag'].str.lower().isin(['bd-r', 'dvd-r', 'on', 'a', 'to'])),
        other='Other'
    )
    chunk.loc[:, 'genres'] = chunk['genres'].where(chunk['genres'].isin(top_genres), other='Other')

    chunk.loc[:, 'tag'] = chunk['tag'].fillna('Other')
    chunk.loc[:, 'genres'] = chunk['genres'].fillna('Other')

    tag_dummies = pd.get_dummies(chunk['tag'], prefix='tag', sparse=True, dtype=np.float32)
    genre_dummies = pd.get_dummies(chunk['genres'], prefix='genre', sparse=True, dtype=np.float32)

    chunk = pd.concat([chunk.drop(['tag', 'genres'], axis=1), tag_dummies, genre_dummies], axis=1)
    return chunk


def optimize_movies_df(movies_df: pd.DataFrame, ratings_agg: pd.DataFrame) -> pd.DataFrame:
    """Оптимизация и кодирование данных по фильмам."""
    TOP_N_TAGS = 2000
    TOP_N_GENRES = 200
    logger.info("Начало оптимизации данных...")

    merged_df = pd.merge(movies_df, ratings_agg, on='movieId', how='left')
    optimize_memory()

    top_tags = merged_df['tag'].value_counts().nlargest(TOP_N_TAGS).index.tolist()
    top_genres = merged_df['genres'].value_counts().nlargest(TOP_N_GENRES).index.tolist()
    optimize_memory()

    logger.info("Выполняем параллельное кодирование тегов и жанров...")

    chunks = [merged_df[i:i + 1000] for i in range(0, len(merged_df), 1000)]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        processed_chunks = list(executor.map(lambda c: one_hot_encode_chunk(c, top_tags, top_genres), chunks))

    result_df = pd.concat(processed_chunks, ignore_index=True)
    optimize_memory()

    for col in result_df.select_dtypes(include=['float64', 'int64']).columns:
        result_df[col] = result_df[col].astype('float32')
    for col in result_df.select_dtypes(include=['object']).columns:
        result_df[col] = result_df[col].astype('category')
    optimize_memory()

    numeric_cols = result_df.select_dtypes(include='number').columns
    result_df[numeric_cols] = result_df[numeric_cols].fillna(result_df[numeric_cols].mean())
    logger.info(f"Оптимизация завершена. Итоговый размер: {result_df.shape}")
    return result_df

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def plot_explained_variance(X, max_components=500, step=10, output_dir="svd_analysis"):
    """Построить график объясненной дисперсии для TruncatedSVD."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Папка '{output_dir}' создана.")

    n_components_range = range(10, min(max_components + 1, X.shape[1]), step)
    explained_variance_ratios = []

    for k in n_components_range:
        logger.info(f"Вычисление TruncatedSVD для k={k}...")
        svd = TruncatedSVD(n_components=k, random_state=42)
        svd.fit(X)
        explained_variance_ratios.append(np.sum(svd.explained_variance_ratio_))
        gc.collect()

    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, explained_variance_ratios, marker='o')
    plt.xlabel('Число компонент (k)')
    plt.ylabel('Доля объясненной дисперсии')
    plt.title('Объясненная дисперсия vs. Число компонент')
    plt.grid(True)
    plt.axvline(x=150, color='r', linestyle='--', label='k=150')
    plt.axvline(x=400, color='g', linestyle='--', label='k=400')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "explained_variance_plot.png"))
    plt.close()
    logger.info(f"График сохранен в {output_dir}/explained_variance_plot.png")

    results_df = pd.DataFrame({
        'n_components': list(n_components_range),
        'explained_variance_ratio': explained_variance_ratios
    })
    results_df.to_csv(os.path.join(output_dir, "explained_variance_results.csv"), index=False)
    logger.info(f"Результаты сохранены в {output_dir}/explained_variance_results.csv")


def compare_clustering_metrics(X, k_values=[150, 400], n_clusters=9, output_dir="svd_analysis"):
    """Сравнить метрики кластеризации для разных k в TruncatedSVD."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Папка '{output_dir}' создана.")

    results = []

    for k in k_values:
        logger.info(f"TruncatedSVD с k={k} и KMeans с n_clusters={n_clusters}...")
        svd = TruncatedSVD(n_components=k, random_state=42)
        X_reduced = svd.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_reduced)

        silhouette = silhouette_score(X_reduced, labels)
        logger.info(f"k={k}, Silhouette Score={silhouette:.4f}")

        results.append({
            'n_components': k,
            'silhouette_score': silhouette
        })

        gc.collect()

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "clustering_metrics.csv"), index=False)
    logger.info(f"Метрики сохранены в {output_dir}/clustering_metrics.csv")
    return results_df


def main():
    movies_csv = "data/movies.csv"
    tags_csv = "data/tags.csv"
    ratings_csv = "data/ratings.csv"
    for file in [movies_csv, tags_csv, ratings_csv]:
        if not os.path.exists(file):
            logger.error(f"Файл не найден: {file}")
            raise FileNotFoundError(f"Файл не найден: {file}")

    # Предобработка данных
    logger.info("Очистка и стандартизация данных...")
    clean_movies(movies_csv, output_path="output/cleaned_movies.csv")
    clean_tags(tags_csv, output_path="output/cleaned_tags.csv")
    clean_ratings(ratings_csv, output_path="output/cleaned_ratings.csv")
    standardize_ratings("output/cleaned_ratings.csv", output_path="output/standardized_ratings.csv")

    # Создание movies_with_tags.csv
    logger.info("Создание movies_with_tags.csv...")
    from scripts.data_processing import prepare_movies_with_tags
    prepare_movies_with_tags(
        movies_csv_path="output/cleaned_movies.csv",
        tags_csv_path="output/cleaned_tags.csv",
        output_path="output/movies_with_tags.csv",
        chunksize=10000
    )
    logger.info("Загружаем предобработанные данные...")
    df = pd.read_csv("output/movies_with_tags.csv")

    logger.info("Формируем агрегированные данные по рейтингам...")
    ratings_df = pd.read_csv("output/standardized_ratings.csv", chunksize=1000)
    ratings_agg = prepare_ratings_agg(ratings_df)
    df = df.merge(ratings_agg, on='movieId', how='left').rename(columns={'avg_rating': 'rating'})

    logger.info("Оптимизация и кодирование данных...")
    result_df = optimize_movies_df(df, ratings_agg)

    # Извлечение числовых признаков и преобразование в разреженную матрицу
    logger.info("Извлечение числовых признаков...")
    X = result_df.select_dtypes(include=['float32', 'int32', 'int64'])
    X = csr_matrix(X.values)
    logger.info(f"Размер X: {X.shape}")

    # Выполнение проверок
    logger.info("Проверка объясненной дисперсии...")
    plot_explained_variance(X, max_components=500, step=10, output_dir="svd_analysis")
    logger.info("Сравнение метрик кластеризации...")
    compare_clustering_metrics(X, k_values=[150, 400], n_clusters=9, output_dir="svd_analysis")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Произошла критическая ошибка: %s", e)
        print(f"Произошла критическая ошибка: {e}")