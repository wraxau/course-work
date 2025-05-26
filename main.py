import time
import logging
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle
from sklearn.decomposition import TruncatedSVD
import gc
import psutil
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

from scripts.clustering_models_comparison import evaluate_clustering_models
from scripts.data_cleaning import clean_movies, clean_tags, clean_ratings, standardize_ratings
from scripts.dbscan_model import evaluate_dbscan_model
from scripts.hierarchical_model import evaluate_hierarchical_model, plot_dendrogram
from scripts.compare_models import compare_models, standardize_metric_names
from scripts.clustering_elbow2 import compute_elbow_method2
from scripts.data_processing import preprocess_categorical_data, standardize_data

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Определяем переменные
RUN_CLUSTERING = True
RUN_VISUALIZATION = True
RUN_ANALYSIS = True
RUN_RECOMMENDATION = True

# Memory optimization settings
CHUNK_SIZE = 1000
MAX_MEMORY_USAGE = 0.7
SVD_COMPONENTS = 400

def get_memory_usage():
    """Get current memory usage in MB"""
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

def data_already_preprocessed():
    """Проверяет, были ли данные предобработаны."""
    return os.path.exists("output/cleaned_movies.csv") and \
        os.path.exists("output/cleaned_ratings.csv") and \
        os.path.exists("output/standardized_ratings.csv") and \
        os.path.exists("output/cleaned_tags.csv")

def ensure_output_directory():
    """Проверяет, существует ли папка 'output', и если нет — создает ее."""
    try:
        if not os.path.exists("output"):
            os.makedirs("output")
            logger.info("Папка 'output' создана.")
        else:
            logger.info("Папка 'output' уже существует.")
    except Exception as e:
        logger.error(f"Ошибка при создании папки 'output': {e}")
        raise

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

    chunks = [merged_df[i:i + CHUNK_SIZE] for i in range(0, len(merged_df), CHUNK_SIZE)]

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

def prepare_movies_with_tags(
        movies_csv_path: str,
        tags_csv_path: str = "output/cleaned_tags.csv",
        output_path: str = "output/movies_with_tags.csv",
        sep: str = ' ',
        chunksize: int = 10_000
):
    """
    Объединяет фильмы с тегами и сохраняет результат сразу в файл по частям.
    Не использует память для хранения всех данных.
    """
    try:
        logger.info(f"Чтение и агрегация тегов из {tags_csv_path}...")
        tags_df = pd.read_csv(tags_csv_path, usecols=["movieId", "tag"])
        tags_df["tag"] = tags_df["tag"].fillna("").astype(str)

        stop_words = {'bd-r', 'dvd-r', 'clv', 'dvd-video', 'on', 'a', 'to', 'itaege', 'than', 'dvd-ram'}
        tags_df["tag"] = tags_df["tag"].apply(
            lambda x: ' '.join(word for word in x.split() if len(word) > 2 and word.lower() not in stop_words)
        )
        tags_df = tags_df[tags_df["tag"] != ""]
        tags_grouped = tags_df.groupby("movieId")["tag"].apply(lambda x: sep.join(set(x))).reset_index()

        is_first_chunk = True
        logger.info(f"Построчная обработка {movies_csv_path} и запись в {output_path}...")

        for chunk in pd.read_csv(movies_csv_path, chunksize=chunksize):
            merged_chunk = chunk.merge(tags_grouped, on="movieId", how="left")
            merged_chunk["tag"] = merged_chunk["tag"].fillna("")

            merged_chunk.to_csv(output_path, mode='w' if is_first_chunk else 'a', index=False, header=is_first_chunk)
            is_first_chunk = False

            del chunk, merged_chunk
            gc.collect()

        logger.info(f"Файл '{output_path}' успешно создан.")

    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {e}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при обработке: {e}")
        raise

def compute_elbow_method(data, k_range=range(3, 20), output_dir="elbow_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Папка '{output_dir}' создана.")

    inertias = []
    for k in k_range:
        print(f"Вычисление инерции для k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    kl = KneeLocator(list(k_range), inertias, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', label='KMeans')
    if optimal_k:
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Оптимальное k = {optimal_k}')
    plt.xlabel('Число кластеров (k)')
    plt.ylabel('Инерция')
    plt.title(f'Метод локтя: KMeans (диапазон {min(k_range), max(k_range)})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"elbow_kmeans_range_{min(k_range)}_{max(k_range)}.png"))
    plt.close()

    results_df = pd.DataFrame({'k': list(k_range), 'inertia': inertias})
    results_df.to_csv(os.path.join(output_dir, f"elbow_results_range_{min(k_range)}_{max(k_range)}.csv"), index=False)

    return optimal_k

def evaluate_kmeans(data, k_values, output_dir="kmeans_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Папка '{output_dir}' создана.")

    results = []
    for k in k_values:
        print(f"Кластеризация с KMeans для k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)

        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)

        results.append({
            "k": k,
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "calinski_harabasz": calinski_harabasz
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "kmeans_metrics.csv"), index=False)

    return results_df

def analyze_clusters(data, labels, df, sub_labels=None, output_dir="cluster_analysis"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    summary_path = os.path.join(output_dir, "cluster_summary.csv")
    if os.path.exists(summary_path):
        os.remove(summary_path)

    df['cluster'] = labels
    df['subcluster'] = sub_labels
    df['main_cluster'] = df['subcluster'] // 1000  # Hierarchical clustering support
    n_clusters = df['main_cluster'].nunique()

    cluster_summary = []

    for cluster in range(n_clusters):
        print(f"\nАнализ кластера {cluster}:")
        cluster_data = df[df['main_cluster'] == cluster]

        cluster_size = len(cluster_data)
        print(f"Размер кластера: {cluster_size} фильмов")

        avg_rating = cluster_data['rating'].mean()
        median_rating = cluster_data['rating'].median()
        rating_variance = cluster_data['rating'].var()
        print(f"Средний рейтинг: {avg_rating:.2f}")
        print(f"Медиана рейтинга: {median_rating:.2f}")
        print(f"Дисперсия рейтинга: {rating_variance:.2f}")

        genres_list = cluster_data['genres'].str.split('|').explode().str.strip()
        genre_counts = genres_list.value_counts()
        print("Топ-5 жанров:")
        print(genre_counts.head(5))

        tags_list = cluster_data['tag'].str.split(' ').explode().str.strip()
        tag_counts = tags_list.value_counts()
        print("Топ-5 тегов:")
        print(tag_counts.head(5))

        print("Примеры фильмов:")
        examples = cluster_data[['title', 'rating', 'genres', 'tag']].head(5)
        print(examples)

        if sub_labels is not None:
            unique_subclusters = sorted(cluster_data['subcluster'].unique())
            for subcluster in unique_subclusters:
                subcluster_data = cluster_data[cluster_data['subcluster'] == subcluster]
                subcluster_size = len(subcluster_data)
                sub_avg_rating = subcluster_data['rating'].mean()
                sub_median_rating = subcluster_data['rating'].median()
                sub_rating_variance = subcluster_data['rating'].var()

                sub_genres_list = subcluster_data['genres'].str.split('|').explode().str.strip()
                sub_genre_counts = sub_genres_list.value_counts().head(5)

                sub_tags_list = subcluster_data['tag'].str.split(' ').explode().str.strip()
                sub_tag_counts = sub_tags_list.value_counts().head(5)

                cluster_summary.append({
                    'cluster': cluster,
                    'subcluster': subcluster,
                    'size': subcluster_size,
                    'avg_rating': sub_avg_rating,
                    'median_rating': sub_median_rating,
                    'rating_variance': sub_rating_variance,
                    'top_genres': sub_genre_counts.to_dict(),
                    'top_tags': sub_tag_counts.to_dict()
                })

                with open(os.path.join(output_dir, f"cluster_{cluster}_subcluster_{subcluster}_analysis.txt"), 'w') as f:
                    f.write(f"Анализ подкластера {subcluster} в кластере {cluster}\n")
                    f.write(f"Размер подкластера: {subcluster_size} фильмов\n")
                    f.write(f"Средний рейтинг: {sub_avg_rating:.2f}\n")
                    f.write(f"Медиана рейтинга: {sub_median_rating:.2f}\n")
                    f.write(f"Дисперсия рейтинга: {sub_rating_variance:.2f}\n")
                    f.write("\nТоп-5 жанров:\n")
                    f.write(sub_genre_counts.to_string() + "\n")
                    f.write("\nТоп-5 тегов:\n")
                    f.write(sub_tag_counts.to_string() + "\n")
                    f.write("\nПримеры фильмов:\n")
                    f.write(subcluster_data[['title', 'rating', 'genres', 'tag']].head(5).to_string() + "\n")

        with open(os.path.join(output_dir, f"cluster_{cluster}_analysis.txt"), 'w') as f:
            f.write(f"Анализ кластера {cluster}\n")
            f.write(f"Размер кластера: {cluster_size} фильмов\n")
            f.write(f"Средний рейтинг: {avg_rating:.2f}\n")
            f.write(f"Медиана рейтинга: {median_rating:.2f}\n")
            f.write(f"Дисперсия рейтинга: {rating_variance:.2f}\n")
            f.write("\nТоп-5 жанров:\n")
            f.write(genre_counts.head(5).to_string() + "\n")
            f.write("\nТоп-5 тегов:\n")
            f.write(tag_counts.head(5).to_string() + "\n")
            f.write("\nПримеры фильмов:\n")
            f.write(examples.to_string() + "\n")

    if cluster_summary:
        summary_df = pd.DataFrame(cluster_summary)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Сводный анализ подкластеров сохранён в {summary_path}")
    else:
        logger.warning("Нет данных о подкластерах для сохранения в cluster_summary.csv")

def subcluster_data_kmeans(
        X_reduced, labels,
        min_k=3, max_k=8,
        silhouette_threshold=0.5,
        max_subclusters_per_cluster=8
):
    """
    Performs K-means subclustering on pre-clustered data, preserving cluster hierarchy.
    """
    unique_clusters = np.unique(labels)
    sub_labels = np.full_like(labels, -1, dtype=np.int32)
    total_subclusters = 0

    for cluster in unique_clusters:
        cluster_indices = np.where(labels == cluster)[0]
        cluster_size = len(cluster_indices)
        logger.info(f"Cluster {cluster}: {cluster_size} movies")

        if cluster_size < min_k:
            logger.warning(f"Cluster {cluster} too small ({cluster_size} movies), keeping as single subcluster")
            sub_labels[cluster_indices] = cluster * 1000
            total_subclusters += 1
            continue

        cluster_data = X_reduced[cluster_indices]
        best_score = -1
        best_labels = None
        best_k = min_k

        for k in range(min_k, min(max_k, cluster_size, max_subclusters_per_cluster) + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels_candidate = kmeans.fit_predict(cluster_data)
            if len(set(labels_candidate)) < min_k:
                continue
            try:
                score = silhouette_score(cluster_data, labels_candidate)
            except Exception as e:
                logger.error(f"Silhouette score failed for Cluster {cluster}, k={k}: {e}")
                continue

            logger.info(f"KMeans for Cluster {cluster}: k={k}, silhouette={score:.2f}")
            if score > best_score and score >= silhouette_threshold:
                best_score = score
                best_labels = labels_candidate
                best_k = k

        if best_labels is not None and best_k >= min_k:
            logger.info(f"Cluster {cluster}: best silhouette={best_score:.2f}, subclusters={best_k}")
            for sub in range(best_k):
                mask = (best_labels == sub)
                sub_labels[cluster_indices[mask]] = cluster * 1000 + sub
            total_subclusters += best_k
        else:
            logger.warning(f"Cluster {cluster}: no good subclusters found, keeping as single subcluster")
            sub_labels[cluster_indices] = cluster * 1000
            total_subclusters += 1

    logger.info(f"Total subclusters created: {total_subclusters}")
    logger.info(f"Unique subcluster labels: {len(np.unique(sub_labels))}")
    return sub_labels

def recommend_movies(movie_id, df, labels, sub_labels, X_reduced, n_recommendations=5):
    try:
        movie_idx = df[df['movieId'] == movie_id].index[0]

        main_cluster = labels[movie_idx]
        subcluster = sub_labels[movie_idx]

        subcluster_indices = np.where((labels == main_cluster) & (sub_labels == subcluster))[0]
        subcluster_indices = subcluster_indices[subcluster_indices != movie_idx]

        if len(subcluster_indices) == 0:
            raise ValueError(f"Нет других фильмов в подкластере {subcluster} для рекомендации.")

        target_genres = set(df.iloc[movie_idx]['genres'].split('|'))
        target_rating = df.iloc[movie_idx]['rating']

        filtered_indices = []
        for idx in subcluster_indices:
            movie_genres = set(df.iloc[idx]['genres'].split('|'))
            common_genres = target_genres & movie_genres
            movie_rating = df.iloc[idx]['rating']
            if common_genres and abs(movie_rating - target_rating) < 1.0:
                filtered_indices.append(idx)

        if not filtered_indices:
            filtered_indices = subcluster_indices

        filtered_indices = np.array(filtered_indices)
        if len(filtered_indices) == 0:
            raise ValueError(f"Нет фильмов с общими жанрами в подкластере {subcluster}.")

        similarities = cosine_similarity(X_reduced[movie_idx].reshape(1, -1), X_reduced[filtered_indices])[0]
        valid_indices = filtered_indices[similarities > 0.8]
        if len(valid_indices) < n_recommendations and len(filtered_indices) > n_recommendations:
            valid_indices = filtered_indices[np.argsort(similarities)[-n_recommendations:][::-1]]
        elif len(valid_indices) == 0:
            valid_indices = filtered_indices[np.argsort(similarities)[-n_recommendations:][::-1]]

        recommendations = df.iloc[valid_indices][['title', 'rating', 'genres', 'tag']].copy()
        recommendations['similarity'] = similarities[np.argsort(similarities)[-(len(valid_indices)):][::-1]]
        logger.info(f"Рекомендации для movieId={movie_id}: {recommendations['title'].tolist()}")
        return recommendations
    except Exception as e:
        logger.error(f"Ошибка при генерации рекомендаций: {e}")
        raise

def main():
    start_time = time.time()
    ensure_output_directory()

    logger.info("Начало выполнения пайплайна...")
    logger.info(f"Initial memory usage: {get_memory_usage():.2f} MB")

    # Загрузка и предобработка данных
    logger.info("Загружаем предобработанные данные...")
    df = pd.read_csv("output/movies_with_tags.csv")

    # Загрузка и агрегация рейтингов
    logger.info("Формируем агрегированные данные по рейтингам...")
    ratings_df = pd.read_csv("output/standardized_ratings.csv", chunksize=CHUNK_SIZE)
    ratings_agg = prepare_ratings_agg(ratings_df)
    df = df.merge(ratings_agg, on='movieId', how='left').rename(columns={'avg_rating': 'rating'})

    # Оптимизация и кодирование данных
    logger.info("Начало оптимизации данных и кодирование тегов/жанров...")
    result_df = optimize_movies_df(df, ratings_agg)

    # Извлекаем числовые признаки для кластеризации
    X = result_df.select_dtypes(include=['float32', 'int32', 'int64']).values
    logger.info(f"Размер X: {X.shape}")

    # Снижение размерности
    logger.info(f"Снижение размерности с помощью TruncatedSVD с {SVD_COMPONENTS} компонентами...")
    svd = TruncatedSVD(n_components=SVD_COMPONENTS)
    X_reduced = svd.fit_transform(X)
    optimize_memory()

    # Перемешиваем данные перед кластеризацией
    logger.info("Перемешиваем данные перед кластеризацией для устранения влияния порядка movieId...")
    movie_ids = df['movieId'].values
    X_reduced, movie_ids = shuffle(X_reduced, movie_ids, random_state=42)

    # Кластеризация с KMeans
    best_k = 9
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(X_reduced)

    # Восстанавливаем порядок labels
    labels_df = pd.DataFrame({'movieId': movie_ids, 'cluster': labels})
    labels_df = labels_df.merge(df[['movieId']], on='movieId', how='right')
    labels = labels_df['cluster'].values

    # Подкластеризация
    logger.info("Выполняем адаптивную подкластеризацию с KMeans...")
    sub_labels = subcluster_data_kmeans(
        X_reduced, labels,
        min_k=3, max_k=8,
        silhouette_threshold=0.5
    )

    # Восстанавливаем порядок sub_labels
    sub_labels_df = pd.DataFrame({'movieId': movie_ids, 'subcluster': sub_labels})
    sub_labels_df = sub_labels_df.merge(df[['movieId']], on='movieId', how='right')
    sub_labels = sub_labels_df['subcluster'].values

    # Сохраняем кластеры и подкластеры
    df['cluster'] = labels
    df['subcluster'] = sub_labels

    # Сохраняем промежуточный файл с обоими наборами меток
    output_path = os.path.join("output", "movies_with_clusters.csv")
    df_final = df[['movieId', 'title', 'genres', 'cluster', 'subcluster', 'tag']]
    df_final.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Полный список фильмов с кластерами сохранён в {output_path}")

    # Анализ кластеров
    try:
        analyze_clusters(X_reduced, labels, df, sub_labels=sub_labels, output_dir="cluster_analysis")
    except KeyError as e:
        logger.error(f"Ошибка в анализе кластеров: отсутствует колонка {e}. Проверь объединение с рейтингами.")
        print(f"Ошибка: отсутствует колонка {e}. Убедись, что рейтинги корректно объединены с df.")

    # Пример рекомендаций
    print("\nРекомендации для фильма с movieId=1 (Toy Story):")
    try:
        recommendations = recommend_movies(movie_id=1, df=df, labels=labels, sub_labels=sub_labels,
                                          X_reduced=X_reduced, n_recommendations=5)
        print(recommendations)
    except Exception as e:
        logger.error(f"Ошибка при генерации рекомендаций: {e}")
        print(f"Ошибка при генерации рекомендаций: {e}")

    while True:
        try:
            user_input = input("\nВведите movieId для рекомендаций (или 'exit' для завершения): ").strip()
            if not user_input:
                print("Ошибка: Пожалуйста, введите число или 'exit'.")
                continue
            if user_input.lower() == 'exit':
                break
            movie_id = int(user_input)
            print(f"\nРекомендации для фильма с movieId={movie_id}:")
            recommendations = recommend_movies(movie_id=movie_id, df=df, labels=labels, sub_labels=sub_labels,
                                              X_reduced=X_reduced, n_recommendations=5)
            print(recommendations)
        except ValueError as e:
            print(f"Ошибка: Введите корректное число, а не '{user_input}'. Попробуйте снова.")
        except Exception as e:
            print(f"Произошла ошибка: {e}")

    # Сохранение итогового файла
    output_file = os.path.join("output", "final_movie_clusters.csv")
    df_final = df[['movieId', 'title', 'genres', 'subcluster', 'tag', 'rating']]
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Итоговый файл сохранён в {output_file}")

    logger.info(f"Пайплайн завершён за {time.time() - start_time:.2f} секунд.")
    logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Произошла критическая ошибка: %s", e)
        print(f"Произошла критическая ошибка: {e}")