import os
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scripts.data_export import save_dataframe, save_model
from scripts.data_visualization import plot_genre_ratings
from sklearn.cluster import MiniBatchKMeans, KMeans
from scripts.clustering import analyze_ratings_by_cluster, recommend_movies, get_favorite_cluster
from scripts.movie_clustering import train_kmeans

# Импорты из cluster_analysis
from scripts.cluster_analysis import (
    analyze_cluster_distribution,
    analyze_genres_by_cluster,
    analyze_ratings_by_cluster,
    get_top_movies_in_clusters,
    popular_genres_in_clusters,
    compare_clusters,
    analyze_sentiment,
    filter_top_movies
)

# Остальные импорты
from scripts.data_cleaning import clean_movies, clean_tags, clean_ratings
from scripts.data_processing import standardize_data
from scripts.movie_clustering import perform_clustering, create_movie_features
from scripts.data_visualization import (
    plot_correlation_matrix,
    plot_rating_distribution,
    plot_user_ratings_distribution,
    plot_ratings_over_time,
    plot_top_movies_by_avg_rating,
    plot_cluster_distribution
)


def ensure_output_directory():
    """Создаёт папку 'output', если её нет."""
    os.makedirs('output', exist_ok=True)


def main():
    start_time = time.time()
    ensure_output_directory()

    print("Очистка данных...")
    with ThreadPoolExecutor() as executor:
        executor.map(lambda func: func(), [clean_movies, clean_tags, clean_ratings])

    print("Загрузка данных...")
    dtype_dict = {"userId": "uint32", "movieId": "uint32", "rating": "float32", "timestamp": "int32"}

    try:
        tags_df = pd.read_csv("output/cleaned_tags.csv", encoding="utf-8", usecols=["movieId", "tag"])
        movies_df = pd.read_csv("output/cleaned_movies.csv", encoding="utf-8", usecols=["movieId", "title", "genres"])
        ratings_df = pd.read_csv("output/cleaned_ratings.csv", encoding="utf-8", dtype=dtype_dict)
    except FileNotFoundError as e:
        print(f"Ошибка: {e}. Проверьте, что файлы существуют.")
        return

    print("Стандартизация рейтингов...")
    ratings_df['rating'] = pd.to_numeric(ratings_df['rating'], errors='coerce')
    ratings_df.dropna(subset=['rating'], inplace=True)
    ratings_df = ratings_df[(ratings_df['rating'] >= 1) & (ratings_df['rating'] <= 5)].drop_duplicates()
    ratings_df_standardized = standardize_data(ratings_df, 'rating')

    print("Сохранение очищенных данных...")
    ratings_df.to_csv('output/cleaned_ratings.csv', index=False)
    ratings_df_standardized.to_csv('output/standardized_ratings.csv', index=False)

    print("Строим графики...")
    plot_correlation_matrix(ratings_df)
    plot_rating_distribution(ratings_df)
    plot_user_ratings_distribution(ratings_df)
    plot_ratings_over_time(ratings_df)
    plot_genre_ratings(pd.merge(movies_df, ratings_df, on="movieId"))

    print("Создание признаков для кластеризации...")
    movie_features = create_movie_features(movies_df, ratings_df, tags_df)
    if movie_features is None or movie_features.empty:
        print("Ошибка: movie_features пуст! Завершаем выполнение.")
        return
    print("Обучаем KMeans...")
    kmeans, movies_df = train_kmeans(movies_df, n_clusters=10)

    print("Выполняем кластеризацию...")
    movies_df, kmeans = perform_clustering(movie_features, movies_df, n_clusters=10)
    if movies_df is None:
        print("Ошибка: кластеризация не удалась! Останавливаем выполнение.")
        return

    print("Строим график распределения фильмов по кластерам...")
    plot_cluster_distribution(movies_df)

    print("Анализируем кластеры...")
    analyze_cluster_distribution(movies_df)
    analyze_genres_by_cluster(movies_df)
    analyze_ratings_by_cluster(movies_df, ratings_df)
    popular_genres_in_clusters(movies_df)
    compare_clusters(movies_df, ratings_df)
    filter_top_movies(movies_df, ratings_df)
    analyze_sentiment(tags_df, movies_df, verbose=True)

    print("Получаем ТОП-5 фильмов в каждом кластере...")
    top_movies = get_top_movies_in_clusters(movies_df, ratings_df, top_n=5)
    print(top_movies)

    save_dataframe(movies_df, "clusters_movies.csv")
    if kmeans is not None:
        save_model(kmeans, "kmeans_model.pkl")
    else:
        print("Ошибка: KMeans не был создан, пропускаем сохранение модели!")

    print("Пример рекомендации фильмов...")
    movie_id = 1
    if movie_id not in movies_df['movieId'].values:
        print(f"Ошибка: фильм с ID {movie_id} не найден!")
    else:
        recommendations = recommend_movies(movie_id, movies_df, ratings_df, n=5)
        print("Рекомендованные фильмы:", recommendations)

    print("Определение любимого кластера пользователя...")
    user_id = 5
    favorite_cluster = get_favorite_cluster(user_id, ratings_df, movies_df)
    print(f"Любимый кластер пользователя {user_id}: {favorite_cluster}")

    print("Классификация нового фильма...")
    genres_vector = np.array([0, 1, 1, 0, 1, 0])
    features = movies_df.drop(columns=["movieId", "title", "cluster", "genres"], errors="ignore")
    if not all(features.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        print("Ошибка: В features есть нечисловые значения! Проверьте данные.")
        return



    print(f"Программа завершена за {time.time() - start_time:.2f} секунд.")


if __name__ == "__main__":
    main()
