import os
import pandas as pd
from scripts.cluster_analysis import analyze_clusters, analyze_genres_and_clusters
from scripts.data_cleaning import load_data, clean_movies, clean_tags, clean_data
from scripts.data_processing import standardize_data
from scripts.movie_clustering import perform_clustering, create_movie_features
from scripts.data_visualization import (
    plot_correlation_matrix,
    plot_rating_distribution,
    plot_user_ratings_distribution,
    plot_ratings_over_time,
    plot_top_movies_by_avg_rating,
    plot_cluster_distribution,
)


def main():
    # Очистка данных
    clean_movies()
    clean_tags()

    # Проверяем наличие папки для сохранения
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка данных
    path = "data"
    filenames = ['movies.csv', 'ratings.csv', 'tags.csv', 'links.csv']
    data = load_data(path, filenames)

    # Очистка данных
    movies_df = clean_data(data['movies.csv'], fillna_values={'genres': ''})
    ratings_df = clean_data(data['ratings.csv'])

    # Предварительная обработка данных
    ratings_df['rating'] = pd.to_numeric(ratings_df['rating'], errors='coerce')
    ratings_df = ratings_df.dropna(subset=['rating'])
    ratings_df = ratings_df[(ratings_df['rating'] >= 1) & (ratings_df['rating'] <= 5)]
    ratings_df = ratings_df.drop_duplicates()
    ratings_df_standardized = standardize_data(ratings_df, 'rating')

    # Сохранение очищенных данных
    ratings_df.to_csv(f'{output_dir}/cleaned_ratings.csv', index=False)
    print("Очищенные данные сохранены в 'output/cleaned_ratings.csv'")

    ratings_df_standardized.to_csv(f'{output_dir}/standardized_ratings.csv', index=False)
    print("Стандартизированные данные сохранены в 'output/standardized_ratings.csv'")

    # Построение графиков
    print("Строим графики")
    plot_correlation_matrix(ratings_df)
    plot_rating_distribution(ratings_df)
    plot_user_ratings_distribution(ratings_df)
    plot_ratings_over_time(ratings_df)
    plot_top_movies_by_avg_rating(ratings_df, movies_df)

    # Создание признаков для кластеризации
    print("Создаем признаки для кластеризации...")
    movie_features = create_movie_features(movies_df, ratings_df, data['tags.csv'])

    # Проверяем, что movie_features не пуст
    if movie_features is None or movie_features.empty:
        print("Ошибка: movie_features пуст! Кластеризация не будет выполнена.")
        return
    print(f"🎯 Перед кластеризацией: movie_features={movie_features.shape}, movies_df={movies_df.shape}")

    # Оставляем только те фильмы, которые есть в обоих DataFrame
    movies_df = movies_df[movies_df['movieId'].isin(movie_features['movieId'])]
    movie_features = movie_features[movie_features['movieId'].isin(movies_df['movieId'])]

    print(f"✅ После синхронизации: movie_features={movie_features.shape}, movies_df={movies_df.shape}")

    # Кластеризация (используется MiniBatchKMeans + PCA)
    print("🔍 Выполняем кластеризацию...")
    movies_df = perform_clustering(movie_features, movies_df, n_clusters=10)

    # Проверяем, что кластеризация завершена успешно
    if movies_df is not None:
        print("Строим график распределения фильмов по кластерам")
        plot_cluster_distribution(movies_df)
    else:
        print("Ошибка: кластеризация не выполнена! Пропускаем построение графика.")
        return

    # Проверяем наличие processed_tags.csv и создаем, если его нет
    file_path = "output/processed_tags.csv"

    if not os.path.exists(file_path):
        print(f"⚠ Файл '{file_path}' не найден. Создаём новый...")

        try:
            tags_df = pd.read_csv("output/cleaned_tags.csv", encoding="utf-8")  # Файл с тегами
            movies_df = pd.read_csv("output/cleaned_movies.csv", encoding="utf-8")  # Файл с фильмами
        except FileNotFoundError:
            print("❌ Ошибка: Один из файлов ('cleaned_tags.csv' или 'cleaned_movies.csv') не найден!")
            exit(1)

        # Объединяем данные по movieId
        processed_tags_df = tags_df.merge(movies_df, on="movieId", how="left")

        # Сохраняем объединённые данные в новый файл
        processed_tags_df.to_csv(file_path, index=False, encoding="utf-8")
        print(f"✅ Файл '{file_path}' успешно создан!")
    else:
        print(f"✅ Файл '{file_path}' уже существует, пропускаем создание.")

    # 🔍 Загружаем объединённые данные tags_with_genres_df
    print(f"🔍 Загружаем {file_path}...")
    try:
        global tags_with_genres_df  # Используем глобальную переменную
        tags_with_genres_df = pd.read_csv(file_path, encoding="utf-8")
        print(f"✅ Загружено! Размерность: {tags_with_genres_df.shape}")
    except Exception as e:
        print(f"❌ Ошибка при загрузке '{file_path}': {e}")
        exit(1)

    # Анализ кластеров
    print("🔎 Анализируем кластеры...")
    analyze_clusters(tags_with_genres_df,n_clusters=10)

    # Анализ жанров по кластерам
    analyze_genres_and_clusters(
        movies_file='output/cleaned_movies.csv',
        tags_file=file_path,  # Используем объединенные теги
        n_clusters=10
    )

    print("Все шаги завершены успешно!")


if __name__ == "__main__":
    main()
