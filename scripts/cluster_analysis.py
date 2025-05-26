'''import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob


def analyze_sentiment(tags_df, movies_df, verbose=True):
    #Анализирует тональность тегов с учётом кластеров

    print("\n Анализируем тональность тегов...")

    tags_df["tag"] = tags_df["tag"].astype(str).fillna("")
    tags_df["sentiment"] = tags_df["tag"].apply(lambda x: TextBlob(x).sentiment.polarity)

    avg_sentiment = tags_df["sentiment"].mean()
    print(f"Средняя тональность всех тегов: {avg_sentiment:.2f}")

    if "cluster" not in tags_df.columns:
        print("`cluster` отсутствует в tags_df, добавляем...")
        tags_df = tags_df.merge(movies_df[["movieId", "cluster"]], on="movieId", how="left")

    if verbose:
        sentiment_by_cluster = tags_df.groupby("cluster")["sentiment"].mean().reset_index()
        print("\n Средняя тональность по кластерам:")
        print(sentiment_by_cluster.sort_values("sentiment", ascending=False))

    return tags_df

def filter_top_movies(movies_df, ratings_df, min_rating=3.5, min_votes=20):
    #Фильтрует фильмы с высоким рейтингом и достаточным числом голосов, учитывая стандартизированные рейтинги

    print("\n Фильтрация топовых фильмов...")

    # Объединяем фильмы с рейтингами
    merged_df = movies_df.merge(ratings_df, on="movieId", how="left")

    # Проверяем, какие есть реальные `rating`
    print("Пример реальных значений рейтингов:")
    print(merged_df["rating"].describe())

    # Считаем средний рейтинг и количество голосов
    movie_stats = merged_df.groupby("movieId").agg(
        avg_rating=("rating", "mean"),  # Средний рейтинг фильма
        vote_count=("rating", "count")  # Количество голосов
    ).reset_index()

    print(f"Всего фильмов в dataset: {movie_stats.shape[0]}")
    print("Пример данных перед фильтрацией:")
    print(movie_stats.head(100))

    # Если рейтинги стандартизированы, переводим обратно к шкале 1-5
    if movie_stats["avg_rating"].min() < 0 and movie_stats["avg_rating"].max() < 5:
        print("Обнаружены стандартизированные рейтинги! Переводим их обратно...")
        movie_stats["avg_rating"] = (movie_stats["avg_rating"] * ratings_df["rating"].std()) + ratings_df["rating"].mean()

    print("Пример после перевода рейтингов в шкалу 1-5:")
    print(movie_stats.head(100))

    # Используем `percentile`, если рейтинги всё ещё не дают результатов
    if movie_stats["avg_rating"].max() < min_rating:
        min_rating = movie_stats["avg_rating"].quantile(0.8)  # Берём топ-20% фильмов

    # Фильтруем фильмы с высоким рейтингом и достаточным числом голосов
    top_movies = movie_stats[(movie_stats["avg_rating"] >= min_rating) & (movie_stats["vote_count"] >= min_votes)]

    # Соединяем с `movies_df`, чтобы вернуть полные данные
    filtered_movies = top_movies.merge(movies_df, on="movieId", how="left")

    print(f"Найдено {filtered_movies.shape[0]} фильмов с рейтингом ≥ {min_rating:.2f} и голосами ≥ {min_votes}")
    print("ТОП-10 фильмов после фильтрации:")
    print(filtered_movies.head(100))

    return filtered_movies

def analyze_cluster_distribution(movies_df):
    #Анализирует количество фильмов в каждом кластере и строит график

    print("\nАнализ распределения фильмов по кластерам...")

    # Проверяем, есть ли нужные колонки
    if "avg_rating" not in movies_df.columns or "rating_count" not in movies_df.columns:
        print("Ошибка: Колонки 'avg_rating' и 'rating_count' отсутствуют в movies_df!")
        return

    cluster_stats = movies_df.groupby("cluster").agg(
        movie_count=("movieId", "count"),
        avg_rating=("avg_rating", "mean"),
        total_votes=("rating_count", "sum")
    ).reset_index()

    print(cluster_stats)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=cluster_stats["cluster"], y=cluster_stats["movie_count"], palette="Set2")
    plt.xlabel("Кластер")
    plt.ylabel("Количество фильмов")
    plt.title("Распределение фильмов по кластерам")
    plt.xticks(rotation=45)
    plt.savefig("output/cluster_distribution_analysis.png")
    print("График 'cluster_distribution_analysis.png' сохранён.")
    plt.close()


def analyze_ratings_by_cluster(movies_df, ratings_df=None):
    print("\nАнализ рейтингов по кластерам...")

    # Убедимся, что нужная колонка есть
    if 'avg_rating' not in movies_df.columns:
        raise ValueError("Колонка 'avg_rating' не найдена в movies_df.")

    cluster_ratings = movies_df.groupby("cluster")["avg_rating"].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x="cluster",
        y="avg_rating",
        hue="cluster",
        data=cluster_ratings,
        dodge=False,
        palette="coolwarm",
        legend=False
    )

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Кластер")
    plt.ylabel("Средний рейтинг")
    plt.title("Средний рейтинг фильмов по кластерам")
    plt.xticks(rotation=45)

    output_path = "output/cluster_ratings_fixed.png"
    plt.savefig(output_path)
    plt.close()
    print(f"График '{output_path}' сохранён.")

    return cluster_ratings


def analyze_genres_by_cluster(movies_df):
    #Анализирует популярные жанры в каждом кластере
    print("Проверка жанров по кластерам:")
    print(movies_df[['movieId', 'genres']].head())

    print("\nАнализ жанров по кластерам...")
    if 'genres' not in movies_df.columns:
        raise ValueError("'genres' столбец отсутствует в данных!")

        # Продолжайте анализ жанров
        # Например, разбиваем жанры на несколько элементов
    movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

    movies_df['genres'] = movies_df['genres'].fillna("").astype(str).apply(lambda x: x.split('|'))
    genre_df = movies_df.explode('genres')[['cluster', 'genres']].groupby(['cluster', 'genres']).size().reset_index(
        name='count')
    genre_counts = genre_df.groupby(["cluster", "genres"]).size().reset_index(name="count")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=genre_counts, x="cluster", y="count", hue="genre", dodge=True, palette="tab10")
    plt.xlabel("Кластер")
    plt.ylabel("Количество фильмов")
    plt.title("Популярные жанры в кластерах")
    plt.legend(loc="upper right", title="Жанры")
    plt.savefig("output/genre_distribution_by_cluster.png")
    print("График 'genre_distribution_by_cluster.png' сохранён.")
    plt.close()

    return genre_counts



def get_top_movies_in_clusters(movies_df, ratings_df, top_n=5):
    # Находит топ-N фильмов в каждом кластере по среднему рейтингу

    print("\n ТОП фильмов в каждом кластере...")

    merged_df = movies_df.merge(ratings_df, on="movieId", how="left")
    avg_ratings = merged_df.groupby(["cluster", "movieId", "title"])["rating"].mean().reset_index()
    top_movies = avg_ratings.sort_values(["cluster", "rating"], ascending=[True, False]).groupby("cluster").head(top_n)

    print(top_movies)
    return top_movies


def compare_clusters(movies_df, ratings_df):
    #Сравнивает кластеры: где больше фильмов с высоким рейтингом

    cluster_ratings = (
        ratings_df.groupby(['movieId'])['rating']
        .mean()
        .reset_index()
        .merge(movies_df, on='movieId', how="left")
    )

    cluster_ratings["rating"] = cluster_ratings["rating"].fillna(0)
    high_rated_clusters = (
        cluster_ratings[cluster_ratings['rating'] > 4]
        .groupby('cluster')
        .size()
        .reset_index(name='count')
    )

    print("Кластеры с наибольшим количеством высокооценённых фильмов:")
    print(high_rated_clusters.sort_values(by='count', ascending=False))

    return high_rated_clusters


def popular_genres_in_clusters(movies_df, top_n=3):
    #Определяет самые популярные жанры в каждом кластере

    print("\nОпределяем популярные жанры в кластерах...")

    genres_df = movies_df[['genres', 'cluster']].explode('genres')
    genre_counts = (
        genres_df.groupby(['cluster', 'genres'])
        .size()
        .reset_index(name='count')
    )

    top_genres = (
        genre_counts.groupby('cluster')
        .apply(lambda x: x.nlargest(top_n, 'count'))
        .reset_index(drop=True)
    )

    print("Популярные жанры в каждом кластере:")
    print(top_genres)

    return top_genres

__all__ = [
    "analyze_cluster_distribution",
    "analyze_genres_by_cluster",
    "analyze_ratings_by_cluster",
    "get_top_movies_in_clusters",
    "compare_clusters",
    "popular_genres_in_clusters",
    "analyze_sentiment",
    "filter_top_movies"
]
'''