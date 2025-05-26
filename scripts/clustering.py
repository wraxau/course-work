import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def analyze_ratings_by_cluster(movies_df, ratings_df):
    logging.info("Анализ рейтингов по кластерам...")
    """Анализ среднего рейтинга фильмов в каждом кластере с визуализацией."""
    print("\nАнализ рейтингов по кластерам...")

    if 'cluster' not in movies_df.columns:
        print(" `cluster` отсутствует в movies_df! Добавляем заглушку...")
        movies_df['cluster'] = -1

    if ratings_df.empty:
        print("Данные о рейтингах пусты!")
        return pd.DataFrame()

    merged_df = movies_df.merge(ratings_df, on="movieId", how="left")
    cluster_ratings = merged_df.groupby("cluster")["rating"].mean().reset_index()
    cluster_ratings = cluster_ratings.sort_values(by="rating", ascending=False)

    print(cluster_ratings)
    plt.figure(figsize=(10, 5))
    sns.barplot(x="cluster", y="rating", data=cluster_ratings, hue="cluster", legend=False, palette="coolwarm")
    plt.xlabel("Кластер")
    plt.ylabel("Средний рейтинг")
    plt.title("Средний рейтинг фильмов по кластерам")
    plt.savefig("output/cluster_ratings.png")
    plt.close()
    print("График 'cluster_ratings.png' сохранён.")

    return cluster_ratings

def recommend_movies(movie_id, movies_df, ratings_df, n=20):
    """Рекомендует n фильмов из того же кластера, что и переданный movie_id."""

    if 'cluster' not in movies_df.columns:
        raise ValueError("Ошибка: В movies_df отсутствует колонка 'cluster'. Проверьте кластеризацию!")

    if movie_id not in movies_df["movieId"].values:
        raise ValueError(f"Ошибка: фильм с ID {movie_id} не найден в данных!")

    cluster_id = movies_df.loc[movies_df['movieId'] == movie_id, 'cluster'].values[0]
    movies_in_cluster = movies_df[movies_df['cluster'] == cluster_id]

    if movies_in_cluster.empty:
        print(f"В кластере {cluster_id} нет фильмов для рекомендации.")
        return pd.DataFrame()

    recommendations = movies_in_cluster.sample(min(n, len(movies_in_cluster)), random_state=42)
    recommended_ids = recommendations['movieId'].tolist()

    rec_ratings = ratings_df[ratings_df['movieId'].isin(recommended_ids)].groupby("movieId")["rating"].mean().reset_index()
    rec_ratings = rec_ratings.merge(recommendations, on="movieId")

    # Визуализация
    plt.figure(figsize=(10, 5))
    sns.barplot(y=rec_ratings["title"], x=rec_ratings["rating"], hue=rec_ratings["title"], palette="Blues_r", legend=False)
    plt.xlabel("Средний рейтинг")
    plt.ylabel("Фильм")
    plt.title("Рекомендуемые фильмы и их средний рейтинг")
    plt.savefig("output/recommended_movies.png")
    plt.close()
    print("График 'recommended_movies.png' сохранён.")

    return recommendations[['movieId', 'title']]
def get_favorite_cluster(user_id, ratings_df, movies_df, rating_threshold=4.0):
    """
    Определяет, к какому кластеру пользователь ставит самые высокие оценки.
    Автоматически адаптируется, если рейтинги стандартизированы.
    """

    if 'cluster' not in movies_df.columns:
        raise ValueError("Ошибка: В movies_df отсутствует колонка 'cluster'. Проверьте кластеризацию!")

    if ratings_df.empty:
        print(f"Нет данных о рейтингах.")
        return None

    user_ratings = ratings_df[ratings_df['userId'] == user_id]

    if user_ratings.empty:
        print(f"У пользователя {user_id} нет рейтингов.")
        return None

    # Проверка стандартизации
    mean_rating = user_ratings["rating"].mean()
    std_rating = user_ratings["rating"].std()
    is_standardized = (-0.5 < mean_rating < 0.5) and (0.8 < std_rating < 1.2)

    if is_standardized:
        print(f" Обнаружены стандартизированные рейтинги. Игнорируем порог {rating_threshold}.")
        high_rated = user_ratings  # не фильтруем
    else:
        high_rated = user_ratings[user_ratings["rating"] >= rating_threshold]
        if high_rated.empty:
            print(f"Пользователь с ID {user_id} не поставил оценок ≥ {rating_threshold}.")
            return None

    # Объединяем с movie_df по movieId
    merged = high_rated.merge(movies_df, on='movieId', how='inner')

    if merged.empty:
        print(f"Нет данных для анализа кластеров пользователя {user_id}.")
        return None

    cluster_ratings = merged.groupby('cluster')['rating'].mean().reset_index()

    if cluster_ratings.empty:
        print(f"Нет оценённых фильмов в известных кластерах.")
        return None

    favorite_cluster = cluster_ratings.loc[cluster_ratings['rating'].idxmax(), 'cluster']
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x="cluster",
        y="rating",
        hue="cluster",
        data=cluster_ratings,
        dodge=False,
        palette="viridis",
        legend=False
    )

    label = "стандартизированные" if is_standardized else f"оценки ≥ {rating_threshold}"
    plt.xlabel("Кластер")
    plt.ylabel("Средний рейтинг пользователя")
    plt.title(f"Средний рейтинг пользователя {user_id} по кластерам ({label})")

    output_path = f"output/user_{user_id}_favorite_cluster.png"
    plt.savefig(output_path)
    plt.close()
    print(f"График сохранён: '{output_path}'")

    return favorite_cluster

