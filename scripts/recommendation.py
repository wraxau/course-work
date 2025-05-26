import pandas as pd
import logging
import numpy as np

from typing import Optional
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def recommend_by_tags(movie_id: int, df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Рекомендует фильмы с похожими тегами в том же кластере.

    :param movie_id: ID исходного фильма
    :param df: DataFrame с фильмами, тегами и кластерами
    :param top_n: Количество рекомендаций
    :return: DataFrame с рекомендованными фильмами
    """
    if "movieId" not in df.columns or "tag" not in df.columns or "cluster" not in df.columns:
        logger.error("DataFrame должен содержать столбцы: 'movieId', 'tag', 'cluster'")
        return pd.DataFrame()

    movie_row = df[df["movieId"] == movie_id]
    if movie_row.empty:
        logger.warning(f"Фильм с ID {movie_id} не найден!")
        return pd.DataFrame()

    target_cluster = movie_row["cluster"].values[0]
    raw_tags = movie_row["tag"].values[0]
    tags = set(raw_tags.split("|")) if isinstance(raw_tags, str) else set()

    cluster_movies = df[(df["cluster"] == target_cluster) & (df["movieId"] != movie_id)].copy()

    def tag_similarity(tags1: str, tags2: str) -> float:
        set1 = set(tags1.split("|"))
        set2 = set(tags2.split("|"))
        return len(set1 & set2) / max(1, len(set1 | set2))

    cluster_movies["similarity"] = cluster_movies["tag"].apply(
        lambda t: tag_similarity(t, "|".join(tags)) if isinstance(t, str) else 0
    )

    recommendations = cluster_movies.sort_values(by="similarity", ascending=False).head(top_n)
    return recommendations[["movieId", "title", "genres", "similarity"]]


def recommend_movies(movie_id: int, movies_df: pd.DataFrame, ratings_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Рекомендует фильмы, похожие по среднему рейтингу.

    :param movie_id: ID исходного фильма
    :param movies_df: DataFrame с фильмами
    :param ratings_df: DataFrame с оценками
    :param top_n: Количество рекомендаций
    :return: DataFrame с рекомендованными фильмами
    """
    if "movieId" not in movies_df.columns or "rating" not in ratings_df.columns:
        logger.error("Недостаточно данных для выполнения рекомендации.")
        return pd.DataFrame()

    target_ratings = ratings_df[ratings_df["movieId"] == movie_id]
    if target_ratings.empty:
        logger.warning(f"Оценки для фильма ID {movie_id} не найдены!")
        return pd.DataFrame()

    target_avg_rating = target_ratings["rating"].mean()

    avg_ratings = ratings_df.groupby("movieId")["rating"].mean().reset_index()
    avg_ratings = avg_ratings.merge(movies_df[["movieId", "title"]], on="movieId", how="left")

    avg_ratings["similarity"] = abs(avg_ratings["rating"] - target_avg_rating)

    recommendations = avg_ratings[avg_ratings["movieId"] != movie_id] \
        .sort_values(by="similarity", ascending=True).head(top_n)

    return recommendations[["movieId", "title", "rating", "similarity"]]


def get_favorite_cluster(user_id: int, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Optional[int]:
    """
    Определяет кластер, в котором пользователь чаще всего ставил высокие оценки (> 4).

    :param user_id: ID пользователя
    :param ratings_df: DataFrame с оценками
    :param movies_df: DataFrame с фильмами и их кластерами
    :return: ID кластера или None, если нет подходящих фильмов
    """
    if "userId" not in ratings_df.columns or "cluster" not in movies_df.columns:
        logger.error("DataFrame должен содержать 'userId' и 'cluster'")
        return None

    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    if user_ratings.empty:
        logger.warning(f"Оценки пользователя ID {user_id} не найдены!")
        return None

    rated_with_clusters = user_ratings.merge(movies_df[['movieId', 'cluster']], on='movieId', how='left')
    liked_movies = rated_with_clusters[rated_with_clusters['rating'] > 4]

    if liked_movies.empty:
        logger.info(f"Пользователь ID {user_id} не поставил высоких оценок.")
        return None

    favorite_cluster = liked_movies['cluster'].value_counts().idxmax()
    return favorite_cluster


if __name__ == "__main__":
    try:
        df = pd.read_csv("output/clusters_movies_with_tags.csv")
        movie_id = 1
        recommended_movies = recommend_by_tags(movie_id, df)

        if not recommended_movies.empty:
            print("\nРекомендованные фильмы:")
            print(recommended_movies.to_string(index=False))
        else:
            print("Нет рекомендаций.")
    except FileNotFoundError:
        logger.error("Файл output/clusters_movies_with_tags.csv не найден!")
