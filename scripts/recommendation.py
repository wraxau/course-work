import pandas as pd

def recommend_by_tags(movie_id, df, top_n=5):
    """Рекомендует фильмы с похожими тегами в том же кластере"""
    movie_row = df[df["movieId"] == movie_id]

    if movie_row.empty:
        print(f"Фильм с ID {movie_id} не найден!")
        return []

    target_cluster = movie_row["cluster"].values[0]
    tags = set(movie_row["tag"].values[0].split("|")) if isinstance(movie_row["tag"].values[0], str) else set()

    cluster_movies = df[df["cluster"] == target_cluster].copy()

    def tag_similarity(tags1, tags2):
        set1, set2 = set(tags1.split("|")), set(tags2.split("|"))
        return len(set1 & set2) / max(1, len(set1 | set2))

    cluster_movies.loc[:, "similarity"] = cluster_movies["tag"].apply(
        lambda t: tag_similarity(t, "|".join(tags)) if isinstance(t, str) else 0
    )

    recommendations = cluster_movies.sort_values(by="similarity", ascending=False).head(top_n)

    return recommendations[["movieId", "title", "genres", "similarity"]]

if __name__ == "__main__":
    df = pd.read_csv("output/clusters_movies_with_tags.csv")

    movie_id = 1  # Например, Toy Story
    recommended_movies = recommend_by_tags(movie_id, df)

    print("\nРекомендованные фильмы:")
    print(recommended_movies)

