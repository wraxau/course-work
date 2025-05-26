import pandas as pd
from collections import Counter
import re

import pandas as pd

def rename_column_in_csv(input_path, output_path, old_name, new_name):
    """
    Загружает CSV, переименовывает столбец и сохраняет в новый файл.
    """
    df = pd.read_csv(input_path)
    if old_name in df.columns:
        df = df.rename(columns={old_name: new_name})
    else:
        print(f"Столбец '{old_name}' не найден в файле '{input_path}'")
    df.to_csv(output_path, index=False)
    print(f"Файл сохранён: {output_path}")


def analyze_movie_clusters(file_path):
    """
    Analyzes a CSV file containing movie data to provide insights on clusters, genres, and movies.

    Parameters “

    Parameters:
    file_path (str): Path to the CSV file with columns movieId, title, genres, cluster, tag

    Returns:
    dict: Analysis results including number of clusters, movies per cluster, and genre distributions
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Validate required columns
        required_columns = ['movieId', 'title', 'genres', 'cluster', 'tag']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV file must contain columns: movieId, title, genres, cluster, tag")

        # Total number of movies
        total_movies = len(df)

        # Number of clusters
        num_clusters = df['cluster'].nunique()

        # Movies per cluster
        movies_per_cluster = df['cluster'].value_counts().sort_index().to_dict()

        # Analyze genres per cluster
        genre_per_cluster = {}
        all_genres = []

        for cluster in df['cluster'].unique():
            # Get movies in this cluster
            cluster_movies = df[df['cluster'] == cluster]

            # Split genres and count frequencies
            genres = []
            for genre_list in cluster_movies['genres']:
                if pd.notna(genre_list):
                    genres.extend(genre_list.split('|'))

            # Count genres
            genre_counts = Counter(genres)
            all_genres.extend(genres)

            # Store top genres (at least 2 occurrences or top 3)
            genre_per_cluster[cluster] = [
                (genre, count) for genre, count in genre_counts.most_common()
                if count >= 2 or len(genre_counts) <= 3
            ]

        # Overall genre distribution
        overall_genre_counts = Counter(all_genres)
        overall_genre_distribution = [
            (genre, count) for genre, count in overall_genre_counts.most_common()
        ]

        # Compile results
        results = {
            'total_movies': total_movies,
            'number_of_clusters': num_clusters,
            'movies_per_cluster': movies_per_cluster,
            'genres_per_cluster': genre_per_cluster,
            'overall_genre_distribution': overall_genre_distribution
        }

        return results

    except FileNotFoundError:
        return {"error": f"File {file_path} not found"}
    except Exception as e:
        return {"error": str(e)}


def print_analysis_results(results):
    """
    Prints the analysis results in a readable format.

    Parameters:
    results (dict): Dictionary containing analysis results
    """
    if 'error' in results:
        print(f"Error: {results['error']}")
        return

    print(f"Total Movies: {results['total_movies']}")
    print(f"Number of Clusters: {results['number_of_clusters']}")
    print("\nMovies per Cluster:")
    for cluster, count in results['movies_per_cluster'].items():
        print(f"  Cluster {cluster}: {count} movies")

    print("\nPredominant Genres per Cluster:")
    for cluster, genres in results['genres_per_cluster'].items():
        print(f"  Cluster {cluster}:")
        for genre, count in genres:
            print(f"    {genre}: {count} occurrences")

    print("\nOverall Genre Distribution:")
    for genre, count in results['overall_genre_distribution']:
        print(f"  {genre}: {count} occurrences")


# Example usage
if __name__ == "__main__":
    rename_column_in_csv(
        input_path="/Users/macbookbro/PycharmProjects/course-work/output/final_movie_clusters.csv",
        # путь к исходному файлу
        output_path="final.csv",  # путь к новому файлу
        old_name="subcluster",  # старое имя столбца
        new_name="cluster"  # новое имя столбца
    )
    file_path = "/Users/macbookbro/PycharmProjects/course-work/scripts/final.csv"
    results = analyze_movie_clusters(file_path)
    print_analysis_results(results)