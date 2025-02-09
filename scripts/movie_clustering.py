import os
import time  # Добавляем импорт
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA

# Создаем папку для результатов, если её нет
output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(output_dir, exist_ok=True)


def filter_rare_values(df, column, threshold=200):
    """Удаляет редкие жанры (менее threshold вхождений)"""
    value_counts = df[column].explode().value_counts()
    rare_values = value_counts[value_counts < threshold].index
    df[column] = df[column].apply(lambda x: [v for v in x if v not in rare_values])
    return df
import ast

def safe_literal_eval(val):
    """Безопасно выполняет ast.literal_eval, проверяя тип данных"""
    try:
        if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
            return ast.literal_eval(val)
        return val  # Если это не строка-список, оставляем как есть
    except (ValueError, SyntaxError):
        return []  # Если ошибка, возвращаем пустой список


def create_movie_features(movies_df, ratings_df, tags_df):
    print("Анализ жанров] Начинаем обработку данных...")

    # Применяем безопасное преобразование жанров
    movies_df['genres'] = movies_df['genres'].apply(safe_literal_eval)

    # Убираем редкие жанры
    movies_df = filter_rare_values(movies_df, 'genres', threshold=200)

    # Разворачиваем список жанров
    movies_genres = movies_df.explode('genres')

    # One-Hot Encoding жанров
    movies_genres = pd.get_dummies(movies_genres, columns=['genres'])

    # Группируем по `movieId`
    movies_genres = movies_genres.groupby('movieId').sum().reset_index()

    # Средний рейтинг и количество оценок
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.rename(columns={'rating': 'avg_rating'}, inplace=True)

    rating_counts = ratings_df.groupby('movieId')['rating'].count().reset_index()
    rating_counts.rename(columns={'rating': 'rating_count'}, inplace=True)

    print(f"[Анализ жанров] Размер movies_df: {movies_df.shape}, tags_df: {tags_df.shape}")

    # Фильтрация тегов: удаляем редкие теги
    tag_counts = tags_df['tag'].value_counts()
    common_tags = tag_counts[tag_counts > 10].index
    tags_df = tags_df[tags_df['tag'].isin(common_tags)]
    print(f"[Фильтрация тегов] Оставлено {len(common_tags)} популярных тегов")

    # Удаляем дубликаты `movieId`, чтобы уменьшить размер `tags_df`
    tags_df = tags_df[['movieId', 'tag']].drop_duplicates()

    print(f"[Объединение данных] Перед объединением: movies_df={movies_df.shape}, tags_df={tags_df.shape}")
    merged_df = pd.concat([
        tags_df.drop_duplicates(subset=["movieId"]).set_index("movieId"),
        movies_genres.drop_duplicates(subset=["movieId"]).set_index("movieId")
    ], axis=1, join="inner")

    print(f"[Объединение данных] После объединения: merged_df={merged_df.shape}")

    # Объединяем с рейтингами
    movie_features = merged_df.merge(avg_ratings, on='movieId', how='left')
    movie_features = movie_features.merge(rating_counts, on='movieId', how='left')

    # Заполняем пропущенные значения
    movie_features.fillna(0, inplace=True)

    print("[Анализ жанров] Признаки успешно созданы!")
    return movie_features
import os
import time
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

def perform_clustering(movie_features, movies_df, n_clusters=10):
    """Выполняет кластеризацию фильмов с MiniBatchKMeans + PCA"""

    print("\n [Кластеризация] Начинаем обработку данных...")

    start_time = time.time()

    # Удаляем NaN
    initial_size = movie_features.shape[0]
    movie_features = movie_features.dropna()
    print(f"[Очистка] Удалены NaN, осталось {movie_features.shape[0]} строк (было {initial_size})")

    if movie_features.shape[0] == 0:
        print("[Ошибка] Нет данных для кластеризации после очистки!")
        return None

    # Убираем строковые столбцы
    movie_features = movie_features.select_dtypes(include=[np.number])

    # Проверяем, что индексы совпадают перед кластеризацией
    if len(movie_features) != len(movies_df):
        print(f"⚠ Предупреждение: число строк в movies_df ({len(movies_df)}) и movie_features ({len(movie_features)}) не совпадает!")
        movies_df = movies_df[movies_df["movieId"].isin(movie_features.index)]
        print(f"movies_df теперь имеет {len(movies_df)} строк")

    # PCA для уменьшения размерности
    print("[PCA] Уменьшаем размерность данных...")
    pca_start = time.time()
    n_components = min(movie_features.shape[1], 300)  # Число компонент не может быть больше количества признаков
    pca = PCA(n_components=n_components, random_state=42)
    movie_features_pca = pca.fit_transform(movie_features)
    print(f" [PCA] Размерность снижена: {movie_features.shape[1]} → {movie_features_pca.shape[1]}")
    print(f"Время выполнения PCA: {time.time() - pca_start:.2f} секунд")
    if len(movie_features_pca) != len(movies_df):
        print(f"Ошибка: размерность PCA ({len(movie_features_pca)}) ≠ movies_df ({len(movies_df)})")
        return None

    # MiniBatchKMeans
    print("[MiniBatchKMeans] Запускаем алгоритм кластеризации...")
    clustering_start = time.time()

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, n_init=10)
    clusters = kmeans.fit_predict(movie_features_pca)

    if len(clusters) == len(movies_df):
        movies_df["cluster"] = clusters
    else:
        print(f"Ошибка: размер кластера {len(clusters)} ≠ {len(movies_df)}")
        return None

    print(f" [MiniBatchKMeans] Кластеры сформированы! (время: {time.time() - clustering_start:.2f} секунд)")

    # Сохраняем результат
    output_file = "output/clusters_movies.csv"
    movies_df.to_csv(output_file, index=False)

    print(f"[Сохранение] Результаты кластеризации сохранены в '{output_file}'")
    print(f"[ИТОГО] Общая длительность кластеризации: {time.time() - start_time:.2f} секунд\n")

    return movies_df
