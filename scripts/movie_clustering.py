import os
import time
import pandas as pd
import numpy as np
import ast
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib  # Для сохранения модели
from sklearn.preprocessing import MultiLabelBinarizer

# Создаем папку для результатов, если её нет
output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(output_dir, exist_ok=True)



def safe_literal_eval(val):
    #Преобразование строки в список (жанры)
    try:
        return ast.literal_eval(val) if isinstance(val, str) and val.startswith("[") else val
    except (ValueError, SyntaxError):
        return []


def filter_rare_values(df, column, threshold=200):
    #Удаляет редкие жанры (менее threshold вхождений)
    df[column] = df[column].apply(lambda x: "|".join(x) if isinstance(x, list) else x)  # Преобразуем в строку

    value_counts = df[column].str.split("|").explode().value_counts()
    rare_values = value_counts[value_counts < threshold].index

    df[column] = df[column].apply(lambda x: "|".join([v for v in x.split("|") if v not in rare_values]))  # Фильтруем жанры
    df[column] = df[column].apply(lambda x: x.split("|") if x else [])  # Преобразуем обратно в списки
    return df


def create_movie_features(movies_df, ratings_df, tags_df):
    print("[Анализ жанров] Начинаем обработку данных...")

    # Оптимизированное чтение жанров
    movies_df['genres'] = movies_df['genres'].apply(safe_literal_eval)

    # One-Hot Encoding жанров
    mlb = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(
        mlb.fit_transform(movies_df['genres']),
        columns=mlb.classes_,
        index=movies_df["movieId"]  # Убедимся, что индекс - это movieId
    )

    # Средний рейтинг и количество оценок
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean().astype('float32').rename("avg_rating")
    rating_counts = ratings_df.groupby('movieId')['rating'].count().astype('int32').rename("rating_count")

    # Убеждаемся, что индекс - movieId, перед `join()`
    genres_encoded = genres_encoded.reset_index().set_index("movieId")
    avg_ratings = avg_ratings.reset_index().set_index("movieId")
    rating_counts = rating_counts.reset_index().set_index("movieId")

    # **Исправленный `join()`**
    movie_features = genres_encoded \
        .join(avg_ratings, how="left") \
        .join(rating_counts, how="left")

    print("[Анализ жанров] Признаки успешно созданы!")
    return movie_features


def perform_clustering(movie_features, movies_df, n_clusters=10, output_dir="output"):
    #Выполняет кластеризацию фильмов с помощью MiniBatchKMeans + PCA
    print("\n[Кластеризация] Начинаем обработку данных...")
    start_time = time.time()

    # Удаляем строки с NaN (если они остались)
    movie_features = movie_features.dropna()

    if movie_features.empty:
        print("[Ошибка] Нет данных для кластеризации после очистки!")
        return None

    # Оставляем только числовые признаки
    movie_features = movie_features.select_dtypes(include=[np.number])

    # PCA для уменьшения размерности (оставляем 95% дисперсии, но минимум 2 компоненты)
    print("[PCA] Уменьшаем размерность данных...")
    n_components = min(0.95, len(movie_features.columns))  # Исправленная ошибка
    pca = PCA(n_components=n_components, random_state=42)
    movie_features_pca = pca.fit_transform(movie_features)

    print(f"[PCA] Итоговая размерность: {movie_features_pca.shape[1]} компонентов")
    print(f"[PCA] Доля сохраненной дисперсии: {sum(pca.explained_variance_ratio_):.4f}")

    # Убираем NaN после PCA (если они появились)
    movie_features_pca = np.nan_to_num(movie_features_pca)

    # MiniBatchKMeans с увеличенным размером пакета
    print("[MiniBatchKMeans] Запускаем алгоритм кластеризации...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=5000, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(movie_features_pca)

    # Сохраняем модель KMeans для дальнейшего использования
    os.makedirs(output_dir, exist_ok=True)  # Создаем папку для вывода, если ее нет
    joblib.dump(kmeans, os.path.join(output_dir, "kmeans_model.pkl"))
    print(f"[KMeans] Модель сохранена в '{output_dir}/kmeans_model.pkl'")

    # Синхронизируем movies_df с movie_features
    movies_df = movies_df.set_index("movieId").reindex(movie_features.index).reset_index()

    # Проверяем, совпадают ли размеры
    if len(clusters) != len(movies_df):
        print(f"Ошибка: размер clusters ({len(clusters)}) ≠ movies_df ({len(movies_df)})")
        print("Оставляем только фильмы, для которых есть кластерные признаки...")
        movies_df = movies_df[movies_df["movieId"].isin(movie_features.index)]

    print(f"Проверка размеров перед записью кластеров:")
    print(f" - кластеры: {len(clusters)}")
    print(f" - movies_df: {len(movies_df)}")
    print(f" - movie_features: {len(movie_features)}")

    # Теперь размеры должны совпадать, записываем кластеры
    movies_df["cluster"] = clusters
    print(f"[MiniBatchKMeans] Кластеры сформированы!")

    # Сохраняем результат
    output_file = os.path.join(output_dir, "clusters_movies.csv")
    movies_df.to_csv(output_file, index=False)
    print(f"[Сохранение] Кластеры сохранены в '{output_file}'")

    print(f"[ИТОГО] Время выполнения: {time.time() - start_time:.2f} сек\n")

    return movies_df, kmeans


def train_kmeans(movies_df, n_clusters=10):
    """Обучает KMeans для кластеризации фильмов по жанрам."""

    # Проверяем, есть ли колонка с жанрами
    if "genres" not in movies_df.columns:
        raise ValueError("Ошибка: В movies_df отсутствует колонка 'genres'!")

    # Разбиваем жанры в отдельные столбцы
    genres_dummies = movies_df["genres"].str.get_dummies(sep="|")  # One-Hot Encoding
    scaler = StandardScaler()
    genres_scaled = scaler.fit_transform(genres_dummies)

    # Обучаем KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    movies_df["cluster"] = kmeans.fit_predict(genres_scaled)

    print(f"KMeans обучен! Количество кластеров: {n_clusters}")

    return kmeans, movies_df