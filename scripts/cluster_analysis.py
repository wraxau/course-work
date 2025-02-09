from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ✅ Указываем путь к файлу
file_path = "output/processed_tags.csv"

# ✅ Проверяем, существует ли файл processed_tags.csv
if not os.path.exists(file_path):
    print(f"⚠ Файл '{file_path}' не найден. Создаём новый...")

    try:
        tags_df = pd.read_csv("output/cleaned_tags.csv", encoding="utf-8")
        movies_df = pd.read_csv("output/cleaned_movies.csv", encoding="utf-8")
    except FileNotFoundError:
        print("❌ Ошибка: Один из файлов ('cleaned_tags.csv' или 'cleaned_movies.csv') не найден!")
        exit(1)

    processed_tags_df = tags_df.merge(movies_df, on="movieId", how="left")
    processed_tags_df.to_csv(file_path, index=False, encoding="utf-8")

    print(f"✅ Файл '{file_path}' успешно создан!")
else:
    print(f"✅ Файл '{file_path}' уже существует, пропускаем создание.")

# ✅ Загружаем processed_tags.csv
print(f"🔍 Загружаем {file_path}...")
try:
    tags_with_genres_df = pd.read_csv(file_path, encoding="utf-8")
    print(f"✅ Загружено! Размерность: {tags_with_genres_df.shape}")
except Exception as e:
    print(f"❌ Ошибка при загрузке '{file_path}': {e}")
    exit(1)

# ✅ Настройка шрифтов для графиков
rcParams["font.family"] = "Arial"
rcParams["axes.unicode_minus"] = False


def analyze_clusters(tags_with_genres_df, n_clusters=10):
    """Анализ распределения тегов по кластерам."""
    print("\n[Анализ кластеров] Начинаем анализ распределения тегов...")
    start_time = time.time()

    if tags_with_genres_df is None or tags_with_genres_df.empty:
        print("❌ Ошибка: tags_with_genres_df пуст или не загружен!")
        return

    # ✅ Заполнение пропущенных значений
    tags_with_genres_df['tag'] = tags_with_genres_df['tag'].fillna('')

    # ✅ Заполняем пропущенные значения, чтобы избежать ошибок
    tags_with_genres_df['tag'] = tags_with_genres_df['tag'].fillna('')

    # ✅ Преобразование тегов в числовые вектора
    print("[Анализ кластеров] Преобразуем теги в числовые вектора...")

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'), max_features=500)
    tag_matrix = vectorizer.fit_transform(tags_with_genres_df['tag'])

    # ✅ Создаём DataFrame с обработанными тегами
    tags_encoded_df = pd.DataFrame(tag_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # ✅ Добавляем векторизованные теги к `tags_with_genres_df`
    tags_with_genres_df = pd.concat([tags_with_genres_df, tags_encoded_df], axis=1)

    print(f"[Анализ кластеров] Теги преобразованы. Новая размерность данных: {tags_with_genres_df.shape}")
    # ✅ Удаляем нечисловые столбцы перед KMeans
    features = tags_with_genres_df.drop(columns=['movieId', 'tag', 'title'], errors='ignore')

    # ✅ Запускаем KMeans-кластеризацию
    print("[Анализ кластеров] Запускаем KMeans-кластеризацию...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    tags_with_genres_df['cluster'] = kmeans.fit_predict(features)

    print("[Анализ кластеров] Кластеризация завершена!")

    # ✅ Анализ популярных тегов по кластерам
    print("[Анализ кластеров] Анализируем популярные теги в каждом кластере...")
    popular_tags_data = []

    for cluster in tags_with_genres_df['cluster'].unique():
        cluster_tags = tags_with_genres_df[tags_with_genres_df['cluster'] == cluster]['tag'].dropna().astype(str)
        all_tags = ' '.join(cluster_tags)
        tag_counts = Counter(all_tags.split('|'))

        for tag, count in tag_counts.items():
            popular_tags_data.append({'cluster': cluster, 'tag': tag, 'count': count})

    popular_tags_df = pd.DataFrame(popular_tags_data)
    print("[Анализ кластеров] Анализ тегов завершен!")

    # ✅ Сохранение графика
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    print("📌 [Анализ кластеров] Строим график распределения тегов по кластерам...")
    plt.figure(figsize=(12, 8))
    for cluster in popular_tags_df['cluster'].unique():
        cluster_data = popular_tags_df[popular_tags_df['cluster'] == cluster]
        plt.bar(cluster_data['tag'], cluster_data['count'], label=f'Cluster {cluster}')

    plt.xticks(rotation=90, fontsize=10)
    plt.xlabel('Tag', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Tag Distribution by Cluster', fontsize=14)
    plt.legend()
    plt.tight_layout(pad=4.0)

    output_file = os.path.join(output_dir, 'tag_distribution_by_cluster.png')
    plt.savefig(output_file, format='png')
    plt.close()
    print(f"[Анализ кластеров] График сохранён в {output_file}")

    print(f"[Анализ кластеров] Завершено за {time.time() - start_time:.2f} секунд\n")


def analyze_genres_and_clusters(movies_file="output/cleaned_movies.csv",
                                tags_file="output/processed_tags.csv",
                                n_clusters=10):
    """Анализирует распределение жанров по кластерам."""
    print("[Анализ жанров] Начинаем анализ жанров по кластерам...")
    start_time = time.time()

    # Загружаем данные
    movies_df = pd.read_csv(movies_file, encoding="utf-8")
    tags_df = pd.read_csv(tags_file, encoding="utf-8")

    # One-hot encoding жанров
    print("[Анализ жанров] Преобразуем жанры в one-hot формат...")
    mlb = MultiLabelBinarizer()
    movies_df["genres"] = movies_df["genres"].apply(lambda x: x.split("|") if isinstance(x, str) else [])
    genres_matrix = mlb.fit_transform(movies_df["genres"])
    genres_df = pd.DataFrame(genres_matrix, columns=mlb.classes_, index=movies_df["movieId"])

    # Объединяем данные по тегам и жанрам
    print("[Анализ жанров] Объединяем данные...")
    tags_with_genres_df = pd.merge(tags_df, movies_df[["movieId"]], on="movieId", how="left")
    tags_with_genres_df = pd.merge(tags_with_genres_df, genres_df, on="movieId", how="left")

    # PCA для снижения размерности
    print("[Анализ жанров] Применяем PCA для снижения размерности...")
    pca = PCA(n_components=min(38, tags_with_genres_df.shape[1] - 2), random_state=42)
    features_pca = pca.fit_transform(tags_with_genres_df.iloc[:, 3:])  # Используем только числовые колонки

    # Кластеризация KMeans
    print("[Анализ жанров] Запускаем KMeans-кластеризацию...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    tags_with_genres_df["cluster"] = kmeans.fit_predict(features_pca)

    print("[Анализ жанров] Кластеризация завершена!")

    # Анализ популярных тегов
    print("[Анализ жанров] Анализируем популярные теги в каждом кластере...")
    popular_tags_data = []
    for cluster in range(n_clusters):
        cluster_tags = tags_with_genres_df[tags_with_genres_df["cluster"] == cluster]["tag"].dropna()
        all_tags = " ".join(cluster_tags.astype(str))
        tag_counts = Counter(all_tags.split("|"))

        for tag, count in tag_counts.items():
            popular_tags_data.append({"cluster": cluster, "tag": tag, "count": count})

    popular_tags_df = pd.DataFrame(popular_tags_data)

    # Построение графика
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    for cluster in popular_tags_df["cluster"].unique():
        cluster_data = popular_tags_df[popular_tags_df["cluster"] == cluster]
        plt.bar(cluster_data["tag"], cluster_data["count"], label=f"Cluster {cluster}")

    plt.xticks(rotation=90)
    plt.xlabel("Tag")
    plt.ylabel("Count")
    plt.title("Tag Distribution by Cluster (with Genres)")
    plt.legend()
    plt.tight_layout()

    output_file = os.path.join(output_dir, "tag_distribution_by_cluster_with_genres.png")
    plt.savefig(output_file)
    plt.close()
    print(f"[Анализ жанров] График сохранён в {output_file}")
    print(f"[Анализ жанров] Завершено за {time.time() - start_time:.2f} секунд\n")


# Указываем, какие функции экспортировать
__all__ = ["analyze_clusters", "analyze_genres_and_clusters"]