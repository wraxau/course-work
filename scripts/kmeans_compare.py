import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

def compute_elbow_method(data, k_range=range(3, 20), output_dir="elbow_results"):
    """
    Вычисляет метод локтя для KMeans и определяет оптимальное число кластеров.

    Параметры:
    ----------
    data : array-like
        Данные для кластеризации.
    k_range : range
        Диапазон значений k для тестирования.
    output_dir : str
        Путь к папке для сохранения результатов.

    Возвращает:
    -----------
    int
        Оптимальное число кластеров (k).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inertias = []
    for k in k_range:
        print(f"Вычисление инерции для k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    kl = KneeLocator(list(k_range), inertias, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', label='KMeans')
    if optimal_k:
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Оптимальное k = {optimal_k}')
    plt.xlabel('Число кластеров (k)')
    plt.ylabel('Инерция')
    plt.title(f'Метод локтя: KMeans (диапазон {min(k_range), max(k_range)})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/elbow_kmeans_range_{min(k_range)}_{max(k_range)}.png")
    plt.close()

    results_df = pd.DataFrame({'k': list(k_range), 'inertia': inertias})
    results_df.to_csv(f"{output_dir}/elbow_results_range_{min(k_range)}_{max(k_range)}.csv", index=False)

    return optimal_k

def evaluate_kmeans(data, k_values, output_dir="kmeans_results"):
    """
    Проводит кластеризацию с KMeans для разных k и вычисляет метрики.

    Параметры:
    ----------
    data : array-like
        Данные для кластеризации.
    k_values : list
        Список значений k для тестирования.
    output_dir : str
        Путь к папке для сохранения результатов.

    Возвращает:
    -----------
    pd.DataFrame
        Таблица с метриками для каждого k.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []
    for k in k_values:
        print(f"Кластеризация с KMeans для k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)

        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)

        results.append({
            "k": k,
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "calinski_harabasz": calinski_harabasz
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/kmeans_metrics.csv", index=False)

    return results_df

def analyze_clusters(data, labels, df, output_dir="cluster_analysis"):
    """
    Анализирует кластеры: какие жанры, теги и рейтинги преобладают.

    Параметры:
    ----------
    data : array-like
        Данные для кластеризации.
    labels : array-like
        Метки кластеров (результат KMeans).
    df : pd.DataFrame
        Исходный датасет с колонками: title, rating, genres, tags.
    output_dir : str
        Путь к папке для сохранения результатов.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df['cluster'] = labels
    n_clusters = len(set(labels))

    for cluster in range(n_clusters):
        print(f"\nАнализ кластера {cluster}:")
        cluster_data = df[df['cluster'] == cluster]

        print(f"Размер кластера: {len(cluster_data)} фильмов")
        avg_rating = cluster_data['rating'].mean()
        print(f"Средний рейтинг: {avg_rating:.2f}")

        genres_list = cluster_data['genres'].str.split(',').explode().str.strip()
        genre_counts = genres_list.value_counts()
        print("Топ-5 жанров:")
        print(genre_counts.head(5))

        tags_list = cluster_data['tags'].str.split(',').explode().str.strip()
        tag_counts = tags_list.value_counts()
        print("Топ-5 тегов:")
        print(tag_counts.head(5))

        print("Примеры фильмов:")
        print(cluster_data[['title', 'rating', 'genres', 'tags']].head(5))

        with open(f"{output_dir}/cluster_{cluster}_analysis.txt", 'w') as f:
            f.write(f"Анализ кластера {cluster}\n")
            f.write(f"Размер кластера: {len(cluster_data)} фильмов\n")
            f.write(f"Средний рейтинг: {avg_rating:.2f}\n")
            f.write("\nТоп-5 жанров:\n")
            f.write(genre_counts.head(5).to_string() + "\n")
            f.write("\nТоп-5 тегов:\n")
            f.write(tag_counts.head(5).to_string() + "\n")
            f.write("\nПримеры фильмов:\n")
            f.write(cluster_data[['title', 'rating', 'genres', 'tags']].head(5).to_string() + "\n")

if __name__ == "__main__":
    X = np.random.rand(35000, 100)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svd = TruncatedSVD(n_components=50)
    X_reduced = svd.fit_transform(X_scaled)

    # Имитация исходного датасета
    df = pd.DataFrame({
        'title': [f"Movie {i}" for i in range(35000)],
        'rating': np.random.uniform(1, 10, 35000),
        'genres': np.random.choice(['Action,Comedy', 'Drama,Romance', 'Horror,Thriller', 'Sci-Fi,Adventure'], 35000),
        'tags': np.random.choice(['funny,exciting', 'sad,emotional', 'scary,intense', 'futuristic,epic'], 35000)
    })

    # Шаг 1: Метод локтя для разных диапазонов
    ranges = [range(3, 20), range(3, 40), range(3, 60)]
    optimal_ks = []
    for k_range in ranges:
        optimal_k = compute_elbow_method(X_reduced, k_range=k_range, output_dir="elbow_results")
        if optimal_k:
            optimal_ks.append(optimal_k)
        print(f"Оптимальное k для диапазона {k_range}: {optimal_k}")

    # Шаг 2: Оценка KMeans для найденных k
    if optimal_ks:
        results_df = evaluate_kmeans(X_reduced, optimal_ks, output_dir="kmeans_results")
        print("\nМетрики для разных k:")
        print(results_df)

        # Выбираем лучшее k по Silhouette Score
        best_k = results_df.loc[results_df['silhouette'].idxmax()]['k']
        print(f"\nЛучшее k по Silhouette Score: {int(best_k)}")

        # Шаг 3: Кластеризация и анализ кластеров для лучшего k
        kmeans = KMeans(n_clusters=int(best_k), random_state=42)
        labels = kmeans.fit_predict(X_reduced)
        analyze_clusters(X_reduced, labels, df, output_dir="cluster_analysis")
    else:
        print("Не удалось определить оптимальное k. Используем k=9 по умолчанию.")
        kmeans = KMeans(n_clusters=9, random_state=42)
        labels = kmeans.fit_predict(X_reduced)
        analyze_clusters(X_reduced, labels, df, output_dir="cluster_analysis")