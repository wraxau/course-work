import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from kneed import KneeLocator

def compute_elbow_method(data, k_range=range(10, 40), output_dir="output"):
    """
    Метод локтя для KMeans и MiniBatchKMeans:
    - Строит графики инерции
    - Определяет оптимальное k по KMeans
    - Сохраняет данные и графики

    Параметры:
    ----------
    data : array-like
        Эмбеддинги или любые числовые представления данных.
    k_range : range
        Диапазон количества кластеров.
    output_dir : str
        Путь к папке для сохранения результатов.

    Возвращает:
    -----------
    int
        Оптимальное число кластеров, определённое по KMeans.
    """

    inertias_kmeans = []
    inertias_mini = []

    for k in k_range:
        # KMeans
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(data)
        inertias_kmeans.append(km.inertia_)

        # MiniBatchKMeans
        mbkm = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100)
        mbkm.fit(data)
        inertias_mini.append(mbkm.inertia_)

    # Определение оптимального k только по KMeans
    kl = KneeLocator(
        x=list(k_range),
        y=inertias_kmeans,
        curve="convex",
        direction="decreasing"
    )
    optimal_k = kl.elbow

    # Проверка существования папки
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Сохраняем график
        plt.figure(figsize=(10, 8))
        plt.plot(k_range, inertias_kmeans, label="KMeans", marker='o')
        plt.plot(k_range, inertias_mini, label="MiniBatchKMeans", marker='s')
        if optimal_k:
            plt.axvline(x=optimal_k, color='red', linestyle='--', label=f"Оптимальное k = {optimal_k}")
        plt.xlabel("Число кластеров")
        plt.ylabel("Inertia")
        plt.title("Метод локтя: KMeans vs MiniBatchKMeans")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/elbow_method_comparison.png")
        plt.close()
    except Exception as e:
        print(f"Ошибка при сохранении графика: {e}")

    try:
        # Сохраняем CSV с инерциями
        df = pd.DataFrame({
            "n_clusters": list(k_range),
            "inertia_kmeans": inertias_kmeans,
            "inertia_minibatch": inertias_mini
        })
        df.to_csv(f"{output_dir}/inertia_comparison.csv", index=False)
    except Exception as e:
        print(f"Ошибка при сохранении CSV: {e}")

    return optimal_k
