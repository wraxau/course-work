import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from kneed import KneeLocator

def compute_elbow_method2(data, k_range=range(3, 30), output_dir="output"):
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

    # Имя диапазона для сохранения файлов (например, "range_3_40")
    range_name = f"range_{min(k_range)}_{max(k_range)}"

    try:
        # Сохраняем график
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias_kmeans, label="KMeans", marker='o')
        plt.plot(k_range, inertias_mini, label="MiniBatchKMeans", marker='s')
        if optimal_k:
            plt.axvline(x=optimal_k, color='red', linestyle='--', label=f"Оптимальное k = {optimal_k}")
        plt.xlabel("Число кластеров")
        plt.ylabel("Inertia")
        plt.title(f"Метод локтя: KMeans vs MiniBatchKMeans ({range_name})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/elbow_method_comparison_{range_name}.png")
        plt.close()
    except Exception as e:
        print(f"Ошибка при сохранении графика для {range_name}: {e}")

    try:
        # Сохраняем CSV с инерциями
        df = pd.DataFrame({
            "n_clusters": list(k_range),
            "inertia_kmeans": inertias_kmeans,
            "inertia_minibatch": inertias_mini
        })
        df.to_csv(f"{output_dir}/inertia_comparison_{range_name}.csv", index=False)
    except Exception as e:
        print(f"Ошибка при сохранении CSV для {range_name}: {e}")

    return optimal_k

# Пример вызова функции с несколькими диапазонами
def run_elbow_method_multiple_ranges2(data, ranges, output_dir="output"):
    """
    Вызывает compute_elbow_method для нескольких диапазонов k_range.

    Параметры:
    ----------
    data : array-like
        Эмбеддинги или числовые данные.
    ranges : list of range
        Список диапазонов для числа кластеров.
    output_dir : str
        Путь к папке для сохранения результатов.

    Возвращает:
    -----------
    dict
        Словарь с оптимальными k для каждого диапазона.
    """
    results = {}

    for k_range in ranges:
        range_name = f"range_{min(k_range)}_{max(k_range)}"
        print(f"Запуск метода локтя для диапазона {range_name}...")
        optimal_k = compute_elbow_method2(data, k_range, output_dir)
        results[range_name] = optimal_k
        print(f"Оптимальное k для {range_name}: {optimal_k}")

    return results

# Пример использования
if __name__ == "__main__":
    # Пример данных (замените на ваши данные)
    import numpy as np
    data = np.random.rand(35000, 50)  # Имитация данных: 35 тысяч строк, 50 признаков

    # Список диапазонов для тестирования
    ranges_to_test = [
        range(3, 20),  # Диапазон 3–19
        range(3, 40),  # Диапазон 3–39
        range(6, 40),  # Диапазон 6–39
        range(3, 30),  # Диапазон 3–29
    ]

    # Вызов функции для всех диапазонов
    results = run_elbow_method_multiple_ranges2(data, ranges_to_test, output_dir="elbow_results")

    # Вывод результатов
    print("\nИтоговые результаты:")
    for range_name, optimal_k in results.items():
        print(f"{range_name}: Оптимальное k = {optimal_k}")