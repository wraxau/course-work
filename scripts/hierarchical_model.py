import os

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

def preprocess_numeric_data(data):
    """
    Предобработка данных: отбор только числовых признаков, если это DataFrame.
    Если вход — массив NumPy, возвращается как есть.

    Параметры:
    ----------
    data : DataFrame или ndarray
        Входные данные для обработки.

    Возвращает:
    -----------
    ndarray
        Числовые данные.
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.DataFrame):
        return data.select_dtypes(include=[np.number])
    else:
        raise TypeError("Ожидался pandas.DataFrame или numpy.ndarray, получен тип: {}".format(type(data)))



from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_hierarchical_model(data, n_clusters_range, output_dir="output"):
    data_numeric = preprocess_numeric_data(data)
    results = []

    for n_clusters in n_clusters_range:
        model = AgglomerativeClustering(n_clusters=n_clusters)

        try:
            labels = model.fit_predict(data_numeric)
            n_actual_clusters = len(set(labels))

            result = {
                "Model": "Hierarchical",
                "n_clusters": n_clusters,
                "num_clusters": n_actual_clusters,
            }

            if n_actual_clusters > 1:
                result["Silhouette"] = silhouette_score(data_numeric, labels)
                result["Calinski-Harabasz Score"] = calinski_harabasz_score(data_numeric, labels)
                result["Davies-Bouldin Score"] = davies_bouldin_score(data_numeric, labels)

            results.append(result)

        except ValueError as e:
            results.append({
                "Model": "Hierarchical",
                "n_clusters": n_clusters,
                "num_clusters": None,
                "error": str(e),
            })

    df_results = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(f"{output_dir}/hierarchical_results.csv", index=False)

    return df_results



def plot_dendrogram(data, method='ward'):
    """
    Функция для построения дендрограммы для иерархической кластеризации.

    Параметры:
    ----------
    data : DataFrame
        Данные, содержащие числовые значения для кластеризации.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    # Подготовка данных (оставляем только числовые столбцы)
    data_numeric = data.select_dtypes(include=[np.number])

    # Заполняем пропущенные значения средним значением
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    data_numeric = imputer.fit_transform(data_numeric)

    # Проводим иерархическую кластеризацию (linkage)
    Z = linkage(data_numeric, method='ward')  # Метод 'ward' для минимизации вариации в кластерах

    # Построение дендрограммы
    plt.figure(figsize=(8, 6))
    dendrogram(Z)
    plt.title('Дендрограмма для иерархической кластеризации')
    plt.xlabel('Индексы объектов')
    plt.ylabel('Расстояние')
    plt.savefig('output/dendrogram.png')
    plt.close()

