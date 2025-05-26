import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, MiniBatchKMeans
import os
import numpy as np
def evaluate_clustering_models(data, k, output_dir="output"):
    """
    Оценка моделей кластеризации KMeans и MiniBatchKMeans по метрикам:
    - Silhouette Score
    - Calinski-Harabasz Index
    - Davies-Bouldin Index

    Параметры:
    ----------
    data : array-like
        Эмбеддинги или числовые представления данных.
    k : int
        Количество кластеров для моделирования.
    output_dir : str
        Путь к папке для сохранения результатов.

    Возвращает:
    -----------
    DataFrame
        Таблица с результатами метрик для каждой модели.
    """

    # Обработка данных в зависимости от типа
    if isinstance(data, pd.DataFrame):
        data_numeric = data.select_dtypes(include=[np.number]).dropna()
    elif isinstance(data, np.ndarray):
        data_numeric = pd.DataFrame(data)  # Преобразуем в DataFrame для совместимости
    else:
        raise TypeError("Параметр 'data' должен быть либо pandas.DataFrame, либо numpy.ndarray.")

    # Модели
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    minibatch_model = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100)

    # Кластеризация KMeans
    kmeans_labels = kmeans_model.fit_predict(data_numeric)
    kmeans_silhouette = silhouette_score(data_numeric, kmeans_labels)
    kmeans_calinski = calinski_harabasz_score(data_numeric, kmeans_labels)
    kmeans_davies = davies_bouldin_score(data_numeric, kmeans_labels)

    # Кластеризация MiniBatchKMeans
    minibatch_labels = minibatch_model.fit_predict(data_numeric)
    minibatch_silhouette = silhouette_score(data_numeric, minibatch_labels)
    minibatch_calinski = calinski_harabasz_score(data_numeric, minibatch_labels)
    minibatch_davies = davies_bouldin_score(data_numeric, minibatch_labels)

    # Сохранение результатов
    results = {
        "Model": ["KMeans", "MiniBatchKMeans"],
        "Silhouette": [kmeans_silhouette, minibatch_silhouette],
        "Calinski-Harabasz Score": [kmeans_calinski, minibatch_calinski],
        "Davies-Bouldin Score": [kmeans_davies, minibatch_davies],
    }

    df_results = pd.DataFrame(results)

    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(f"{output_dir}/clustering_model_comparison.csv", index=False)

    return df_results
