import os
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
import numpy as np
def preprocess_numeric_data(data):
    """
    Приводит входные данные к числовому формату и удаляет строки с NaN.

    Параметры:
    ----------
    data : pandas.DataFrame или numpy.ndarray

    Возвращает:
    -----------
    pandas.DataFrame
        Чистые числовые данные.
    """
    if isinstance(data, pd.DataFrame):
        data_numeric = data.select_dtypes(include=[np.number]).dropna()
    elif isinstance(data, np.ndarray):
        data_numeric = pd.DataFrame(data)  # Преобразуем для унификации
    else:
        raise TypeError("Ожидался pandas.DataFrame или numpy.ndarray.")

    return data_numeric

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_dbscan_model(data, eps_values, min_samples_values, output_dir="output"):
    data_numeric = preprocess_numeric_data(data)
    results = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
            try:
                labels = dbscan_model.fit_predict(data_numeric)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                result = {
                    "Model": "DBSCAN",
                    "eps": eps,
                    "min_samples": min_samples,
                    "num_clusters": n_clusters,
                }

                # Вычисляем метрики только если кластеров больше 1
                if n_clusters > 1:
                    result["Silhouette"] = silhouette_score(data_numeric, labels)
                    result["Calinski-Harabasz Score"] = calinski_harabasz_score(data_numeric, labels)
                    result["Davies-Bouldin Score"] = davies_bouldin_score(data_numeric, labels)

                results.append(result)

            except ValueError as e:
                results.append({
                    "Model": "DBSCAN",
                    "eps": eps,
                    "min_samples": min_samples,
                    "num_clusters": None,
                    "error": str(e),
                })

    df_results = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(f"{output_dir}/dbscan_results.csv", index=False)

    return df_results
