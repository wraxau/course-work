import os
import pandas as pd

def standardize_metric_names(df):
    return df.rename(columns={
        "Silhouette": "Silhouette",
        "CH Score": "Calinski-Harabasz Score",
        "DB Index": "Davies-Bouldin Score"
    })

def compare_models(*results):
    # Приводим все DataFrame к единому формату
    standardized = [standardize_metric_names(df.copy()) for df in results]

    # Объединяем все результаты
    all_results = pd.concat(standardized, ignore_index=True)

    # Проверяем наличие нужных колонок
    required_columns = ['Silhouette', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
    missing = [col for col in required_columns if col not in all_results.columns]
    if missing:
        raise ValueError(f"Отсутствуют нужные метрики: {missing}. Доступные колонки: {list(all_results.columns)}")

    # Рассчитываем рейтинг (Silhouette и CH — чем выше, тем лучше; DB — чем ниже)
    all_results['Rank'] = (
        all_results[['Silhouette', 'Calinski-Harabasz Score']].rank(ascending=False) +
        all_results[['Davies-Bouldin Score']].rank(ascending=True)
    ).mean(axis=1)

    all_results.sort_values('Rank', inplace=True)

    return all_results
