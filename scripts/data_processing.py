import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize_data(df, column_name, verbose=True):
    """
    Нормализация столбца в DataFrame.
    :param df: DataFrame с данными.
    :param column_name: Название столбца для стандартизации.
    :param verbose: Логирование процесса.
    :return: DataFrame с нормализованными данными.
    """
    if verbose:
        print(f"Проверяем, что столбец '{column_name}' существует в DataFrame.")

    # Проверяем существование столбца
    if column_name not in df.columns:
        raise ValueError(f"Ошибка: столбец '{column_name}' отсутствует в DataFrame!")

    # Используем NumPy для более быстрой стандартизации
    mean, std = df[column_name].mean(), df[column_name].std()
    df[column_name] = (df[column_name] - mean) / std

    if verbose:
        print(f"Стандартизация '{column_name}' выполнена!")

    return df


def merge_data(tags_df, movies_df, verbose=True):
    """
    Объединяет данные по movieId с оптимизированной памятью.
    :param tags_df: DataFrame с тегами.
    :param movies_df: DataFrame с фильмами.
    :param verbose: Логирование процесса.
    :return: Объединенный DataFrame.
    """
    if verbose:
        print("Объединяем данные...")

    # Оптимизация типов данных
    tags_df["movieId"] = tags_df["movieId"].astype("uint32")
    movies_df["movieId"] = movies_df["movieId"].astype("uint32")

    # Удаление дубликатов перед объединением (ускоряет join)
    tags_df.drop_duplicates(subset=["movieId"], inplace=True)
    movies_df.drop_duplicates(subset=["movieId"], inplace=True)

    # Используем merge вместо join (быстрее и эффективнее)
    merged_df = tags_df.merge(movies_df, on="movieId", how="left")

    if verbose:
        print(f"Объединенные данные: {merged_df.shape}")

    return merged_df
