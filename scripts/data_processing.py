import logging
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import os
from sklearn.preprocessing import OneHotEncoder

# Предобработка категориальных данных
def preprocess_categorical_data(dataframe, categorical_columns):
    missing_columns = [col for col in categorical_columns if col not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Следующие столбцы отсутствуют в DataFrame: {', '.join(missing_columns)}")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categories = encoder.fit_transform(dataframe[categorical_columns])
    encoded_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(categorical_columns))
    dataframe_encoded = dataframe.drop(categorical_columns, axis=1)
    dataframe_encoded = pd.concat([dataframe_encoded, encoded_df], axis=1)
    return dataframe_encoded

# Подготовка данных с тегами
"""def prepare_movies_with_tags():
    try:
        movies_df = pd.read_csv("output/clusters_movies.csv")
        tags_df = pd.read_csv("output/cleaned_tags.csv", usecols=["movieId", "tag"])

        tags_df["tag"] = tags_df["tag"].fillna("").astype(str)
        tags_grouped = tags_df.groupby("movieId")["tag"].apply(lambda x: "|".join(set(x))).reset_index()
        merged_df = movies_df.merge(tags_grouped, on="movieId", how="left")
        merged_df["tag"] = merged_df["tag"].fillna("")
        merged_df.to_csv("output/clusters_movies_with_tags.csv", index=False)
        logging.info("Файл 'clusters_movies_with_tags.csv' успешно создан.")

        return merged_df
    except FileNotFoundError as e:
        logging.error(f"Файл не найден: {e}. Проверьте, что все необходимые файлы существуют.")
    except Exception as e:
        logging.error(f"Неизвестная ошибка при подготовке данных с тегами: {e}")
"""
# Стандартизация данных
def standardize_data(df: pd.DataFrame, column_name: str, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        logging.info(f"Проверяем наличие столбца '{column_name}'.")

    if column_name not in df.columns:
        raise ValueError(f"Столбец '{column_name}' отсутствует в DataFrame!")

    mean, std = df[column_name].mean(), df[column_name].std()
    if std == 0:
        logging.warning(f"Стандартное отклонение столбца '{column_name}' равно 0. Стандартизация невозможна.")
        return df

    df[column_name] = (df[column_name] - mean) / std
    if verbose:
        logging.info(f"Стандартизация '{column_name}' выполнена успешно.")
    return df