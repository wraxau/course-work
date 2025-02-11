import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

def ensure_output_directory():
    #Создаёт папку 'output' если её нет
    os.makedirs('output', exist_ok=True)

def clean_data(file_name, output_name, drop_columns=None):
    #Общая функция для очистки данных `movies.csv`, `tags.csv` и `ratings.csv`
    ensure_output_directory()
    file_path = os.path.join('data', file_name)

    if not os.path.exists(file_path):
        print(f"Ошибка: Файл {file_path} не найден!")
        return

    print(f"Загружаем {file_name}...")
    df = pd.read_csv(file_path, low_memory=False)

    # Удаление ненужных столбцов, но оставляем `title`
    if drop_columns:
        drop_columns = [col for col in drop_columns if col != "title"]  # НЕ УДАЛЯЕМ `title`
        df = df.drop(columns=drop_columns, errors="ignore")

    # Очистка данных
    df.drop_duplicates(inplace=True)

    # Оптимизация памяти
    if "movieId" in df.columns:
        df["movieId"] = df["movieId"].astype("uint32")
    if "userId" in df.columns:
        df["userId"] = df["userId"].astype("uint32")

    print(f"Размер после очистки {file_name}: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # Сохранение очищенных данных
    output_path = os.path.join("output", output_name)
    df.to_csv(output_path, index=False)
    print(f"Очищенные данные сохранены в {output_path}")

def clean_movies():
    """Очистка данных `movies.csv`, НЕ удаляем `title`!"""
    print("Очистка данных `movies.csv`...")
    clean_data("movies.csv", "cleaned_movies.csv", drop_columns=[])

def clean_tags():
    """Очистка данных `tags.csv`."""
    print("Очистка данных `tags.csv`...")
    clean_data("tags.csv", "cleaned_tags.csv")
def clean_ratings():
    """Очистка данных `ratings.csv`."""
    print("Очистка данных `ratings.csv`...")
    clean_data("ratings.csv", "cleaned_ratings.csv", drop_columns=[])

def load_data():
    """Загружает и оптимизирует данные, уменьшая потребление памяти."""
    print("Загружаем данные с оптимизированными типами...")

    dtype_dict = {
        "userId": "uint32",
        "movieId": "uint32",
        "rating": "float32",
        "timestamp": "int32",
    }

    tags_df = pd.read_csv("output/cleaned_tags.csv", encoding="utf-8", usecols=["movieId", "tag"], na_filter=False)
    movies_df = pd.read_csv("output/cleaned_movies.csv", encoding="utf-8")

    if "title" not in movies_df.columns:
        print("Ошибка: 'title' отсутствует в movies_df! Перезагружаем из оригинального movies.csv...")
        movies_df = pd.read_csv("data/movies.csv", encoding="utf-8", usecols=["movieId", "title", "genres"])
        movies_df.to_csv("output/cleaned_movies.csv", index=False, encoding="utf-8")

    ratings_df = pd.read_csv("output/cleaned_ratings.csv", encoding="utf-8", dtype=dtype_dict)

    print(f"Размер movies_df: {movies_df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"Размер tags_df: {tags_df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"Размер ratings_df: {ratings_df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    return tags_df, movies_df, ratings_df

def main():
    print("Очистка данных...")

    # Используем ThreadPoolExecutor для многопоточной очистки
    with ThreadPoolExecutor() as executor:
        executor.submit(clean_movies)
        executor.submit(clean_tags)
        executor.submit(clean_ratings)

    print("Очистка завершена!")

if __name__ == "__main__":
    main()
