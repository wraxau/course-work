import pandas as pd
import os
import logging
logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output')

def ensure_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info("Папка 'output' создана или уже существует.")

def clean_data(file_name, output_name, drop_columns=None):
    """Общая функция для очистки данных `movies.csv`, `tags.csv` и `ratings.csv`"""
    ensure_output_directory()
    file_path = os.path.join(DATA_DIR, file_name)

    if not os.path.exists(file_path):
        logging.error(f"Ошибка: Файл {file_path} не найден!")
        return

    logging.info(f"Загружаем {file_name}...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        logging.error(f"Ошибка при загрузке файла {file_name}: {e}")
        return

    # Удаление ненужных столбцов, но оставляем `title`
    if drop_columns:
        drop_columns = [col for col in drop_columns if col != "title"]
        df = df.drop(columns=drop_columns, errors="ignore")

    df.drop_duplicates(inplace=True)

    if "movieId" in df.columns:
        df["movieId"] = df["movieId"].astype("uint32")
    if "userId" in df.columns:
        df["userId"] = df["userId"].astype("uint32")

    logging.info(f"Размер после очистки {file_name}: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    output_path = os.path.join(OUTPUT_DIR, output_name)
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Очищенные данные сохранены в {output_path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении файла {output_name}: {e}")

def clean_movies():
    logging.info("Очистка данных `movies.csv`...")
    clean_data("movies.csv", "cleaned_movies.csv", drop_columns=[])


def clean_tags():
    logging.info("Очистка данных `tags.csv`...")
    file_path = os.path.join(DATA_DIR, "tags.csv")

    if not os.path.exists(file_path):
        logging.error(f"Файл {file_path} не найден!")
        return

    try:
        df = pd.read_csv(file_path)

        if "tag" not in df.columns:
            logging.warning("Столбец 'tag' отсутствует в файле tags.csv!")
            return

        df.drop_duplicates(inplace=True)
        df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_tags.csv"), index=False)
        logging.info("Данные из `tags.csv` очищены и сохранены.")
    except Exception as e:
        logging.error(f"Ошибка при очистке данных `tags.csv`: {e}")


def clean_ratings(input_path=None, output_path=None):
    """Очистка данных рейтингов"""
    ensure_output_directory()

    if input_path is None:
        input_path = os.path.join(DATA_DIR, "ratings.csv")
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "cleaned_ratings.csv")

    if not os.path.exists(input_path):
        logging.error(f"Файл {input_path} не найден.")
        return

    try:
        df = pd.read_csv(input_path)
        df.drop_duplicates(inplace=True)
        df["userId"] = df["userId"].astype("uint32")
        df["movieId"] = df["movieId"].astype("uint32")
        df["rating"] = df["rating"].astype("float32")
        df["timestamp"] = df["timestamp"].astype("int32")
        df.to_csv(output_path, index=False)
        logging.info(f"Очищенные рейтинги сохранены в {output_path}")
    except Exception as e:
        logging.error(f"Ошибка при очистке рейтингов: {e}")

def standardize_ratings(input_path=None, output_path=None):
    """Стандартизация рейтингов (z-score)"""
    if input_path is None:
        input_path = os.path.join(OUTPUT_DIR, "cleaned_ratings.csv")
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "standardized_ratings.csv")

    if not os.path.exists(input_path):
        logging.error(f"Файл {input_path} не найден.")
        return

    try:
        df = pd.read_csv(input_path)

        if "rating" not in df.columns or df.empty:
            logging.warning("Файл пуст или не содержит столбец 'rating'")
            return

        mean = df["rating"].mean()
        std = df["rating"].std()
        df["standardized_rating"] = (df["rating"] - mean) / std

        df.to_csv(output_path, index=False)
        logging.info(f"Стандартизованные рейтинги сохранены в {output_path}")

    except Exception as e:
        logging.error(f"Ошибка при стандартизации рейтингов: {e}")

def main():
    logging.info("Очистка и стандартизация данных...")

    # Очистка данных
    clean_movies()  # очистка данных о фильмах
    clean_tags()    # очистка данных о тегах
    clean_ratings() # очистка данных о рейтингах
    standardize_ratings() # стандартизация рейтингов

    logging.info("Очистка и стандартизация завершены!")

if __name__ == "__main__":
    main()
