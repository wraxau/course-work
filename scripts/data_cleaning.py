import pandas as pd
import os
import ast  # Для безопасного преобразования строк в списки

def ensure_output_directory():
    """Создаёт папку 'output', если её нет."""
    if not os.path.exists('output'):
        os.makedirs('output')


def clean_data(df, drop_columns=None, fillna_values=None):
    """
    Общая функция для очистки данных.
    :param df: DataFrame для очистки.
    :param drop_columns: Список колонок для удаления.
    :param fillna_values: Словарь для заполнения пропусков.
    :return: Очищенный DataFrame.
    """
    # Удалим дубликаты
    df = df.drop_duplicates()

    # Удалим указанные столбцы
    if drop_columns:
        df = df.drop(columns=drop_columns)

    # Заполним пропущенные значения
    if fillna_values:
        for column, value in fillna_values.items():
            df[column] = df[column].fillna(value)

    return df


def load_data(path, filenames):
    """Загружает данные из CSV файлов."""
    return {filename: pd.read_csv(os.path.join(path, filename)) for filename in filenames}


def safe_literal_eval(value):
    """Преобразует строковые представления списков в настоящие списки с обработкой ошибок."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []  # Если ошибка, возвращаем пустой список


def clean_movies():
    """Очистка данных movies.csv."""
    ensure_output_directory()

    movies_path = os.path.join('data', 'movies.csv')

    if not os.path.exists(movies_path):
        print(f"Ошибка: Файл {movies_path} не найден!")
        return

    movies_df = pd.read_csv(movies_path)

    # Проверяем, есть ли столбец genres
    if 'genres' not in movies_df.columns:
        print("Ошибка: столбец 'genres' отсутствует в файле!")
        return

    print(f"Размерность до очистки: {movies_df.shape}")
    print("Первые 5 строк до очистки:")
    print(movies_df.head())

    # Обработка жанров
    if movies_df['genres'].apply(lambda x: isinstance(x, str) and x.startswith("[")).all():
        # Если жанры хранятся как строки списков (например, "['Comedy', 'Drama']")
        movies_df['genres'] = movies_df['genres'].apply(ast.literal_eval)
    else:
        # Если жанры хранятся в строковом формате с разделителем "|"
        movies_df['genres'] = movies_df['genres'].str.split('|')

    # Заполняем пустые значения
    movies_df['genres'] = movies_df['genres'].apply(lambda x: x if isinstance(x, list) else [])

    # Проверка: если после обработки столбец пуст
    if movies_df['genres'].apply(len).sum() == 0:
        print("Ошибка: после обработки столбец 'genres' пуст!")
        print("Посмотрим, что в первых 5 строках 'genres':")
        print(movies_df['genres'].head())  # Выводит первые строки столбца genres
        return

    print(f"Размерность после очистки: {movies_df.shape}")
    print("Первые 5 строк после очистки:")
    print(movies_df.head())

    # Сохраняем очищенные данные
    output_path = os.path.join('output', 'cleaned_movies.csv')
    movies_df.to_csv(output_path, index=False)

    print(f"Очищенные данные сохранены в {output_path}")

def clean_tags():
    """Очистка данных tags.csv."""
    ensure_output_directory()

    # Загрузим данные
    tags_df = pd.read_csv(os.path.join('data', 'tags.csv'))

    # Очистим данные с использованием универсальной функции
    tags_df = clean_data(tags_df)

    # Сохраним очищенные данные
    tags_df.to_csv(os.path.join('output', 'cleaned_tags.csv'), index=False)
    print("Очищенные данные tags.csv сохранены в 'output/cleaned_tags.csv'")


def main():
    clean_movies()
    clean_tags()


if __name__ == "__main__":
    main()
