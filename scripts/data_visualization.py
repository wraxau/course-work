import ast
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Используем бэкенд 'Agg' для работы без GUI
matplotlib.use('Agg')

# Директория для сохранения графиков
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Настройки для корректного отображения графиков
plt.rcParams['font.sans-serif'] = ['Arial']  # Убедимся, что поддерживается кириллица
plt.rcParams['axes.unicode_minus'] = False


def save_plot(filename):
    """Функция для стандартизированного сохранения графиков"""
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"График '{filename}' сохранён.")


def plot_rating_distribution(df, bins=20, color='blue'):
    """Распределение стандартизированных рейтингов"""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['rating'], bins=bins, kde=True, color=color, stat='density')
    plt.title("Распределение стандартизированных рейтингов пользователей")
    plt.xlabel('Стандартизированный рейтинг')
    plt.ylabel('Плотность вероятности')
    save_plot('rating_distribution_standardized.png')


def plot_user_ratings_distribution(df):
    """Распределение количества оценок на пользователя"""
    user_ratings_count = df.groupby('userId').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_ratings_count, bins=20, kde=True, color='green')
    plt.title("Распределение количества оценок на пользователя")
    plt.xlabel('Количество оценок')
    plt.ylabel('Частота пользователей')
    save_plot('user_ratings_distribution.png')


def plot_top_movies_by_avg_rating(df, movies_df, top_n=10):
    """ТОП-10 фильмов по среднему рейтингу"""
    avg_ratings = df.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings = avg_ratings.merge(movies_df, on='movieId')
    top_movies = avg_ratings.nlargest(top_n, 'rating')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='rating', y='title', data=top_movies, hue='title', palette='magma', legend=False)
    plt.xlabel("Средний рейтинг")
    plt.ylabel("Фильм")
    plt.title("ТОП-10 фильмов по среднему рейтингу")
    save_plot('top_movies_by_avg_rating.png')


def plot_genre_ratings(df):
    """Средний рейтинг по жанрам"""
    df = df.copy()  # Избегаем предупреждений Pandas
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)

    df_exploded = df.explode('genres')
    avg_ratings = df_exploded.groupby('genres')['rating'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='genres', y='rating', data=avg_ratings, palette='coolwarm')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Жанр')
    plt.ylabel('Средний рейтинг')
    plt.title("Средний рейтинг по жанрам")
    save_plot('avg_rating_by_genre.png')



def plot_cluster_distribution(df):
    """Распределение фильмов по кластерам"""
    cluster_counts = df['cluster'].value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='Set2')
    plt.xlabel("Кластер")
    plt.ylabel("Количество фильмов")
    plt.title("Распределение фильмов по кластерам")
    save_plot('cluster_distribution.png')


def plot_ratings_over_time(df):
    """Динамика количества оценок по годам"""
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df['year'] = df['date'].dt.year
    ratings_by_year = df.groupby('year').size()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=ratings_by_year.index, y=ratings_by_year.values, marker='o', color='purple')
    plt.title('Количество оценок по годам')
    plt.xlabel('Год')
    plt.ylabel('Количество оценок')
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot('ratings_over_time.png')


def plot_correlation_matrix(df, filename="correlation_matrix.png"):
    """
    Строит и сохраняет корреляционную матрицу числовых переменных в DataFrame.

    :param df: DataFrame, содержащий числовые данные.
    :param filename: Название файла для сохранения графика.
    """
    # Выбираем только числовые столбцы
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.empty:
        print("Ошибка: в данных нет числовых столбцов для построения корреляционной матрицы.")
        return

    corr_matrix = numeric_df.corr()

    # Строим график
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Корреляционная матрица")

    # Сохраняем график
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"График корреляционной матрицы сохранён в {save_path}")