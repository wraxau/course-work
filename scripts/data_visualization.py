import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import ast

matplotlib.use('Agg')
sns.set_style("whitegrid")

# Директория для сохранения графиков
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


def plot_cluster_rating_distribution(movies_df):
    print("\nСтроим график распределения рейтингов по кластерам...")

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="cluster", y="rating", data=movies_df, palette="coolwarm")
    plt.xlabel("Кластер")
    plt.ylabel("Рейтинг")
    plt.title("Распределение рейтингов фильмов по кластерам")
    plt.savefig("output/cluster_rating_distribution.png")
    print("График 'cluster_rating_distribution.png' сохранён.")
    plt.close()


def save_plot(filename):
    #Функция для стандартизированного сохранения графиков
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"График '{filename}' сохранён.")


def plot_rating_distribution(df, bins=20):
    plt.figure(figsize=(10, 6))

    # Проверим реальный диапазон значений
    min_rating, max_rating = df['rating'].min(), df['rating'].max()
    print(f"Диапазон рейтингов: {min_rating} - {max_rating}")

    # Построим распределение
    sns.histplot(df['rating'], bins=bins, kde=False, color='blue', stat='density', alpha=0.7)

    plt.title("Распределение рейтингов пользователей")
    plt.xlabel('Рейтинг')
    plt.ylabel('Плотность вероятности')

    plt.tight_layout()
    plt.savefig('output/rating_distribution.png')


def plot_user_ratings_distribution(df):
    # Считаем количество оценок на каждого пользователя
    ratings_per_user = df.groupby('userId')['movieId'].count()

    # Ограничим диапазон значений, чтобы убрать редкие выбросы
    upper_limit = np.percentile(ratings_per_user, 99)  # Обрежем 1% самых активных пользователей

    plt.figure(figsize=(12, 6))
    sns.histplot(ratings_per_user[ratings_per_user <= upper_limit], bins=50, kde=False, color='green')

    plt.title("Распределение количества оценок на пользователя")
    plt.xlabel('Количество оценок')
    plt.ylabel('Частота пользователей')

    plt.tight_layout()
    plt.savefig('output/user_ratings_distribution.png')  # Сохраняем в output


def plot_top_movies_by_avg_rating(df, movies_df, top_n=10):
    # Вычисляем средний рейтинг для каждого фильма
    avg_ratings = df.groupby('movieId')['rating'].mean().nlargest(top_n).reset_index()

    # Добавляем названия фильмов
    top_movies = avg_ratings.merge(movies_df, on='movieId')

    plt.figure(figsize=(12, 6))
    sns.barplot(x='rating', y='title', data=top_movies, palette="Set2")

    plt.xlabel("Средний рейтинг")
    plt.ylabel("Фильм")
    plt.title("ТОП-10 фильмов по среднему рейтингу")

    plt.tight_layout()
    plt.savefig('output/top_movies_by_avg_rating.png')  # Сохраняем в папку output


def plot_genre_ratings(df):
    # Копируем DataFrame, чтобы избежать изменений в оригинале
    df = df.copy()

    # Обрабатываем жанры: превращаем строки в списки
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)

    # Удаляем строки, где genres = NaN
    df = df.dropna(subset=['genres'])

    # Разворачиваем список жанров в отдельные строки
    df_exploded = df.explode('genres')

    # Считаем средний рейтинг для каждого жанра
    avg_ratings = df_exploded.groupby('genres')['rating'].mean().reset_index()

    # Фильтруем топ-20 самых частых жанров
    top_genres = df_exploded['genres'].value_counts().index[:20]
    avg_ratings = avg_ratings[avg_ratings['genres'].isin(top_genres)]

    # Сортируем жанры по среднему рейтингу
    avg_ratings = avg_ratings.sort_values(by='rating', ascending=False)

    # Визуализация: горизонтальный barplot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=avg_ratings, y='genres', x='rating', hue='genres', dodge=False, palette='coolwarm', legend=False)

    plt.xlabel('Средний рейтинг')
    plt.ylabel('Жанр')
    plt.title("Средний рейтинг по топ-20 жанрам")
    plt.tight_layout()

    output_path = 'output/avg_rating_by_genre.png'
    plt.savefig(output_path)
    plt.close()
    print(f"График '{output_path}' сохранён.")


def plot_cluster_distribution(df):
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=df['cluster'].value_counts().index,
        y=df['cluster'].value_counts().values,
        hue=df['cluster'].value_counts().index,  # Добавляем hue
        dodge=False,
        palette='Set2',
        legend=False
    )

    plt.xlabel("Кластер")
    plt.ylabel("Количество фильмов")
    plt.title("Распределение фильмов по кластерам")
    plt.savefig("output/cluster_distribution.png")
    plt.close()
    print(f" График 'распределение фильмов по кластерам' сохранён.")


def plot_ratings_over_time(df):
    #График динамика количества оценок по годам
    df['date'] = pd.to_datetime(df['timestamp'], unit='s', cache=True)
    df['year'] = df['date'].dt.year

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df['year'].value_counts().sort_index().index,
                 y=df['year'].value_counts().sort_index().values,
                 marker='o', color='purple')

    plt.title('Количество оценок по годам')
    plt.xlabel('Год')
    plt.ylabel('Количество оценок')
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot('ratings_over_time.png')
    plt.savefig("output/ratings_over_time.png")


def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        print("Ошибка: в данных нет числовых столбцов для построения корреляционной матрицы.")
        return

    plt.figure(figsize=(10, 7))
    sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Корреляционная матрица")

    plt.tight_layout()
    plt.savefig("output/correlation_matrix.png")
