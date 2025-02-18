# всем прив это типо инструкция, если чето непонятно пишите, буду стараться объянить че к чему
#сори графики могут дублироваться или там мб какие-то не оч понятные я это почищу, мне это скорее в отчете надо будет, но позже
# файлы standartized_rating.csv и processed_tags.csv - весят очень много, поэтому я отправила их в телеге, если что они должны лежать в папке output в данном проекте

Основные этапы работы 
1- Очистка и загрузка данных
Файлы:
cleaned_movies.csv → очищенный список фильмов
cleaned_tags.csv → отфильтрованные теги
cleaned_ratings.csv → очищенные и стандартизированные оценки пользователей
Файлы читаются с помощью pandas.read_csv().
dtype_dict = {"userId": "uint32", "movieId": "uint32", "rating": "float32", "timestamp": "int32"}

tags_df = pd.read_csv("output/cleaned_tags.csv", usecols=["movieId", "tag"])
movies_df = pd.read_csv("output/cleaned_movies.csv", usecols=["movieId", "title", "genres"])
ratings_df = pd.read_csv("output/cleaned_ratings.csv", dtype=dtype_dict)

была применена оптимизация:
uint32 (беззнаковое 32-битное число) уменьшает использование памяти по сравнению с int64.
float32 занимает в 2 раза меньше памяти, чем float64.
Оптимизация полезна, если датасет очень большой (миллионы записей).
Какие проверки делаются?

dropna() → удаляет NaN (пустые значения).
drop_duplicates() → убирает дублирующиеся строки.
astype(str).str.split('|') → разбиение тегов по |, так как один фильм может иметь несколько тегов.
2️ -Стандартизация оценок
Алгоритм KMeans чувствителен к различным масштабам данных. Если рейтинги в диапазоне [1, 5], а количество оценок в [10,000, 500,000], то большие значения будут доминировать при расчёте кластеров.

Поэтому нормализуем данные:

def standardize_data(df, column):
    df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df
​
 
Альтернативы:

MinMax Scaling → масштабирует данные в диапазон [0,1], но чувствительно к выбросам.
Robust Scaling → менее чувствителен к выбросам, но теряет относительные масштабы данных.
3- Создание признаков для кластеризации
Чтобы алгоритм мог определить, какие фильмы похожи, каждому фильму добавляются характеристики (фичи).
def create_movie_features(movies_df, ratings_df, tags_df):
    # One-Hot Encoding жанров
    genres_encoded = movies_df['genres'].str.get_dummies('|')

    # Средний рейтинг фильма
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean().rename('avg_rating')

    # Количество оценок фильма
    rating_counts = ratings_df.groupby('movieId')['rating'].count().rename('rating_count')

    # Объединяем фичи
    features = movies_df.set_index('movieId').join([genres_encoded, avg_ratings, rating_counts])

    # Заполняем NaN нулями (например, если у фильма нет рейтинга)
    return features.fillna(0)

Жанры → преобразуются в One-Hot Encoding (get_dummies()).
Средний рейтинг → groupby().mean().
Количество оценок → groupby().count().
Почему важен One-Hot Encoding?
KMeans работает с числовыми данными, поэтому текстовые значения (Action|Sci-Fi) нужно преобразовать в бинарные столбцы:

nginx
Копировать
Редактировать
Action | Sci-Fi | Comedy
   1   |   1    |   0
Альтернативы:

TF-IDF для тегов, если бы использовали текстовые описания.
Word2Vec/Embeddings для более сложного представления жанров.
4️- Кластеризация фильмов
Используется алгоритм KMeans, но в варианте MiniBatchKMeans.
from sklearn.cluster import MiniBatchKMeans

def train_kmeans(movies_df, n_clusters=10):
    features = movies_df.drop(columns=['title', 'genres'])  # Убираем нечисловые признаки
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=42)
    movies_df['cluster'] = kmeans.fit_predict(features)
    return kmeans, movies_df
    
MiniBatchKMeans обновляет кластеры порциями по 1000 записей (юзает меньше памяти).
Работает намного быстрее на больших датасетах.
Качество решения примерно такое же, как у обычного KMeans.
Альтернативы:

DBSCAN (для кластеров произвольной формы, но требует подбора eps).
Agglomerative Clustering (иерархическая кластеризация, но медленный).
5️- Визуализация результатов
После кластеризации строятся графики.
def plot_cluster_distribution(movies_df):
    import matplotlib.pyplot as plt
    movies_df['cluster'].value_counts().sort_index().plot(kind='bar')
    plt.xlabel("Кластеры")
    plt.ylabel("Число фильмов")
    plt.title("Распределение фильмов по кластерам")
    plt.show()
Bar Chart → показывает, сколько фильмов в каждом кластере.
Scatter Plot (если много фичей) → используется PCA или TSNE для уменьшения размерности.
6️-  Анализ данных по кластерам
После разбиения анализируем:

Какие жанры преобладают в каждом кластере?
Какой средний рейтинг фильмов в кластере?
def analyze_genres_by_cluster(movies_df):
    return movies_df.groupby('cluster')['genres'].apply(lambda x: x.value_counts().head(5))
.groupby('cluster') → группируем по кластеру.
.apply(lambda x: x.value_counts().head(5)) → ищем топ-5 жанров в каждом кластере.
7️ - Рекомендации фильмов пользователям
Если у пользователя больше всего оценок в одном кластере, он, скорее всего, любит такие фильмы.
def get_favorite_cluster(user_id, ratings_df, movies_df):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    user_movies = user_ratings.merge(movies_df, on="movieId")
    return user_movies['cluster'].value_counts().idxmax()
Берём оценки пользователя.
Определяем, в каких кластерах находятся эти фильмы.
Выбираем самый частый кластер → это любимый тип фильмов.
8️- Сохранение результатов
movies_df.to_csv('output/clusters_movies.csv', index=False)
Также сохраняется модель KMeans:
import joblib
joblib.dump(kmeans, 'output/kmeans_model.pkl')

Что уже реализовано?
 Чистка и предобработка данных
 Анализ и стандартизация рейтингов
 Обучение модели кластеризации (MiniBatchKMeans)
 Анализ кластеров (жанры, рейтинги, популярность)
 Выделение любимого кластера пользователя
 Генерация базовых рекомендаций
 Визуализация результатов


что рекомендуент gpt (видимо для Даши задачки исходя из нашего разбиения ролей)
Системный администратор
Развернуть FastAPI для REST API
Подключить PostgreSQL / ClickHouse для хранения данных
Настроить автоматическое обновление кластеров (например, раз в неделю)
Оптимизировать хранение модели (сжатие, joblib vs ONNX)

Для Маши еще раз, более целенаправленно для того, что нужно делать тебе как мы с gpt это поняли))
Инструкция для бэкенд-разработчика: Разворачивание рекомендательной системы
Этот документ предназначен для бэкенд-разработчика, который будет развивать рекомендательную систему фильмов. Здесь подробно описано, где брать данные, как их использовать и что нужно реализовать для развертывания API.

1️Данные: Где что лежит и как использовать
Вся обработка данных уже выполнена, поэтому брать информацию нужно из output/:

Файл	Что содержит	Как использовать
cleaned_movies.csv	ID фильма, название, жанры	Подгружать мета-информацию о фильмах
cleaned_tags.csv	ID фильма, теги пользователей	Анализировать тематику фильмов
cleaned_ratings.csv	ID пользователя, ID фильма, рейтинг	Для рекомендаций на основе оценок
clusters_movies.csv	ID фильма, название, кластер	Определять, в каком кластере фильм
kmeans_model.pkl	Обученная модель KMeans	Использовать для кластеризации новых фильмов
2️ - Развертывание API (FastAPI)
Рекомендации должны быть доступны через API.
Используем FastAPI – он быстрый, асинхронный и легко разворачивается.

 Установка зависимостей
pip install fastapi uvicorn joblib pandas numpy
 Базовая структура API
from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Загружаем данные
movies_df = pd.read_csv("output/clusters_movies.csv")
kmeans = joblib.load("output/kmeans_model.pkl")

@app.get("/")
def home():
    return {"message": "Recommender System API is running"}

Теперь API запустится командой:
uvicorn main:app --host 0.0.0.0 --port 8000
3️ -  Реализация основных эндпоинтов

1. Получить информацию о фильме
Этот эндпоинт возвращает информацию о фильме по ID.
@app.get("/movie/{movie_id}")
def get_movie(movie_id: int):
    movie = movies_df[movies_df["movieId"] == movie_id]
    if movie.empty:
        return {"error": "Movie not found"}
    return movie.to_dict(orient="records")[0]
Пример запроса:
GET http://localhost:8000/movie/1

Пример ответа:
{
  "movieId": 1,
  "title": "Toy Story (1995)",
  "genres": "Animation|Children|Comedy",
  "cluster": 3
}
2-  Получить рекомендации по фильму
@app.get("/recommend/movie/{movie_id}")
def recommend_movies(movie_id: int, n: int = 5):
    movie = movies_df[movies_df["movieId"] == movie_id]
    if movie.empty:
        return {"error": "Movie not found"}
    
    cluster = movie["cluster"].values[0]
    recommendations = movies_df[movies_df["cluster"] == cluster].sample(n)
    return recommendations[["movieId", "title"]].to_dict(orient="records")
Пример запроса:
GET http://localhost:8000/recommend/movie/1?n=5
 Пример ответа:
[
  {"movieId": 45, "title": "Casablanca (1942)"},
  {"movieId": 67, "title": "Citizen Kane (1941)"},
  {"movieId": 128, "title": "Schindler's List (1993)"},
  {"movieId": 255, "title": "Pulp Fiction (1994)"},
  {"movieId": 302, "title": "Forrest Gump (1994)"}
]
3. Получить рекомендации для пользователя
Этот эндпоинт анализирует, какие кластеры пользователь смотрел чаще всего, и рекомендует фильмы оттуда.
ratings_df = pd.read_csv("output/cleaned_ratings.csv")

@app.get("/recommend/user/{user_id}")
def recommend_for_user(user_id: int, n: int = 5):
    user_ratings = ratings_df[ratings_df["userId"] == user_id]
    if user_ratings.empty:
        return {"error": "User not found"}
    
    top_cluster = user_ratings.merge(movies_df, on="movieId")["cluster"].value_counts().idxmax()
    recommendations = movies_df[movies_df["cluster"] == top_cluster].sample(n)
    
    return recommendations[["movieId", "title"]].to_dict(orient="records")
 Пример запроса:
GET http://localhost:8000/recommend/user/5?n=5

 Пример ответа:
[
  {"movieId": 90, "title": "The Matrix (1999)"},
  {"movieId": 128, "title": "Inception (2010)"},
  {"movieId": 215, "title": "Interstellar (2014)"},
  {"movieId": 312, "title": "The Dark Knight (2008)"},
  {"movieId": 499, "title": "Fight Club (1999)"}
]


 Итоги
Где брать данные?

clusters_movies.csv → для рекомендаций
kmeans_model.pkl → для обновления кластеров
cleaned_ratings.csv → для анализа пользователей
Что реализовано в API?

/movie/{movie_id} → Получение информации о фильме
/recommend/movie/{movie_id} → Рекомендации по фильму
/recommend/user/{user_id} → Рекомендации для пользователя
Как развернуть?

FastAPI + Uvicorn
Docker-контейнер
Готово к интеграции в сервис!

