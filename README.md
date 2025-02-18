ЕЩЕ НЕ ГОТОВ

## Инструкция для бэкенд-разработчика: Разворачивание рекомендательной системы (Сокращенная версия)

Фокус на трех главных задачах:  
1. Определение предпочтений пользователя (в каком кластере он смотрит фильмы).  
2. Поиск похожих фильмов (из его кластера).  
3. Генерация рекомендаций (на основе его рейтингов и тегов).  

---

## Какие файлы использует бэкенд  
!! файл standartized_rating.csv прислала в тг !!
| Файл | Для чего нужен |
|--------------|--------------------------------------------------|
| `cleaned_movies.csv` | Список фильмов (ID, название, жанры). |
| `cleaned_ratings.csv` | Оценки пользователей (ID фильма, ID пользователя, оценка). |
| `standardized_ratings.csv` | Оценки в стандартизированном формате. | 
| `clusters_movies.csv` | Фильмы с кластерами (какой фильм в каком кластере). |
| `sorted_tags_by_cluster.csv` | Популярные теги в каждом кластере. |
| `kmeans_model.pkl` | Модель кластеризации (используется для предсказаний). |

---

## 1. Определение предпочтений пользователя  
### Что делает бэкенд  
1. Получает `user_id`.  
2. Находит фильмы, которые пользователь оценивал (`cleaned_ratings.csv`).  
3. Определяет, в каких кластерах находятся эти фильмы (`clusters_movies.csv`).  
4. Выбирает самый популярный кластер (в котором больше всего оценок).  
5. Возвращает любимый кластер пользователя.  

### Какие файлы нужны  
- `cleaned_ratings.csv` – оценки пользователя.  
- `clusters_movies.csv` – в каком кластере находится каждый фильм.  

### Какой код использовать  
```python
from scripts.clustering import get_favorite_cluster

user_id = 5
favorite_cluster = get_favorite_cluster(user_id, ratings_df, movies_df)
print(f"Любимый кластер пользователя {user_id}: {favorite_cluster}")
```

### Пример API-запроса  
```http
GET /favorite-cluster?user_id=5
```
### Ответ  
```json
{
  "favorite_cluster": 3
}
```

---

## 2. Поиск похожих фильмов  
### Что делает бэкенд  
1. Получает `movie_id`.  
2. Определяет, в каком кластере этот фильм (`clusters_movies.csv`).  
3. Выбирает другие фильмы из этого же кластера.  
4. Сортирует по популярности (количеству оценок, среднему рейтингу).  
5. Возвращает список похожих фильмов.  

### Какие файлы нужны  
- `clusters_movies.csv` – группировка фильмов по кластерам.  
- `cleaned_ratings.csv` – рейтинги для сортировки фильмов по популярности.  

### Какой код использовать  
```python
from scripts.clustering import recommend_movies

movie_id = 1
recommendations = recommend_movies(movie_id, movies_df, ratings_df, n=5)
print("Рекомендованные фильмы:", recommendations)
```

### Пример API-запроса  
```http
GET /recommend?movie_id=123
```
### Ответ  
```json
{
  "recommendations": ["The Matrix", "Inception", "Blade Runner 2049"]
}
```

---

## 3. Рекомендации фильмов на основе тегов  
### Что делает бэкенд  
1. Определяет любимый кластер пользователя.  
2. Загружает `sorted_tags_by_cluster.csv` (популярные теги в этом кластере).  
3. Ищет фильмы с похожими тегами.  
4. Фильтрует по популярности и рейтингу.  
5. Возвращает список рекомендаций.  

### Какие файлы нужны  
- `sorted_tags_by_cluster.csv` – теги по кластерам.  
- `cleaned_movies.csv` – список фильмов.  
- `cleaned_ratings.csv` – популярность фильмов.  

### Какой код использовать  
```python
from scripts.cluster_analysis import filter_top_movies

recommended_movies = filter_top_movies(movies_df, ratings_df)
print("Фильмы на основе тегов:", recommended_movies)
```

### Пример API-запроса  
```http
GET /recommend-by-tags?user_id=5
```
### Ответ  
```json
{
  "recommendations": ["Interstellar", "Gravity", "2001: A Space Odyssey"]
}
```

---

```

### Пример API-запроса  
```http
POST /train-kmeans
```
### Ответ  
```json
{
  "status": "training started"
}
```

---
