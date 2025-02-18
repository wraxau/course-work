import os
import pandas as pd
from collections import Counter

# Проверяем, существует ли файл
file_path = './output/movies_with_clusters.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Ошибка: Файл '{file_path}' не найден.")

# Загружаем данные
movies_df = pd.read_csv(file_path, dtype={'cluster': 'int16'})

# Проверяем наличие столбца 'tag'
if 'tag' not in movies_df.columns:
    raise KeyError("Ошибка: В данных нет столбца 'tag'.")

# Удаляем строки, где 'tag' = NaN
movies_df = movies_df.dropna(subset=['tag'])

# Разделяем теги в списки
movies_df['tag'] = movies_df['tag'].str.split('|')

# "Взрываем" DataFrame по тегам
exploded_tags = movies_df.explode('tag').reset_index(drop=True)

# Удаляем пустые теги
exploded_tags = exploded_tags[exploded_tags['tag'] != '']

# Оптимизированная группировка
tag_counts_per_cluster = exploded_tags.groupby('cluster')['tag'].agg(lambda x: Counter(x.tolist())).reset_index()

# Преобразуем в удобный формат
tag_summary = [{'cluster': row['cluster'], 'tag': tag, 'count': count}
               for _, row in tag_counts_per_cluster.iterrows()
               for tag, count in row['tag'].items()]

# Создаем DataFrame и сортируем
tag_summary_df = pd.DataFrame(tag_summary).sort_values(['cluster', 'count'], ascending=[True, False])

# Сохраняем в сжатом формате
output_path = './output/sorted_tags_by_cluster.csv.gz'
tag_summary_df.to_csv(output_path, index=False, float_format='%.0f', compression='gzip')

print(f"Файл '{output_path}' успешно сохранен в сжатом формате!")
