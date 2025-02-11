import pandas as pd
from collections import Counter

# Оптимизированное чтение CSV
movies_df = pd.read_csv(
    './output/movies_with_clusters.csv',
    dtype={'cluster': 'int16'}  # Используем int16, так как число кластеров обычно небольшое
)

# Проверяем, есть ли столбец 'tag' в данных
if 'tag' not in movies_df.columns:
    raise KeyError("Ошибка: В данных нет столбца 'tag'.")

# Преобразуем теги в списки и развернем их
movies_df['tag'] = movies_df['tag'].fillna('').astype(str).str.split('|')

# "Взрываем" (explode) DataFrame по тегам
exploded_tags = movies_df.explode('tag')

# Удаляем пустые теги, если они есть
exploded_tags = exploded_tags[exploded_tags['tag'] != '']

# Оптимизированная группировка: считаем частоту тегов в каждом кластере
tag_counts_per_cluster = exploded_tags.groupby('cluster')['tag'].apply(lambda x: Counter(x)).reset_index()

# Преобразуем данные в удобный формат
tag_summary = []
for _, row in tag_counts_per_cluster.iterrows():
    cluster = row['cluster']
    for tag, count in row['tag'].items():
        tag_summary.append({'cluster': cluster, 'tag': tag, 'count': count})

# Создаем DataFrame и сортируем
tag_summary_df = pd.DataFrame(tag_summary).sort_values(by=['cluster', 'count'], ascending=[True, False])

# Сохраняем результаты (с оптимизацией записи)
output_path = './output/sorted_tags_by_cluster.csv'
tag_summary_df.to_csv(output_path, index=False, float_format='%.0f')

print(f"Сортированные теги по кластерам сохранены в '{output_path}'")
