import pandas as pd
from collections import Counter

# Загружаем данные, например, результаты кластеризации
movies_df = pd.read_csv('./output/movies_with_clusters.csv')

# Группировка тегов по кластерам и подсчет их частоты
tag_counts_per_cluster = movies_df.groupby('cluster')['tag'].apply(lambda tags: Counter(tag for tag_list in tags.str.split('|') for tag in tag_list))

# Сортировка тегов в каждом кластере по убыванию их количества
sorted_tags_by_cluster = {}
for cluster, tag_counts in tag_counts_per_cluster.items():
    sorted_tags_by_cluster[cluster] = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

# Преобразуем отсортированные данные в DataFrame для удобства
tag_summary = []
for cluster, tags in sorted_tags_by_cluster.items():
    for tag, count in tags:
        tag_summary.append({'cluster': cluster, 'tag': tag, 'count': count})

tag_summary_df = pd.DataFrame(tag_summary)

# Сортируем по кластеру и количеству тегов
tag_summary_df = tag_summary_df.sort_values(by=['cluster', 'count'], ascending=[True, False])

# Сохраняем результаты
output_path = './output/sorted_tags_by_cluster.csv'
tag_summary_df.to_csv(output_path, index=False)
print(f"Сортированные теги по кластерам сохранены в '{output_path}'")
