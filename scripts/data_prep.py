import pandas as pd

cluster_data = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 1000), (1, 1001), (1, 1002), (1, 1003), (1, 1004),
    (2, 2000), (2, 2001), (2, 2002), (2, 2003), (2, 2004),
    (3, 3000), (3, 3001), (3, 3002), (3, 3003), (3, 3004), (3, 3005),
    (4, 4000), (4, 4001), (4, 4002), (4, 4003), (4, 4004), (4, 4005),
    (5, 5000), (5, 5001), (5, 5002), (5, 5003), (5, 5004), (5, 5005),
    (6, 6000), (6, 6001), (6, 6002),
    (7, 7000), (7, 7001), (7, 7002), (7, 7003), (7, 7004),
    (8, 8000), (8, 8001), (8, 8002), (8, 8003), (8, 8004)
]

# Создание отображения для subcluster >= 1000
cluster_mapping = {}
new_cluster_id = 5
for _, subcluster in cluster_data:
    if subcluster >= 1000:
        cluster_mapping[subcluster] = new_cluster_id
        new_cluster_id += 1

df = pd.read_csv('final.csv')

df['cluster'] = df['cluster'].map(cluster_mapping).fillna(df['cluster']).astype(int)

df.to_csv('final_updated.csv', index=False)

print("Файл успешно обновлен и сохранен как 'final_updated.csv'")