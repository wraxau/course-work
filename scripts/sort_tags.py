import pandas as pd

def generate_sorted_tags_by_cluster(input_file, output_file):
    """Генерирует файл sorted_tags_by_cluster.csv с популярными тегами по каждому кластеру"""

    # Загружаем данные с кластерами и тегами
    df = pd.read_csv(input_file)

    # Разбиваем теги по '|', собираем частоту тегов в каждом кластере
    def get_tag_counts(tags):
        tag_list = tags.split("|")  # Разделяем теги по '|'
        tag_counts = pd.Series(tag_list).value_counts()  # Считаем частоту
        return "|".join(tag_counts.index)  # Сортируем по популярности


    cluster_tags = df.groupby("cluster")["tag"].apply(
        lambda x: get_tag_counts("|".join(x.fillna("").astype(str)))).reset_index()
    cluster_tags.columns = ["cluster", "sorted_tags"]

    # Сохраняем результат
    cluster_tags.to_csv(output_file, index=False)
    print(f" Файл {output_file} сохранён!")

if __name__ == "__main__":
    input_csv = "output/clusters_movies_with_tags.csv"
    output_csv = "output/sorted_tags_by_cluster.csv"
    generate_sorted_tags_by_cluster(input_csv, output_csv)
