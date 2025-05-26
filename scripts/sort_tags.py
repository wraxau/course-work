import pandas as pd
import logging
import numpy as np
from typing import Optional
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sorted_tags_by_cluster(input_file: str, output_file: str) -> Optional[pd.DataFrame]:
    """
    Генерирует CSV-файл с отсортированными по популярности тегами для каждого кластера.

    :param input_file: Путь к входному CSV-файлу с колонками 'cluster' и 'tag'
    :param output_file: Путь для сохранения выходного файла
    :return: DataFrame с результатом (или None при ошибке)
    """
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        logger.error(f"Файл не найден: {input_file}")
        return None
    except Exception as e:
        logger.exception(f"Ошибка при чтении файла: {e}")
        return None

    if "cluster" not in df.columns or "tag" not in df.columns:
        logger.error("Файл должен содержать колонки 'cluster' и 'tag'")
        return None

    logger.info("Обработка тегов по кластерам...")

    def get_sorted_tags(tag_str: str) -> str:
        """Возвращает строку уникальных тегов, отсортированных по популярности"""
        tag_list = [tag.strip() for tag in tag_str.split("|") if tag.strip()]
        if not tag_list:
            return ""
        tag_counts = pd.Series(tag_list).value_counts()
        return "|".join(tag_counts.index)

    # Группировка по кластеру и сортировка тегов
    cluster_tags = df.groupby("cluster")["tag"].apply(
        lambda tags: get_sorted_tags("|".join(tags.dropna().astype(str)))
    ).reset_index(name="sorted_tags")

    try:
        cluster_tags.to_csv(output_file, index=False)
        logger.info(f"Файл сохранён: {output_file}")
    except Exception as e:
        logger.exception(f"Ошибка при сохранении файла: {e}")
        return None

    return cluster_tags


if __name__ == "__main__":
    input_csv = "output/clusters_movies_with_tags.csv"
    output_csv = "output/sorted_tags_by_cluster.csv"
    result = generate_sorted_tags_by_cluster(input_csv, output_csv)

    if result is not None:
        print(result)
