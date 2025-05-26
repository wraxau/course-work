import pandas as pd

def rename_column_in_csv(input_path, output_path, old_name, new_name):
    """
    Загружает CSV, переименовывает столбец и сохраняет в новый файл.
    """
    df = pd.read_csv(input_path)
    if old_name in df.columns:
        df = df.rename(columns={old_name: new_name})
    else:
        print(f"Столбец '{old_name}' не найден в файле '{input_path}'")
    df.to_csv(output_path, index=False)
    print(f"Файл сохранён: {output_path}")

# Пример использования:
rename_column_in_csv(
    input_path="/Users/macbookbro/PycharmProjects/course-work/output/movies_with_clusters.csv",
# путь к исходному файлу
    output_path="final.csv",      # путь к новому файлу
    old_name="subcluster",        # старое имя столбца
    new_name="cluster"            # новое имя столбца
)
