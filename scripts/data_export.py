import os
import pandas as pd
import joblib  # Для сохранения моделей

OUTPUT_DIR = "output"

def save_dataframe(df, filename):
    """Сохраняет DataFrame в CSV-файл."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"[Сохранение] Файл '{filename}' сохранён в {OUTPUT_DIR}")

def save_model(model, filename):
    #Сохраняет модель машинного обучения??
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    joblib.dump(model, filepath)
    print(f"Модель сохранена в {filepath}")


def load_model(filename):
    #Загружает ранее сохранённую модель из .pkl файла
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Ошибка: файл '{filename}' не найден!")
        return None
    model = joblib.load(filepath)
    print(f"[Загрузка модели] '{filename}' успешно загружен.")
    return model
