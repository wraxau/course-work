import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
df = pd.read_csv(csv_path)

# Удаление строк с пропущенными метриками качества
df_clean = df.dropna(subset=["Silhouette", "Calinski-Harabasz Score", "Davies-Bouldin Score"]).copy()

df_clean["Silhouette_Rank"] = df_clean["Silhouette"].rank(ascending=False)
df_clean["CH_Rank"] = df_clean["Calinski-Harabasz Score"].rank(ascending=False)
df_clean["DB_Rank"] = df_clean["Davies-Bouldin Score"].rank(ascending=True)

df_clean["Avg_Rank"] = df_clean[["Silhouette_Rank", "CH_Rank", "DB_Rank"]].mean(axis=1)
df_clean["Rank"] = df_clean["Avg_Rank"].rank(ascending=True).astype(int)

# Сохранение ранжированной таблицы
ranked_csv_path = os.path.join(OUTPUT_DIR, "ranked_models.csv")
df_clean.sort_values("Rank").to_csv(ranked_csv_path, index=False)

# График: топ-10 моделей по среднему рангу
top10 = df_clean.sort_values("Avg_Rank").head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top10, x="Model", y="Avg_Rank", hue="Rank")
plt.title("Топ-10 моделей по среднему рангу")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top10_avg_rank.png"))
plt.close()

# График: сравнение всех моделей по метрикам
metrics = ["Silhouette", "Calinski-Harabasz Score", "Davies-Bouldin Score"]
for metric in metrics:
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_clean, x="Model", y=metric)
    plt.title(f"Сравнение моделей по метрике {metric}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f"{metric.lower().replace(' ', '_')}_by_model.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
