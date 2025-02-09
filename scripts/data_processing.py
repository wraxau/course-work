from sklearn.preprocessing import StandardScaler


def standardize_data(df, column_name):
    # Проверяем, какие столбцы есть в ratings_df
    print(f"Проверяем, что столбец '{column_name}' существует в DataFrame.")
    print(f"Список столбцов: {df.columns}")

    # Проверяем, что столбец существует в DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Следующие столбцы отсутствуют в DataFrame: {column_name}")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # Применяем стандартизацию
    df[column_name] = scaler.fit_transform(df[[column_name]])

    return df

