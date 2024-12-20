import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f_oneway, pearsonr

# Загружаем данные из файла
def load_data(file_path):
    data = pd.read_csv(file_path, sep=',')
    y = data['y']
    X = data.drop(columns=['y'])
    return X, y

# Оценка параметров линейной модели
def fit_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Расчет корреляций между факторами и откликом
def compute_correlations(X, y):
    factor_correlations = {}
    for col in X.columns:
        corr, _ = pearsonr(X[col], y)
        factor_correlations[col] = corr
    return factor_correlations

# Функция для фильтрации факторов на основе корреляций
def filter_factors(X, y, correlation_threshold):
    correlations = compute_correlations(X, y)
    filtered_factors = [col for col, corr in correlations.items() if abs(corr) >= correlation_threshold]
    return X[filtered_factors]

# Оценка адекватности модели
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    r_squared = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    n, k = X.shape
    if r_squared == 1:
        f_stat = float('inf')  # Если r_squared = 1, F-статистика становится бесконечной
    else:
        f_stat = (r_squared / (1 - r_squared)) * ((n - k - 1) / k)
    return r_squared, mse, f_stat

# Вывод результатов в консоль и файл
def save_results(file_path, results):
    with open(file_path, 'w') as f:
        f.write(results)

# Предсказание с использованием новой модели
def predict_new_data(model, new_data_file):
    new_data = pd.read_csv(new_data_file, sep=',')
    predictions = model.predict(new_data)
    return predictions

if __name__ == "__main__":
    data_file = input("Введите путь к файлу с данными: ")
    X, y = load_data(data_file)

    significance_level = float(input("Введите уровень значимости (например, 0.05): "))
    correlation_threshold = float(input("Введите порог корреляции для отбора факторов: "))

    # Первичная модель
    model = fit_model(X, y)
    r_squared, mse, f_stat = evaluate_model(model, X, y)

    print(f"Коэффициент детерминации: {r_squared}")
    print(f"Среднеквадратичная ошибка: {mse}")
    print(f"F-статистика: {f_stat}")

    # Фильтрация факторов
    X_filtered = filter_factors(X, y, correlation_threshold)
    print(f"Выбраны факторы: {list(X_filtered.columns)}")

    # Модель с отфильтрованными факторами
    model_filtered = fit_model(X_filtered, y)
    r_squared_filtered, mse_filtered, f_stat_filtered = evaluate_model(model_filtered, X_filtered, y)

    print(f"Коэффициент детерминации (после фильтрации): {r_squared_filtered}")
    print(f"Среднеквадратичная ошибка (после фильтрации): {mse_filtered}")
    print(f"F-статистика (после фильтрации): {f_stat_filtered}")

    # Сохранение результатов
    output_file = "results.txt"
    results = (f"R^2: {r_squared_filtered}\n" +
               f"MSE: {mse_filtered}\n" +
               f"F-stat: {f_stat_filtered}\n")
    save_results(output_file, results)
    print(f"Результаты сохранены в файл {output_file}")

    # Предсказания
    new_data_file = input("Введите путь к файлу с новыми данными для предсказания: ")
    predictions = predict_new_data(model_filtered, new_data_file)
    print(f"Предсказания: {predictions}")
