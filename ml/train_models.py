import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


OUTPUT_DIR = Path("ml_output")
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv_dataset(path: Path, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Загрузка произвольного CSV.

    path: путь к csv
    target_col: имя целевой колонки
    """

    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Столбец '{target_col}' не найден в CSV {path}")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Простейший препроцессинг: выбрасываем нечисловые колонки
    X = X.select_dtypes(include=[np.number]).copy()
    return X, y


def prepare_classification_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_bins: int = 6,
    min_per_class: int = 2,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], pd.Series]:
    """Приводит целевую переменную к дискретным классам и убирает редкие метки.

    Возвращает (X_filtered, y_filtered, class_counts). Если после фильтрации классов
    меньше двух или выборка слишком мала, возвращает (None, None, class_counts).
    """

    y_proc = y.copy()

    # Если target числовой и уникальных значений много, бинним по квантилям,
    # чтобы избежать классов с одним объектом.
    if np.issubdtype(y_proc.dtype, np.number) and y_proc.nunique() > max_bins:
        y_proc = pd.qcut(y_proc, q=max_bins, duplicates="drop")

    y_proc = y_proc.astype(str)

    # Убираем классы реже заданного порога
    value_counts = y_proc.value_counts()
    rare_classes = value_counts[value_counts < min_per_class].index
    if len(rare_classes):
        mask = ~y_proc.isin(rare_classes)
        y_proc = y_proc[mask]
        X = X.loc[mask]

    value_counts = y_proc.value_counts()

    if y_proc.nunique() < 2 or len(y_proc) < 4:
        return None, None, value_counts

    return X, y_proc, value_counts


def run_regression(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    print("=== Линейная регрессия ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"R^2: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")

    # Графики: предсказание vs фактические, распределение ошибок
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Фактическое значение")
    plt.ylabel("Предсказание")
    plt.title("Линейная регрессия: y_true vs y_pred")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "regression_scatter.png")
    plt.close()

    errors = y_test - y_pred
    plt.figure(figsize=(6, 5))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel("Ошибка (y_true - y_pred)")
    plt.ylabel("Частота")
    plt.title("Распределение ошибок регрессии")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "regression_errors_hist.png")
    plt.close()

    # Сохраняем модель вместе с порядком фичей, чтобы использовать в боте
    payload = {
        "model": model,
        "feature_names": list(X.columns),
    }
    joblib.dump(payload, MODELS_DIR / "ocr_time_regression.joblib")

    return {"r2": float(r2), "mae": float(mae), "mse": float(mse)}


def run_classification(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
    print("=== Классификация: логистическая регрессия, SVM, MLP ===")
    n_classes = len(np.unique(y))
    n_samples = len(y)
    test_size = max(0.2, n_classes / n_samples + 0.01)
    test_size = min(test_size, 0.5)  # не отдаём в тест больше половины датасета
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Для F1 различаем бинарный и многоклассовый случаи
    average_type = "binary" if len(np.unique(y_train)) == 2 else "weighted"

    # Логистическая регрессия
    log_reg = LogisticRegression(max_iter=500)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    results["LogReg"] = {
        "acc": accuracy_score(y_test, y_pred_lr),
        "f1": f1_score(y_test, y_pred_lr, average=average_type),
    }

    # SVM (RBF)
    svm = SVC(kernel="rbf", probability=True)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    results["SVM"] = {
        "acc": accuracy_score(y_test, y_pred_svm),
        "f1": f1_score(y_test, y_pred_svm, average=average_type),
    }

    # MLP (нейросеть)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp.predict(X_test_scaled)
    results["MLP"] = {
        "acc": accuracy_score(y_test, y_pred_mlp),
        "f1": f1_score(y_test, y_pred_mlp, average=average_type),
    }

    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {metrics['acc']:.3f}")
        print(f"  F1-score: {metrics['f1']:.3f}")

    print("\nОтчёт по лучшей модели (по F1):")
    best_name = max(results, key=lambda k: results[k]["f1"])
    if best_name == "LogReg":
        best_pred = y_pred_lr
    elif best_name == "SVM":
        best_pred = y_pred_svm
    else:
        best_pred = y_pred_mlp

    print(classification_report(y_test, best_pred))

    # График: сравнение моделей по accuracy и F1
    names = list(results.keys())
    accs = [results[n]["acc"] for n in names]
    f1s = [results[n]["f1"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(6, 5))
    plt.bar(x - width / 2, accs, width, label="Accuracy")
    plt.bar(x + width / 2, f1s, width, label="F1-score")
    plt.xticks(x, names)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.title("Качество моделей классификации")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "classification_models.png")
    plt.close()

    joblib.dump(mlp, MODELS_DIR / "mlp_classifier.joblib")

    return {name: {"acc": float(m["acc"]), "f1": float(m["f1"])} for name, m in results.items()}


def run_clustering(X: pd.DataFrame, y_true: Optional[pd.Series] = None) -> Dict[str, Any]:
    print("=== Кластеризация: k-means ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Если есть истинные метки, берём число их уникальных значений
    if y_true is not None:
        unique_classes = np.unique(y_true)
        k = max(2, len(unique_classes))
    else:
        k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    print(f"Silhouette score: {sil:.3f}")

    ari = None
    if y_true is not None and len(np.unique(y_true)) == k:
        from sklearn.metrics import adjusted_rand_score

        ari = adjusted_rand_score(y_true, labels)
        print(f"Adjusted Rand Index (кластеры vs истинные классы): {ari:.3f}")

    # Берём первые две главные компоненты для наглядного scatter
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("k-means кластеры (PCA 2D)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "clustering_kmeans.png")
    plt.close()

    joblib.dump(kmeans, MODELS_DIR / "kmeans.joblib")

    return {"k": int(k), "silhouette": float(sil), "ari": float(ari) if ari is not None else None}


def run_all(csv_path: Optional[str] = None, target_col: Optional[str] = None) -> Dict[str, Any]:
    """Запустить полный ML-эксперимент и вернуть метрики.

    Используется как из CLI (main), так и из Telegram-бота.
    Ожидается табличный датасет (например, логи бота), переданный в виде CSV.
    """

    ensure_dirs()

    if not csv_path or not target_col:
        raise ValueError(
            "Нужно указать --csv и --target-col. Рекомендуется использовать CSV с логами бота или другой табличный датасет."
        )

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")

    print(f"Загружаю CSV датасет: {csv_path}")
    X_full, y = load_csv_dataset(csv_path, target_col)

    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(
            "Для регрессии нужен числовой числовой target; используйте CSV, где целевая колонка — число (например, время обработки)."
        )
    y_reg = y.astype(float)

    # Для регрессии стараемся использовать те же признаки, что и в боте
    preferred_cols = [
        "width",
        "height",
        "megapixels",
        "brightness",
        "contrast",
        "word_count",
    ]
    available = [c for c in preferred_cols if c in X_full.columns]
    if available:
        X_reg = X_full[available].copy()
    else:
        X_reg = X_full

    print(f"Форма X_reg (для регрессии): {X_reg.shape}")
    print(f"Форма X_full (для классификации/кластеризации): {X_full.shape}")

    metrics_reg = run_regression(X_reg, y_reg)

    X_cls, y_cls, class_counts = prepare_classification_dataset(X_full, y_reg)
    if X_cls is None or y_cls is None:
        print(
            "Пропускаю классификацию/кластеризацию: после биннинга/фильтрации осталось меньше двух классов."
        )
        if not class_counts.empty:
            print("Распределение классов:")
            print(class_counts.to_string())
        metrics_cls = {}
        metrics_clu = {}
    else:
        print("Распределение классов после биннинга/фильтрации:")
        print(class_counts.to_string())
        metrics_cls = run_classification(X_cls, y_cls)
        metrics_clu = run_clustering(X_cls, y_cls)

    return {
        "n_samples": int(X_full.shape[0]),
        "n_features": int(X_full.shape[1]),
        "regression": metrics_reg,
        "classification": metrics_cls,
        "clustering": metrics_clu,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Учебный ML-проект: регрессия, классификация, кластеризация и нейросеть "
            "на табличных данных. По умолчанию используется ваш CSV (например, логи бота)."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Путь к своему CSV (например, ml_output/events.csv с логами бота)",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help=(
            "Имя целевой колонки для CSV. Рекомендуется использовать логи бота или другой табличный датасет."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = run_all(args.csv, args.target_col)
    print("\nВсе результаты сохранены в папке:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
