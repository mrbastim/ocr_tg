import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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


def load_default_dataset() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Загружаем встроенный ТЕКСТОВЫЙ датасет документов (20newsgroups).

    Идея: приближаем задачу к работе с документами, как в боте.

    Возвращаем:
    - X: TF-IDF признаки документов (DataFrame)
    - y_cls: целевая переменная для классификации (тематика письма)
    - y_reg: целевая переменная для регрессии — длина документа (в словах)
    """

    categories = [
        "comp.graphics",
        "sci.space",
        "rec.sport.hockey",
        "talk.politics.mideast",
    ]

    data = datasets.fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )

    texts = data.data
    y_cls = pd.Series(data.target, name="target")

    # Цель для регрессии — длина документа в словах
    doc_lengths = pd.Series([len(t.split()) for t in texts], name="doc_len")

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
    X_sparse = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    X = pd.DataFrame(X_sparse.toarray(), columns=feature_names)

    y_reg = doc_lengths

    return X, y_cls, y_reg


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


def run_regression(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    print("=== Линейная регрессия ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

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

    joblib.dump(model, MODELS_DIR / "linear_regression.joblib")

    return {"r2": float(r2), "mae": float(mae), "mse": float(mse)}


def run_classification(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
    print("=== Классификация: логистическая регрессия, SVM, MLP ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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
    """

    ensure_dirs()

    if csv_path:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
        if not target_col:
            raise ValueError("Для своего CSV укажите target_col")
        print(f"Загружаю CSV датасет: {csv_path}")
        X, y = load_csv_dataset(csv_path, target_col)

        y_cls = y
        if not np.issubdtype(y_cls.dtype, np.number):
            raise ValueError(
                "Для регрессии нужен числовой target; используйте встроенный датасет или другой CSV."
            )
        y_reg = y_cls.astype(float)
    else:
        print("Использую встроенный датасет 20newsgroups из sklearn")
        X, y_cls, y_reg = load_default_dataset()

    print(f"Форма X: {X.shape}")

    metrics_reg = run_regression(X, y_reg)
    metrics_cls = run_classification(X, y_cls)
    metrics_clu = run_clustering(X, y_cls)

    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "regression": metrics_reg,
        "classification": metrics_cls,
        "clustering": metrics_clu,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Учебный ML-проект: регрессия, классификация, кластеризация и нейросеть "
            "на табличных данных (по умолчанию 20newsgroups из sklearn)."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Путь к своему CSV (если не задан, используем встроенный датасет)",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help=(
            "Имя целевой колонки для CSV. Если не задано, для встроенного датасета "
            "используется 'target' для классификации и 'mean radius' для регрессии."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = run_all(args.csv, args.target_col)
    print("\nВсе результаты сохранены в папке:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
