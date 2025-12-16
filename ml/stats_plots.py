from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .event_logger import LOG_FILE, LOG_DIR


PLOTS_DIR = LOG_DIR / "plots_stats"


def _ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def build_basic_plots() -> List[Tuple[Path, str]]:
    """Построить несколько базовых графиков по логам обработки.

    Возвращает список (путь_к_png, подпись).
    """

    _ensure_dirs()

    if not LOG_FILE.exists():
        raise FileNotFoundError("Файл логов не найден. Пока нет обработанных изображений.")

    df = pd.read_csv(LOG_FILE)
    imgs = []

    # Гистограмма времени обработки (факт и прогноз)
    if {"predicted_time", "ocr_time", "total_time"}.issubset(df.columns):
        plt.figure(figsize=(6, 4))
        df["predicted_time"].plot(kind="hist", bins=20, alpha=0.5, label="Прогноз")
        df["total_time"].plot(kind="hist", bins=20, alpha=0.5, label="Факт (всё)")
        plt.xlabel("Время, секунды")
        plt.ylabel("Частота")
        plt.title("Распределение времени обработки")
        plt.legend()
        plt.tight_layout()
        path_time = PLOTS_DIR / "time_distribution.png"
        plt.savefig(path_time)
        plt.close()
        imgs.append((path_time, "Распределение времени обработки (прогноз vs факт)"))

    # Столбчатая диаграмма по типам документов
    if "doc_type" in df.columns:
        counts = df["doc_type"].value_counts().sort_values(ascending=False)
        plt.figure(figsize=(6, 4))
        counts.plot(kind="bar")
        plt.xlabel("Тип документа")
        plt.ylabel("Количество")
        plt.title("Распределение типов документов")
        plt.tight_layout()
        path_types = PLOTS_DIR / "doc_types.png"
        plt.savefig(path_types)
        plt.close()
        imgs.append((path_types, "Распределение типов изображений (по тексту)"))

    return imgs
