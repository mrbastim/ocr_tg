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
    imgs: List[Tuple[Path, str]] = []

    # 1. Гистограмма времени обработки (факт и прогноз)
    if {"predicted_time", "total_time"}.issubset(df.columns):
        plt.figure(figsize=(6, 4))
        df["predicted_time"].plot(kind="hist", bins=20, alpha=0.5, label="Прогноз")
        df["total_time"].plot(kind="hist", bins=20, alpha=0.5, label="Факт (общее время)")
        plt.xlabel("Время, секунды")
        plt.ylabel("Частота")
        plt.title("Распределение времени обработки")
        plt.legend()
        plt.tight_layout()
        path_time = PLOTS_DIR / "time_distribution.png"
        plt.savefig(path_time)
        plt.close()
        imgs.append((path_time, "Распределение времени обработки (прогноз против факта)"))

        # 2. Диаграмма рассеяния: прогноз vs факт
        mask = df[["predicted_time", "total_time"]].dropna()
        if not mask.empty:
            plt.figure(figsize=(6, 4))
            plt.scatter(mask["predicted_time"], mask["total_time"], alpha=0.6)
            plt.xlabel("Прогноз времени, сек")
            plt.ylabel("Фактическое время, сек")
            plt.title("Прогноз против фактического времени обработки")
            plt.tight_layout()
            path_scatter = PLOTS_DIR / "time_pred_vs_actual.png"
            plt.savefig(path_scatter)
            plt.close()
            imgs.append((path_scatter, "Прогноз vs факт времени обработки"))

        # 3. Распределение ошибки прогноза
        if not mask.empty:
            errors = mask["total_time"] - mask["predicted_time"]
            plt.figure(figsize=(6, 4))
            plt.hist(errors, bins=20, alpha=0.7)
            plt.xlabel("Ошибка прогноза (факт − прогноз), сек")
            plt.ylabel("Частота")
            plt.title("Распределение ошибки прогноза времени")
            plt.tight_layout()
            path_err = PLOTS_DIR / "time_error_hist.png"
            plt.savefig(path_err)
            plt.close()
            imgs.append((path_err, "Распределение ошибки прогноза времени"))

    # 4. Столбчатая диаграмма по типам документов
    if "doc_type" in df.columns:
        counts = df["doc_type"].value_counts().sort_values(ascending=False)
        if not counts.empty:
            plt.figure(figsize=(6, 4))
            counts.plot(kind="bar")
            plt.xlabel("Тип изображения")
            plt.ylabel("Количество")
            plt.title("Распределение типов изображений")
            plt.tight_layout()
            path_types = PLOTS_DIR / "doc_types.png"
            plt.savefig(path_types)
            plt.close()
            imgs.append((path_types, "Распределение типов изображений (по классификатору)"))

        # 5. Среднее время обработки по типам
        if {"doc_type", "total_time"}.issubset(df.columns):
            grouped = df.dropna(subset=["doc_type", "total_time"]).groupby("doc_type")[
                "total_time"
            ].mean().sort_values(ascending=False)
            if not grouped.empty:
                plt.figure(figsize=(6, 4))
                grouped.plot(kind="bar")
                plt.xlabel("Тип изображения")
                plt.ylabel("Среднее время, сек")
                plt.title("Среднее время обработки по типам изображений")
                plt.tight_layout()
                path_types_time = PLOTS_DIR / "doc_types_mean_time.png"
                plt.savefig(path_types_time)
                plt.close()
                imgs.append((path_types_time, "Среднее время обработки по типам изображений"))

    # 6. Зависимость времени обработки от размера изображения
    if {"megapixels", "total_time"}.issubset(df.columns):
        mp_df = df.dropna(subset=["megapixels", "total_time"])
        if not mp_df.empty:
            plt.figure(figsize=(6, 4))
            plt.scatter(mp_df["megapixels"], mp_df["total_time"], alpha=0.6)
            plt.xlabel("Размер изображения, мегапиксели")
            plt.ylabel("Фактическое время, сек")
            plt.title("Время обработки в зависимости от размера изображения")
            plt.tight_layout()
            path_mp = PLOTS_DIR / "time_vs_megapixels.png"
            plt.savefig(path_mp)
            plt.close()
            imgs.append((path_mp, "Время обработки vs размер изображения (мегапиксели)"))

    # 7. Среднее время обработки по провайдерам LLM
    if {"provider", "total_time"}.issubset(df.columns):
        prov_df = df.dropna(subset=["provider", "total_time"]).copy()
        # Нормализуем имена провайдеров: внутренний 'api' и 'gemini' считаем Gemini
        prov_df["provider"] = prov_df["provider"].replace(
            {
                "api": "Gemini",
                "gemini": "Gemini",
                "Gemini": "Gemini",
                "gigachat": "GigaChat",
                "GigaChat": "GigaChat",
            }
        )
        if not prov_df.empty:
            grouped = prov_df.groupby("provider")["total_time"].mean().sort_values(ascending=False)
            plt.figure(figsize=(6, 4))
            grouped.plot(kind="bar")
            plt.xlabel("Провайдер LLM")
            plt.ylabel("Среднее время, сек")
            plt.title("Среднее время обработки по провайдерам LLM")
            plt.tight_layout()
            path_prov = PLOTS_DIR / "providers_mean_time.png"
            plt.savefig(path_prov)
            plt.close()
            imgs.append((path_prov, "Среднее время обработки по провайдерам LLM"))

    return imgs
