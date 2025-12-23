from __future__ import annotations

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .processing_regression import ImageFeatures, extract_image_features
from .doc_classifier import classify_document_text


LOG_DIR = Path("ml_output")
LOG_FILE = LOG_DIR / "events.csv"


@dataclass
class EventRecord:
    ts: float
    user_id: int
    provider: str
    source: str  # "photo" | "document" | "pdf_page"
    is_pdf: bool
    width: int
    height: int
    megapixels: float
    brightness: float
    contrast: float
    word_count: int
    text_length: int  # количество символов в тексте
    line_count: int   # количество строк
    avg_word_length: float  # средняя длина слова
    ocr_time: float
    total_time: float
    doc_type: str


def log_event(
    *,
    image_path: str,
    text: str,
    user_id: int,
    provider: str,
    source: str,
    is_pdf: bool,
    t_ocr: float,
    t_total: float,
) -> None:
    """Записать один факт обработки изображения в CSV-лог.

    Если файл ещё не существует, создаётся с заголовком.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    feats: ImageFeatures = extract_image_features(image_path, text)

    try:
        doc_label = classify_document_text(text)[0]
    except Exception:
        doc_label = "unknown"

    # Вычисляем текстовые признаки
    text_clean = text or ""
    text_length = len(text_clean)
    line_count = text_clean.count('\n') + 1 if text_clean else 0
    words = text_clean.split()
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0.0

    rec = EventRecord(
        ts=time.time(),
        user_id=int(user_id),
        provider=str(provider),
        source=str(source),
        is_pdf=bool(is_pdf),
        width=feats.width,
        height=feats.height,
        megapixels=float(feats.megapixels),
        brightness=float(feats.brightness),
        contrast=float(feats.contrast),
        word_count=int(feats.word_count),
        text_length=int(text_length),
        line_count=int(line_count),
        avg_word_length=float(avg_word_length),
        ocr_time=float(t_ocr),
        total_time=float(t_total),
        doc_type=str(doc_label),
    )

    is_new = not LOG_FILE.exists()
    row = asdict(rec)

    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def get_log_file() -> Optional[Path]:
    """Вернуть путь к лог-файлу, если он есть."""

    return LOG_FILE if LOG_FILE.exists() else None
