from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass
class ImageFeatures:
    width: int
    height: int
    megapixels: float
    brightness: float
    contrast: float
    word_count: int


def extract_image_features(image_path: str, text: str) -> ImageFeatures:
    """Извлечь простые признаки изображения + текста для регрессии.

    - размер (ширина/высота, мегапиксели)
    - яркость и контраст (по grayscale)
    - количество слов в OCR-тексте
    """

    word_count = len((text or "").split())

    if not image_path or not os.path.exists(image_path):
        return ImageFeatures(
            width=0,
            height=0,
            megapixels=0.0,
            brightness=0.5,
            contrast=0.3,
            word_count=word_count,
        )

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ImageFeatures(
            width=0,
            height=0,
            megapixels=0.0,
            brightness=0.5,
            contrast=0.3,
            word_count=word_count,
        )

    h, w = img.shape[:2]
    megapixels = (w * h) / 1_000_000.0
    brightness = float(np.mean(img)) / 255.0
    contrast = float(np.std(img)) / 255.0

    return ImageFeatures(
        width=int(w),
        height=int(h),
        megapixels=float(megapixels),
        brightness=float(brightness),
        contrast=float(contrast),
        word_count=int(word_count),
    )


def predict_ocr_time(features: ImageFeatures) -> float:
    """Простая "регрессия" для оценки времени OCR в секундах.

    Используем линейную модель по нескольким признакам.
    Это скорее обучающий пример, чем точный прогноз.
    """

    # Базовое время
    base = 0.4

    # Чем больше мегапикселей, тем дольше
    mp_term = 0.5 * features.megapixels

    # Больше слов — немного дольше
    wc_term = 0.001 * features.word_count

    # Слабый контраст ухудшает OCR → чуть больше времени
    contrast_term = 0.4 * (1.0 - features.contrast)

    t = base + mp_term + wc_term + contrast_term
    t = max(0.3, min(t, 15.0))  # ограничим диапазон
    return float(t)


def build_processing_summary(image_path: str | None, text: str) -> str:
    """Сформировать человекочитаемую строку о времени обработки.

    Пример: "Это займёт примерно 2.5 секунды обработки.".
    """

    feats = extract_image_features(image_path or "", text)
    t = predict_ocr_time(feats)

    # Округляем до десятых
    return f"Это займёт примерно {t:.1f} секунды обработки."