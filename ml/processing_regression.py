from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import cv2
import numpy as np
import joblib
import pandas as pd


@dataclass
class ImageFeatures:
    width: int
    height: int
    megapixels: float
    brightness: float
    contrast: float
    word_count: int


MODEL_PATH = Path("ml_output") / "models" / "ocr_time_regression.joblib"
_cached_model: Optional[Dict[str, Any]] = None


def _load_trained_model() -> Optional[Dict[str, Any]]:
    """Попробовать загрузить обученную регрессионную модель времени.

    Возвращает dict {"model": ..., "feature_names": [...]} или None,
    если модель ещё не обучена или не читается.
    """

    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if not MODEL_PATH.exists():
        return None

    try:
        payload = joblib.load(MODEL_PATH)
        if not isinstance(payload, dict):
            return None
        if "model" not in payload or "feature_names" not in payload:
            return None
        _cached_model = payload
        return _cached_model
    except Exception:
        return None


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
    """Оценка времени OCR в секундах.

    1. Если есть обученная модель (ocr_time_regression.joblib), используем её.
    2. Иначе используем простую эвристику по размеру, контрасту и числу слов.
    """

    model_payload = _load_trained_model()
    if model_payload is not None:
        model = model_payload["model"]
        scaler = model_payload.get("scaler")  # может быть None для старых моделей
        feature_names = list(model_payload.get("feature_names", []))
        
        # Базовые признаки
        feat_map: Dict[str, float] = {
            "width": float(features.width),
            "height": float(features.height),
            "megapixels": float(features.megapixels),
            "brightness": float(features.brightness),
            "contrast": float(features.contrast),
            "word_count": float(features.word_count),
        }
        
        # Добавляем производные признаки (как в train_models.py)
        if "megapixels" in feat_map:
            feat_map["megapixels_squared"] = feat_map["megapixels"] ** 2
            feat_map["log_megapixels"] = np.log1p(feat_map["megapixels"])
        if "width" in feat_map and "height" in feat_map:
            feat_map["aspect_ratio"] = feat_map["width"] / (feat_map["height"] + 1)
            feat_map["total_pixels"] = feat_map["width"] * feat_map["height"]
            feat_map["log_pixels"] = np.log1p(feat_map["total_pixels"])
        
        if feature_names:
            # Передаём именованные признаки в правильном порядке
            x_vec = pd.DataFrame([
                {name: feat_map.get(name, 0.0) for name in feature_names}
            ])
            try:
                # Применяем scaler если он есть
                if scaler is not None:
                    x_vec = scaler.transform(x_vec)
                
                pred = model.predict(x_vec)[0]
                t = float(pred)
                if np.isfinite(t):
                    # Ограничиваем разумный диапазон
                    return float(max(0.1, min(t, 120.0)))
            except Exception:
                # Падаем в эвристику ниже
                pass

    # Базовая эвристика, если модели ещё нет
    base = 1.0
    mp_term = 0.8 * features.megapixels
    wc_term = 0.002 * features.word_count
    contrast_term = 0.8 * (1.0 - features.contrast)

    t = base + mp_term + wc_term + contrast_term
    t = max(0.5, min(t, 20.0))
    return float(t)


def build_processing_summary(image_path: str | None, text: str) -> str:
    """Сформировать человекочитаемую строку о времени обработки.

    Пример: "Это займёт примерно 2.5 секунды обработки.".
    """

    feats = extract_image_features(image_path or "", text)
    t = predict_ocr_time(feats)

    # Округляем до десятых
    return f"Это займёт примерно {t:.1f} секунды обработки."