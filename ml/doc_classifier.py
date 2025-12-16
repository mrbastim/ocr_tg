import re
from typing import Dict, Tuple


LabelProbs = Dict[str, float]


def _normalize(text: str) -> str:
    return (text or "").lower()


def _score_receipt(text: str) -> float:
    score = 0.0
    keywords = [
        "чек",
        "касса",
        "кассовый",
        "итог",
        "оплата",
        "покупк",
        "сдача",
        "ккм",
        "инн",
        "сумма",
        "total",
        "subtotal",
        "cash",
        "change",
        "receipt",
    ]
    for kw in keywords:
        if kw in text:
            score += 1.0

    price_pattern = re.compile(r"\b\d+[.,]\d{2}\b")
    prices = price_pattern.findall(text)
    score += min(len(prices), 10) * 0.4

    if "шт" in text or "кол-во" in text or "quantity" in text:
        score += 0.8

    return score


def _score_screenshot(text: str) -> float:
    score = 0.0
    keywords = [
        "скриншот",
        "screenshot",
        "android",
        "ios",
        "windows",
        "chrome",
        "mozilla",
        "telegram",
        "vk.com",
        "instagram.com",
        "http://",
        "https://",
        "www.",
        "ctrl+",
        "shift+",
        "alt+",
        "кликайте",
        "нажмите",
    ]
    for kw in keywords:
        if kw in text:
            score += 1.0

    if text.count("@") > 3:
        score += 1.0

    return score


def _score_document(text: str) -> float:
    score = 0.0
    keywords = [
        "договор",
        "заявление",
        "приказ",
        "акт",
        "справка",
        "паспорт",
        "протокол",
        "резолюция",
        "отчёт",
        "отчет",
        "приложение",
        "страница",
        "organization",
        "university",
        "certificate",
        "agreement",
    ]
    for kw in keywords:
        if kw in text:
            score += 1.0

    if len(text.split()) > 200:
        score += 1.0

    return score


def classify_document_text(text: str) -> Tuple[str, LabelProbs]:
    """Грубая эвристическая классификация типа документа по OCR-тексту.

    Возвращает метку (receipt|screenshot|document) и словарь вероятностей.
    """

    t = _normalize(text)
    base = 0.1
    scores = {
        "receipt": base + _score_receipt(t),
        "screenshot": base + _score_screenshot(t),
        "document": base + _score_document(t),
    }

    total = sum(scores.values()) or 1.0
    probs: LabelProbs = {k: v / total for k, v in scores.items()}
    label = max(probs, key=probs.get)
    return label, probs
