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
        "руб",
        "р.",
        "eur",
        "usd",
        "total",
        "subtotal",
        "cash",
        "change",
        "receipt",
    ]
    found_kw = False
    for kw in keywords:
        if kw in text:
            score += 1.0
            found_kw = True

    price_pattern = re.compile(r"\b\d+[.,]\d{2}\b")
    prices = price_pattern.findall(text)
    # Если нет ни одного "кассового" слова, не воспринимаем одни только числа как чек
    if found_kw:
        score += min(len(prices), 10) * 0.4
    else:
        score += min(len(prices), 10) * 0.1

    if "шт" in text or "кол-во" in text or "quantity" in text:
        score += 0.8

    # Дополнительный фильтр: очень короткий текст и нет ключевых слов — скорее не чек
    if not found_kw and len(text.split()) < 20:
        score *= 0.3

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
        "отчет",
        "отчёт",
        "записка",
        "инструкция",
        "руководство",
        "protocol",
        "report",
        "manual",
    ]
    for kw in keywords:
        if kw in text:
            score += 1.0

    if len(text.split()) > 200:
        score += 1.0

    return score


def _score_form(text: str) -> float:
    score = 0.0
    keywords = [
        "анкета",
        "форма",
        "заполните",
        "фио",
        "фамилия",
        "имя",
        "отчество",
        "подпись",
        "дата",
        "паспорт",
        "телефон",
        "e-mail",
        "email",
        "адрес",
        "подпись",
        "печать",
        "заявитель",
    ]
    for kw in keywords:
        if kw in text:
            score += 1.0

    # формы часто содержат много коротких полей → двоеточия
    score += text.count(":") * 0.1
    return score


def _score_diagram(text: str) -> float:
    score = 0.0
    keywords = [
        "схема",
        "схемы",
        "диаграмма",
        "график",
        "чертеж",
        "чертёж",
        "генератор",
        "двигатель",
        "насос",
        "датчик",
        "температура",
        "давление",
        "напряжение",
        "ток",
        "квт",
        "kw",
        "rpm",
        "об/мин",
        "voltage",
        "current",
        "power",
        "pressure",
        "flow",
    ]
    for kw in keywords:
        if kw in text:
            score += 1.0

    # много чисел без кассовых слов тоже признак схем/измерений
    numbers = re.findall(r"\b\d+[.,]?\d*\b", text)
    if numbers:
        score += min(len(numbers), 20) * 0.05

    return score


def classify_document_text(text: str) -> Tuple[str, LabelProbs]:
    """Грубая эвристическая классификация типа документа по OCR-тексту.

    Возвращает метку (receipt|screenshot|document|form|diagram|unknown) и словарь вероятностей.
    """

    t = _normalize(text)
    base = 0.1
    scores = {
        "receipt": base + _score_receipt(t),
        "screenshot": base + _score_screenshot(t),
        "document": base + _score_document(t),
        "form": base + _score_form(t),
        "diagram": base + _score_diagram(t),
    }

    total = sum(scores.values()) or 1.0
    probs: LabelProbs = {k: v / total for k, v in scores.items()}

    # Порог для "неопределённого" класса: если лучший класс слишком слабо выражен,
    # считаем, что надёжной информации о типе нет.
    label = max(probs, key=probs.get)
    max_prob = probs[label]
    if max_prob < 0.45:
        return "unknown", {"unknown": 1.0}

    return label, probs
