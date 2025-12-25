# Телеграм-бот для пост-OCR коррекции с LLM

## Быстрый старт

### 1) Установка Tesseract (Windows)
- Скачайте инсталлятор: https://github.com/UB-Mannheim/tesseract/wiki
- Установите в `C:\\Program Files\\Tesseract-OCR\\`
- Добавьте путь к `tesseract.exe` в переменную окружения `PATH` или укажите его в коде (`ocr/base.py`).
- Для русского языка установите пакет `rus`. В инсталляторе отметьте дополнительные языки.

### 2) Установка зависимостей
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3) Мини-датасет
- Положите 15-20 изображений в папку `dataset_mini/`.
- Для каждого изображения создайте эталонный Markdown-файл (`.md`).

### 4) Запуск базового OCR
- В `ocr/base.py` есть функция `get_raw_text(image_path, lang="rus")`.
- Пример запуска:
```
python bot/bot.py --ocr-only --image "dataset_mini/sample.jpg" --lang rus
```

### 5) Настройка LLM (GigaChat/Gemini/Local)
- Получите API ключ и сохраните как переменную окружения `GIGACHAT_API_KEY`.
- Для Gemini используйте переменные `AI_API_BASE/AI_API_USER/AI_API_PASS` или `GEMINI_API_KEY`.
- Для локальной LLM через Ollama (qwen2:1.5b по умолчанию) задайте `LOCAL_LLM_MODEL` и, при необходимости, `LOCAL_LLM_MAX_CHARS`.
- В `bot/bot.py` добавлены заготовки промптов (A/B/C).

### 6) Запуск Telegram-бота
- Создайте бота и получите токен у @BotFather.
- Сохраните токен как `TELEGRAM_BOT_TOKEN`.
- Запустите:
```
python bot/bot.py --run-bot
```

## Цели НИР
- Сбор мини-датасета и эталонов.
- Базовый пайплайн: OCR → LLM коррекция → Markdown.
- Эксперименты с промптами (A: коррекция, B: структуризация, C: семантическое восстановление).
- Сравнение результатов с эталонами.
