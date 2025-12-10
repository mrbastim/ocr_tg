FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Ставим системные зависимости для Tesseract, PDF и OpenCV (libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Стандартный путь к языкам Tesseract в Debian/Ubuntu
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ожидаем, что TELEGRAM_BOT_TOKEN и прочие переменные заданы через .env или ENV
CMD ["python", "-m", "bot.aiogram_bot"]
