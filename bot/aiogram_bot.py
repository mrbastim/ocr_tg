import os
import sys
import asyncio
import logging

# Добавляем корневую директорию проекта в sys.path для импорта `ocr`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Загружаем переменные окружения из файла .env, если установлен python-dotenv
try:
	from dotenv import load_dotenv
	env_path = os.path.join(os.path.dirname(__file__), ".env")
	if os.path.exists(env_path):
		load_dotenv(env_path)
except Exception:
	pass

from aiogram import Bot, Dispatcher

try:
    # запуск как модуля: python -m bot.aiogram_bot
    from .handlers import register_handlers
except ImportError:
    # запуск как обычного скрипта: python bot/aiogram_bot.py
    from bot.handlers import register_handlers

logger = logging.getLogger(__name__)

async def main():
    # Всегда включаем подробное логирование в терминал
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("aiogram").setLevel(logging.DEBUG)
    logger.debug("Logger initialized")
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Не найден TELEGRAM_BOT_TOKEN в переменных окружения")

    logger.debug("Creating Bot and Dispatcher")
    bot = Bot(token=token)
    dp = Dispatcher()

    # Регистрация всех хендлеров вынесена в отдельный модуль
    register_handlers(dp)

    logger.debug("Start polling")
    await dp.start_polling(bot)
    logger.debug("Polling stopped")


if __name__ == "__main__":
    asyncio.run(main())
