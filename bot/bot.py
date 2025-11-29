import os
import sys
import argparse
from typing import Optional

# Добавляем корневую директорию проекта в sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.base import get_raw_text, normalize_whitespace

# Загружаем переменные окружения из файла .env, если установлен python-dotenv
try:
	from dotenv import load_dotenv
	env_path = os.path.join(os.path.dirname(__file__), ".env")
	if os.path.exists(env_path):
		load_dotenv(env_path)
except Exception:
	# Тихо игнорируем, если пакет не установлен
	pass

try:
	from gigachat import GigaChat
	from gigachat.models import Chat, Messages, MessagesRole
	GIGACHAT_AVAILABLE = True
except ImportError:
	GIGACHAT_AVAILABLE = False

# Попытка подключения Gemini (Google Generative AI)
try:
	import google.generativeai as genai
	GEMINI_AVAILABLE = True
except ImportError:
	GEMINI_AVAILABLE = False


def gigachat_complete(prompt: str, api_key: Optional[str] = None) -> str:
	"""
	Вызов LLM через официальный Python SDK GigaChat.
	Ожидается переменная окружения GIGACHAT_CREDENTIALS (авторизационные данные).
	Можно настроить модель через GIGACHAT_MODEL (по умолчанию GigaChat-Pro).
	"""
	if not GIGACHAT_AVAILABLE:
		return f"[LLM SDK NOT INSTALLED]\nУстановите: pip install gigachat\n\n{prompt[:200]}..."

	credentials = api_key or os.getenv("GIGACHAT_CREDENTIALS")
	if not credentials:
		return f"[LLM OUTPUT MOCK]\nНе найден GIGACHAT_CREDENTIALS\n\n{prompt[:200]}..."

	try:
		# Инициализация клиента GigaChat
		with GigaChat(
			credentials=credentials,
			model=os.getenv("GIGACHAT_MODEL", "GigaChat-2"),
			verify_ssl_certs=False,  # Для тестовой среды
			scope="GIGACHAT_API_PERS"  # Или GIGACHAT_API_CORP для корпоративной версии
		) as giga:
			# Формирование запроса
			response = giga.chat(
				Chat(
					messages=[
						Messages(
							role=MessagesRole.SYSTEM,
							content="Ты помощник по коррекции OCR и Markdown."
						),
						Messages(
							role=MessagesRole.USER,
							content=prompt
						),
					],
					temperature=float(os.getenv("GIGACHAT_TEMPERATURE", "0.3")),
				)
			)
			return response.choices[0].message.content
	except Exception as e:
		return f"[LLM ERROR] {e}\n\n[LLM OUTPUT MOCK]\n{prompt[:200]}..."


def gemini_complete(prompt: str, api_key: Optional[str] = None, model_name: Optional[str] = None) -> str:
	"""
	Вызов LLM Gemini через официальный пакет google-generativeai.
	Ожидается переменная окружения GEMINI_API_KEY. Модель настраивается через GEMINI_MODEL.
	"""
	if not GEMINI_AVAILABLE:
		return f"[LLM SDK NOT INSTALLED]\nУстановите: pip install google-generativeai\n\n{prompt[:200]}..."

	key = api_key or os.getenv("GEMINI_API_KEY")
	if not key:
		return f"[LLM OUTPUT MOCK]\nНе найден GEMINI_API_KEY\n\n{prompt[:200]}..."

	try:
		genai.configure(api_key=key)
		model_id = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
		model = genai.GenerativeModel(model_id)
		resp = model.generate_content(prompt)
		# У API может быть разная структура, стараемся безопасно извлечь текст
		return getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else str(resp))
	except Exception as e:
		return f"[LLM ERROR] {e}\n\n[LLM OUTPUT MOCK]\n{prompt[:200]}..."


def prompt_strategy_A(raw_text: str) -> str:
	return (
		"Исправь орфографические ошибки в тексте, полученном после OCR. "
		f"Верни исправленный текст без комментариев. Текст:\n\n{raw_text}"
	)


def prompt_strategy_B(raw_text: str) -> str:
	return (
		"Преобразуй следующий текст в формат Markdown. Выдели заголовки через #, "
		"списки через -, жирный шрифт через **. Верни только валидный Markdown. "
		f"Исходный текст:\n\n{raw_text}"
	)


def prompt_strategy_C(raw_text: str) -> str:
	return (
		"Ты — редактор. Твоя задача — восстановить поврежденный текст документа. "
		"Исправь ошибки OCR, опираясь на контекст. Восстанови логическую структуру "
		"(заголовки, абзацы). Верни ТОЛЬКО валидный Markdown код. Текст:\n\n"
		f"{raw_text}"
	)


def run_ocr(image_path: str, lang: str = "rus") -> str:
	raw = get_raw_text(image_path, lang=lang)
	return normalize_whitespace(raw)


def run_llm_correction(text: str, strategy: str = "A", llm: str = "gigachat") -> str:
	if strategy == "A":
		prompt = prompt_strategy_A(text)
	elif strategy == "B":
		prompt = prompt_strategy_B(text)
	else:
		prompt = prompt_strategy_C(text)
	llm_choice = (llm or os.getenv("LLM_PROVIDER", "gigachat")).lower()
	if llm_choice == "gemini":
		return gemini_complete(prompt, api_key=os.getenv("GEMINI_API_KEY"), model_name=os.getenv("GEMINI_MODEL"))
	else:
		return gigachat_complete(prompt, api_key=os.getenv("GIGACHAT_CREDENTIALS"))


def main_cli():
	parser = argparse.ArgumentParser(description="OCR → LLM коррекция → Markdown")
	parser.add_argument("--ocr-only", action="store_true", help="Только OCR и вывод сырого текста")
	parser.add_argument("--run-bot", action="store_true", help="Запуск Telegram-бота")
	parser.add_argument("--image", type=str, help="Путь к изображению для OCR")
	parser.add_argument("--lang", type=str, default="rus", help="Язык Tesseract (rus/eng)")
	parser.add_argument("--strategy", type=str, default="A", choices=["A", "B", "C"], help="Промпт-стратегия")
	parser.add_argument("--llm", type=str, default=os.getenv("LLM_PROVIDER", "gigachat"), choices=["gigachat", "gemini"], help="Выбор LLM-провайдера")
	args = parser.parse_args()

	if args.ocr_only:
		if not args.image:
			raise SystemExit("Укажите --image для OCR")
		raw = run_ocr(args.image, lang=args.lang)
		print("--- RAW OCR OUTPUT ---")
		print(raw)
		return

	if args.run_bot:
		run_telegram_bot()
		return

	if args.image:
		raw = run_ocr(args.image, lang=args.lang)
		print("--- RAW OCR OUTPUT ---")
		print(raw)
		print("\n--- LLM CORRECTED ---")
		corrected = run_llm_correction(raw, strategy=args.strategy, llm=args.llm)
		print(corrected)
	else:
		parser.print_help()


def run_telegram_bot():
	# Полноценная реализация на python-telegram-bot v21
	from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
	from telegram.constants import ParseMode
	from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

	bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
	if not bot_token:
		raise SystemExit("Не найден TELEGRAM_BOT_TOKEN в переменных окружения")

	default_lang = os.getenv("OCR_LANG", "rus")

	async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
		keyboard = [
			[
				InlineKeyboardButton("Стратегия A", callback_data="set_strategy:A"),
				InlineKeyboardButton("Стратегия B", callback_data="set_strategy:B"),
				InlineKeyboardButton("Стратегия C", callback_data="set_strategy:C"),
			],
			[
				InlineKeyboardButton("LLM: GigaChat", callback_data="set_llm:gigachat"),
				InlineKeyboardButton("LLM: Gemini", callback_data="set_llm:gemini"),
			]
		]
		await update.message.reply_text(
			"Привет! Пришлите изображение или PDF.\n"
			"Выберите стратегию (по умолчанию A) и LLM:",
			reply_markup=InlineKeyboardMarkup(keyboard),
		)

	async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
		await update.message.reply_text(
			"/start — начать и выбрать стратегию\n"
			"/strategy A|B|C — выбрать стратегию\n"
			"/lang rus|eng — выбрать язык OCR\n"
			"/llm gigachat|gemini — выбрать провайдера LLM\n"
			"Пришлите фото/скан или документ для OCR и коррекции"
		)

	async def set_strategy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
		# /strategy A|B|C
		if not context.args:
			await update.message.reply_text("Укажите стратегию: A, B или C")
			return
		strategy = context.args[0].upper()
		if strategy not in {"A", "B", "C"}:
			await update.message.reply_text("Допустимые значения: A, B, C")
			return
		context.user_data["strategy"] = strategy
		await update.message.reply_text(f"Стратегия установлена: {strategy}")

	async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
		query = update.callback_query
		await query.answer()
		if not query.data:
			return
		if query.data.startswith("set_strategy:"):
			_, val = query.data.split(":", 1)
			strategy = val.upper()
			if strategy in {"A", "B", "C"}:
				context.user_data["strategy"] = strategy
				await query.edit_message_text(f"Стратегия установлена: {strategy}")
		elif query.data.startswith("set_llm:"):
			_, val = query.data.split(":", 1)
			llm = val.lower()
			if llm in {"gigachat", "gemini"}:
				context.user_data["llm"] = llm
				await query.edit_message_text(f"LLM провайдер: {llm}")

	async def set_lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
		# /lang rus|eng
		if not context.args:
			await update.message.reply_text("Укажите язык: rus или eng")
			return
		lang = context.args[0].lower()
		if lang not in {"rus", "eng"}:
			await update.message.reply_text("Допустимые значения: rus, eng")
			return
		context.user_data["lang"] = lang
		await update.message.reply_text(f"Язык OCR установлен: {lang}")

	async def set_llm_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
		# /llm gigachat|gemini
		if not context.args:
			await update.message.reply_text("Укажите LLM: gigachat или gemini")
			return
		llm = context.args[0].lower()
		if llm not in {"gigachat", "gemini"}:
			await update.message.reply_text("Допустимые значения: gigachat, gemini")
			return
		context.user_data["llm"] = llm
		await update.message.reply_text(f"LLM провайдер установлен: {llm}")

	async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
		# Берем фото наилучшего качества
		photo = update.message.photo[-1]
		file = await photo.get_file()
		# Скачиваем во временный файл
		tmp_dir = os.path.join(os.getcwd(), "tmp")
		os.makedirs(tmp_dir, exist_ok=True)
		local_path = os.path.join(tmp_dir, f"{file.file_id}.jpg")
		await file.download_to_drive(local_path)

		lang = context.user_data.get("lang", default_lang)
		await update.message.reply_text(f"Выполняю OCR (язык {lang})...")
		try:
			raw = run_ocr(local_path, lang=lang)
		except Exception as e:
			await update.message.reply_text(f"Ошибка OCR: {e}")
			return

		strategy = context.user_data.get("strategy", "A")
		llm = context.user_data.get("llm", os.getenv("LLM_PROVIDER", "gigachat"))
		await update.message.reply_text(f"Коррекция LLM (стратегия {strategy}, {llm})...")
		corrected = run_llm_correction(raw, strategy=strategy, llm=llm)

		# Отправляем как текст (Markdown, если стратегия B/C)
		parse_mode = ParseMode.MARKDOWN if strategy in {"B", "C"} else None
		await update.message.reply_text(corrected[:4000], parse_mode=parse_mode)

	async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
		# Поддержка изображений/сканов, возможно PDF → пока обрабатываем как неподдерживаемое
		doc = update.message.document
		file_name = doc.file_name or "document"
		mime = doc.mime_type or ""
		file = await doc.get_file()
		tmp_dir = os.path.join(os.getcwd(), "tmp")
		os.makedirs(tmp_dir, exist_ok=True)
		local_path = os.path.join(tmp_dir, f"{file.file_id}_{file_name}")
		await file.download_to_drive(local_path)

		lang = context.user_data.get("lang", default_lang)
		if mime.startswith("image/"):
			await update.message.reply_text("Обнаружено изображение в документе. Выполняю OCR...")
			try:
				raw = run_ocr(local_path, lang=lang)
			except Exception as e:
				await update.message.reply_text(f"Ошибка OCR: {e}")
				return
			strategy = context.user_data.get("strategy", "A")
			llm = context.user_data.get("llm", os.getenv("LLM_PROVIDER", "gigachat"))
			corrected = run_llm_correction(raw, strategy=strategy, llm=llm)
			parse_mode = ParseMode.MARKDOWN if strategy in {"B", "C"} else None
			await update.message.reply_text(corrected[:4000], parse_mode=parse_mode)
		elif mime == "application/pdf" or file_name.lower().endswith(".pdf"):
			await update.message.reply_text("Получен PDF. Пытаюсь извлечь страницы как изображения...")
			# Легкая попытка: если установлен pdf2image и poppler в PATH
			try:
				from pdf2image import convert_from_path
				pages = convert_from_path(local_path, dpi=200)
				if not pages:
					await update.message.reply_text("Не удалось извлечь страницы из PDF.")
					return
				# Обрабатываем первые 3 страницы, чтобы не спамить
				max_pages = min(3, len(pages))
				all_text = []
				for i in range(max_pages):
					img_path = os.path.join(tmp_dir, f"{file.file_id}_page_{i+1}.jpg")
					pages[i].save(img_path, "JPEG")
					try:
						raw = run_ocr(img_path, lang=lang)
						all_text.append(raw)
					except Exception as e:
						all_text.append(f"[Ошибка OCR стр.{i+1}] {e}")
				combined = "\n\n".join(all_text)
				strategy = context.user_data.get("strategy", "A")
				llm = context.user_data.get("llm", os.getenv("LLM_PROVIDER", "gigachat"))
				corrected = run_llm_correction(combined, strategy=strategy, llm=llm)
				parse_mode = ParseMode.MARKDOWN if strategy in {"B", "C"} else None
				await update.message.reply_text(corrected[:4000], parse_mode=parse_mode)
			except Exception:
				await update.message.reply_text(
					"Для PDF требуется установленный poppler и пакет pdf2image. Пока обработка PDF недоступна."
				)
		else:
			await update.message.reply_text("Пока поддерживаются изображения (jpg/png) и PDF при наличии pdf2image.")

	application = Application.builder().token(bot_token).build()

	application.add_handler(CommandHandler("start", start))
	application.add_handler(CommandHandler("help", help_cmd))
	application.add_handler(CommandHandler("strategy", set_strategy_cmd))
	application.add_handler(CommandHandler("lang", set_lang_cmd))
	application.add_handler(CommandHandler("llm", set_llm_cmd))
	application.add_handler(CallbackQueryHandler(buttons))
	application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
	application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

	application.run_polling()


if __name__ == "__main__":
	main_cli()
