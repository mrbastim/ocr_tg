import os
import sys
import asyncio
from typing import Optional

# Добавляем корневую директорию проекта в sys.path для импорта `ocr`
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

# Optional SDKs
try:
    from gigachat import GigaChat
    from gigachat.models import Chat, Messages, MessagesRole
    GIGACHAT_AVAILABLE = True
except Exception:
    GIGACHAT_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram import BaseMiddleware


def gigachat_complete(prompt: str, api_key: Optional[str] = None) -> str:
    if not GIGACHAT_AVAILABLE:
        return f"[LLM SDK NOT INSTALLED]\nInstall: pip install gigachat\n\n{prompt[:200]}..."
    credentials = api_key or os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        return f"[LLM OUTPUT MOCK]\nGIGACHAT_CREDENTIALS missing\n\n{prompt[:200]}..."
    try:
        with GigaChat(
            credentials=credentials,
            model=os.getenv("GIGACHAT_MODEL", "GigaChat-2"),
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
        ) as giga:
            response = giga.chat(
                Chat(
                    messages=[
                        Messages(role=MessagesRole.SYSTEM, content="Ты помощник по коррекции OCR и Markdown."),
                        Messages(role=MessagesRole.USER, content=prompt),
                    ],
                    temperature=float(os.getenv("GIGACHAT_TEMPERATURE", "0.3")),
                )
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"[LLM ERROR] {e}\n\n[LLM OUTPUT MOCK]\n{prompt[:200]}..."


def gemini_complete(prompt: str, api_key: Optional[str] = None, model_name: Optional[str] = None) -> str:
    if not GEMINI_AVAILABLE:
        return f"[LLM SDK NOT INSTALLED]\nInstall: pip install google-generativeai\n\n{prompt[:200]}..."
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return f"[LLM OUTPUT MOCK]\nGEMINI_API_KEY missing\n\n{prompt[:200]}..."
    try:
        genai.configure(api_key=key)
        model_id = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_id)
        resp = model.generate_content(prompt)
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


_user_state: dict[int, dict] = {}


def _get_state(user_id: int) -> dict:
    st = _user_state.get(user_id)
    if not st:
        st = {
            "strategy": "A",
            "lang": os.getenv("OCR_LANG", "rus"),
            "llm": os.getenv("LLM_PROVIDER", "gigachat"),
            "debug": False,
        }
        _user_state[user_id] = st
    return st


def kb_main(user_id: int) -> InlineKeyboardMarkup:
    st = _get_state(user_id)
    strat = st["strategy"]
    llm = st["llm"]
    lang = st["lang"]
    debug = st["debug"]
    def mark(label: str, active: bool) -> str:
        return f"{label}{' ✅' if active else ''}"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=mark("Стратегия A", strat == "A"), callback_data="set_strategy:A"),
                InlineKeyboardButton(text=mark("Стратегия B", strat == "B"), callback_data="set_strategy:B"),
                InlineKeyboardButton(text=mark("Стратегия C", strat == "C"), callback_data="set_strategy:C"),
            ],
            [
                InlineKeyboardButton(text=mark("LLM: GigaChat", llm == "gigachat"), callback_data="set_llm:gigachat"),
                InlineKeyboardButton(text=mark("LLM: Gemini", llm == "gemini"), callback_data="set_llm:gemini"),
            ],
            [
                InlineKeyboardButton(text=mark("Язык: RU", lang == "rus"), callback_data="set_lang:rus"),
                InlineKeyboardButton(text=mark("Язык: EN", lang == "eng"), callback_data="set_lang:eng"),
            ],
            [
                InlineKeyboardButton(text=mark("Debug", debug), callback_data="toggle_debug"),
            ],
            [
                InlineKeyboardButton(text="Обновить меню", callback_data="refresh_menu"),
            ],
        ]
    )


async def cmd_start(message: Message):
    st = _get_state(message.from_user.id)
    header = (
        f"<b>Стратегия:</b> {st['strategy']}\n"
        f"<b>LLM:</b> {st['llm']}\n"
        f"<b>Язык OCR:</b> {st['lang']}\n"
        f"<b>Debug:</b> {'on' if st['debug'] else 'off'}"
    )
    await message.answer(
        header,
        reply_markup=kb_main(message.from_user.id),
        parse_mode=ParseMode.HTML,
    )


async def cmd_help(message: Message):
    await message.answer(
        "/start — начать и выбрать стратегию\n"
        "/strategy A|B|C — выбрать стратегию\n"
        "/lang rus|eng — выбрать язык OCR\n"
        "/llm gigachat|gemini — выбрать провайдера LLM\n"
        "/debug on|off — включить/выключить вывод OCR и LLM\n"
        "Пришлите фото/скан или документ для OCR и коррекции"
    )


async def cmd_strategy(message: Message):
    args = (message.text or "").split()
    if len(args) < 2:
        await message.answer("Укажите стратегию: A, B или C")
        return
    strategy = args[1].upper()
    if strategy not in {"A", "B", "C"}:
        await message.answer("Допустимые значения: A, B, C")
        return
    st = _get_state(message.from_user.id)
    st["strategy"] = strategy
    await message.answer(f"Стратегия установлена: {strategy}", reply_markup=kb_main(message.from_user.id))


async def cmd_lang(message: Message):
    args = (message.text or "").split()
    if len(args) < 2:
        await message.answer("Укажите язык: rus или eng")
        return
    lang = args[1].lower()
    if lang not in {"rus", "eng"}:
        await message.answer("Допустимые значения: rus, eng")
        return
    st = _get_state(message.from_user.id)
    st["lang"] = lang
    await message.answer(
        f"Язык OCR установлен: {lang}",
        reply_markup=kb_main(message.from_user.id)
    )


async def cmd_llm(message: Message):
    args = (message.text or "").split()
    if len(args) < 2:
        await message.answer("Укажите LLM: gigachat или gemini")
        return
    llm = args[1].lower()
    if llm not in {"gigachat", "gemini"}:
        await message.answer("Допустимые значения: gigachat, gemini")
        return
    st = _get_state(message.from_user.id)
    st["llm"] = llm
    await message.answer(f"LLM провайдер установлен: {llm}", reply_markup=kb_main(message.from_user.id))


async def cmd_debug(message: Message):
    args = (message.text or "").split()
    if len(args) < 2 or args[1].lower() not in {"on", "off"}:
        await message.answer("Использование: /debug on|off")
        return
    st = _get_state(message.from_user.id)
    st["debug"] = args[1].lower() == "on"
    await message.answer(
        f"Debug: {'on' if st['debug'] else 'off'}",
        reply_markup=kb_main(message.from_user.id)
    )


async def on_btn(query: CallbackQuery):
    data = query.data or ""
    st = _get_state(query.from_user.id)
    edited = False
    if data.startswith("set_strategy:"):
        _, val = data.split(":", 1)
        strategy = val.upper()
        if strategy in {"A", "B", "C"}:
            st["strategy"] = strategy
            edited = True
    elif data.startswith("set_llm:"):
        _, val = data.split(":", 1)
        llm = val.lower()
        if llm in {"gigachat", "gemini"}:
            st["llm"] = llm
            edited = True
    elif data.startswith("set_lang:"):
        _, val = data.split(":", 1)
        if val in {"rus", "eng"}:
            st["lang"] = val
            edited = True
    elif data == "toggle_debug":
        st["debug"] = not st["debug"]
        edited = True
    elif data == "refresh_menu":
        edited = True
    if edited:
        header = (
            f"<b>Стратегия:</b> {st['strategy']}\n"
            f"<b>LLM:</b> {st['llm']}\n"
            f"<b>Язык OCR:</b> {st['lang']}\n"
            f"<b>Debug:</b> {'on' if st['debug'] else 'off'}"
        )
        await query.message.edit_text(
            header,
            reply_markup=kb_main(query.from_user.id),
            parse_mode=ParseMode.HTML,
        )
    await query.answer()


async def on_photo(message: Message):
    photo = message.photo[-1]
    file = await message.bot.get_file(photo.file_id)
    tmp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    local_path = os.path.join(tmp_dir, f"{photo.file_id}.jpg")
    await message.bot.download_file(file.file_path, local_path)

    st = _get_state(message.from_user.id)
    lang = st['lang']
    await message.answer(f"Выполняю OCR (язык {lang})...")
    try:
        raw = run_ocr(local_path, lang=lang)
    except Exception as e:
        await message.answer(f"Ошибка OCR: {e}")
        return

    strategy = st['strategy']
    llm = st['llm']
    await message.answer(f"Коррекция LLM (стратегия {strategy}, {llm})...")
    corrected = run_llm_correction(raw, strategy=strategy, llm=llm)

    if st["debug"]:
        def html_escape(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        ocr_part = html_escape(raw)[:3500]
        await message.answer(f"<b>OCR ({lang})</b>\n<pre>{ocr_part}</pre>", parse_mode=ParseMode.HTML)
        parse_mode = ParseMode.MARKDOWN if strategy in {"B", "C"} else None
        await message.answer(corrected[:4000], parse_mode=parse_mode)
    else:
        parse_mode = ParseMode.MARKDOWN if strategy in {"B", "C"} else None
        await message.answer(corrected[:4000], parse_mode=parse_mode)


async def on_document(message: Message):
    doc = message.document
    file_name = doc.file_name or "document"
    mime = doc.mime_type or ""
    file = await message.bot.get_file(doc.file_id)
    tmp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    local_path = os.path.join(tmp_dir, f"{doc.file_id}_{file_name}")
    await message.bot.download_file(file.file_path, local_path)

    st = _get_state(message.from_user.id)
    lang = st['lang']
    if mime.startswith("image/"):
        await message.answer("Обнаружено изображение в документе. Выполняю OCR...")
        try:
            raw = run_ocr(local_path, lang=lang)
        except Exception as e:
            await message.answer(f"Ошибка OCR: {e}")
            return
        strategy = st['strategy']
        llm = st['llm']
        corrected = run_llm_correction(raw, strategy=strategy, llm=llm)
        if st["debug"]:
            def html_escape(s: str) -> str:
                return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            ocr_part = html_escape(raw)[:3500]
            await message.answer(f"<b>OCR ({lang})</b>\n<pre>{ocr_part}</pre>", parse_mode=ParseMode.HTML)
            parse_mode = ParseMode.MARKDOWN if strategy in {"B", "C"} else None
            await message.answer(corrected[:4000], parse_mode=parse_mode)
        else:
            parse_mode = ParseMode.MARKDOWN if strategy in {"B", "C"} else None
            await message.answer(corrected[:4000], parse_mode=parse_mode)
    elif mime == "application/pdf" or file_name.lower().endswith(".pdf"):
        await message.answer("Получен PDF. Пытаюсь извлечь страницы как изображения...")
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(local_path, dpi=200)
            if not pages:
                await message.answer("Не удалось извлечь страницы из PDF.")
                return
            max_pages = min(3, len(pages))
            all_text = []
            for i in range(max_pages):
                img_path = os.path.join(tmp_dir, f"{doc.file_id}_page_{i+1}.jpg")
                pages[i].save(img_path, "JPEG")
                try:
                    raw = run_ocr(img_path, lang=lang)
                    all_text.append(raw)
                except Exception as e:
                    all_text.append(f"[Ошибка OCR стр.{i+1}] {e}")
            combined = "\n\n".join(all_text)
            strategy = st['strategy']
            llm = st['llm']
            corrected = run_llm_correction(combined, strategy=strategy, llm=llm)
            if st["debug"]:
                def html_escape(s: str) -> str:
                    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                ocr_part = html_escape(combined)[:3500]
                await message.answer(f"<b>OCR ({lang})</b>\n<pre>{ocr_part}</pre>", parse_mode=ParseMode.HTML)
                parse_mode = ParseMode.MARKDOWN if strategy in {"B", "C"} else None
                await message.answer(corrected[:4000], parse_mode=parse_mode)
            else:
                parse_mode = ParseMode.MARKDOWN if strategy in {"B", "C"} else None
                await message.answer(corrected[:4000], parse_mode=parse_mode)
        except Exception:
            await message.answer("Для PDF требуется poppler и пакет pdf2image. Пока обработка PDF недоступна.")
    else:
        await message.answer("Пока поддерживаются изображения (jpg/png) и PDF при наличии pdf2image.")


async def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Не найден TELEGRAM_BOT_TOKEN в переменных окружения")

    bot = Bot(token=token)
    dp = Dispatcher()

    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_help, Command("help"))
    dp.message.register(cmd_strategy, F.text.startswith("/strategy"))
    dp.message.register(cmd_lang, F.text.startswith("/lang"))
    dp.message.register(cmd_llm, F.text.startswith("/llm"))
    dp.message.register(cmd_debug, F.text.startswith("/debug"))

    dp.callback_query.register(on_btn)
    dp.message.register(on_photo, F.photo)
    dp.message.register(on_document, F.document)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
