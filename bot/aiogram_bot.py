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

# HTTP интеграция с внешним сервером (JWT + endpoints)
import json
import urllib.request
import urllib.error
import time
from pathlib import Path
import re

API_BASE = os.getenv("AI_API_BASE") or os.getenv("GEMINI_API_BASE")  # например: https://your.server/api
API_USERNAME = os.getenv("AI_API_USER") or os.getenv("GEMINI_API_USER")
API_PASSWORD = os.getenv("AI_API_PASS") or os.getenv("GEMINI_API_PASS")
API_JWT: str | None = None
API_DEBUG = os.getenv("AI_API_DEBUG", "0").lower() not in {"0", "false", "off"}
API_LOG_DIR = Path(os.getenv("AI_API_LOG_DIR", os.path.join(os.getcwd(), "tmp")))
API_LOG_FILE = API_LOG_DIR / "api_debug.log"

def _api_log(event: str, **fields):
    if not API_DEBUG:
        return
    try:
        API_LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        parts = [f"[{ts}] {event}"]
        for k, v in fields.items():
            if k.lower() == 'authorization':
                continue
            text = str(v)
            if len(text) > 1500:
                text = text[:1500] + ' …<truncated>'
            parts.append(f"{k}={text}")
        with open(API_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(' | '.join(parts) + '\n')
    except Exception:
        pass

def api_login() -> bool:
    """Логин на внешний сервер: отправляем JSON {username, password}, извлекаем JWT из разных возможных ключей."""
    global API_JWT
    if not API_BASE or not API_USERNAME or not API_PASSWORD:
        _api_log('login_skip', reason='missing_credentials', base=API_BASE, user=API_USERNAME)
        return False
    url = f"{API_BASE}/login"
    payload_dict = {"username": API_USERNAME, "password": API_PASSWORD}
    payload = json.dumps(payload_dict).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    _api_log('login_request', url=url, body=payload.decode('utf-8'))
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body_raw = resp.read().decode("utf-8")
            _api_log('login_response_raw', status=getattr(resp, 'status', None), body=body_raw)
            try:
                data = json.loads(body_raw)
            except Exception as parse_err:
                _api_log('login_parse_error', error=parse_err)
                return False
            token_candidates = []
            if isinstance(data, dict):
                for k in ["token", "jwt", "access", "access_token", "auth", "bearer", "bearerToken"]:
                    v = data.get(k)
                    if isinstance(v, str):
                        token_candidates.append(v)
                for k, v in data.items():
                    if isinstance(v, dict):
                        for kk in ["token", "jwt", "access", "access_token"]:
                            vv = v.get(kk)
                            if isinstance(vv, str):
                                token_candidates.append(vv)
            # Regex поиск JWT (header.payload.signature)
            jwt_regex = re.compile(r"^[A-Za-z0-9-_]+=*\.[A-Za-z0-9-_]+=*\.[A-Za-z0-9-_]+=*$")
            for s in re.findall(r"[A-Za-z0-9\-_=]+\.[A-Za-z0-9\-_=]+\.[A-Za-z0-9\-_=]+", body_raw):
                if jwt_regex.match(s):
                    token_candidates.append(s)
            API_JWT = next(iter(token_candidates), None)
            _api_log('login_ok', token_present=API_JWT is not None, found=len(token_candidates))
            return API_JWT is not None
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode('utf-8')
        except Exception:
            err_body = str(e)
        _api_log('login_http_error', code=e.code, body=err_body[:1500])
        return False
    except Exception as e:
        _api_log('login_error', error=e)
        return False

def api_ask_text(prompt: str) -> str:
    if not API_BASE:
        _api_log('ask_skip', reason='no_base')
        return "[API NOT CONFIGURED] Set AI_API_BASE, AI_API_USER, AI_API_PASS"
    if not API_JWT and not api_login():
        _api_log('ask_auth_failed')
        return "[API AUTH FAILED] Check AI_API_USER/AI_API_PASS"
    url = f"{API_BASE}/user/ai/text"
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_JWT}",
    }
    _api_log('ask_request', url=url, body=payload.decode('utf-8'))
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body_raw = resp.read().decode("utf-8")
            _api_log('ask_raw_response', status=getattr(resp, 'status', None), body=body_raw)
            try:
                data = json.loads(body_raw)
            except Exception as parse_err:
                _api_log('ask_parse_error', error=parse_err)
                return body_raw[:4000]
            # Приоритетные пути извлечения текста
            if isinstance(data, dict):
                # 1) data.text
                if isinstance(data.get("data"), dict) and isinstance(data["data"].get("text"), str):
                    return data["data"]["text"]
                # 2) text на верхнем уровне
                if isinstance(data.get("text"), str):
                    return data["text"]
                # 3) success.text
                if isinstance(data.get("success"), dict) and isinstance(data["success"].get("text"), str):
                    return data["success"]["text"]
                # 4) success.data.text
                if isinstance(data.get("success"), dict):
                    sd = data["success"].get("data")
                    if isinstance(sd, dict) and isinstance(sd.get("text"), str):
                        return sd["text"]
            # Не удалось найти — вернём JSON целиком (для отладки)
            return json.dumps(data, ensure_ascii=False)
    except urllib.error.HTTPError as e:
        try:
            err = e.read().decode("utf-8")
        except Exception:
            err = str(e)
        _api_log('ask_http_error', code=e.code, body=err)
        return f"[API ERROR] {e.code}: {err[:2000]}"
    except Exception as e:
        _api_log('ask_error', error=e)
        return f"[API ERROR] {e}"

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram import BaseMiddleware
from aiogram.exceptions import TelegramBadRequest


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


def external_api_complete(prompt: str) -> str:
    """Вызов внешнего сервера: POST /user/ai/text с JWT."""
    return api_ask_text(prompt)


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
    """Формирует промпт по стратегии и отправляет в выбранный LLM.
    Gemini теперь всегда через внешний сервер при наличии API_BASE (сервер сам общается с Gemini).
    Чтобы принудительно использовать локальный SDK Gemini, установите GEMINI_LOCAL=1.
    """
    if strategy == "A":
        prompt = prompt_strategy_A(text)
    elif strategy == "B":
        prompt = prompt_strategy_B(text)
    else:
        prompt = prompt_strategy_C(text)
    llm_choice = (llm or os.getenv("LLM_PROVIDER", "gigachat")).lower()
    force_local_gemini = os.getenv("GEMINI_LOCAL", "0").lower() in {"1", "true", "yes"}
    if llm_choice == "gemini":
        # Если есть внешний сервер — используем его (JWT); так выполняется требование работы через сервер.
        if API_BASE and not force_local_gemini:
            return external_api_complete(prompt)
        # Иначе локальный SDK, если есть ключ
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            return gemini_complete(prompt, api_key=gemini_key, model_name=os.getenv("GEMINI_MODEL"))
        if API_BASE:  # fallback ещё раз, вдруг force_local_gemini был включен но ключ отсутствует
            return external_api_complete(prompt)
        return "[GEMINI CONFIG MISSING] Set GEMINI_API_KEY or AI_API_BASE/AI_API_USER/AI_API_PASS"
    elif llm_choice in {"api", "gemini_api", "external"}:
        return external_api_complete(prompt)
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
                InlineKeyboardButton(text=mark("LLM: External API", llm in {"api", "gemini_api", "external"}), callback_data="set_llm:api"),
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
        "/llm gigachat|gemini|api — выбрать провайдера LLM (api = внешний сервер)\n"
        "/debug on|off — включить/выключить вывод OCR и LLM\n"
        "/apilog — последние строки лога интеграции (AI_API_DEBUG=1)\n"
        "/testlogin — выполнить попытку логина и показать сырой ответ (AI_API_DEBUG рекомендуется)\n"
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
        await message.answer("Укажите LLM: gigachat | gemini | api")
        return
    llm = args[1].lower()
    if llm not in {"gigachat", "gemini", "api"}:
        await message.answer("Допустимые значения: gigachat, gemini, api")
        return
    st = _get_state(message.from_user.id)
    st["llm"] = llm
    await message.answer(f"LLM провайдер установлен: {llm}", reply_markup=kb_main(message.from_user.id))

async def cmd_testlogin(message: Message):
    ok = api_login()
    if ok:
        await message.answer("Логин успешен: токен получен.")
    else:
        # Попытаемся показать последние строки лога
        if API_DEBUG and API_LOG_FILE.exists():
            try:
                with open(API_LOG_FILE, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-20:]
                text = ''.join(lines)
                esc = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                await message.answer(f"<b>Логин неудачен</b>\n<pre>{esc}</pre>", parse_mode=ParseMode.HTML)
            except Exception as e:
                await message.answer(f"Логин неудачен. Ошибка чтения лога: {e}")
        else:
            await message.answer("Логин неудачен. Включите AI_API_DEBUG=1 для деталей.")

async def cmd_apilog(message: Message):
    if not API_DEBUG:
        await message.answer("Логирование отключено. Установите AI_API_DEBUG=1")
        return
    if not API_LOG_FILE.exists():
        await message.answer("Файл лога отсутствует")
        return
    try:
        with open(API_LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()[-40:]
        text = ''.join(lines)
        esc = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        await message.answer(f"<b>API LOG (последние)</b>\n<pre>{esc}</pre>", parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Ошибка чтения лога: {e}")


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
        if llm in {"gigachat", "gemini", "api"}:
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

    def safe_send(text: str):
        pm = ParseMode.MARKDOWN if strategy in {"B", "C"} else None
        try:
            return message.answer(text[:4000], parse_mode=pm)
        except TelegramBadRequest:
            # Fallback: отправляем как обычный текст без Markdown, минимально экранируем обратные кавычки
            return message.answer(text[:4000])

    if st["debug"]:
        def html_escape(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        ocr_part = html_escape(raw)[:3500]
        await message.answer(f"<b>OCR ({lang})</b>\n<pre>{ocr_part}</pre>", parse_mode=ParseMode.HTML)
        await safe_send(corrected)
    else:
        await safe_send(corrected)


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
        def safe_send(text: str):
            pm = ParseMode.MARKDOWN if strategy in {"B", "C"} else None
            try:
                return message.answer(text[:4000], parse_mode=pm)
            except TelegramBadRequest:
                return message.answer(text[:4000])
        if st["debug"]:
            def html_escape(s: str) -> str:
                return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            ocr_part = html_escape(raw)[:3500]
            await message.answer(f"<b>OCR ({lang})</b>\n<pre>{ocr_part}</pre>", parse_mode=ParseMode.HTML)
            await safe_send(corrected)
        else:
            await safe_send(corrected)
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
            def safe_send(text: str):
                pm = ParseMode.MARKDOWN if strategy in {"B", "C"} else None
                try:
                    return message.answer(text[:4000], parse_mode=pm)
                except TelegramBadRequest:
                    return message.answer(text[:4000])
            if st["debug"]:
                def html_escape(s: str) -> str:
                    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                ocr_part = html_escape(combined)[:3500]
                await message.answer(f"<b>OCR ({lang})</b>\n<pre>{ocr_part}</pre>", parse_mode=ParseMode.HTML)
                await safe_send(corrected)
            else:
                await safe_send(corrected)
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
    dp.message.register(cmd_apilog, F.text.startswith("/apilog"))
    dp.message.register(cmd_testlogin, F.text.startswith("/testlogin"))

    dp.callback_query.register(on_btn)
    dp.message.register(on_photo, F.photo)
    dp.message.register(on_document, F.document)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
