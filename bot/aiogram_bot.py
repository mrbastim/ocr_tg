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
import json as _json
import logging
logger = logging.getLogger(__name__)

API_BASE = os.getenv("AI_API_BASE") or os.getenv("GEMINI_API_BASE")  # например: https://your.server/api
API_BASE_PATH = os.getenv("AI_API_BASE_PATH", "/api")  # базовый префикс, по умолчанию "/api"
API_USERNAME = os.getenv("AI_API_USER") or os.getenv("GEMINI_API_USER")
API_PASSWORD = os.getenv("AI_API_PASS") or os.getenv("GEMINI_API_PASS")
API_JWT: str | None = None
# Персональные JWT на пользователя Telegram (tg_id -> token)
API_JWT_BY_USER: dict[int, str] = {}
API_JWT_TS_BY_USER: dict[int, float] = {}  # unix timestamp получения токена
API_DEBUG = os.getenv("AI_API_DEBUG", "0").lower() not in {"0", "false", "off"}
API_LOG_DIR = Path(os.getenv("AI_API_LOG_DIR", os.path.join(os.getcwd(), "tmp")))
API_LOG_FILE = API_LOG_DIR / "api_debug.log"

def _api_log(event: str, **fields):
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
        line = ' | '.join(parts)
        print(line)
        if API_DEBUG:
            with open(API_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
    except Exception:
        pass

def _api_url(path: str) -> str:
    base = API_BASE or ""
    prefix = API_BASE_PATH or ""
    # Нормализуем слеши
    if prefix and not prefix.startswith("/"):
        prefix = "/" + prefix
    if base.endswith("/"):
        base = base[:-1]
    if path and not path.startswith("/"):
        path = "/" + path
    return f"{base}{prefix}{path}"

def api_set_key(tg_id: int, username: str, provider: str, key: str) -> bool:
    """Отправить ключ Gemini на сервер: POST /user/ai/key {api_key}."""
    # На сервер отправляем только ключи для Gemini
    if provider != "gemini":
        _api_log('set_key_skip_local_only', provider=provider)
        return False
    if not API_BASE:
        _api_log('set_key_skip', reason='no_base')
        return False
    jwt = API_JWT_BY_USER.get(tg_id)
    ts = API_JWT_TS_BY_USER.get(tg_id, 0)
    if jwt and (time.time() - ts > 3600):
        _api_log('set_key_token_expired', tg_id=tg_id)
        jwt = None
    if not jwt:
        # обеспечим регистрацию/логин
        if not api_login(tg_id, username):
            api_register(tg_id, username)
            if not api_login(tg_id, username):
                _api_log('set_key_auth_failed', tg_id=tg_id)
                return False
        jwt = API_JWT_BY_USER.get(tg_id)
    url = _api_url("/user/ai/key")
    payload = json.dumps({"api_key": key}).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {jwt}"}
    _api_log('set_key_request', url=url, body=payload.decode('utf-8'))
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body_raw = resp.read().decode("utf-8")
            _api_log('set_key_response', status=getattr(resp, 'status', None), body=body_raw)
            return True
    except urllib.error.HTTPError as e:
        if e.code == 401:
            _api_log('set_key_401_retry', tg_id=tg_id)
            if api_login(tg_id, username):
                jwt2 = API_JWT_BY_USER.get(tg_id)
                headers["Authorization"] = f"Bearer {jwt2}"
                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=15) as resp2:
                        body_raw = resp2.read().decode("utf-8")
                        _api_log('set_key_response_retry', status=getattr(resp2, 'status', None), body=body_raw)
                        return True
                except Exception as e2:
                    _api_log('set_key_retry_error', error=e2)
        try:
            err_body = e.read().decode('utf-8')
        except Exception:
            err_body = str(e)
        _api_log('set_key_http_error', code=e.code, body=err_body[:1500])
        return False
    except Exception as e:
        _api_log('set_key_error', error=e)
        return False

def api_clear_key(tg_id: int, username: str, provider: str) -> bool:
    """Удалить ключ Gemini у пользователя: DELETE /user/ai/key без тела."""
    # На сервере храним только ключи для Gemini
    if provider != "gemini":
        _api_log('clear_key_skip_local_only', provider=provider)
        return False
    if not API_BASE:
        _api_log('clear_key_skip', reason='no_base')
        return False
    jwt = API_JWT_BY_USER.get(tg_id)
    ts = API_JWT_TS_BY_USER.get(tg_id, 0)
    if jwt and (time.time() - ts > 3600):
        _api_log('clear_key_token_expired', tg_id=tg_id)
        jwt = None
    if not jwt:
        if not api_login(tg_id, username):
            api_register(tg_id, username)
            if not api_login(tg_id, username):
                _api_log('clear_key_auth_failed', tg_id=tg_id)
                return False
        jwt = API_JWT_BY_USER.get(tg_id)
    url = _api_url("/user/ai/key")
    headers = {"Authorization": f"Bearer {jwt}"}
    _api_log('clear_key_request', url=url)
    req = urllib.request.Request(url, headers=headers, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body_raw = resp.read().decode("utf-8")
            _api_log('clear_key_response', status=getattr(resp, 'status', None), body=body_raw)
            return True
    except urllib.error.HTTPError as e:
        if e.code == 401:
            _api_log('clear_key_401_retry', tg_id=tg_id)
            if api_login(tg_id, username):
                jwt2 = API_JWT_BY_USER.get(tg_id)
                headers["Authorization"] = f"Bearer {jwt2}"
                req = urllib.request.Request(url, headers=headers, method="DELETE")
                try:
                    with urllib.request.urlopen(req, timeout=15) as resp2:
                        body_raw = resp2.read().decode("utf-8")
                        _api_log('clear_key_response_retry', status=getattr(resp2, 'status', None), body=body_raw)
                        return True
                except Exception as e2:
                    _api_log('clear_key_retry_error', error=e2)
        try:
            err_body = e.read().decode('utf-8')
        except Exception:
            err_body = str(e)
        _api_log('clear_key_http_error', code=e.code, body=err_body[:1500])
        return False
    except Exception as e:
        _api_log('clear_key_error', error=e)
        return False

def api_key_status(tg_id: int, username: str) -> dict:
    """Проверить наличие ключей: GET /user/ai/key -> {gigachat: bool, gemini: bool} (ожидаемый формат)."""
    if not API_BASE:
        _api_log('key_status_skip', reason='no_base')
        return {}
    jwt = API_JWT_BY_USER.get(tg_id)
    ts = API_JWT_TS_BY_USER.get(tg_id, 0)
    if jwt and (time.time() - ts > 3600):
        _api_log('key_status_token_expired', tg_id=tg_id)
        jwt = None
    if not jwt:
        if not api_login(tg_id, username):
            api_register(tg_id, username)
            if not api_login(tg_id, username):
                _api_log('key_status_auth_failed', tg_id=tg_id)
                return {}
        jwt = API_JWT_BY_USER.get(tg_id)
    url = _api_url("/user/ai/key")
    headers = {"Authorization": f"Bearer {jwt}"}
    _api_log('key_status_request', url=url)
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body_raw = resp.read().decode("utf-8")
            _api_log('key_status_response', status=getattr(resp, 'status', None), body=body_raw)
            try:
                data = json.loads(body_raw)
                # Универсальный парсинг: ищем флаги в data/keys/success/data
                result = {}
                if isinstance(data, dict):
                    # прямые поля
                    for k in ("gemini", "gigachat"):
                        if k in data and isinstance(data[k], (bool, int)):
                            result[k] = bool(data[k])
                    if isinstance(data.get("has_key"), (bool, int)):
                        result["gemini"] = bool(data["has_key"])  # для простого эндпоинта одного ключа
                    # вложенное data
                    if isinstance(data.get("data"), dict):
                        for k in ("gemini", "gigachat"):
                            v = data["data"].get(k)
                            if isinstance(v, (bool, int)):
                                result[k] = bool(v)
                        if isinstance(data["data"].get("has_key"), (bool, int)):
                            result["gemini"] = bool(data["data"]["has_key"]) 
                    # success.data
                    if isinstance(data.get("success"), dict) and isinstance(data["success"].get("data"), dict):
                        sd = data["success"]["data"]
                        for k in ("gemini", "gigachat"):
                            v = sd.get(k)
                            if isinstance(v, (bool, int)):
                                result[k] = bool(v)
                        if isinstance(sd.get("has_key"), (bool, int)):
                            result["gemini"] = bool(sd["has_key"]) 
                return result
            except Exception as e:
                _api_log('key_status_parse_error', error=e)
                return {}
    except urllib.error.HTTPError as e:
        if e.code == 401:
            _api_log('key_status_401_retry', tg_id=tg_id)
            if api_login(tg_id, username):
                jwt2 = API_JWT_BY_USER.get(tg_id)
                headers["Authorization"] = f"Bearer {jwt2}"
                req = urllib.request.Request(url, headers=headers, method="GET")
                try:
                    with urllib.request.urlopen(req, timeout=15) as resp2:
                        body_raw = resp2.read().decode("utf-8")
                        _api_log('key_status_response_retry', status=getattr(resp2, 'status', None), body=body_raw)
                        try:
                            data = json.loads(body_raw)
                            return data if isinstance(data, dict) else {}
                        except Exception:
                            return {}
                except Exception as e2:
                    _api_log('key_status_retry_error', error=e2)
        try:
            err_body = e.read().decode('utf-8')
        except Exception:
            err_body = str(e)
        _api_log('key_status_http_error', code=e.code, body=err_body[:1500])
        return {}
    except Exception as e:
        _api_log('key_status_error', error=e)
        return {}

def api_register(tg_id: int, username: str) -> bool:
    """Регистрация пользователя: POST /register {tg_id, username}."""
    if not API_BASE:
        return False
    url = _api_url("/register")
    payload = json.dumps({"tg_id": tg_id, "username": username}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    _api_log('register_request', url=url, body=payload.decode('utf-8'))
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body_raw = resp.read().decode("utf-8")
            _api_log('register_response', status=getattr(resp, 'status', None), body=body_raw)
            # Успешная регистрация не возвращает токен — просто True
            return True
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode('utf-8')
        except Exception:
            err_body = str(e)
        _api_log('register_http_error', code=e.code, body=err_body[:1500])
        return False
    except Exception as e:
        _api_log('register_error', error=e)
        return False

def api_login(tg_id: int, username: str) -> bool:
    """Логин на внешний сервер: отправляем JSON {username, password}, извлекаем JWT из разных возможных ключей."""
    global API_JWT
    if not API_BASE:
        _api_log('login_skip', reason='no_base')
        return False
    url = _api_url("/login")
    # Согласно swagger: LoginRequest {tg_id, username}
    payload_dict = {"tg_id": tg_id, "username": username}
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
            if API_JWT:
                API_JWT_BY_USER[tg_id] = API_JWT
                API_JWT_TS_BY_USER[tg_id] = time.time()
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

def api_ask_text(prompt: str, tg_id: int, username: str) -> str:
    if not API_BASE:
        _api_log('ask_skip', reason='no_base')
        return "[API NOT CONFIGURED] Set AI_API_BASE, AI_API_USER, AI_API_PASS"
    jwt = API_JWT_BY_USER.get(tg_id)
    # Проверка валидности 1 час
    ts = API_JWT_TS_BY_USER.get(tg_id, 0)
    if jwt and (time.time() - ts > 3600):
        _api_log('token_expired', tg_id=tg_id)
        jwt = None
    if not jwt:
        # Пытаемся логиниться, если 401 — пробуем регистрацию и снова логин
        if not api_login(tg_id, username):
            # Попробуем регистрацию
            api_register(tg_id, username)
            if not api_login(tg_id, username):
                _api_log('ask_auth_failed', tg_id=tg_id)
                return "[API AUTH FAILED]"
        jwt = API_JWT_BY_USER.get(tg_id)
    # если всё ещё нет токена
    if not jwt:
        return "[API AUTH FAILED]"
    url = _api_url("/user/ai/text")
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {jwt}",
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
        if e.code == 401:
            # Пробуем перелогиниться и повторить один раз
            _api_log('ask_401_retry', tg_id=tg_id)
            if api_login(tg_id, username):
                jwt = API_JWT_BY_USER.get(tg_id)
                headers["Authorization"] = f"Bearer {jwt}"
                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp2:
                        body_raw = resp2.read().decode("utf-8")
                        _api_log('ask_raw_response_retry', status=getattr(resp2, 'status', None), body=body_raw)
                        try:
                            data = json.loads(body_raw)
                        except Exception:
                            return body_raw[:4000]
                        if isinstance(data, dict) and isinstance(data.get("data"), dict) and isinstance(data["data"].get("text"), str):
                            return data["data"]["text"]
                        if isinstance(data, dict) and isinstance(data.get("text"), str):
                            return data["text"]
                        return json.dumps(data, ensure_ascii=False)
                except Exception as e2:
                    _api_log('ask_retry_error', error=e2)
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

# Хранилище пользовательских API-ключей (persist JSON)
USER_KEYS_PATH = os.path.join(os.getcwd(), "tmp", "user_keys.json")

def _load_user_keys() -> dict:
    try:
        with open(USER_KEYS_PATH, 'r', encoding='utf-8') as f:
            return _json.load(f)
    except Exception:
        return {}

def _save_user_keys(data: dict) -> None:
    os.makedirs(os.path.dirname(USER_KEYS_PATH), exist_ok=True)
    with open(USER_KEYS_PATH, 'w', encoding='utf-8') as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)

def _get_user_key(user_id: int, provider: str) -> str | None:
    data = _load_user_keys()
    u = data.get(str(user_id), {})
    return u.get(provider)

def _set_user_key(user_id: int, provider: str, key: str) -> None:
    data = _load_user_keys()
    u = data.get(str(user_id)) or {}
    u[provider] = key
    data[str(user_id)] = u
    _save_user_keys(data)

def _del_user_key(user_id: int, provider: str) -> bool:
    data = _load_user_keys()
    u = data.get(str(user_id))
    if not u or provider not in u:
        return False
    del u[provider]
    data[str(user_id)] = u
    _save_user_keys(data)
    return True


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


def external_api_complete(prompt: str, tg_id: int, username: str) -> str:
    """Вызов внешнего сервера: POST /user/ai/text с JWT."""
    return api_ask_text(prompt, tg_id=tg_id, username=username)

def _ensure_gemini_key(tg_id: int, username: str) -> bool:
    """Проверяет наличие ключа Gemini на сервере.
    Если нет — пытается отправить локальный ключ пользователя на сервер и повторно проверяет.
    """
    status = api_key_status(tg_id, username)
    if bool(status.get("gemini")):
        return True
    # Попробуем подтолкнуть локальный ключ на сервер
    local_key = _get_user_key(tg_id, "gemini")
    if local_key:
        if api_set_key(tg_id, username, "gemini", local_key):
            status2 = api_key_status(tg_id, username)
            return bool(status2.get("gemini"))
    return False


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
    # Оставляем только стратегию C
    prompt = prompt_strategy_C(text)
    llm_choice = (llm or os.getenv("LLM_PROVIDER", "gigachat")).lower()
    force_local_gemini = os.getenv("GEMINI_LOCAL", "0").lower() in {"1", "true", "yes"}
    if llm_choice == "gemini":
        # Если есть внешний сервер — используем его (JWT); так выполняется требование работы через сервер.
        if API_BASE and not force_local_gemini:
            if not _ensure_gemini_key(_current_user_id, _current_username):
                return "[GEMINI API KEY MISSING]\nОтправьте ключ через настройки: Ключ Gemini."
            return external_api_complete(prompt, tg_id=_current_user_id, username=_current_username)
        # Иначе локальный SDK, если есть ключ
        # Сначала берём персональный ключ пользователя, затем системный
        gemini_key = _get_user_key(_current_user_id, "gemini") or os.getenv("GEMINI_API_KEY")
        if gemini_key:
            return gemini_complete(prompt, api_key=gemini_key, model_name=os.getenv("GEMINI_MODEL"))
        if API_BASE:  # fallback ещё раз, вдруг force_local_gemini был включен но ключ отсутствует
            return external_api_complete(prompt)
        return "[GEMINI CONFIG MISSING] Set GEMINI_API_KEY or AI_API_BASE/AI_API_USER/AI_API_PASS"
    elif llm_choice in {"api", "gemini_api", "external"}:
        if not _ensure_gemini_key(_current_user_id, _current_username):
            return "[GEMINI API KEY MISSING]\nОтправьте ключ через настройки: Ключ Gemini."
        return external_api_complete(prompt, tg_id=_current_user_id, username=_current_username)
    else:
        # Сначала берём персональный ключ пользователя, затем системный
        giga_key = _get_user_key(_current_user_id, "gigachat") or os.getenv("GIGACHAT_CREDENTIALS")
        return gigachat_complete(prompt, api_key=giga_key)


_user_state: dict[int, dict] = {}
_current_user_id: int = 0  # заполняется в хендлерах перед вызовами LLM
_current_username: str = ""


def _get_state(user_id: int) -> dict:
    st = _user_state.get(user_id)
    if not st:
        st = {
            "strategy": "C",
            "lang": os.getenv("OCR_LANG", "rus"),
            "llm": os.getenv("LLM_PROVIDER", "gigachat"),
            "debug": False,
            "settings_open": False,
        }
        _user_state[user_id] = st
    return st

def _token_status(user_id: int) -> tuple[bool, int]:
    """Возвращает (валиден, оставшиеся_минуты)."""
    jwt = API_JWT_BY_USER.get(user_id)
    ts = API_JWT_TS_BY_USER.get(user_id, 0)
    if not jwt or not ts:
        return (False, 0)
    age = time.time() - ts
    if age > 3600:
        return (False, 0)
    remain = int((3600 - age) // 60)
    return (True, max(remain, 0))


def kb_main(user_id: int) -> InlineKeyboardMarkup:
    st = _get_state(user_id)
    def mark(label: str, active: bool) -> str:
        return f"{label}{' ✅' if active else ''}"
    # Главный экран: только стратегия C и кнопка настроек
    valid, mins = _token_status(user_id)
    login_text = "🔐 Вход ✅" if valid else "🔐 Вход"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="⚙️ Настройки", callback_data="open_settings"),
            ],
            [
                InlineKeyboardButton(text=login_text, callback_data="do_login"),
            ],
        ]
    )

def kb_settings(user_id: int) -> InlineKeyboardMarkup:
    st = _get_state(user_id)
    llm = st["llm"]
    lang = st["lang"]
    debug = st["debug"]
    def mark(label: str, active: bool) -> str:
        return f"{label}{' ✅' if active else ''}"
    valid, mins = _token_status(user_id)
    login_text = "🔐 Вход ✅" if valid else "🔐 Вход"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=mark("LLM: GigaChat", llm == "gigachat"), callback_data="set_llm:gigachat"),
                InlineKeyboardButton(text=mark("LLM: Gemini", llm == "gemini" or llm == "api"), callback_data="set_llm:gemini"),
            ],
            [
                InlineKeyboardButton(text=mark("Язык: RU", lang == "rus"), callback_data="set_lang:rus"),
                InlineKeyboardButton(text=mark("Язык: EN", lang == "eng"), callback_data="set_lang:eng"),
            ],
            [
                InlineKeyboardButton(text=mark("Debug", debug), callback_data="toggle_debug"),
            ],
            [
                InlineKeyboardButton(text="🔑 Ключ GigaChat", callback_data="set_key:gigachat"),
                InlineKeyboardButton(text="🔑 Ключ Gemini", callback_data="set_key:gemini"),
            ],
            [
                InlineKeyboardButton(text="❌ Удалить GigaChat", callback_data="del_key:gigachat"),
                InlineKeyboardButton(text="❌ Удалить Gemini", callback_data="del_key:gemini"),
            ],
            [
                InlineKeyboardButton(text="📝 Регистрация", callback_data="do_register"),
                InlineKeyboardButton(text=login_text, callback_data="do_login"),
            ],
            [
                InlineKeyboardButton(text="⬅️ Назад", callback_data="close_settings"),
            ],
        ]
    )


async def cmd_start(message: Message):
    logger.debug(f"/start from={message.from_user.id} username={message.from_user.username}")
    st = _get_state(message.from_user.id)
    valid, mins = _token_status(message.from_user.id)
    ttl = f" | Token: {'валиден' if valid else 'нет'}{f' (~{mins} мин)' if valid else ''}"
    header = (
        f"<b>Стратегия:</b> C\n"
        f"<b>LLM:</b> {st['llm']}\n"
        f"<b>Язык OCR:</b> {st['lang']}\n"
        f"<b>Debug:</b> {'on' if st['debug'] else 'off'}{ttl}"
    )
    await message.answer(
        header,
        reply_markup=kb_main(message.from_user.id),
        parse_mode=ParseMode.HTML,
    )


async def cmd_help(message: Message):
    logger.debug(f"/help from={message.from_user.id}")
    await message.answer(
        "/start — начать и выбрать стратегию\n"
        "/strategy C — выбрать стратегию\n"
        "/lang rus|eng — выбрать язык OCR\n"
        "/llm gigachat|gemini|api — выбрать провайдера LLM (api = внешний сервер)\n"
        "/debug on|off — включить/выключить вывод OCR и LLM\n"
        "/apilog — последние строки лога интеграции (AI_API_DEBUG=1)\n"
        "/testlogin — выполнить попытку логина и показать сырой ответ (AI_API_DEBUG рекомендуется)\n"
        "/setkey <gigachat|gemini> <ключ> — сохранить личный API-ключ\n"
        "/delkey <gigachat|gemini> — удалить личный API-ключ\n"
        "/mykeys — показать, какие ключи сохранены (без самих значений)\n"
        "Пришлите фото/скан или документ для OCR и коррекции"
    )


async def cmd_strategy(message: Message):
    logger.debug(f"/strategy from={message.from_user.id} text={message.text}")
    args = (message.text or "").split()
    if len(args) < 2:
        await message.answer("Укажите стратегию: C")
        return
    strategy = args[1].upper()
    if strategy not in {"C"}:
        await message.answer("Допустимое значение: C")
        return
    st = _get_state(message.from_user.id)
    st["strategy"] = strategy
    await message.answer(f"Стратегия установлена: {strategy}", reply_markup=kb_main(message.from_user.id))


async def cmd_lang(message: Message):
    logger.debug(f"/lang from={message.from_user.id} text={message.text}")
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
    logger.debug(f"/llm from={message.from_user.id} text={message.text}")
    args = (message.text or "").split()
    if len(args) < 2:
        await message.answer("Укажите LLM: gigachat | gemini | api")
        return
    llm = args[1].lower()
    if llm not in {"gigachat", "gemini", "api"}:
        await message.answer("Допустимые значения: gigachat, gemini, api")
        return
    st = _get_state(message.from_user.id)
    # Привязка Gemini к внешнему API: выбор gemini приводит к режиму api
    st["llm"] = "api" if llm == "gemini" else llm
    await message.answer(f"LLM провайдер установлен: {st['llm']}", reply_markup=kb_main(message.from_user.id))

async def cmd_setkey(message: Message):
    logger.debug(f"/setkey from={message.from_user.id} text_len={len(message.text or '')}")
    args = (message.text or "").split(maxsplit=2)
    if len(args) < 3 or args[1].lower() not in {"gigachat", "gemini"}:
        await message.answer("Использование: /setkey <gigachat|gemini> <ключ>")
        return
    provider = args[1].lower()
    key = args[2].strip()
    # Всегда сохраняем локально
    _set_user_key(message.from_user.id, provider, key)
    if provider == "gemini":
        uid = message.from_user.id
        uname = (message.from_user.username or str(uid))
        ok = api_set_key(uid, uname, provider, key)
        if ok:
            await message.answer(f"Ключ для {provider} сохранён на сервере и локально.")
        else:
            await message.answer(f"Ключ для {provider} сохранён локально. Сервер: ошибка, смотрите /apilog.")
    else:
        await message.answer(f"Ключ для {provider} сохранён локально.")

async def cmd_delkey(message: Message):
    logger.debug(f"/delkey from={message.from_user.id} text={message.text}")
    args = (message.text or "").split(maxsplit=1)
    if len(args) < 2 or args[1].lower() not in {"gigachat", "gemini"}:
        await message.answer("Использование: /delkey <gigachat|gemini>")
        return
    provider = args[1].lower()
    ok_local = _del_user_key(message.from_user.id, provider)
    if provider == "gemini":
        uid = message.from_user.id
        uname = (message.from_user.username or str(uid))
        ok = api_clear_key(uid, uname, provider)
        await message.answer(f"Ключ для {provider} удалён локально и {'удалён на сервере' if ok else 'сервер: не найден/ошибка'}.")
    else:
        await message.answer(f"Ключ для {provider} {'удалён' if ok_local else 'не найден'} локально.")

async def cmd_mykeys(message: Message):
    logger.debug(f"/mykeys from={message.from_user.id}")
    # GigaChat — локально, Gemini — сервер
    local = _load_user_keys().get(str(message.from_user.id), {})
    has_giga_local = '✅' if 'gigachat' in local else '—'
    uid = message.from_user.id
    uname = (message.from_user.username or str(uid))
    status = api_key_status(uid, uname)
    has_gem_srv = '✅' if bool(status.get('gemini')) else '—'
    await message.answer(f"Ключи:\nGigaChat (локально): {has_giga_local}\nGemini (сервер): {has_gem_srv}")

async def cmd_testlogin(message: Message):
    logger.debug(f"/testlogin from={message.from_user.id} username={message.from_user.username}")
    uid = message.from_user.id
    uname = (message.from_user.username or str(uid))
    ok = api_login(uid, uname)
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

async def cmd_testregister(message: Message):
    uid = message.from_user.id
    uname = (message.from_user.username or str(uid))
    ok = api_register(uid, uname)
    if ok:
        await message.answer("Регистрация выполнена. Пробую логин...")
        if api_login(uid, uname):
            await message.answer("Логин успешен: токен получен.")
        else:
            await message.answer("Логин неудачен. Используйте /apilog для подробностей.")
    else:
        await message.answer("Регистрация не удалась. Используйте /apilog для подробностей.")

async def cmd_apilog(message: Message):
    logger.debug(f"/apilog from={message.from_user.id}")
    if not API_DEBUG:
        await message.answer("Логирование отключено. Установите AI_API_DEBUG=1")
        return
    if not API_LOG_FILE.exists():
        await message.answer("Файл лога отсутствует")
        return
    try:
        with open(API_LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()[-10:]
        text = ''.join(lines)
        esc = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        await message.answer(f"<b>API LOG (последние 10)</b>\n<pre>{esc}</pre>", parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Ошибка чтения лога: {e}")


async def cmd_debug(message: Message):
    logger.debug(f"/debug from={message.from_user.id} text={message.text}")
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
    logger.debug(f"on_btn from={query.from_user.id} data={query.data}")
    data = query.data or ""
    st = _get_state(query.from_user.id)
    edited = False
    if data.startswith("set_strategy:"):
        # Всегда C
        st["strategy"] = "C"
        edited = True
    elif data == "open_settings":
        st["settings_open"] = True
        edited = True
    elif data == "close_settings":
        st["settings_open"] = False
        edited = True
    elif data.startswith("set_llm:"):
        _, val = data.split(":", 1)
        llm = val.lower()
        if llm in {"gigachat", "gemini"}:
            # Привязка: gemini -> api
            st["llm"] = "api" if llm == "gemini" else llm
            edited = True
    elif data.startswith("set_lang:"):
        _, val = data.split(":", 1)
        if val in {"rus", "eng"}:
            st["lang"] = val
            edited = True
    elif data == "toggle_debug":
        st["debug"] = not st["debug"]
        edited = True
    elif data == "do_login":
        uid = query.from_user.id
        uname = (query.from_user.username or str(uid))
        # Если нет токена или истёк час — логинимся; если неудачно, пытаемся регистрировать
        need_login = True
        jwt = API_JWT_BY_USER.get(uid)
        ts = API_JWT_TS_BY_USER.get(uid, 0)
        if jwt and (time.time() - ts <= 3600):
            need_login = False
        if need_login:
            if not api_login(uid, uname):
                await query.message.answer("Логин неудачен, пробую регистрацию...")
                if api_register(uid, uname) and api_login(uid, uname):
                    valid, mins = _token_status(uid)
                    await query.message.answer(f"Регистрация и вход выполнены. Токен ~{mins} мин.")
                else:
                    await query.message.answer("Не удалось выполнить вход. Проверьте /apilog.")
            else:
                valid, mins = _token_status(uid)
                await query.message.answer(f"Вход выполнен: токен получен. Токен ~{mins} мин.")
        else:
            valid, mins = _token_status(uid)
            await query.message.answer(f"Вы уже вошли. Токен ещё действителен (~{mins} мин).")
        edited = True
    elif data == "do_register":
        uid = query.from_user.id
        uname = (query.from_user.username or str(uid))
        ok = api_register(uid, uname)
        if ok:
            await query.message.answer("Регистрация выполнена. Пробую логин...")
            if api_login(uid, uname):
                await query.message.answer("Логин успешен: токен получен.")
            else:
                await query.message.answer("Логин неудачен. Проверьте логи через /apilog.")
        else:
            await query.message.answer("Регистрация не удалась. Проверьте логи через /apilog.")
    elif data.startswith("set_key:"):
        _, provider = data.split(":", 1)
        if provider in {"gigachat", "gemini"}:
            st.setdefault("await_key_provider", provider)
            await query.message.answer(
                f"Отправьте одним сообщением ключ для {provider}. Он будет сохранён как ваш личный."
                + (" И отправлен на сервер." if provider == "gemini" else ""))
    elif data.startswith("del_key:"):
        _, provider = data.split(":", 1)
        if provider in {"gigachat", "gemini"}:
            uid = query.from_user.id
            uname = (query.from_user.username or str(uid))
            ok_local = _del_user_key(query.from_user.id, provider)
            if provider == "gemini":
                ok_srv = api_clear_key(uid, uname, provider)
                await query.message.answer(
                    f"Ключ для {provider} удалён локально и {'удалён на сервере' if ok_srv else 'сервер: не найден/ошибка'}.")
            else:
                await query.message.answer(
                    f"Ключ для {provider} {'удалён' if ok_local else 'не найден'} локально.")
    if edited:
        valid, mins = _token_status(query.from_user.id)
        ttl = f" | Token: {'валиден' if valid else 'нет'}{f' (~{mins} мин)' if valid else ''}"
        header = (
            f"<b>Стратегия:</b> C\n"
            f"<b>LLM:</b> {st['llm']}\n"
            f"<b>Язык OCR:</b> {st['lang']}\n"
            f"<b>Debug:</b> {'on' if st['debug'] else 'off'}{ttl}"
        )
        kb = kb_settings(query.from_user.id) if st.get("settings_open") else kb_main(query.from_user.id)
        try:
            await query.message.edit_text(
                header,
                reply_markup=kb,
                parse_mode=ParseMode.HTML,
            )
        except TelegramBadRequest as e:
            # Игнорируем "message is not modified" и подобные
            logger.debug(f"edit_text skipped: {e}")
    await query.answer()


async def on_photo(message: Message):
    logger.debug(f"on_photo from={message.from_user.id} file_id={message.photo[-1].file_id}")
    photo = message.photo[-1]
    file = await message.bot.get_file(photo.file_id)
    tmp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    local_path = os.path.join(tmp_dir, f"{photo.file_id}.jpg")
    await message.bot.download_file(file.file_path, local_path)

    st = _get_state(message.from_user.id)
    global _current_user_id, _current_username
    _current_user_id = message.from_user.id
    _current_username = (message.from_user.username or str(message.from_user.id))
    lang = st['lang']
    # Если пользователь прислал фото как ключ — игнорируем, ключ ожидается только как текст
    await message.answer(f"Выполняю OCR (язык {lang})...")
    try:
        raw = run_ocr(local_path, lang=lang)
        logger.debug(f"OCR done len={len(raw)}")
    except Exception as e:
        logger.exception("OCR error")
        await message.answer(f"Ошибка OCR: {e}")
        return

    strategy = st['strategy']
    llm = st['llm']
    await message.answer(f"Коррекция LLM (стратегия {strategy}, {llm})...")
    corrected = run_llm_correction(raw, strategy=strategy, llm=llm)
    logger.debug(f"LLM corrected len={len(corrected)}")

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
    logger.debug(f"on_document from={message.from_user.id} name={message.document.file_name} mime={message.document.mime_type}")
    doc = message.document
    file_name = doc.file_name or "document"
    mime = doc.mime_type or ""
    file = await message.bot.get_file(doc.file_id)
    tmp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    local_path = os.path.join(tmp_dir, f"{doc.file_id}_{file_name}")
    await message.bot.download_file(file.file_path, local_path)

    st = _get_state(message.from_user.id)
    global _current_user_id, _current_username
    _current_user_id = message.from_user.id
    _current_username = (message.from_user.username or str(message.from_user.id))
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

async def on_text(message: Message):
    logger.debug(f"on_text from={message.from_user.id} len={len(message.text or '')}")
    # Перехватываем ввод ключа, если ожидается
    st = _get_state(message.from_user.id)
    provider = st.pop("await_key_provider", None)
    if provider:
        key = (message.text or "").strip()
        if not key:
            await message.answer("Пустой ключ — отправьте непустой текст.")
            return
        # Сохраняем локально и, если gemini, пробуем отправить на сервер
        _set_user_key(message.from_user.id, provider, key)
        if provider == "gemini":
            uid = message.from_user.id
            uname = (message.from_user.username or str(uid))
            ok = api_set_key(uid, uname, provider, key)
            if ok:
                await message.answer(f"Ключ для {provider} сохранён на сервере и локально.")
            else:
                await message.answer(f"Ключ для {provider} сохранён локально. Сервер: ошибка, смотрите /apilog.")
        else:
            await message.answer(f"Ключ для {provider} сохранён локально.")
        return
    # Иначе игнор, можно добавить помощь


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

    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_help, Command("help"))
    dp.message.register(cmd_strategy, F.text.startswith("/strategy"))
    dp.message.register(cmd_lang, F.text.startswith("/lang"))
    dp.message.register(cmd_llm, F.text.startswith("/llm"))
    dp.message.register(cmd_debug, F.text.startswith("/debug"))
    dp.message.register(cmd_apilog, F.text.startswith("/apilog"))
    dp.message.register(cmd_testlogin, F.text.startswith("/testlogin"))
    dp.message.register(cmd_testregister, F.text.startswith("/testregister"))
    dp.message.register(cmd_setkey, F.text.startswith("/setkey"))
    dp.message.register(cmd_delkey, F.text.startswith("/delkey"))
    dp.message.register(cmd_mykeys, F.text.startswith("/mykeys"))

    dp.callback_query.register(on_btn)
    dp.message.register(on_photo, F.photo)
    dp.message.register(on_document, F.document)
    dp.message.register(on_text, F.text)

    logger.debug("Start polling")
    await dp.start_polling(bot)
    logger.debug("Polling stopped")


if __name__ == "__main__":
    asyncio.run(main())
