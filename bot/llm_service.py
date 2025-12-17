import os
from typing import Optional
import logging

from ocr.base import get_raw_text, normalize_whitespace

from .user_keys import get_user_key
from .api_client import API_BASE, api_ask_text, api_key_status, api_set_key

logger = logging.getLogger(__name__)

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
        return getattr(resp, "text", None) or (
            resp.candidates[0].content.parts[0].text
            if getattr(resp, "candidates", None)
            else str(resp)
        )
    except Exception as e:
        return f"[LLM ERROR] {e}\n\n[LLM OUTPUT MOCK]\n{prompt[:200]}..."


def external_api_complete(prompt: str, tg_id: int, username: str) -> str:
    return api_ask_text(prompt, tg_id=tg_id, username=username)


def _ensure_gemini_key(tg_id: int, username: str) -> bool:
    # Всегда делаем прямой запрос к серверу, проверяя есть ли ключ
    status = api_key_status(tg_id, username, skip_cache=True)
    
    # Проверяем наличие ошибок
    if "error" in status:
        # Если ошибка 401, значит не авторизован
        if status.get("error_code") == 401:
            logger.warning(f"User {tg_id} not authorized on API server")
            return False
        # Другие ошибки также означают, что ключ недоступен
        return False
    
    # Если ключ есть на сервере - хорошо
    if bool(status.get("gemini")):
        return True

    # Ключа на сервере нет — пробуем отправить локальный, если он сохранен.
    local_key = get_user_key(tg_id, "gemini")
    if local_key:
        # api_set_key при успехе сам обновит кэш API_HAS_GEMINI_KEY
        if api_set_key(tg_id, username, "gemini", local_key):
            return True
    return False


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


def run_llm_correction(
    text: str,
    strategy: str,
    llm: str,
    user_id: int,
    username: str,
) -> str:
    # Сейчас используем только стратегию C, но оставляем параметр для будущего
    prompt = prompt_strategy_C(text)
    llm_choice = (llm or os.getenv("LLM_PROVIDER", "gigachat")).lower()
    force_local_gemini = os.getenv("GEMINI_LOCAL", "0").lower() in {"1", "true", "yes"}

    if llm_choice in {"gemini", "api", "gemini_api", "external"}:
        if API_BASE and not force_local_gemini:
            if not _ensure_gemini_key(user_id, username):
                return "[GEMINI API KEY MISSING]\nОтправьте ключ через настройки: Ключ Gemini."
            return external_api_complete(prompt, tg_id=user_id, username=username)
        gemini_key = get_user_key(user_id, "gemini") or os.getenv("GEMINI_API_KEY")
        if gemini_key:
            return gemini_complete(prompt, api_key=gemini_key, model_name=os.getenv("GEMINI_MODEL"))
        if API_BASE:
            return external_api_complete(prompt, tg_id=user_id, username=username)
        return "[GEMINI CONFIG MISSING] Set GEMINI_API_KEY or AI_API_BASE/AI_API_USER/AI_API_PASS"

    giga_key = get_user_key(user_id, "gigachat") or os.getenv("GIGACHAT_CREDENTIALS")
    return gigachat_complete(prompt, api_key=giga_key)
