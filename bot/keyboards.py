import os
import time
from typing import Dict, Tuple

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from .api_client import API_JWT_BY_USER, API_JWT_TS_BY_USER

_user_state: Dict[int, Dict] = {}


def get_state(user_id: int) -> Dict:
    st = _user_state.get(user_id)
    if not st:
        st = {
            "strategy": "C",
            "lang": os.getenv("OCR_LANG", "rus+eng"),
            "llm": os.getenv("LLM_PROVIDER", "gigachat"),
            "debug": False,
            "settings_open": False,
            "has_gemini": False,
        }
        _user_state[user_id] = st
    return st


def token_status(user_id: int) -> Tuple[bool, int]:
    jwt = API_JWT_BY_USER.get(user_id)
    ts = API_JWT_TS_BY_USER.get(user_id, 0)
    if not jwt or not ts:
        return False, 0
    age = time.time() - ts
    if age > 3600:
        return False, 0
    remain = int((3600 - age) // 60)
    return True, max(remain, 0)


def kb_main(user_id: int) -> InlineKeyboardMarkup:
    def mark(label: str, active: bool) -> str:
        return f"{label}{' ✅' if active else ''}"

    valid, _ = token_status(user_id)
    login_text = "🔐 Вход ✅" if valid else "🔐 Вход"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⚙️ Настройки", callback_data="open_settings")],
            [InlineKeyboardButton(text=login_text, callback_data="do_login")],
        ]
    )


def kb_settings(user_id: int) -> InlineKeyboardMarkup:
    st = get_state(user_id)
    llm = st["llm"]
    lang = st["lang"]
    debug = st["debug"]
    has_gemini = bool(st.get("has_gemini"))

    def mark(label: str, active: bool) -> str:
        return f"{label}{' ✅' if active else ''}"

    valid, _ = token_status(user_id)
    login_text = "🔐 Вход ✅" if valid else "🔐 Вход"

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=mark("LLM: GigaChat", llm == "gigachat"), callback_data="set_llm:gigachat"),
                InlineKeyboardButton(text=mark("LLM: Yandex", llm == "yandex"), callback_data="set_llm:yandex"),
            ],
            [
                InlineKeyboardButton(
                    text=mark("LLM: Gemini", llm in {"gemini", "api"}), callback_data="set_llm:gemini"
                ),
            ],
            [
                InlineKeyboardButton(text=mark("Язык: RU", lang == "rus"), callback_data="set_lang:rus"),
                InlineKeyboardButton(text=mark("Язык: EN", lang == "eng"), callback_data="set_lang:eng"),
                InlineKeyboardButton(
                    text=mark("Язык: RU+EN", lang == "rus+eng"), callback_data="set_lang:rus+eng"
                ),
            ],
            [InlineKeyboardButton(text=mark("Debug", debug), callback_data="toggle_debug")],
            [
                InlineKeyboardButton(text="🔑 Ключ GigaChat", callback_data="set_key:gigachat"),
                InlineKeyboardButton(text="🔑 Ключ Yandex", callback_data="set_key:yandex"),
            ],
            [
                InlineKeyboardButton(
                    text=f"🔑 Ключ Gemini {'✅' if has_gemini else '❌'}",
                    callback_data="set_key:gemini",
                ),
            ],
            [
                InlineKeyboardButton(text="❌ Удалить GigaChat", callback_data="del_key:gigachat"),
                InlineKeyboardButton(text="❌ Удалить Yandex", callback_data="del_key:yandex"),
            ],
            [
                InlineKeyboardButton(text="❌ Удалить Gemini", callback_data="del_key:gemini"),
            ],
            [
                InlineKeyboardButton(text="📝 Регистрация", callback_data="do_register"),
                InlineKeyboardButton(text=login_text, callback_data="do_login"),
            ],
            [
                InlineKeyboardButton(text="📋 ML требования", callback_data="ml_requirements"),
            ],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="close_settings")],
        ]
    )
