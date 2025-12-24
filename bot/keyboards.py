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
            "strategy": "strong",
            "lang": os.getenv("OCR_LANG", "rus"),
            "llm": os.getenv("LLM_PROVIDER", "gigachat"),
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "debug": False,
            "settings_open": False,
            "llm_menu_open": False,
            "prompt_settings_open": False,
            "has_gemini": False,
            "models_cache": {},  # Кэш доступных моделей для быстрого доступа
            "custom_prompt": None,
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


def kb_llm_settings(user_id: int) -> InlineKeyboardMarkup:
    """Клавиатура настройки LLM (выбор провайдера, модели, ключей, промта)."""
    st = get_state(user_id)
    llm = st["llm"]
    has_gemini = bool(st.get("has_gemini"))
    current_model = st.get("model", "gemini-2.5-flash")

    def mark(label: str, active: bool) -> str:
        return f"{label}{' ✅' if active else ''}"

    keyboard = [
        # Выбор провайдера LLM
        [
            InlineKeyboardButton(text=mark("GigaChat", llm == "gigachat"), callback_data="set_llm:gigachat"),
            InlineKeyboardButton(text=mark("Yandex", llm == "yandex"), callback_data="set_llm:yandex"),
            InlineKeyboardButton(text=mark("Gemini", llm in {"gemini", "api"}), callback_data="set_llm:gemini"),
        ],
        [InlineKeyboardButton(text="🧠 Настройки промта", callback_data="open_prompt")],
    ]
    
    # Выбор модели только если выбран Gemini
    if llm in {"gemini", "api"}:
        models_cache = st.get("models_cache", {})
        display_model = models_cache.get(current_model, {}).get("display_name", current_model)
        keyboard.append([
            InlineKeyboardButton(
                text=f"🤖 Модель: {display_model}", 
                callback_data="select_model"
            ),
        ])
    
    # Управление ключами
    keyboard.extend([
        [
            InlineKeyboardButton(text="➕ Добавить GigaChat", callback_data="set_key:gigachat"),
            InlineKeyboardButton(text="➖ Удалить GigaChat", callback_data="del_key:gigachat"),
        ],
        [
            InlineKeyboardButton(text="➕ Добавить Yandex", callback_data="set_key:yandex"),
            InlineKeyboardButton(text="➖ Удалить Yandex", callback_data="del_key:yandex"),
        ],
        [
            InlineKeyboardButton(text=f"➕ Добавить Gemini {'✅' if has_gemini else ''}", callback_data="set_key:gemini"),
            InlineKeyboardButton(text="➖ Удалить Gemini", callback_data="del_key:gemini"),
        ],
        # Настройка промта
        [InlineKeyboardButton(text="📝 Настройка промта", callback_data="set_prompt")],
        # Назад в главные настройки
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="open_settings")],
    ])

    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def kb_settings(user_id: int) -> InlineKeyboardMarkup:
    st = get_state(user_id)
    lang = st["lang"]
    debug = st["debug"]

    def mark(label: str, active: bool) -> str:
        return f"{label}{' ✅' if active else ''}"

    valid, _ = token_status(user_id)
    login_text = "🔐 Вход ✅" if valid else "🔐 Вход"

    keyboard = [
        # Подраздел настройки LLM
        [InlineKeyboardButton(text="⚙️ Настройка LLM", callback_data="open_llm_settings")],
        # Выбор языка
        [
            InlineKeyboardButton(text=mark("RU", lang == "rus"), callback_data="set_lang:rus"),
            InlineKeyboardButton(text=mark("EN", lang == "eng"), callback_data="set_lang:eng"),
            InlineKeyboardButton(
                text=mark("RU+EN", lang == "rus+eng"), callback_data="set_lang:rus+eng"
            ),
        ],
        [InlineKeyboardButton(text=mark("Debug", debug), callback_data="toggle_debug")],
        [
            InlineKeyboardButton(text="📝 Регистрация", callback_data="do_register"),
            InlineKeyboardButton(text=login_text, callback_data="do_login"),
        ],
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="close_settings")],
    ]

    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def kb_models(user_id: int, models: Dict[str, dict]) -> InlineKeyboardMarkup:
    """Создать клавиатуру для выбора модели.
    
    Args:
        user_id: ID пользователя
        models: Словарь моделей в формате {name: {display_name, ...}, ...}
    
    Returns:
        InlineKeyboardMarkup с кнопками моделей
    """
    st = get_state(user_id)
    current_model = st.get("model", "gemini-2.5-flash")
    
    keyboard = []
    
    # Добавляем по 2 кнопки в ряд
    model_names = sorted(models.keys())
    for i in range(0, len(model_names), 2):
        row = []
        for j in range(2):
            if i + j < len(model_names):
                model_name = model_names[i + j]
                model_info = models[model_name]
                display_name = model_info.get("display_name", model_name)
                is_available = model_info.get("is_available", True)
                is_selected = model_name == current_model
                
                # Используем полное название без обрезания
                btn_text = display_name
                if is_selected:
                    btn_text = f"✅ {btn_text}"
                elif not is_available:
                    btn_text = f"⚠️ {btn_text}"
                
                row.append(InlineKeyboardButton(
                    text=btn_text,
                    callback_data=f"set_model:{model_name}"
                ))
        if row:
            keyboard.append(row)
    
    # Добавляем кнопку "Назад"
    keyboard.append([
        InlineKeyboardButton(text="⬅️ Назад", callback_data="close_models")
    ])
    
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def kb_prompt_settings(user_id: int, st: Dict) -> InlineKeyboardMarkup:
    strategy = (st.get("strategy") or "strong").lower()

    def mark(label: str, active: bool) -> str:
        return f"{label}{' ✅' if active else ''}"

    keyboard = [
        [
            InlineKeyboardButton(text=mark("Слабый", strategy == "weak"), callback_data="set_prompt:weak"),
            InlineKeyboardButton(text=mark("Средний", strategy == "medium"), callback_data="set_prompt:medium"),
        ],
        [
            InlineKeyboardButton(text=mark("Сильный", strategy == "strong"), callback_data="set_prompt:strong"),
            InlineKeyboardButton(text=mark("Свой", strategy == "custom"), callback_data="set_prompt:custom"),
        ],
        [InlineKeyboardButton(text="👁 Посмотреть промт", callback_data="show_prompt")],
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="close_prompt")],
    ]

    return InlineKeyboardMarkup(inline_keyboard=keyboard)

