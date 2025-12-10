import os
import json
from pathlib import Path
from typing import Dict, Optional

USER_KEYS_PATH = Path(os.getcwd()) / "tmp" / "user_keys.json"


def _load_user_keys() -> Dict[str, Dict[str, str]]:
    try:
        with open(USER_KEYS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_user_keys(data: Dict[str, Dict[str, str]]) -> None:
    USER_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_KEYS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_user_key(user_id: int, provider: str) -> Optional[str]:
    data = _load_user_keys()
    user = data.get(str(user_id), {})
    return user.get(provider)


def set_user_key(user_id: int, provider: str, key: str) -> None:
    data = _load_user_keys()
    user = data.get(str(user_id)) or {}
    user[provider] = key
    data[str(user_id)] = user
    _save_user_keys(data)


def delete_user_key(user_id: int, provider: str) -> bool:
    data = _load_user_keys()
    user = data.get(str(user_id))
    if not user or provider not in user:
        return False
    del user[provider]
    data[str(user_id)] = user
    _save_user_keys(data)
    return True


def get_all_user_keys(user_id: int) -> Dict[str, str]:
    data = _load_user_keys()
    return data.get(str(user_id), {})
