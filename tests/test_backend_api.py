import os
import json
import time
import random
import string

import requests

AI_API_BASE = os.getenv("AI_API_BASE", "http://net.unequaled-earthquake.ru:5555")
AI_API_BASE_PATH = os.getenv("AI_API_BASE_PATH", "/api")


def _base_url(path: str) -> str:
    base = AI_API_BASE.rstrip("/")
    prefix = AI_API_BASE_PATH or ""
    if prefix and not prefix.startswith("/"):
        prefix = "/" + prefix
    if path and not path.startswith("/"):
        path = "/" + path
    return f"{base}{prefix}{path}"


def _make_user() -> tuple[int, str]:
    """Сгенерировать уникальные tg_id и username для тестов."""
    tg_id = int(time.time()) + random.randint(1, 1_000_000)
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    username = f"test_user_{suffix}"
    return tg_id, username


def _login_token(tg_id: int, username: str) -> str:
    url = _base_url("/login")
    resp = requests.post(url, json={"tg_id": tg_id, "username": username}, timeout=10)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # Согласно текущему backend, успешный ответ логина имеет вид
    # {"data": {"token": "..."}, "status": "success"}
    assert isinstance(data.get("data"), dict), data
    token = data["data"].get("token")
    assert isinstance(token, str) and token, data
    return token


def test_register_and_login():
    """Проверяем, что можно зарегистрироваться и залогиниться, получить JWT."""
    tg_id, username = _make_user()

    # /register
    url_reg = _base_url("/register")
    resp_reg = requests.post(url_reg, json={"tg_id": tg_id, "username": username}, timeout=10)
    assert resp_reg.status_code == 200, resp_reg.text

    # /login
    token = _login_token(tg_id, username)
    assert token


def test_gemini_key_lifecycle():
    """Проверяем полный цикл работы с /user/ai/key: GET -> POST -> GET -> DELETE -> GET."""
    tg_id, username = _make_user()

    # сначала регистрация и логин
    url_reg = _base_url("/register")
    resp_reg = requests.post(url_reg, json={"tg_id": tg_id, "username": username}, timeout=10)
    assert resp_reg.status_code == 200, resp_reg.text

    token = _login_token(tg_id, username)
    headers = {"Authorization": f"Bearer {token}"}

    # начальный статус ключа
    url_key = _base_url("/user/ai/key")
    resp_status = requests.get(url_key, headers=headers, timeout=10)
    assert resp_status.status_code == 200, resp_status.text
    data_status = resp_status.json()
    # Фактический API: {"data": {"has_key": bool}, "status": "success"}
    assert isinstance(data_status.get("data"), dict)
    assert isinstance(data_status["data"].get("has_key"), bool)

    # устанавливаем тестовый ключ
    test_key = "test-gemini-key-for-ci"
    resp_set = requests.post(url_key, headers=headers, json={"api_key": test_key}, timeout=10)
    assert resp_set.status_code == 200, resp_set.text

    # статус после установки ключа
    resp_status2 = requests.get(url_key, headers=headers, timeout=10)
    assert resp_status2.status_code == 200, resp_status2.text
    data_status2 = resp_status2.json()
    assert isinstance(data_status2.get("data"), dict)
    assert data_status2["data"].get("has_key") is True

    # удаляем ключ
    resp_del = requests.delete(url_key, headers=headers, timeout=10)
    assert resp_del.status_code == 200, resp_del.text

    # статус после удаления ключа
    resp_status3 = requests.get(url_key, headers=headers, timeout=10)
    assert resp_status3.status_code == 200, resp_status3.text
    data_status3 = resp_status3.json()
    assert isinstance(data_status3.get("data"), dict)
    assert data_status3["data"].get("has_key") is False


def test_ai_text_generation_optional():
    """Тест генерации текста. Опционален: срабатывает только если есть тестовый ключ.

    Чтобы включить, нужно заранее задать переменную окружения TEST_GEMINI_KEY
    и один раз сохранить ключ для пользователя.
    """
    test_key = os.getenv("TEST_GEMINI_KEY")
    if not test_key:
        # пропускаем тест, если не выдан реальный ключ для CI
        import pytest

        pytest.skip("TEST_GEMINI_KEY is not set; skipping real text generation test")

    tg_id, username = _make_user()

    # регистрация и логин
    url_reg = _base_url("/register")
    requests.post(url_reg, json={"tg_id": tg_id, "username": username}, timeout=10)
    token = _login_token(tg_id, username)
    headers = {"Authorization": f"Bearer {token}"}

    # устанавливаем боевой ключ
    url_key = _base_url("/user/ai/key")
    resp_set = requests.post(url_key, headers=headers, json={"api_key": test_key}, timeout=15)
    assert resp_set.status_code == 200, resp_set.text

    # пробуем генерацию текста
    url_text = _base_url("/user/ai/text")
    resp_text = requests.post(url_text, headers=headers, json={"prompt": "Test prompt from CI"}, timeout=60)
    assert resp_text.status_code == 200, resp_text.text
    data = resp_text.json()
    # по swagger ожидается {"data": {"text": "..."}, "status": "ok"}
    assert isinstance(data.get("data"), dict)
    assert isinstance(data["data"].get("text"), str)
    assert data["data"]["text"].strip() != ""
