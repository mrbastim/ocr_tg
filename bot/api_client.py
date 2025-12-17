import os
import json
import time
import re
import logging
from pathlib import Path
from typing import Dict, Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

API_BASE = os.getenv("AI_API_BASE") or os.getenv("GEMINI_API_BASE")
API_BASE_PATH = os.getenv("AI_API_BASE_PATH", "/api")
API_USERNAME = os.getenv("AI_API_USER") or os.getenv("GEMINI_API_USER")
API_PASSWORD = os.getenv("AI_API_PASS") or os.getenv("GEMINI_API_PASS")
API_JWT: Optional[str] = None
API_JWT_BY_USER: Dict[int, str] = {}
API_JWT_TS_BY_USER: Dict[int, float] = {}
API_HAS_GEMINI_KEY: Dict[int, bool] = {}
API_DEBUG = os.getenv("AI_API_DEBUG", "0").lower() not in {"0", "false", "off"}
API_LOG_DIR = Path(os.getenv("AI_API_LOG_DIR", os.path.join(os.getcwd(), "tmp")))
API_LOG_FILE = API_LOG_DIR / "api_debug.log"


def _api_log(event: str, **fields):
    try:
        API_LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        parts = [f"[{ts}] {event}"]
        for k, v in fields.items():
            if k.lower() == "authorization":
                continue
            text = str(v)
            if len(text) > 1500:
                text = text[:1500] + " …<truncated>"
            parts.append(f"{k}={text}")
        line = " | ".join(parts)
        print(line)
        if API_DEBUG:
            with open(API_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass


def _api_url(path: str) -> str:
    base = API_BASE or ""
    prefix = API_BASE_PATH or ""
    if prefix and not prefix.startswith("/"):
        prefix = "/" + prefix
    if base.endswith("/"):
        base = base[:-1]
    if path and not path.startswith("/"):
        path = "/" + path
    return f"{base}{prefix}{path}"


def api_register(tg_id: int, username: str) -> bool:
    if not API_BASE:
        return False
    url = _api_url("/register")
    payload = json.dumps({"tg_id": tg_id, "username": username}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    _api_log("register_request", url=url, body=payload.decode("utf-8"))
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body_raw = resp.read().decode("utf-8")
            _api_log("register_response", status=getattr(resp, "status", None), body=body_raw)
            return True
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        _api_log("register_http_error", code=e.code, body=err_body[:1500])
        return False
    except Exception as e:
        _api_log("register_error", error=e)
        return False


def api_login(tg_id: int, username: str) -> bool:
    global API_JWT
    if not API_BASE:
        _api_log("login_skip", reason="no_base")
        return False
    url = _api_url("/login")
    payload_dict = {"tg_id": tg_id, "username": username}
    payload = json.dumps(payload_dict).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    _api_log("login_request", url=url, body=payload.decode("utf-8"))
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body_raw = resp.read().decode("utf-8")
            _api_log("login_response_raw", status=getattr(resp, "status", None), body=body_raw)
            try:
                data = json.loads(body_raw)
            except Exception as parse_err:
                _api_log("login_parse_error", error=parse_err)
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
            jwt_regex = re.compile(r"^[A-Za-z0-9-_]+=*\.[A-Za-z0-9-_]+=*\.[A-Za-z0-9-_]+=*$")
            for s in re.findall(r"[A-Za-z0-9\-_=]+\.[A-Za-z0-9\-_=]+\.[A-Za-z0-9\-_=]+", body_raw):
                if jwt_regex.match(s):
                    token_candidates.append(s)
            API_JWT = next(iter(token_candidates), None)
            if API_JWT:
                API_JWT_BY_USER[tg_id] = API_JWT
                API_JWT_TS_BY_USER[tg_id] = time.time()
            _api_log("login_ok", token_present=API_JWT is not None, found=len(token_candidates))
            return API_JWT is not None
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        _api_log("login_http_error", code=e.code, body=err_body[:1500])
        return False
    except Exception as e:
        _api_log("login_error", error=e)
        return False


def _ensure_jwt(tg_id: int, username: str) -> Optional[str]:
    jwt = API_JWT_BY_USER.get(tg_id)
    ts = API_JWT_TS_BY_USER.get(tg_id, 0)
    if jwt and (time.time() - ts <= 3600):
        return jwt
    if not api_login(tg_id, username):
        api_register(tg_id, username)
        if not api_login(tg_id, username):
            _api_log("auth_failed", tg_id=tg_id)
            return None
    return API_JWT_BY_USER.get(tg_id)


def api_ask_text(prompt: str, tg_id: int, username: str, model: Optional[str] = None) -> str:
    if not API_BASE:
        _api_log("ask_skip", reason="no_base")
        return "[API NOT CONFIGURED] Set AI_API_BASE, AI_API_USER, AI_API_PASS"
    jwt = _ensure_jwt(tg_id, username)
    if not jwt:
        return "[API AUTH FAILED]"
    url = _api_url("/user/ai/text")
    payload_dict = {"prompt": prompt}
    if model:
        payload_dict["model"] = model
    payload = json.dumps(payload_dict).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {jwt}"}
    _api_log("ask_request", url=url, body=payload.decode("utf-8"), model=model)
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body_raw = resp.read().decode("utf-8")
            _api_log("ask_raw_response", status=getattr(resp, "status", None), body=body_raw)
            try:
                data = json.loads(body_raw)
            except Exception as parse_err:
                _api_log("ask_parse_error", error=parse_err)
                return body_raw[:4000]
            if isinstance(data, dict):
                if isinstance(data.get("data"), dict) and isinstance(data["data"].get("text"), str):
                    return data["data"]["text"]
                if isinstance(data.get("text"), str):
                    return data["text"]
                if isinstance(data.get("success"), dict) and isinstance(data["success"].get("text"), str):
                    return data["success"]["text"]
                if isinstance(data.get("success"), dict):
                    sd = data["success"].get("data")
                    if isinstance(sd, dict) and isinstance(sd.get("text"), str):
                        return sd["text"]
            return json.dumps(data, ensure_ascii=False)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            _api_log("ask_401_retry", tg_id=tg_id)
            jwt = _ensure_jwt(tg_id, username)
            if jwt:
                headers["Authorization"] = f"Bearer {jwt}"
                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp2:
                        body_raw = resp2.read().decode("utf-8")
                        _api_log("ask_raw_response_retry", status=getattr(resp2, "status", None), body=body_raw)
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
                    _api_log("ask_retry_error", error=e2)
        try:
            err = e.read().decode("utf-8")
        except Exception:
            err = str(e)
        _api_log("ask_http_error", code=e.code, body=err)
        return f"[API ERROR] {e.code}: {err[:2000]}"
    except Exception as e:
        _api_log("ask_error", error=e)
        return f"[API ERROR] {e}"


def api_set_key(tg_id: int, username: str, provider: str, key: str) -> bool:
    # На сервер отправляем только ключи для Gemini
    if provider != "gemini":
        _api_log("set_key_skip_local_only", provider=provider)
        return False
    if not API_BASE:
        _api_log("set_key_skip", reason="no_base")
        return False
    jwt = _ensure_jwt(tg_id, username)
    if not jwt:
        return False
    url = _api_url("/user/ai/key")
    payload = json.dumps({"api_key": key}).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {jwt}"}
    _api_log("set_key_request", url=url, body=payload.decode("utf-8"))
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body_raw = resp.read().decode("utf-8")
            _api_log("set_key_response", status=getattr(resp, "status", None), body=body_raw)
            API_HAS_GEMINI_KEY[tg_id] = True
            return True
    except urllib.error.HTTPError as e:
        if e.code == 401:
            _api_log("set_key_401_retry", tg_id=tg_id)
            jwt2 = _ensure_jwt(tg_id, username)
            if jwt2:
                headers["Authorization"] = f"Bearer {jwt2}"
                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=15) as resp2:
                        body_raw = resp2.read().decode("utf-8")
                        _api_log("set_key_response_retry", status=getattr(resp2, "status", None), body=body_raw)
                        API_HAS_GEMINI_KEY[tg_id] = True
                        return True
                except Exception as e2:
                    _api_log("set_key_retry_error", error=e2)
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        _api_log("set_key_http_error", code=e.code, body=err_body[:1500])
        return False
    except Exception as e:
        _api_log("set_key_error", error=e)
        return False


def api_clear_key(tg_id: int, username: str, provider: str) -> bool:
    if provider != "gemini":
        _api_log("clear_key_skip_local_only", provider=provider)
        return False
    if not API_BASE:
        _api_log("clear_key_skip", reason="no_base")
        return False
    jwt = _ensure_jwt(tg_id, username)
    if not jwt:
        return False
    url = _api_url("/user/ai/key")
    headers = {"Authorization": f"Bearer {jwt}"}
    _api_log("clear_key_request", url=url)
    req = urllib.request.Request(url, headers=headers, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body_raw = resp.read().decode("utf-8")
            _api_log("clear_key_response", status=getattr(resp, "status", None), body=body_raw)
            API_HAS_GEMINI_KEY.pop(tg_id, None)
            return True
    except urllib.error.HTTPError as e:
        if e.code == 401:
            _api_log("clear_key_401_retry", tg_id=tg_id)
            jwt2 = _ensure_jwt(tg_id, username)
            if jwt2:
                headers["Authorization"] = f"Bearer {jwt2}"
                req = urllib.request.Request(url, headers=headers, method="DELETE")
                try:
                    with urllib.request.urlopen(req, timeout=15) as resp2:
                        body_raw = resp2.read().decode("utf-8")
                        _api_log("clear_key_response_retry", status=getattr(resp2, "status", None), body=body_raw)
                        API_HAS_GEMINI_KEY.pop(tg_id, None)
                        return True
                except Exception as e2:
                    _api_log("clear_key_retry_error", error=e2)
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        _api_log("clear_key_http_error", code=e.code, body=err_body[:1500])
        return False
    except Exception as e:
        _api_log("clear_key_error", error=e)
        return False


def api_key_status(tg_id: int, username: str, skip_cache: bool = False) -> Dict:
    """Проверить статус ключа API на сервере.
    
    Возвращает словарь с ключами:
    - 'gemini': bool - наличие ключа
    - 'error': str - описание ошибки (если есть)
    - 'error_code': int - код ошибки HTTP (если есть)
    """
    if not API_BASE:
        _api_log("key_status_skip", reason="no_base")
        return {}
    jwt = _ensure_jwt(tg_id, username)
    if not jwt:
        return {"error": "auth_failed", "error_code": 401}
    url = _api_url("/user/ai/key")
    headers = {"Authorization": f"Bearer {jwt}"}
    _api_log("key_status_request", url=url, skip_cache=skip_cache)
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body_raw = resp.read().decode("utf-8")
            _api_log("key_status_response", status=getattr(resp, "status", None), body=body_raw)
            try:
                data = json.loads(body_raw)
                # Фактический формат backend: {"data": {"has_key": bool}, "status": "success"}
                result: Dict = {}
                if isinstance(data, dict) and isinstance(data.get("data"), dict):
                    has_key_val = data["data"].get("has_key")
                    if isinstance(has_key_val, (bool, int)):
                        result["gemini"] = bool(has_key_val)

                # обновляем кэш наличия ключа Gemini на сервере
                if "gemini" in result:
                    if result["gemini"]:
                        API_HAS_GEMINI_KEY[tg_id] = True
                    else:
                        API_HAS_GEMINI_KEY.pop(tg_id, None)
                return result
            except Exception as e:
                _api_log("key_status_parse_error", error=e)
                return {"error": "parse_error"}
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        _api_log("key_status_http_error", code=e.code, body=err_body[:1500])
        return {"error": "http_error", "error_code": e.code, "error_body": err_body[:500]}
    except Exception as e:
        _api_log("key_status_error", error=e)
        return {"error": str(e)}


def api_has_gemini_key_cached(tg_id: int) -> Optional[bool]:
    """Вернуть кэшированный признак наличия ключа Gemini на сервере для пользователя.

    None означает, что информации в кэше нет.
    """
    return API_HAS_GEMINI_KEY.get(tg_id)


def api_get_text_models(tg_id: int, username: str) -> Dict[str, dict]:
    """Получить список доступных текстовых моделей Gemini с сервера.
    
    Возвращает словарь вида:
    {
        "model_name": {
            "display_name": "Display Name",
            "description": "Description",
            "is_available": bool,
            ...
        },
        ...
    }
    
    При ошибке возвращает пустой словарь.
    """
    if not API_BASE:
        _api_log("get_models_skip", reason="no_base")
        return {}
    
    jwt = _ensure_jwt(tg_id, username)
    if not jwt:
        _api_log("get_models_auth_failed", tg_id=tg_id)
        return {}
    
    url = _api_url("/user/ai/models")
    headers = {"Authorization": f"Bearer {jwt}"}
    _api_log("get_models_request", url=url)
    req = urllib.request.Request(url, headers=headers, method="GET")
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body_raw = resp.read().decode("utf-8")
            _api_log("get_models_response", status=getattr(resp, "status", None), body_len=len(body_raw))
            try:
                data = json.loads(body_raw)
                result: Dict[str, dict] = {}
                
                # Ожидаем структуру: {"data": {"models": [...]}, "status": "success"}
                if isinstance(data, dict):
                    models_list = None
                    
                    # Пробуем найти список моделей в разных местах
                    if isinstance(data.get("data"), dict) and isinstance(data["data"].get("models"), list):
                        models_list = data["data"]["models"]
                    elif isinstance(data.get("models"), list):
                        models_list = data["models"]
                    
                    if models_list:
                        for model in models_list:
                            if not isinstance(model, dict):
                                continue
                            
                            name = model.get("name")
                            if not name:
                                continue
                            
                            # Фильтруем только текстовые модели
                            category = model.get("category", "").lower()
                            supported = model.get("supported_actions", [])
                            
                            # Берём модели, у которых категория "text" или поддерживают generateContent
                            if category == "text" or "generateContent" in supported:
                                is_available = model.get("is_available", True)
                                result[name] = {
                                    "display_name": model.get("display_name", name),
                                    "description": model.get("description", ""),
                                    "is_available": is_available,
                                    "category": category,
                                    "input_token_limit": model.get("input_token_limit", 0),
                                    "output_token_limit": model.get("output_token_limit", 0),
                                }
                
                _api_log("get_models_parsed", count=len(result))
                return result
            except Exception as parse_err:
                _api_log("get_models_parse_error", error=parse_err)
                return {}
    except urllib.error.HTTPError as e:
        if e.code == 401:
            _api_log("get_models_401_retry", tg_id=tg_id)
            jwt2 = _ensure_jwt(tg_id, username)
            if jwt2:
                headers["Authorization"] = f"Bearer {jwt2}"
                req = urllib.request.Request(url, headers=headers, method="GET")
                try:
                    with urllib.request.urlopen(req, timeout=15) as resp2:
                        body_raw = resp2.read().decode("utf-8")
                        _api_log("get_models_response_retry", status=getattr(resp2, "status", None), body_len=len(body_raw))
                        try:
                            data = json.loads(body_raw)
                            result: Dict[str, dict] = {}
                            
                            if isinstance(data, dict):
                                models_list = None
                                if isinstance(data.get("data"), dict) and isinstance(data["data"].get("models"), list):
                                    models_list = data["data"]["models"]
                                elif isinstance(data.get("models"), list):
                                    models_list = data["models"]
                                
                                if models_list:
                                    for model in models_list:
                                        if not isinstance(model, dict):
                                            continue
                                        name = model.get("name")
                                        if not name:
                                            continue
                                        category = model.get("category", "").lower()
                                        supported = model.get("supported_actions", [])
                                        if category == "text" or "generateContent" in supported:
                                            is_available = model.get("is_available", True)
                                            result[name] = {
                                                "display_name": model.get("display_name", name),
                                                "description": model.get("description", ""),
                                                "is_available": is_available,
                                                "category": category,
                                                "input_token_limit": model.get("input_token_limit", 0),
                                                "output_token_limit": model.get("output_token_limit", 0),
                                            }
                            
                            return result
                        except Exception as parse_err:
                            _api_log("get_models_retry_parse_error", error=parse_err)
                            return {}
                except Exception as e2:
                    _api_log("get_models_retry_error", error=e2)
                    return {}
        
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        _api_log("get_models_http_error", code=e.code, body_len=len(err_body))
        return {}
    except Exception as e:
        _api_log("get_models_error", error=e)
        return {}
