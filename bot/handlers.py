import os
import logging
from typing import Tuple

from aiogram import F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message
from aiogram.exceptions import TelegramBadRequest

from .api_client import API_DEBUG, API_LOG_FILE, api_login, api_register, api_key_status, api_set_key, api_clear_key, api_get_text_models
from .user_keys import set_user_key, delete_user_key, get_all_user_keys
from .keyboards import get_state, kb_main, kb_settings, kb_models, token_status
from .llm_service import run_ocr, run_llm_correction

logger = logging.getLogger(__name__)


async def cmd_start(message: Message):
    logger.debug(f"/start from={message.from_user.id} username={message.from_user.username}")
    st = get_state(message.from_user.id)
    valid, mins = token_status(message.from_user.id)
    ttl = f" | Token: {'валиден' if valid else 'нет'}{f' (~{mins} мин)' if valid else ''}"
    header = (
        f"<b>Стратегия:</b> C\n"
        f"<b>LLM:</b> {st['llm']}\n"
        f"<b>Язык OCR:</b> {st['lang']}\n"
        f"<b>Debug:</b> {'on' if st['debug'] else 'off'}{ttl}"
    )
    await message.answer(header, reply_markup=kb_main(message.from_user.id), parse_mode=ParseMode.HTML)


async def cmd_help(message: Message):
    logger.debug(f"/help from={message.from_user.id}")
    await message.answer(
        "/start — начать и выбрать стратегию\n"
        "/strategy C — выбрать стратегию\n"
        "/lang rus|eng — выбрать язык OCR\n"
        "/llm gigachat|gemini|yandex|api — выбрать провайдера LLM (api = внешний сервер)\n"
        "/debug on|off — включить/выключить вывод OCR и LLM\n"
        "/apilog — последние строки лога интеграции (AI_API_DEBUG=1)\n"
        "/testlogin — выполнить попытку логина и показать сырой ответ\n"
        "/setkey <gigachat|gemini|yandex> <ключ> — сохранить личный API-ключ\n"
        "/delkey <gigachat|gemini|yandex> — удалить личный API-ключ\n"
        "/mykeys — показать, какие ключи сохранены\n"
        "Пришлите фото/скан или документ для OCR и коррекции",
    )


async def cmd_strategy(message: Message):
    logger.debug(f"/strategy from={message.from_user.id} text={message.text}")
    args = (message.text or "").split()
    if len(args) < 2:
        await message.answer("Укажите стратегию: C")
        return
    if args[1].upper() != "C":
        await message.answer("Допустимое значение: C")
        return
    st = get_state(message.from_user.id)
    st["strategy"] = "C"
    await message.answer("Стратегия установлена: C", reply_markup=kb_main(message.from_user.id))


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
    st = get_state(message.from_user.id)
    st["lang"] = lang
    await message.answer(f"Язык OCR установлен: {lang}", reply_markup=kb_main(message.from_user.id))


async def cmd_llm(message: Message):
    logger.debug(f"/llm from={message.from_user.id} text={message.text}")
    args = (message.text or "").split()
    if len(args) < 2:
        await message.answer("Укажите LLM: gigachat | gemini | yandex | api")
        return
    llm = args[1].lower()
    if llm not in {"gigachat", "gemini", "yandex", "api"}:
        await message.answer("Допустимые значения: gigachat, gemini, yandex, api")
        return
    st = get_state(message.from_user.id)
    st["llm"] = "api" if llm == "gemini" else llm
    await message.answer(f"LLM провайдер установлен: {st['llm']}", reply_markup=kb_main(message.from_user.id))


async def cmd_setkey(message: Message):
    logger.debug(f"/setkey from={message.from_user.id} text_len={len(message.text or '')}")
    args = (message.text or "").split(maxsplit=2)
    if len(args) < 3 or args[1].lower() not in {"gigachat", "gemini", "yandex"}:
        await message.answer("Использование: /setkey <gigachat|gemini|yandex> <ключ>\nДля Yandex: <folder_id>:<api_key>")
        return
    provider = args[1].lower()
    key = args[2].strip()
    set_user_key(message.from_user.id, provider, key)
    if provider == "gemini":
        uid = message.from_user.id
        uname = message.from_user.username or str(uid)
        ok = api_set_key(uid, uname, provider, key)
        if ok:
            await message.answer("Ключ для gemini сохранён на сервере и локально.")
        else:
            await message.answer("Ключ для gemini сохранён локально. Сервер: ошибка, смотрите /apilog.")
    elif provider == "yandex":
        await message.answer("Ключ для yandex сохранён локально. Формат: <folder_id>:<api_key>")
    else:
        await message.answer("Ключ для gigachat сохранён локально.")


async def cmd_delkey(message: Message):
    logger.debug(f"/delkey from={message.from_user.id} text={message.text}")
    args = (message.text or "").split(maxsplit=1)
    if len(args) < 2 or args[1].lower() not in {"gigachat", "gemini", "yandex"}:
        await message.answer("Использование: /delkey <gigachat|gemini|yandex>")
        return
    provider = args[1].lower()
    ok_local = delete_user_key(message.from_user.id, provider)
    if provider == "gemini":
        uid = message.from_user.id
        uname = message.from_user.username or str(uid)
        ok_srv = api_clear_key(uid, uname, provider)
        await message.answer(
            f"Ключ для gemini удалён локально и {'удалён на сервере' if ok_srv else 'сервер: не найден/ошибка'}."
        )
    else:
        await message.answer(f"Ключ для {provider} {'удалён' if ok_local else 'не найден'} локально.")


async def cmd_mykeys(message: Message):
    logger.debug(f"/mykeys from={message.from_user.id}")
    local = get_all_user_keys(message.from_user.id)
    has_giga_local = "✅" if "gigachat" in local else "—"
    has_yandex_local = "✅" if "yandex" in local else "—"
    uid = message.from_user.id
    uname = message.from_user.username or str(uid)
    status = api_key_status(uid, uname)
    
    # Обрабатываем возможные ошибки
    if "error_code" in status and status["error_code"] == 401:
        has_gem_srv = "⚠️ (не авторизован)"
    elif "error" in status:
        has_gem_srv = "⚠️ (ошибка)"
    else:
        has_gem_srv = "✅" if bool(status.get("gemini")) else "—"
    
    await message.answer(f"Ключи:\nGigaChat (локально): {has_giga_local}\nYandex (локально): {has_yandex_local}\nGemini (сервер): {has_gem_srv}")


async def cmd_testlogin(message: Message):
    logger.debug(f"/testlogin from={message.from_user.id} username={message.from_user.username}")
    uid = message.from_user.id
    uname = message.from_user.username or str(uid)
    ok = api_login(uid, uname)
    if ok:
        await message.answer("Логин успешен: токен получен.")
    else:
        if API_DEBUG and API_LOG_FILE.exists():
            try:
                with open(API_LOG_FILE, "r", encoding="utf-8") as f:
                    lines = f.readlines()[-20:]
                text = "".join(lines)
                esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                await message.answer(f"<b>Логин неудачен</b>\n<pre>{esc}</pre>", parse_mode=ParseMode.HTML)
            except Exception as e:
                await message.answer(f"Логин неудачен. Ошибка чтения лога: {e}")
        else:
            await message.answer("Логин неудачен. Включите AI_API_DEBUG=1 для деталей.")


async def cmd_testregister(message: Message):
    uid = message.from_user.id
    uname = message.from_user.username or str(uid)
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
        with open(API_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()[-10:]
        text = "".join(lines)
        esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        await message.answer(f"<b>API LOG (последние 10)</b>\n<pre>{esc}</pre>", parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Ошибка чтения лога: {e}")


async def cmd_debug(message: Message):
    logger.debug(f"/debug from={message.from_user.id} text={message.text}")
    args = (message.text or "").split()
    if len(args) < 2 or args[1].lower() not in {"on", "off"}:
        await message.answer("Использование: /debug on|off")
        return
    st = get_state(message.from_user.id)
    st["debug"] = args[1].lower() == "on"
    await message.answer(f"Debug: {'on' if st['debug'] else 'off'}", reply_markup=kb_main(message.from_user.id))


async def on_btn(query: CallbackQuery):
    logger.debug(f"on_btn from={query.from_user.id} data={query.data}")
    data = query.data or ""
    st = get_state(query.from_user.id)
    edited = False

    if data.startswith("set_strategy:"):
        st["strategy"] = "C"
        edited = True
    elif data == "open_settings":
        st["settings_open"] = True
        # При открытии настроек проверяем статус ключа Gemini напрямую у сервера
        uid = query.from_user.id
        uname = query.from_user.username or str(uid)
        status = api_key_status(uid, uname, skip_cache=True)
        
        if "error_code" in status and status["error_code"] == 401:
            await query.answer("⚠️ Ключ не обновлён. Требуется вход на API сервер.", show_alert=True)
        elif "error" in status:
            await query.answer(f"⚠️ Ошибка при проверке ключа: {status.get('error', 'unknown')}", show_alert=True)
        else:
            st["has_gemini"] = bool(status.get("gemini"))
        edited = True
    elif data == "close_settings":
        st["settings_open"] = False
        edited = True
    elif data.startswith("set_llm:"):
        _, val = data.split(":", 1)
        llm = val.lower()
        if llm in {"gigachat", "gemini", "yandex"}:
            st["llm"] = "api" if llm == "gemini" else llm
            edited = True
    elif data.startswith("set_lang:"):
        _, val = data.split(":", 1)
        if val in {"rus", "eng", "rus+eng"}:
            st["lang"] = val
            edited = True
    elif data == "toggle_debug":
        st["debug"] = not st["debug"]
        edited = True
    elif data == "do_login":
        uid = query.from_user.id
        uname = query.from_user.username or str(uid)
        need_login = True
        jwt_valid, _ = token_status(uid)
        if jwt_valid:
            need_login = False
        if need_login:
            if not api_login(uid, uname):
                await query.message.answer("Логин неудачен, пробую регистрацию...")
                if api_register(uid, uname) and api_login(uid, uname):
                    valid, mins = token_status(uid)
                    await query.message.answer(f"Регистрация и вход выполнены. Токен ~{mins} мин.")
                else:
                    await query.message.answer("Не удалось выполнить вход. Проверьте /apilog.")
            else:
                valid, mins = token_status(uid)
                await query.message.answer(f"Вход выполнен: токен получен. Токен ~{mins} мин.")
        else:
            valid, mins = token_status(uid)
            await query.message.answer(f"Вы уже вошли. Токен ещё действителен (~{mins} мин).")
        # после входа (или при уже валидном токене) проверяем наличие ключа Gemini на сервере
        try:
            status = api_key_status(uid, uname)
            if "error" not in status:
                st["has_gemini"] = bool(status.get("gemini"))
        except Exception as e:
            logger.debug(f"key_status check failed: {e}")
        edited = True
    elif data == "do_register":
        uid = query.from_user.id
        uname = query.from_user.username or str(uid)
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
        if provider in {"gigachat", "gemini", "yandex"}:
            st.setdefault("await_key_provider", provider)
            extra = " И отправлен на сервер." if provider == "gemini" else (" Формат: <folder_id>:<api_key>" if provider == "yandex" else "")
            await query.message.answer(
                f"Отправьте одним сообщением ключ для {provider}. Он будет сохранён как ваш личный." + extra
            )
    elif data == "select_model":
        # Загружаем доступные модели с сервера
        uid = query.from_user.id
        uname = query.from_user.username or str(uid)
        models = api_get_text_models(uid, uname)
        
        if not models:
            await query.answer("❌ Не удалось загрузить модели. Проверьте подключение и токен.", show_alert=True)
        else:
            # Кэшируем модели в состояние
            st["models_cache"] = models
            kb = kb_models(uid, models)
            try:
                await query.message.edit_text(
                    "🤖 Выберите модель для Gemini:",
                    reply_markup=kb,
                    parse_mode=ParseMode.HTML
                )
            except TelegramBadRequest as e:
                logger.debug(f"edit_text skipped: {e}")
        await query.answer()
        return
    elif data.startswith("set_model:"):
        _, model_name = data.split(":", 1)
        # Проверяем, есть ли эта модель в кэше (безопасность)
        models_cache = st.get("models_cache", {})
        if model_name in models_cache or len(models_cache) == 0:
            # Если кэш пуст, позволяем всё равно установить (может быть юзер скопировал вручную)
            st["model"] = model_name
            logger.debug(f"set_model from={query.from_user.id} model={model_name}")
            
            # Возвращаемся в настройки
            st["settings_open"] = True
            edited = True
    elif data == "close_models":
        # Возвращаемся в настройки
        st["settings_open"] = True
        edited = True
    elif data.startswith("del_key:"):
        _, provider = data.split(":", 1)
        if provider in {"gigachat", "gemini", "yandex"}:
            uid = query.from_user.id
            uname = query.from_user.username or str(uid)
            ok_local = delete_user_key(uid, provider)
            if provider == "gemini":
                ok_srv = api_clear_key(uid, uname, provider)
                st["has_gemini"] = False
                await query.message.answer(
                    f"Ключ для {provider} удалён локально и {'удалён на сервере' if ok_srv else 'сервер: не найден/ошибка'}."
                )
            else:
                await query.message.answer(
                    f"Ключ для {provider} {'удалён' if ok_local else 'не найден'} локально."
                )

    if edited:
        valid, mins = token_status(query.from_user.id)
        ttl = f" | Token: {'валиден' if valid else 'нет'}{f' (~{mins} мин)' if valid else ''}"
        header = (
            f"<b>Стратегия:</b> C\n"
            f"<b>LLM:</b> {st['llm']}\n"
            f"<b>Язык OCR:</b> {st['lang']}\n"
            f"<b>Debug:</b> {'on' if st['debug'] else 'off'}{ttl}"
        )
        kb = kb_settings(query.from_user.id) if st.get("settings_open") else kb_main(query.from_user.id)
        try:
            await query.message.edit_text(header, reply_markup=kb, parse_mode=ParseMode.HTML)
        except TelegramBadRequest as e:
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

    st = get_state(message.from_user.id)
    lang = st["lang"]
    await message.answer(f"Выполняю OCR (язык {lang})...")
    try:
        raw = run_ocr(local_path, lang=lang)
        logger.debug(f"OCR done len={len(raw)}")
    except Exception as e:
        logger.exception("OCR error")
        await message.answer(f"Ошибка OCR: {e}")
        return

    strategy = st["strategy"]
    llm = st["llm"]
    model = st.get("model", "gemini-2.5-flash")
    await message.answer(f"Коррекция LLM (стратегия {strategy}, {llm})...")
    corrected = run_llm_correction(raw, strategy=strategy, llm=llm, user_id=message.from_user.id, username=message.from_user.username or str(message.from_user.id), model_name=model)
    logger.debug(f"LLM corrected len={len(corrected)}")

    async def safe_send(text: str):
        pm = ParseMode.MARKDOWN if strategy in {"B", "C"} else None
        try:
            return await message.answer(text[:4000], parse_mode=pm)
        except TelegramBadRequest:
            return await message.answer(text[:4000])

    if st["debug"]:
        def html_escape(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        ocr_part = html_escape(raw)[:3500]
        await message.answer(f"<b>OCR ({lang})</b>\n<pre>{ocr_part}</pre>", parse_mode=ParseMode.HTML)
        await safe_send(corrected)
    else:
        await safe_send(corrected)


async def on_document(message: Message):
    logger.debug(
        f"on_document from={message.from_user.id} name={message.document.file_name} mime={message.document.mime_type}"
    )
    doc = message.document
    file_name = doc.file_name or "document"
    mime = doc.mime_type or ""
    file = await message.bot.get_file(doc.file_id)
    tmp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    local_path = os.path.join(tmp_dir, f"{doc.file_id}_{file_name}")
    await message.bot.download_file(file.file_path, local_path)

    st = get_state(message.from_user.id)
    lang = st["lang"]

    async def safe_send(text: str, strategy: str):
        pm = ParseMode.MARKDOWN if strategy in {"B", "C"} else None
        try:
            return await message.answer(text[:4000], parse_mode=pm)
        except TelegramBadRequest:
            return await message.answer(text[:4000])

    if mime.startswith("image/"):
        await message.answer("Обнаружено изображение в документе. Выполняю OCR...")
        try:
            raw = run_ocr(local_path, lang=lang)
        except Exception as e:
            await message.answer(f"Ошибка OCR: {e}")
            return
        strategy = st["strategy"]
        llm = st["llm"]
        model = st.get("model", "gemini-2.5-flash")
        corrected = run_llm_correction(
            raw,
            strategy=strategy,
            llm=llm,
            user_id=message.from_user.id,
            username=message.from_user.username or str(message.from_user.id),
            model_name=model,
        )
        if st["debug"]:
            def html_escape(s: str) -> str:
                return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            ocr_part = html_escape(raw)[:3500]
            await message.answer(f"<b>OCR ({lang})</b>\n<pre>{ocr_part}</pre>", parse_mode=ParseMode.HTML)
            await safe_send(corrected, strategy)
        else:
            await safe_send(corrected, strategy)
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
                img_path = os.path.join(tmp_dir, f"{doc.file_id}_page_{i + 1}.jpg")
                pages[i].save(img_path, "JPEG")
                try:
                    raw = run_ocr(img_path, lang=lang)
                    all_text.append(raw)
                except Exception as e:
                    all_text.append(f"[Ошибка OCR стр.{i + 1}] {e}")
            combined = "\n\n".join(all_text)
            strategy = st["strategy"]
            llm = st["llm"]
            model = st.get("model", "gemini-2.5-flash")
            corrected = run_llm_correction(
                combined,
                strategy=strategy,
                llm=llm,
                user_id=message.from_user.id,
                username=message.from_user.username or str(message.from_user.id),
                model_name=model,
            )
            if st["debug"]:
                def html_escape(s: str) -> str:
                    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                ocr_part = html_escape(combined)[:3500]
                await message.answer(
                    f"<b>OCR ({lang})</b>\n<pre>{ocr_part}</pre>", parse_mode=ParseMode.HTML
                )
                await safe_send(corrected, strategy)
            else:
                await safe_send(corrected, strategy)
        except Exception:
            await message.answer(
                "Для PDF требуется poppler и пакет pdf2image. Пока обработка PDF недоступна."
            )
    else:
        await message.answer("Пока поддерживаются изображения (jpg/png) и PDF при наличии pdf2image.")


async def on_text(message: Message):
    logger.debug(f"on_text from={message.from_user.id} len={len(message.text or '')}")
    st = get_state(message.from_user.id)
    provider = st.pop("await_key_provider", None)
    if provider:
        key = (message.text or "").strip()
        if not key:
            await message.answer("Пустой ключ — отправьте непустой текст.")
            return
        set_user_key(message.from_user.id, provider, key)
        if provider == "gemini":
            uid = message.from_user.id
            uname = message.from_user.username or str(uid)
            ok = api_set_key(uid, uname, provider, key)
            if ok:
                st["has_gemini"] = True
                await message.answer("Ключ для gemini сохранён на сервере и локально.")
            else:
                await message.answer("Ключ для gemini сохранён локально. Сервер: ошибка, смотрите /apilog.")
        elif provider == "yandex":
            await message.answer("Ключ для yandex сохранён локально. Формат поддерживается: <folder_id>:<api_key>.")
        else:
            await message.answer("Ключ для gigachat сохранён локально.")
        return


def register_handlers(dp):
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
