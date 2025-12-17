import os
import logging
import asyncio
from typing import Tuple

from aiogram import F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message
from aiogram.exceptions import TelegramBadRequest

from .api_client import API_DEBUG, API_LOG_FILE, api_login, api_register, api_key_status, api_set_key, api_clear_key
from .user_keys import set_user_key, delete_user_key, get_all_user_keys
from .keyboards import get_state, kb_main, kb_settings, token_status
from .llm_service import run_ocr, run_llm_correction

try:
    from ml.train_models import run_all as ml_run_all
except Exception:
    ml_run_all = None

try:
    from ml.doc_classifier import classify_document_text
except Exception:
    classify_document_text = None

try:
    from ml.processing_regression import (
        build_processing_summary,
        extract_image_features,
        predict_ocr_time,
    )
except Exception:
    build_processing_summary = None
    extract_image_features = None
    predict_ocr_time = None

try:
    from ml.event_logger import log_event
except Exception:
    log_event = None

try:
    from ml.stats_plots import build_basic_plots
except Exception:
    build_basic_plots = None

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
        "/lang rus|eng|rus+eng — выбрать язык OCR\n"
        "/llm gigachat|gemini|yandex|api — выбрать провайдера LLM (api = внешний сервер)\n"
        "/debug on|off — включить/выключить вывод OCR и LLM\n"
        "/apilog — последние строки лога интеграции (AI_API_DEBUG=1)\n"
        "/testlogin — выполнить попытку логина и показать сырой ответ\n"
        "/setkey <gigachat|gemini|yandex> <ключ> — сохранить личный API-ключ\n"
        "/delkey <gigachat|gemini|yandex> — удалить личный API-ключ\n"
        "/mykeys — показать, какие ключи сохранены\n"
        "/ml_demo — учебный пример моделирования на внешнем датасете (для отчёта)\n"
        "/ml_stats — построить графики по реальным логам обработки изображений\n"
        "/ml_requirements — показать, как ML-подпроект закрывает пункты 3–6 задания\n"
        "Пришлите фото/скан или документ для OCR и коррекции",
    )


async def cmd_ml_demo(message: Message):
    logger.debug(f"/ml_demo from={message.from_user.id}")
    if ml_run_all is None:
        await message.answer("ML-модуль недоступен на этом развёртывании.")
        return

    await message.answer(
        "Запускаю учебный ML-эксперимент на текстовом датасете "
        "(20 Newsgroups). Это может занять некоторое время..."
    )

    loop = asyncio.get_running_loop()
    try:
        metrics = await loop.run_in_executor(None, ml_run_all, None, None)
    except Exception as e:
        logger.exception("ML demo failed")
        await message.answer(f"Ошибка при выполнении ML-демо: {e}")
        return

    def fmt(v):
        try:
            return f"{float(v):.3f}"
        except Exception:
            return "—"

    reg = metrics.get("regression", {})
    cls = metrics.get("classification", {})
    clu = metrics.get("clustering", {})

    lines = [
        "Учебный ML-эксперимент завершён.",
        f"Объектов: {metrics.get('n_samples')} | Признаков: {metrics.get('n_features')}",
        "",
        "Регрессия (длина документа в словах):",
        f"R² = {fmt(reg.get('r2'))}, MAE = {fmt(reg.get('mae'))}, MSE = {fmt(reg.get('mse'))}",
        "",
        "Классификация (тематика документа):",
    ]

    for name, m in cls.items():
        lines.append(f"{name}: accuracy = {fmt(m.get('acc'))}, F1 = {fmt(m.get('f1'))}")

    lines.extend(
        [
            "",
            "Кластеризация (k-means по тем же признакам):",
            f"k = {clu.get('k')}, silhouette = {fmt(clu.get('silhouette'))}, ARI = {fmt(clu.get('ari'))}",
            "",
            "Графики (распределения ошибок, сравнение моделей, кластеры) и сохранённые модели",
            "лежат в папке ml_output/ на сервере.",
        ]
    )

    await message.answer("\n".join(lines))


async def cmd_ml_stats(message: Message):
    logger.debug(f"/ml_stats from={message.from_user.id}")
    if build_basic_plots is None:
        await message.answer("Аналитика по логам недоступна в этом окружении.")
        return

    try:
        plots = build_basic_plots()
    except FileNotFoundError:
        await message.answer("Пока нет логов обработки изображений. Отправьте несколько картинок боту.")
        return
    except Exception as e:
        logger.exception("build_basic_plots failed")
        await message.answer(f"Ошибка при построении графиков: {e}")
        return

    if not plots:
        await message.answer("Логи найдены, но нечего визуализировать.")
        return

    from aiogram.types import FSInputFile

    for path, caption in plots:
        try:
            photo = FSInputFile(path)
            await message.answer_photo(photo, caption=caption)
        except Exception as e:
            logger.debug(f"send plot failed for {path}: {e}")


async def cmd_ml_requirements(message: Message):
    logger.debug(f"/ml_requirements from={message.from_user.id}")
    text = (
        "3. Визуализация результатов моделирования (графики, диаграммы, таблицы).\n"
        "   • Команда /ml_stats строит и сохраняет графики на основе реальных логов работы бота:\n"
        "     - time_distribution.png — распределение прогнозируемого и фактического времени обработки,\n"
        "     - doc_types.png — распределение типов изображений (чек, форма, схема и т.п.).\n"
        "\n"
        "4. Документированный и структурированный код.\n"
        "   • Код ML-подпроекта разнесён по модулям: ml/train_models.py, ml/doc_classifier.py, ml/processing_regression.py, ml/event_logger.py, ml/stats_plots.py.\n"
        "   • В файлах используются docstring-и и комментарии, описывающие назначение функций и шаги обработки.\n"
        "\n"
        "5. Оценка качества моделей и сравнительный анализ.\n"
        "   • /ml_demo по-прежнему показывает пример метрик на внешнем датасете,\n"
        "     а логи позволяют дополнительно оценивать качество и скорость реальной обработки,\n"
        "     например сравнивать прогнозируемое и фактическое время.\n"
        "\n"
        "6. Подробный отчёт о проделанной работе.\n"
        "   • В качестве отчёта можно использовать связку: /ml_demo, /ml_stats + данный текст:\n"
        "     - описание лог-датасета (реальные события обработки изображений),\n"
        "     - перечисление использованных моделей и эвристик (регрессия времени, классификация типов документов),\n"
        "     - анализ: сравнение прогноз/факт по времени и распределение типов документов.\n"
        "   • При необходимости этот текст можно перенести в отчёт (PDF/Docx) и дополнить скриншотами графиков.\n"
    )
    await message.answer(text)


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
        await message.answer("Укажите язык: rus, eng или rus+eng")
        return
    lang = args[1].lower()
    if lang not in {"rus", "eng", "rus+eng"}:
        await message.answer("Допустимые значения: rus, eng, rus+eng")
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


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_llm_label(llm_raw: str) -> str:
    """Красивое имя LLM для заголовка.

    Внутри состояния 'api' сейчас означает Gemini (внешний сервер).
    """

    llm = (llm_raw or "").lower()
    if llm == "gigachat":
        return "GigaChat"
    if llm in {"gemini", "api"}:
        return "Gemini"
    return llm_raw or "LLM"


def _build_extra_info(image_path: str | None, raw_text: str) -> str:
    """Собрать дополнительную строку для ответа: тип документа.

    Время обработки теперь показывается отдельным сообщением ДО OCR.
    """

    parts = []

    if classify_document_text and raw_text and raw_text.strip():
        try:
            label, _ = classify_document_text(raw_text)
            label_map = {
                "receipt": "чек / кассовый документ",
                "screenshot": "скриншот экрана",
                "document": "скан текста / отчёт",
                "form": "форма / анкета",
                "diagram": "техническая схема / график",
                "unknown": "тип не распознан / недостаточно текста",
            }
            human = label_map.get(label, label)
            parts.append(f"Тип изображения (по тексту): {human}")
        except Exception as e:
            logger.debug(f"doc_type classification failed: {e}")

    return "\n".join(parts) if parts else ""


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
    elif data == "ml_requirements":
        await cmd_ml_requirements(query.message)

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
        import time as _time

        t0 = _time.perf_counter()
        raw = run_ocr(local_path, lang=lang)
        t1 = _time.perf_counter()
        logger.debug(f"OCR done len={len(raw)}")
    except Exception as e:
        logger.exception("OCR error")
        await message.answer(f"Ошибка OCR: {e}")
        return

    strategy = st["strategy"]
    llm = st["llm"]
    
    # Оценка времени ПОСЛЕ OCR (когда известен реальный text для word_count)
    predicted_time = None
    feats = None
    if extract_image_features and predict_ocr_time:
        try:
            feats = extract_image_features(local_path, raw)
            predicted_time = predict_ocr_time(feats)
            await message.answer(f"Это займёт примерно {predicted_time:.1f} секунды обработки.")
        except Exception as e:
            logger.debug(f"processing time estimate failed: {e}")
    
    await message.answer(f"Коррекция LLM (стратегия {strategy}, {llm})...")
    corrected = run_llm_correction(
        raw,
        strategy=strategy,
        llm=llm,
        user_id=message.from_user.id,
        username=message.from_user.username or str(message.from_user.id),
    )
    t2 = _time.perf_counter()
    logger.debug(f"LLM corrected len={len(corrected)}")

    extra = _build_extra_info(local_path, raw)

    async def send_llm_result(text: str, extra_text: str):
        body = _html_escape(text)[:3500]
        html = f"<b>LLM ({_format_llm_label(llm)})</b>\n<pre>{body}</pre>"
        if extra_text:
            html += f"\n\n{_html_escape(extra_text)}"
        try:
            return await message.answer(html, parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            return await message.answer(text[:4000])

    if st["debug"]:
        ocr_part = _html_escape(raw)[:3500]
        await message.answer(f"<b>OCR ({lang})</b>\n<pre>{ocr_part}</pre>", parse_mode=ParseMode.HTML)
        await send_llm_result(corrected, extra)
    else:
        await send_llm_result(corrected, extra)

    # Логируем событие обработки для последующего анализа
    if log_event:
        try:
            if feats is None and extract_image_features:
                try:
                    feats = extract_image_features(local_path, raw)
                except Exception:
                    pass
            ocr_time = t1 - t0
            total_time = t2 - t0
            log_event(
                image_path=local_path,
                text=raw,
                user_id=message.from_user.id,
                provider=_format_llm_label(llm),
                source="photo",
                is_pdf=False,
                t_ocr=ocr_time,
                t_total=total_time,
                predicted_time=predicted_time,
            )
        except Exception as e:
            logger.debug(f"ml log_event (photo) failed: {e}")


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

    async def send_llm_result(text: str, extra_text: str, llm_label: str):
        body = _html_escape(text)[:3500]
        html = f"<b>LLM ({_format_llm_label(llm_label)})</b>\n<pre>{body}</pre>"
        if extra_text:
            html += f"\n\n{_html_escape(extra_text)}"
        try:
            return await message.answer(html, parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            return await message.answer(text[:4000])

    if mime.startswith("image/"):
        await message.answer("Обнаружено изображение в документе. Выполняю OCR...")
        try:
            import time as _time

            t0 = _time.perf_counter()
            raw = run_ocr(local_path, lang=lang)
            t1 = _time.perf_counter()
        except Exception as e:
            await message.answer(f"Ошибка OCR: {e}")
            return
        strategy = st["strategy"]
        llm = st["llm"]
        
        # Оценка времени ПОСЛЕ OCR
        predicted_time = None
        feats = None
        if extract_image_features and predict_ocr_time:
            try:
                feats = extract_image_features(local_path, raw)
                predicted_time = predict_ocr_time(feats)
                await message.answer(f"Это займёт примерно {predicted_time:.1f} секунды обработки.")
            except Exception as e:
                logger.debug(f"processing time estimate (document) failed: {e}")
        
        corrected = run_llm_correction(
            raw,
            strategy=strategy,
            llm=llm,
            user_id=message.from_user.id,
            username=message.from_user.username or str(message.from_user.id),
        )
        t2 = _time.perf_counter()
        extra = _build_extra_info(local_path, raw)

        if st["debug"]:
            ocr_part = _html_escape(raw)[:3500]
            await message.answer(f"<b>OCR ({lang})</b>\n<pre>{ocr_part}</pre>", parse_mode=ParseMode.HTML)
            await send_llm_result(corrected, extra, llm)
        else:
            await send_llm_result(corrected, extra, llm)
        # логирование события
        if log_event:
            try:
                if feats is None and extract_image_features:
                    try:
                        feats = extract_image_features(local_path, raw)
                    except Exception:
                        pass
                ocr_time = t1 - t0
                total_time = t2 - t0
                log_event(
                    image_path=local_path,
                    text=raw,
                    user_id=message.from_user.id,
                    provider=_format_llm_label(llm),
                    source="document_image",
                    is_pdf=False,
                    t_ocr=ocr_time,
                    t_total=total_time,
                    predicted_time=predicted_time,
                )
            except Exception as e:
                logger.debug(f"ml log_event (document image) failed: {e}")
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
            corrected = run_llm_correction(
                combined,
                strategy=strategy,
                llm=llm,
                user_id=message.from_user.id,
                username=message.from_user.username or str(message.from_user.id),
            )
            # для PDF используем первую сохранённую страницу как представителя изображения
            first_page_path = os.path.join(tmp_dir, f"{doc.file_id}_page_1.jpg") if pages else None
            extra = _build_extra_info(first_page_path, combined)
            if st["debug"]:
                ocr_part = _html_escape(combined)[:3500]
                await message.answer(
                    f"<b>OCR ({lang})</b>\n<pre>{ocr_part}</pre>", parse_mode=ParseMode.HTML
                )
                await send_llm_result(corrected, extra, llm)
            else:
                await send_llm_result(corrected, extra, llm)
            # логирование события для PDF (по первой странице)
            if log_event and first_page_path is not None:
                try:
                    import time as _time

                    # Здесь точное разбиение времени сложнее, поэтому считаем только total
                    # как время прохода цикла, без отдельного OCR/LLM per page.
                    # Чтобы не усложнять, ставим ocr_time = total_time.
                    # В реальном проекте это можно детализировать.
                    total_time = 0.0
                    log_event(
                        image_path=str(first_page_path),
                        text=combined,
                        user_id=message.from_user.id,
                        provider=_format_llm_label(llm),
                        source="pdf",
                        is_pdf=True,
                        t_ocr=total_time,
                        t_total=total_time,
                        predicted_time=None,
                    )
                except Exception as e:
                    logger.debug(f"ml log_event (pdf) failed: {e}")
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
    dp.message.register(cmd_ml_demo, F.text.startswith("/ml_demo"))
    dp.message.register(cmd_ml_stats, F.text.startswith("/ml_stats"))
    dp.message.register(cmd_ml_requirements, F.text.startswith("/ml_requirements"))

    dp.callback_query.register(on_btn)
    dp.message.register(on_photo, F.photo)
    dp.message.register(on_document, F.document)
    dp.message.register(on_text, F.text)
