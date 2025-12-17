from __future__ import annotations

"""Пакетная загрузка локальных изображений в ML-лог бота.

Сценарий использования:

1. Скинуть папку с картинками (jpg/png/jpeg) в корень проекта, например `dataset/`.
2. Запустить из корня:

    python -m ml.batch_import --dir dataset --lang rus+eng

3. Скрипт для КАЖДОГО изображения:
   - запустит OCR (через ocr.base.get_raw_text),
   - измерит время OCR,
   - запишет событие в ml_output/events.csv через ml.event_logger.log_event.

После этого можно строить графики через /ml_stats и обучать модели через

    python -m ml.train_models --csv ml_output/events.csv --target-col total_time
"""

import argparse
import random
import time
from pathlib import Path
from typing import Iterable, List, Optional

from ocr.base import get_raw_text, normalize_whitespace
from .event_logger import log_event, LOG_FILE


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(root: Path, recursive: bool = True) -> Iterable[Path]:
    """Найти все изображения в указанной папке."""

    if recursive:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path
    else:
        for path in root.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def process_image(path: Path, *, lang: str, user_id: int, provider: str, source: str) -> None:
    """Обработать одно изображение и записать запись в events.csv."""

    t0 = time.perf_counter()
    raw = get_raw_text(str(path), lang=lang)
    text = normalize_whitespace(raw)
    t1 = time.perf_counter()

    ocr_time = t1 - t0
    total_time = ocr_time  # без LLM считаем, что всё время ушло на OCR

    log_event(
        image_path=str(path),
        text=text,
        user_id=user_id,
        provider=provider,
        source=source,
        is_pdf=False,
        t_ocr=ocr_time,
        t_total=total_time,
        predicted_time=None,
    )


def run_batch_import(
    directory: str,
    *,
    lang: str,
    user_id: int,
    provider: str,
    recursive: bool,
    max_files: Optional[int] = None,
) -> int:
    """Запустить пакетную обработку для всех изображений в каталоге.

    Возвращает количество успешно обработанных файлов.
    """

    root = Path(directory)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Каталог не найден или не является папкой: {root}")

    images = list(iter_images(root, recursive=recursive))

    if max_files is not None:
        random.shuffle(images)
        images = images[:max_files]

    count = 0
    for img_path in images:
        try:
            process_image(img_path, lang=lang, user_id=user_id, provider=provider, source="offline_batch")
            count += 1
        except Exception as e:
            # В учебном скрипте просто печатаем ошибку и идём дальше
            print(f"[SKIP] {img_path}: {e}")

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Пакетная обработка локальных изображений: запустить OCR и записать события "
            "в ml_output/events.csv так же, как это делает Telegram-бот."
        )
    )
    parser.add_argument("--dir", required=True, help="Каталог с изображениями (jpg/png/jpeg/bmp/tiff)")
    parser.add_argument(
        "--lang",
        default="rus+eng",
        help="Язык OCR для Tesseract (по умолчанию rus+eng, как у бота)",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=0,
        help="Идентификатор пользователя для логов (по умолчанию 0)",
    )
    parser.add_argument(
        "--provider",
        default="offline",
        help="Имя провайдера LLM для логов (например, offline/Gemini/GigaChat). Влияет только на графики.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Не искать файлы рекурсивно, только в указанной папке",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Ограничить число обрабатываемых файлов (берём случайные)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recursive = not args.no_recursive
    print(f"Каталог: {args.dir}")
    print(f"Язык OCR: {args.lang}")
    print(f"user_id для логов: {args.user_id}, provider: {args.provider}")

    n = run_batch_import(
        args.dir,
        lang=args.lang,
        user_id=args.user_id,
        provider=args.provider,
        recursive=recursive,
        max_files=args.max_files,
    )
    print(f"Готово. Обработано файлов: {n}.")
    if LOG_FILE.exists():
        print(f"Логи записаны в: {LOG_FILE.resolve()}")


if __name__ == "__main__":
    main()
