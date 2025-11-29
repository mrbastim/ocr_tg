import cv2
import pytesseract
import os

# Настройка пути к Tesseract (если Windows)
pytesseract.pytesseract.tesseract_cmd = r"D:\tesseract-ocr\tesseract.exe"

# Явно указываем путь к языковым данным
os.environ['TESSDATA_PREFIX'] = r"D:\tesseract-ocr\tessdata"

def get_raw_text(image_path, lang='rus'):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть изображение по пути: {image_path}. Проверьте существование файла и рабочую директорию.")
    # Перевод в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Получение сырого текста с указанием конфигурации для Windows
    try:
        # Используем указанный язык
        text = pytesseract.image_to_string(gray, lang=lang, config='--psm 6')
    except pytesseract.pytesseract.TesseractError:
        # Попытка без языка (используется английский по умолчанию)
        print(f"Предупреждение: не удалось загрузить язык '{lang}', используется английский")
        text = pytesseract.image_to_string(gray, config='--psm 6')
    return text

def normalize_whitespace(text):
    # Замена нескольких пробелов на один и удаление лишних переносов строк
    lines = text.splitlines()
    normalized_lines = [' '.join(line.split()) for line in lines if line.strip()]
    return '\n'.join(normalized_lines)

if __name__ == "__main__":
    # Укажите корректный путь к файлу изображения
    image_path = r"D:\IMG_20251129_172545_204.jpg"
    try:
        raw_text = get_raw_text(image_path)
        print("--- RAW OCR OUTPUT ---")
        print(raw_text)
    except FileNotFoundError as e:
        print(str(e))
        print("Подсказка: запустите скрипт из папки с изображением или передайте абсолютный путь.")
    except Exception as e:
        print(f"Ошибка: {e}")
