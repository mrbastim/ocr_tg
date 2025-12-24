import pytest

from bot.keyboards import get_prompt_label, prompt_preview


def test_get_prompt_label_uses_preset_when_not_custom():
    custom = "Мой кастомный промт"
    assert get_prompt_label("weak", custom) == "Слабый"
    assert get_prompt_label("medium", custom) == "Средний"
    assert get_prompt_label("strong", custom) == "Сильный"


def test_get_prompt_label_custom_only_when_strategy_custom():
    custom = "Мой кастомный промт"
    assert "Свой" in get_prompt_label("custom", custom)


def test_prompt_preview_uses_preset_when_not_custom():
    custom = "<CUSTOM>"
    assert "опечатки" in prompt_preview("weak", custom)
    assert "пунктуацию" in prompt_preview("medium", custom)
    assert "Полная коррекция" in prompt_preview("strong", custom)


def test_prompt_preview_uses_custom_when_strategy_custom():
    custom = "<CUSTOM>"
    assert prompt_preview("custom", custom) == custom
