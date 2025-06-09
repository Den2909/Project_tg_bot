import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import importlib
from unittest.mock import patch, MagicMock

# Моки для Telegram API
@pytest.fixture(autouse=True)
def mock_telegram_api():
    def selective_getenv(key, default=None):
        if key == "TELEGRAM_BOT_TOKEN":
            return "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
        return os.getenv(key, default)
    
    with patch("os.getenv", side_effect=selective_getenv), \
         patch("aiogram.Bot", return_value=MagicMock()):
        yield

# Тест загрузки модели
def test_load_model(mocker):
    # Мокаем torch.load
    mocker.patch("torch.load", return_value={})
    # Динамически импортируем app_v3
    app_v3 = importlib.import_module("app_v3")
    model = app_v3.load_model("vangogh")
    assert model is not None, "Model failed to load"
