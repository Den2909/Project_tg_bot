import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import importlib
import torch
from unittest.mock import patch, MagicMock
from collections import OrderedDict

# Сохраняем реальный os.getenv
real_getenv = os.getenv

# Моки для Telegram API
@pytest.fixture(autouse=True)
def mock_telegram_api():
    def selective_getenv(key, default=None):
        if key == "TELEGRAM_BOT_TOKEN":
            return "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
        return real_getenv(key, default)
    
    with patch("os.getenv", side_effect=selective_getenv), \
         patch("aiogram.Bot", return_value=MagicMock()), \
         patch("aiogram.Dispatcher", return_value=MagicMock()) as mock_dispatcher:
        mock_dispatcher.return_value.message_handler = MagicMock(return_value=lambda x: x)
        mock_dispatcher.return_value.callback_query_handler = MagicMock(return_value=lambda x: x)
        yield

# Тест загрузки модели
def test_load_model(mocker):
    # Мокаем torch.load для load_model
    mocker.patch("torch.load", return_value={})
    # Мокаем VGG19_Weights.IMAGENET1K_V1.get_state_dict
    mock_state_dict = OrderedDict([
        # Features
        (f"features.{i}.{param}", torch.tensor(0.0))
        for i in [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        for param in ["weight", "bias"]
    ] + [
        # Classifier
        (f"classifier.{i}.{param}", torch.tensor(0.0))
        for i in [0, 3, 6]
        for param in ["weight", "bias"]
    ])
    mocker.patch(
        "torchvision.models.vgg.VGG19_Weights.IMAGENET1K_V1.get_state_dict",
        return_value=mock_state_dict
    )
    # Динамически импортируем app_v3
    app_v3 = importlib.import_module("app_v3")
    model = app_v3.load_model("vangogh")
    assert model is not None, "Model failed to load"
