import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import importlib
import torch
from unittest.mock import MagicMock
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
         patch("aiogram.Bot", return_value=MagicMock()):
        yield

# Тест загрузки модели
def test_load_model(mocker):
    # Мокаем aiogram.Dispatcher до импорта app_v3
    mock_dispatcher = MagicMock()
    mock_dispatcher.message_handler = MagicMock(return_value=lambda x: x)
    mock_dispatcher.callback_query_handler = MagicMock(return_value=lambda x: x)
    mocker.patch("aiogram.Dispatcher", return_value=mock_dispatcher)
    
    # Мокаем torch.load для load_model
    mocker.patch("torch.load", return_value={})
    # Мокаем VGG19_Weights.IMAGENET1K_V1.get_state_dict
    feature_shapes = {
        0: ([64, 3, 3, 3], [64]),      # weight: [out, in, k, k], bias: [out]
        2: ([64, 64, 3, 3], [64]),
        5: ([128, 64, 3, 3], [128]),
        7: ([128, 128, 3, 3], [128]),
        10: ([256, 128, 3, 3], [256]),
        12: ([256, 256, 3, 3], [256]),
        14: ([256, 256, 3, 3], [256]),
        16: ([256, 256, 3, 3], [256]),
        19: ([512, 256, 3, 3], [512]),
        21: ([512, 512, 3, 3], [512]),
        23: ([512, 512, 3, 3], [512]),
        25: ([512, 512, 3, 3], [512]),
        28: ([512, 512, 3, 3], [512]),
        30: ([512, 512, 3, 3], [512]),
        32: ([512, 512, 3, 3], [512]),
        34: ([512, 512, 3, 3], [512]),
    }
    classifier_shapes = {
        0: ([4096, 25088], [4096]),    # weight: [out, in], bias: [out]
        3: ([4096, 4096], [4096]),
        6: ([1000, 4096], [1000]),
    }
    mock_state_dict = OrderedDict([
        (f"features.{i}.{param}", torch.zeros(shape))
        for i, (weight_shape, bias_shape) in feature_shapes.items()
        for param, shape in [("weight", weight_shape), ("bias", bias_shape)]
    ] + [
        (f"classifier.{i}.{param}", torch.zeros(shape))
        for i, (weight_shape, bias_shape) in classifier_shapes.items()
        for param, shape in [("weight", weight_shape), ("bias", bias_shape)]
    ])
    mocker.patch(
        "torchvision.models.vgg.VGG19_Weights.IMAGENET1K_V1.get_state_dict",
        return_value=mock_state_dict
    )
    # Динамически импортируем app_v3
    app_v3 = importlib.import_module("app_v3")
    model = app_v3.load_model("vangogh")
    assert model is not None, "Model failed to load"
