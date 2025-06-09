import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
from unittest.mock import patch, MagicMock

# Моки для Telegram API
@pytest.fixture(autouse=True)
def mock_telegram_api():
    with patch("os.getenv", return_value="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"), \
         patch("aiogram.Bot", return_value=MagicMock()), \
         patch("aiogram.Dispatcher", return_value=MagicMock()) as mock_dispatcher:
        mock_dispatcher.return_value.message_handler = MagicMock(return_value=lambda x: x)
        mock_dispatcher.return_value.callback_query_handler = MagicMock(return_value=lambda x: x)
        yield

# Мок для окружения без GPU
@pytest.fixture(autouse=True)
def mock_no_gpu():
    with patch("torch.cuda.is_available", return_value=False):
        yield

# Импорт после моков
from app_v3 import process_image, enhance_image, style_transfer_nst_sync, STYLE_MODELS, load_model

# Фикстура для временного изображения
@pytest.fixture
def temp_image():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        img = Image.new("RGB", (256, 256), color="red")
        img.save(temp.name)
        yield temp.name
    os.unlink(temp.name)

# Тест загрузки модели
def test_load_model(mocker):
    mocker.patch("torch.load", return_value={})
    model = load_model("vangogh")
    assert model is not None, "Model failed to load"

# Тест обработки изображения
def test_process_image(temp_image, mocker):
    mocker.patch("torch.load", return_value={})
    output_path = process_image(temp_image, "vangogh")
    assert os.path.exists(output_path), "Output image not created"
    os.unlink(output_path)

# Тест улучшения изображения
def test_enhance_image(temp_image, mocker):
    mocker.patch("realesrgan.RealESRGANer.enhance", return_value=(np.zeros((512, 512, 3), dtype=np.uint8), None))
    output_path = enhance_image(temp_image)
    assert os.path.exists(output_path), "Enhanced image not created"
    os.unlink(output_path)

# Тест NST
def test_style_transfer_nst_sync(temp_image, mocker):
    mocker.patch("torch.load", return_value={})
    mocker.patch("app_v3.denormalize_and_save", return_value=np.zeros((512, 512, 3), dtype=np.uint8))
    output_path = tempfile.mktemp(suffix=".jpg")
    style_transfer_nst_sync(temp_image, temp_image, output_path, num_steps=1)
    assert os.path.exists(output_path), "NST output image not created"
    os.unlink(output_path)
