import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import numpy as np
from PIL import Image
import cv2
import tempfile
from unittest.mock import patch

# Фикстура для создания временного изображения
@pytest.fixture
def temp_image():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        img = Image.new("RGB", (256, 256), color="red")
        img.save(temp.name)
        yield temp.name
    os.unlink(temp.name)

# Мок для os.getenv, чтобы обойти проверку TELEGRAM_BOT_TOKEN
@pytest.fixture(autouse=True)
def mock_telegram_token():
    with patch("os.getenv") as mock_getenv:
        mock_getenv.return_value = "fake_token"
        yield

# Импорт после мока
from app_v3 import process_image, enhance_image, style_transfer_nst_sync, STYLE_MODELS, load_model, load_enhance_model

# Пропускаем тесты, если нет GPU, чтобы не зависеть от окружения
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

# Тест для проверки загрузки модели
def test_load_model(mocker):
    mocker.patch("torch.load", return_value={})
    for style_key in STYLE_MODELS:
        model = load_model(style_key)
        assert model is not None, f"Model for {style_key} failed to load"
        assert isinstance(model, torch.nn.Module), f"Model for {style_key} is not a torch module"

# Тест для process_image
def test_process_image(temp_image, mocker):
    style_key = "vangogh"
    mocker.patch("torch.load", return_value={})
    output_path = process_image(temp_image, style_key)
    assert os.path.exists(output_path), "Output image was not created"
    img = Image.open(output_path)
    assert img.size == (256, 256), "Output image size is incorrect"
    os.unlink(output_path)

# Тест для enhance_image
def test_enhance_image(temp_image, mocker):
    mocker.patch("realesrgan.RealESRGANer.enhance", return_value=(np.zeros((512, 512, 3), dtype=np.uint8), None))
    output_path = enhance_image(temp_image)
    assert os.path.exists(output_path), "Enhanced image was not created"
    img = Image.open(output_path)
    assert img.size[0] >= 256 * 2, "Enhanced image size is too small"
    os.unlink(output_path)

# Тест для style_transfer_nst_sync
def test_style_transfer_nst_sync(temp_image, mocker):
    mocker.patch("torch.load", return_value={})
    mocker.patch("app_v3.denormalize_and_save", return_value=np.zeros((512, 512, 3), dtype=np.uint8))
    output_path = tempfile.mktemp(suffix=".jpg")
    style_transfer_nst_sync(temp_image, temp_image, output_path, num_steps=1)
    assert os.path.exists(output_path), "NST output image was not created"
    img = Image.open(output_path)
    assert img.size == (512, 512), "NST output image size is incorrect"
    os.unlink(output_path)
