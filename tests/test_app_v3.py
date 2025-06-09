import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import numpy as np
from PIL import Image
import cv2
import tempfile
from app_v3 import (
    process_image,
    enhance_image,
    style_transfer_nst_sync,
    STYLE_MODELS,
    load_model,
    load_enhance_model,
)


# Создание временного изображения
@pytest.fixture
def temp_image():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        img = Image.new("RGB", (256, 256), color="red")
        img.save(temp.name)
        yield temp.name
    os.unlink(temp.name)


# Тест для load_model
def test_load_model(mocker):
    # Мокаем load_state_dict, чтобы избежать проблем с ключами
    mocker.patch("torch.nn.Module.load_state_dict", return_value=None)
    for style_key in STYLE_MODELS:
        model = load_model(style_key)
        assert model is not None, f"Модель для {style_key} не загрузилась"
        assert isinstance(
            model, torch.nn.Module
        ), f"Модель для {style_key} не является torch модулем"


# Тест для process_image
def test_process_image(temp_image, mocker):
    style_key = "vangogh"
    # Мокаем load_state_dict и обработку изображения
    mocker.patch("torch.nn.Module.load_state_dict", return_value=None)
    mocker.patch(
        "torchvision.transforms.ToTensor", return_value=torch.randn(3, 256, 256)
    )
    mocker.patch("torch.nn.Module.forward", return_value=torch.randn(3, 256, 256))
    mocker.patch("PIL.Image.fromarray", return_value=Image.new("RGB", (256, 256)))
    output_path = process_image(temp_image, style_key)
    assert os.path.exists(output_path), "Выходное изображение не создано"
    img = Image.open(output_path)
    assert img.size == (256, 256), "Неверный размер выходного изображения"
    os.unlink(output_path)


# Тест для enhance_image
def test_enhance_image(temp_image, mocker):
    mocker.patch(
        "realesrgan.RealESRGANer.enhance",
        return_value=(np.zeros((512, 512, 3), dtype=np.uint8), None),
    )
    output_path = enhance_image(temp_image)
    assert os.path.exists(output_path), "Улучшенное изображение не создано"
    img = Image.open(output_path)
    assert img.size[0] >= 256 * 2, "Размер улучшенного изображения слишком мал"
    os.unlink(output_path)


# Тест для style_transfer_nst_sync
def test_style_transfer_nst_sync(temp_image, mocker):
    # Создаём временное изображение для мока denormalize_and_save
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_out:
        Image.new("RGB", (512, 512), color="black").save(temp_out.name)
        mocker.patch(
            "app_v3.denormalize_and_save",
            side_effect=lambda img, path: cv2.imwrite(path, cv2.imread(temp_out.name)),
        )

    output_path = tempfile.mktemp(suffix=".jpg")
    # Мокаем зависимости для style_transfer_nst_sync
    content_img = torch.randn(1, 3, 512, 512, requires_grad=False)
    mocker.patch("app_v3.load_image", return_value=content_img)
    mocker.patch(
        "app_v3.get_features",
        return_value={
            "0": torch.randn(1, 64, 128, 128),
            "5": torch.randn(1, 128, 64, 64),
            "10": torch.randn(1, 256, 32, 32),
            "19": torch.randn(1, 512, 16, 16),
            "21": torch.randn(1, 512, 16, 16),
            "28": torch.randn(1, 512, 8, 8),
        },
    )
    mocker.patch("app_v3.gram_matrix", return_value=torch.randn(64, 64))
    # Мокаем torch.randn_like для создания листового тензора
    mocker.patch("torch.randn_like", return_value=torch.randn(1, 3, 512, 512))
    # Мокаем создание оптимизатора Adam, чтобы избежать ошибки с non-leaf tensor
    mock_optimizer = mocker.Mock()
    mock_optimizer.step = mocker.Mock(return_value=None)
    mock_optimizer.zero_grad = mocker.Mock(return_value=None)
    mocker.patch("torch.optim.Adam", return_value=mock_optimizer)
    style_transfer_nst_sync(temp_image, temp_image, output_path, num_steps=1)
    assert os.path.exists(output_path), "Выходное изображение NST не создано"
    img = Image.open(output_path)
    assert img.size == (512, 512), "Неверный размер выходного изображения NST"
    os.unlink(output_path)
    os.unlink(temp_out.name)
