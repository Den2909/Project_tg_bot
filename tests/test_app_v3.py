import pytest
import torch
import torch.nn as nn
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict
from aiogram import Bot, Dispatcher
from aiogram.types import Message, InputFile
from app_v3 import MODELS_CACHE, load_model, start_handler, style_handler

@pytest.fixture
def mock_bot():
    """Фикстура для мока Bot."""
    bot = AsyncMock(spec=Bot)
    return bot

@pytest.fixture
def mock_dispatcher():
    """Фикстура для мока Dispatcher."""
    dp = MagicMock(spec=Dispatcher)
    return dp

@pytest.fixture
def mock_message():
    """Фикстура для мока Message."""
    message = AsyncMock(spec=Message)
    message.chat.id = 12345
    message.from_user.id = 67890
    return message

def test_models_cache_initialization():
    """Тест: MODELS_CACHE инициализируется как пустой словарь."""
    assert isinstance(MODELS_CACHE, dict)
    assert len(MODELS_CACHE) == 0
    assert MODELS_CACHE == {}

@patch("app_v3.torch.load")
@patch("app_v3.nn.Module")
def test_load_model(mock_module, mock_torch_load):
    """Тест: load_model загружает модель и добавляет в MODELS_CACHE."""
    mock_model = MagicMock(spec=nn.Module)
    mock_torch_load.return_value = {"state_dict": "mocked_state"}
    mock_module.return_value = mock_model

    model_path = "checkpoints/model.pth"
    model_name = "test_model"
    model = load_model(model_name, model_path)

    assert model_name in MODELS_CACHE
    assert MODELS_CACHE[model_name] == mock_model
    mock_torch_load.assert_called_once_with(model_path, map_location="cpu")
    mock_model.load_state_dict.assert_called_once_with("mocked_state")

@pytest.mark.asyncio
async def test_start_handler(mock_bot, mock_message):
    """Тест: start_handler отправляет приветственное сообщение."""
    with patch.object(mock_message, "answer", new=AsyncMock()) as mock_answer:
        await start_handler(mock_message)
        mock_answer.assert_called_once()
        args, _ = mock_answer.call_args
        assert "Привет! Я бот для переноса стиля" in args[0]

@pytest.mark.asyncio
async def test_style_handler_no_photo(mock_bot, mock_message):
    """Тест: style_handler запрашивает фото, если оно не отправлено."""
    with patch.object(mock_message, "answer", new=AsyncMock()) as mock_answer:
        await style_handler(mock_message)
        mock_answer.assert_called_once_with("Пожалуйста, отправьте изображение для стилизации.")

@pytest.mark.asyncio
@patch("app_v3.load_model")
@patch("app_v3.torchvision.transforms.ToTensor")
@patch("app_v3.PIL.Image.open")
async def test_style_handler_with_photo(mock_pil_open, mock_to_tensor, mock_load_model, mock_bot, mock_message):
    """Тест: style_handler обрабатывает фото и отправляет результат."""
    mock_message.photo = [MagicMock(file_id="photo_id")]
    mock_bot.get_file = AsyncMock(return_value=MagicMock(file_path="photo.jpg"))
    mock_bot.download_file = AsyncMock(return_value=b"image_data")
    mock_pil_image = MagicMock()
    mock_pil_open.return_value = mock_pil_image
    mock_tensor = MagicMock()
    mock_to_tensor.return_value = mock_tensor
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_model.return_value = mock_tensor
    mock_message.answer_photo = AsyncMock()

    await style_handler(mock_message)

    mock_load_model.assert_called_once()
    mock_pil_open.assert_called_once()
    mock_to_tensor.assert_called_once()
    mock_message.answer_photo.assert_called_once()

if __name__ == "__main__":
    pytest.main()
