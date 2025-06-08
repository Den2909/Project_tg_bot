import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app_v3 import MODELS_CACHE, load_model  # Основные импорты

@pytest.fixture
def mock_message():
    """Фикстура для мока Message."""
    message = AsyncMock()
    message.chat.id = 12345
    message.from_user.id = 67890
    return message

def test_models_cache_initialization():
    """Тест: MODELS_CACHE — пустой словарь."""
    assert isinstance(MODELS_CACHE, dict)
    assert not MODELS_CACHE

@patch("app_v3.torch.load")
@patch("app_v3.nn.Module")
def test_load_model(mock_module, mock_torch_load):
    """Тест: load_model загружает модель в кэш."""
    mock_model = MagicMock()
    mock_torch_load.return_value = {"state_dict": "mocked"}
    mock_module.return_value = mock_model
    model = load_model("test_model", "checkpoints/model.pth")
    assert "test_model" in MODELS_CACHE
    assert MODELS_CACHE["test_model"] == mock_model

# Тест для функции из degradations.py, не использующей rgb_to_grayscale
def test_random_bivariate_gaussian():
    """Тест: random_bivariate_Gaussian создаёт нормализованный кернел."""
    import numpy as np
    from app_v3 import random_bivariate_Gaussian  # Предполагаем, что функция скопирована в app_v3.py
    kernel = random_bivariate_Gaussian(
        kernel_size=21,
        sigma_x_range=(0.6, 5),
        sigma_y_range=(0.6, 5),
        rotation_range=(-np.pi, np.pi),
        isotropic=True
    )
    assert kernel.shape == (21, 21)
    assert np.isclose(np.sum(kernel), 1.0, atol=1e-6)  # Нормализация
    assert np.all(kernel >= 0)  # Кернел неотрицательный

# Опциональный тест для хэндлера
@patch("app_v3.realesrgan", None)  # Изолируем realesrgan
@pytest.mark.asyncio
async def test_start_handler(mock_message):
    """Тест: start_handler отправляет приветствие."""
    from app_v3 import start_handler  # Ленивый импорт
    await start_handler(mock_message)
    mock_message.answer.assert_called_once_with("Привет! Отправь фото для стилизации с помощью /style.")
