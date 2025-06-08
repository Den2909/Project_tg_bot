import pytest
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.fixture
def mock_env():
    """Мок переменных окружения."""
    with patch("app_v3.os.getenv") as mock_getenv:
        mock_getenv.return_value = "mocked_token"
        yield mock_getenv

@pytest.fixture
def mock_message():
    """Фикстура для мока Message."""
    message = AsyncMock()
    message.chat.id = 12345
    message.from_user.id = 67890
    return message

@patch("app_v3.realesrgan", None)  # Изолируем realesrgan
@patch("app_v3.torchvision", None)  # Изолируем torchvision
def test_models_cache_initialization(mock_env):
    """Тест: MODELS_CACHE — пустой словарь."""
    from app_v3 import MODELS_CACHE
    assert isinstance(MODELS_CACHE, dict)
    assert not MODELS_CACHE

@patch("app_v3.realesrgan", None)
@patch("app_v3.torchvision", None)
@patch("app_v3.torch.load")
@patch("app_v3.nn.Module")
def test_load_model(mock_module, mock_torch_load, mock_env):
    """Тест: load_model загружает модель в кэш."""
    from app_v3 import load_model, MODELS_CACHE
    mock_model = MagicMock()
    mock_torch_load.return_value = {"state_dict": "mocked"}
    mock_module.return_value = mock_model
    model = load_model("test_model", "checkpoints/model.pth")
    assert "test_model" in MODELS_CACHE
    assert MODELS_CACHE["test_model"] == mock_model

@pytest.mark.asyncio
@patch("app_v3.realesrgan", None)
@patch("app_v3.torchvision", None)
async def test_start_handler(mock_message, mock_env):
    """Тест: start_handler отправляет приветствие."""
    from app_v3 import start_handler
    await start_handler(mock_message)
    mock_message.answer.assert_called_once_with("Привет! Отправь фото для стилизации с помощью /style.")
