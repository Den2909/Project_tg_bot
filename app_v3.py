import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils import executor
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from PIL import Image
import cv2
import numpy as np
from models.networks import define_G, init_weights
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import sys


# Инициализация бота
if "pytest" not in sys.modules:
    API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not API_TOKEN:
        raise ValueError("Не задан TELEGRAM_BOT_TOKEN в переменных окружения!")
    bot = Bot(token=API_TOKEN)
    storage = MemoryStorage()
    dp = Dispatcher(bot, storage=storage)
else:
    # Заглушки для тестов
    API_TOKEN = None
    bot = MagicMock()
    storage = None
    dp = MagicMock()
    dp.message_handler = MagicMock(return_value=lambda x: x)
    dp.callback_query_handler = MagicMock(return_value=lambda x: x)

# Словарь с доступными стилями и путями к моделям
STYLE_MODELS = {
    "vangogh": {
        "path": "checkpoints/style_vangogh_pretrained/latest_net_G.pth",
        "name": "Ван Гог",
    },
    "monet": {
        "path": "checkpoints/style_monet_pretrained/latest_net_G.pth",
        "name": "Моне",
    },
    "ukiyoe": {
        "path": "checkpoints/style_ukiyoe_pretrained/latest_net_G.pth",
        "name": "Укиё-э",
    },
    "cezanne": {
        "path": "checkpoints/style_cezanne_pretrained/latest_net_G.pth",
        "name": "Сезанн",
    },
}

# Глобальный словарь для хранения загруженных моделей
MODELS_CACHE = {}  # type: ignore

# Загрузка VGG19 для NST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# Нормализация для NST
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose(
    [transforms.Resize((512, 512)), transforms.ToTensor(), normalize]
)


# Загрузка модели стилизации
def load_model(style_key):
    if style_key in MODELS_CACHE:
        print(f"Using cached model for style: {style_key}")
        return MODELS_CACHE[style_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # noqa: F841
    net_G = define_G(input_nc=3, output_nc=3, ngf=64, netG="resnet_9blocks").to(device)

    print("Initializing network weights before loading checkpoint")
    init_weights(net_G, init_type="xavier", init_gain=0.02)

    checkpoint = torch.load(STYLE_MODELS[style_key]["path"], map_location=device)
    net_G.load_state_dict(checkpoint, strict=True)

    net_G.eval()
    MODELS_CACHE[style_key] = net_G
    print(f"Loaded and cached model for style: {style_key}")
    return net_G


# Загрузка модели Real-ESRGAN
def load_enhance_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # noqa: F841
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path="weights/RealESRGAN_x4plus.pth",
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device,
    )
    return upsampler


# Функция обработки изображения для стилизации (предобученной моделью)
def process_image(content_path, style_key):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # noqa: F841
    content_img = Image.open(content_path).convert("RGB")

    content_img = cv2.resize(np.array(content_img), (256, 256))
    content_tensor = (
        torch.from_numpy(content_img).permute(2, 0, 1).float().unsqueeze(0).to(device)
    )
    content_tensor = (content_tensor / 127.5) - 1.0

    net_G = load_model(style_key)

    with torch.no_grad():
        styled_tensor = net_G(content_tensor)

    styled_tensor = (styled_tensor + 1.0) * 127.5
    styled_img = styled_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    styled_img = styled_img.clip(0, 255).astype(np.uint8)

    output_path = f"output_{style_key}.jpg"
    Image.fromarray(styled_img).save(output_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_path


# Функция улучшения качества изображения с Real-ESRGAN
def enhance_image(content_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # noqa: F841
    upsampler = load_enhance_model()

    img = Image.open(content_path).convert("RGB")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

    output, _ = upsampler.enhance(img, outscale=2)

    output_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    output_path = "enhanced_image.jpg"
    output_img.save(output_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_path


# Функции для NST
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image


def denormalize_and_save(tensor, output_path):
    tensor = tensor.clone().cpu().squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(output_path)
    return cv2.imread(output_path)


def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features


def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram.div(c * h * w)


# Функция NST (синхронная часть с оптимизатором Adam)
def style_transfer_nst_sync(content_path, style_path, output_path, num_steps=1200):
    # Предобработка стилевого изображения
    style_img = cv2.imread(style_path)
    style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
    # Гистограммное выравнивание для усиления контраста
    style_img_yuv = cv2.cvtColor(style_img, cv2.COLOR_RGB2YUV)
    style_img_yuv[:, :, 0] = cv2.equalizeHist(style_img_yuv[:, :, 0])
    style_img = cv2.cvtColor(style_img_yuv, cv2.COLOR_YUV2RGB)
    cv2.imwrite(style_path, cv2.cvtColor(style_img, cv2.COLOR_RGB2BGR))

    content_img = load_image(content_path)
    style_img = load_image(style_path)

    # Инициализация целевого изображения
    target = content_img.clone() + torch.randn_like(content_img) * 0.05
    target = target.requires_grad_(True).to(device)

    optimizer = optim.Adam([target], lr=0.01)

    content_layers = ["21"]  # conv4_2
    style_layers = ["0", "5", "10", "19", "28"]  # Все слои

    content_features = get_features(content_img, vgg, content_layers)
    style_features = get_features(style_img, vgg, style_layers)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

    content_weight = 1.5
    style_weight = 1e6
    # Адаптивный tv_weight: начинаем с большего значения, затем уменьшаем
    initial_tv_weight = 1e-4
    style_layer_weights = {"0": 0.5, "5": 0.7, "10": 0.9, "19": 1.0, "28": 1.2}

    def total_variation_loss(img):
        batch, channel, height, width = img.size()
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return tv_h + tv_w

    print("Starting NST style transfer...")
    for step in range(num_steps):
        # Адаптивный tv_weight
        tv_weight = initial_tv_weight * (1 - step / num_steps)  # Уменьшаем со временем

        def closure():
            optimizer.zero_grad()

            target_features = get_features(target, vgg, content_layers + style_layers)

            content_loss = 0
            for layer in content_layers:
                content_loss += torch.mean(
                    (target_features[layer] - content_features[layer]) ** 2
                )
            content_loss *= content_weight

            style_loss = 0
            for layer in style_layers:
                target_gram = gram_matrix(target_features[layer])
                style_gram = style_grams[layer]
                layer_loss = torch.mean((target_gram - style_gram) ** 2)
                style_loss += layer_loss * style_layer_weights[layer]
            style_loss *= style_weight

            tv_loss = total_variation_loss(target) * tv_weight

            total_loss = content_loss + style_loss + tv_loss

            if torch.isnan(total_loss):
                print("NaN detected in loss. Stopping optimization.")
                return total_loss

            total_loss.backward()
            return total_loss

        optimizer.step(closure)

        if step % 150 == 0:
            loss_value = closure().item()
            print(f"Step {step}, Loss: {loss_value}")
            target_features = get_features(target, vgg, content_layers + style_layers)
            content_loss = 0
            for layer in content_layers:
                content_loss += torch.mean(
                    (target_features[layer] - content_features[layer]) ** 2
                )
            content_loss *= content_weight
            style_loss = 0
            for layer in style_layers:
                target_gram = gram_matrix(target_features[layer])
                style_gram = style_grams[layer]
                layer_loss = torch.mean((target_gram - style_gram) ** 2)
                style_loss += layer_loss * style_layer_weights[layer]
            style_loss *= style_weight
            tv_loss = total_variation_loss(target) * tv_weight
            print(
                f"Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}, TV Loss: {tv_loss.item()}"
            )

    # Сохранение промежуточного результата
    denormalize_and_save(target, output_path)

    # Постобработка: сглаживание с сохранением краёв
    img = cv2.imread(output_path)
    img_smooth = cv2.bilateralFilter(img, 9, 75, 75)  # Сохраняет края
    cv2.imwrite(output_path, img_smooth)

    # Улучшение резкости с Real-ESRGAN
    upsampler = load_enhance_model()
    img = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2RGB)
    output, _ = upsampler.enhance(
        img, outscale=1
    )  # Только резкость, без масштабирования
    output_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    output_img.save(output_path)

    print(f"NST style transfer completed. Result saved to {output_path}")


# Функция NST (асинхронная обёртка)
async def style_transfer_nst(content_path, style_path, output_path):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, lambda: style_transfer_nst_sync(content_path, style_path, output_path)
    )


# Определение состояний для FSM
class StyleTransferStates(StatesGroup):
    waiting_for_style = State()
    waiting_for_content = State()


class EnhanceStates(StatesGroup):
    waiting_for_enhance = State()


class NSTStates(StatesGroup):
    waiting_for_style_image = State()
    waiting_for_content_image = State()


# Обработчик команды /start
@dp.message_handler(commands=["start"])
async def start_command(message: types.Message):
    await message.answer(
        "Привет! Я бот для переноса стиля и улучшения качества.\n"
        "Отправь /style для стилизации с предобученной моделью,\n"
        "/enhance для улучшения качества,\n"
        "или /nst для переноса стиля с помощью NST (выберите стилевое изображение)."
    )


# Обработчик команды /style
@dp.message_handler(commands=["style"])
async def style_command(message: types.Message, state: FSMContext):
    keyboard = InlineKeyboardMarkup()
    for style_key, style_info in STYLE_MODELS.items():
        keyboard.add(
            InlineKeyboardButton(style_info["name"], callback_data=f"style_{style_key}")
        )
    await message.answer("Выберите стиль для переноса:", reply_markup=keyboard)
    await StyleTransferStates.waiting_for_style.set()


# Обработчик выбора стиля
@dp.callback_query_handler(
    lambda c: c.data.startswith("style_"), state=StyleTransferStates.waiting_for_style
)
async def process_style_selection(
    callback_query: types.CallbackQuery, state: FSMContext
):
    style_key = callback_query.data.split("_")[1]
    if style_key not in STYLE_MODELS:
        await callback_query.message.answer(
            "Ошибка: стиль не найден. Попробуйте снова с помощью /style."
        )
        return

    await state.update_data(style_key=style_key)
    await callback_query.message.answer(
        f"Выбран стиль: {STYLE_MODELS[style_key]['name']}. Теперь отправь контентное изображение."
    )
    await StyleTransferStates.waiting_for_content.set()
    await callback_query.answer()


# Обработчик контентного изображения для стилизации
@dp.message_handler(
    content_types=["photo"], state=StyleTransferStates.waiting_for_content
)
async def handle_content_photo(message: types.Message, state: FSMContext):
    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    file_path = file_info.file_path
    content_path = f"content_{message.from_user.id}.jpg"
    await bot.download_file(file_path, content_path)

    user_data = await state.get_data()
    style_key = user_data["style_key"]

    await message.answer(
        f"Обрабатываю изображение в стиле {STYLE_MODELS[style_key]['name']}, пожалуйста, подождите..."
    )

    output_path = await asyncio.to_thread(process_image, content_path, style_key)

    try:
        with open(output_path, "rb") as photo:
            await message.answer_photo(
                photo,
                caption=f"Ваше стилизованное изображение в стиле {STYLE_MODELS[style_key]['name']}!",
            )
    finally:
        os.remove(content_path)
        os.remove(output_path)

    await state.finish()


# Обработчик команды /enhance
@dp.message_handler(commands=["enhance"])
async def enhance_command(message: types.Message, state: FSMContext):
    await message.answer("Отправьте фотографию для улучшения качества.")
    await EnhanceStates.waiting_for_enhance.set()


# Обработчик фото для улучшения качества
@dp.message_handler(content_types=["photo"], state=EnhanceStates.waiting_for_enhance)
async def handle_enhance_photo(message: types.Message, state: FSMContext):
    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    file_path = file_info.file_path
    content_path = f"content_{message.from_user.id}.jpg"
    await bot.download_file(file_path, content_path)

    await message.answer("Улучшаю качество изображения, пожалуйста, подождите...")

    output_path = await asyncio.to_thread(enhance_image, content_path)

    try:
        with open(output_path, "rb") as photo:
            await message.answer_photo(
                photo, caption="Ваше изображение с улучшенным качеством!"
            )
    finally:
        os.remove(content_path)
        os.remove(output_path)

    await state.finish()


# Обработчик команды /nst
@dp.message_handler(commands=["nst"])
async def nst_command(message: types.Message, state: FSMContext):
    await message.answer(
        "Отправьте стилевое изображение (например, картину, стиль которой хотите перенести)."
    )
    await NSTStates.waiting_for_style_image.set()


# Обработчик стилевого изображения для NST
@dp.message_handler(content_types=["photo"], state=NSTStates.waiting_for_style_image)
async def handle_nst_style_photo(message: types.Message, state: FSMContext):
    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    file_path = file_info.file_path
    style_path = f"style_{message.from_user.id}.jpg"
    await bot.download_file(file_path, style_path)

    await state.update_data(style_path=style_path)
    await message.answer(
        "Теперь отправьте контентное изображение (фото, на которое нужно перенести стиль)."
    )
    await NSTStates.waiting_for_content_image.set()


# Обработчик контентного изображения для NST
@dp.message_handler(content_types=["photo"], state=NSTStates.waiting_for_content_image)
async def handle_nst_content_photo(message: types.Message, state: FSMContext):
    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    file_path = file_info.file_path
    content_path = f"content_{message.from_user.id}.jpg"
    await bot.download_file(file_path, content_path)

    user_data = await state.get_data()
    style_path = user_data["style_path"]
    output_path = f"nst_output_{message.from_user.id}.jpg"

    await message.answer("Переношу стиль на изображение, пожалуйста, подождите...")

    # Вызываем NST с использованием run_in_executor
    await style_transfer_nst(content_path, style_path, output_path)

    try:
        with open(output_path, "rb") as photo:
            await message.answer_photo(
                photo, caption="Ваше стилизованное изображение (NST)!"
            )
    finally:
        os.remove(content_path)
        os.remove(style_path)
        os.remove(output_path)

    await state.finish()


# Запуск бота
if __name__ == "__main__":
    # Проверка наличия моделей
    for style_key, style_info in STYLE_MODELS.items():
        if not os.path.exists(style_info["path"]):
            print(
                f"Ошибка: Модель для стиля {style_info['name']} не найдена по пути {style_info['path']}"
            )
            exit(1)

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(executor.start_polling(dp, skip_updates=True))
    except KeyboardInterrupt:
        print("\nБот остановлен пользователем.")
    finally:
        loop.run_until_complete(dp.storage.close())
        loop.run_until_complete(dp.storage.wait_closed())
        loop.run_until_complete(bot.session.close())
        loop.close()
        print("Бот завершил работу.")
