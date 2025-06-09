# Telegram Style Transfer Bot (@MyStyleBot)

Этот проект представляет собой Telegram-бот [`@MyStyleBot`](https://t.me/MyStyleBot), который позволяет стилизовать изображения с использованием предобученных моделей (на основе CycleGAN), выполнять нейронный перенос стиля (NST) и улучшать качество изображений с помощью Real-ESRGAN. Бот развёрнут с использованием Docker и может работать как на CPU, так и на GPU.

## Описание проекта

- **Стилизация изображений**: Использует предобученные модели CycleGAN для переноса стиля (например, Ван Гог, Моне, Укиё-э, Сезанн).
- **Нейронный перенос стиля (NST)**: Применяет стиль одного изображения к другому с использованием VGG19.
- **Улучшение качества**: Использует Real-ESRGAN для повышения резкости и качества изображений.
- **Развёртывание**: Бот упакован в Docker-контейнер для универсального запуска на CPU или GPU.

## Требования

- Docker (для запуска контейнера)
- Telegram API-токен (токен был прикреплён к ссылке на GitHub в Stepik)

## ⚙️ Доступ к боту и ограничения

Бот [`@MyStyleBot`](https://t.me/MyStyleBot) (для перехода, нажать по ссылке или ввести имя в Telegram) развёрнут на сервере с ограниченными ресурсами:

- **Процессор**: 2 ядра (только CPU, без GPU)
- **Оперативная память**: 2 ГБ
- **Платформа**: Telegram API + Docker

**Ограничения по обработке изображений**:
-  **Стилизация (CycleGAN)** — входное изображение масштабируется до **256×256**
-  **Нейронный перенос стиля (NST)** — масштабируется до **256×256**
-  **Улучшение качества (Real-ESRGAN)** — увеличивает изображение в **2 раза**

Это обусловлено ограничениями по производительности.

## Инструкция по запуску Docker-контейнера самостоятельно

1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/Den2909/Project_tg_bot.git
   cd Project_tg_bot
   ```

2. **Соберите Docker-образ**:
   ```bash
   docker build -t tg-bot --build-arg TELEGRAM_BOT_TOKEN="token" .
   ```

3. **Запустите контейнер**:

   - На машине с **GPU**:
     ```bash
     docker run --gpus all -d --name style-bot tg-bot
     ```

   - На машине с **CPU**:
     ```bash
     docker run -d --name style-bot tg-bot
     ```

4. **Взаимодействие с ботом**:

- Найдите бота в Telegram: [@MyStyleBot](https://t.me/MyStyleBot)
- Используйте команды:
  - `/start` — начать работу.
  - `/style` — стилизация с предобученной моделью.
  - `/enhance` — улучшение качества изображения.
  - `/nst` — нейронный перенос стиля.

## Использование в Google Colab

Вы можете попробовать функциональность бота напрямую в Google Colab (без установки и запуска локально):  
👉 [Открыть в Colab](https://colab.research.google.com/drive/1uYcQuI8COUMZYSkhF5_Jzups56Jx9nBH?usp=sharing)

## Примеры изображений 
- Преобразования проводились на локальной машине с GPU, разрешение в коде было увеличено

### 🎨 Стилизация (CycleGAN) 

### 🖼️ Нейронный перенос стиля (NST)

| Контент      | Стиль        | Результат  |
|--------------|--------------|------------|
![Контент](https://raw.githubusercontent.com/Den2909/Project_tg_bot/main/images/NST/IMG_8186.jpg) | ![Стиль](https://raw.githubusercontent.com/Den2909/Project_tg_bot/main/images/NST/style_cont.jpg) | ![Результат](https://raw.githubusercontent.com/Den2909/Project_tg_bot/main/images/NST/photo_2025-06-09_21-24-02.jpg) |

### 🔧 Улучшение качества (Real-ESRGAN)

| Входное изображение | Результат |
|---------------------|-----------|
| ![Пример](https://raw.githubusercontent.com/Den2909/Project_tg_bot/main/images/ESRG/photo_2025-06-09_21-28-08.jpg) | ![Результат](https://raw.githubusercontent.com/Den2909/Project_tg_bot/main/images/ESRG/photo_2025-06-09_21-28-13.jpg) |

| Стиль        | Пример входа | Результат |
|--------------|--------------|-----------|
| Ван Гог      | ![Пример](https://raw.githubusercontent.com/Den2909/Project_tg_bot/main/images/style/vang/vang.jpg) | ![Результат](https://raw.githubusercontent.com/Den2909/Project_tg_bot/main/images/style/vang/res_1.jpg) |
| Моне         | ![Пример](https://raw.githubusercontent.com/Den2909/Project_tg_bot/main/images/style/mone/mone.jpg) | ![Результат](https://raw.githubusercontent.com/Den2909/Project_tg_bot/main/images/style/mone/res_2.jpg) |
| Укиё-э       | ![Пример](https://raw.githubusercontent.com/Den2909/Project_tg_bot/main/images/style/Uki/uki.jpg) | ![Результат](https://raw.githubusercontent.com/Den2909/Project_tg_bot/main/images/style/Uki/res_4.jpg) |
| Сезанн       | ![Пример](https://raw.githubusercontent.com/Den2909/Project_tg_bot/main/images/style/cezan/h-364747.jpg) | ![Результат](https://raw.githubusercontent.com/Den2909/Project_tg_bot/main/images/style/cezan/res_3.jpg) |

##  Форматирование, линтинг и тестирование

Проект использует следующие инструменты для обеспечения качества кода:

- **Black** — автоматическое форматирование (PEP 8, длина строки 88).
- **Flake8** — проверка стиля. Игнорирует каталоги `BasicSR`, `tests`.
- **Pylint** — статический анализ кода.
- **Mypy** — проверка типов с флагом 
- **Pytest** — тестирование с поддержкой асинхронного режима (`--asyncio-mode=auto`).

###  Проверка при сборке Docker

Во время сборки Docker-образа автоматически запускаются все проверки и тестирование:

```Dockerfile
RUN python -m black --check . && \
    python -m flake8 . --config=.flake8 && \
    python -m pylint *.py && \
    python -m mypy . && \
    python -m pytest tests/ -v --asyncio-mode=auto
```

> ❗ GitHub Actions для тестирования **не используется**, поскольку Telegram-токен нельзя безопасно хранить в CI/CD.

###  Что тестируется

Тесты расположены в папке `tests/` и проверяют ключевые функции из `app_v3.py`:

- `test_load_model`: загружает модель (например, стиль "vangogh").
- `test_process_image`: проверяет работу стилизации.
- `test_enhance_image`: проверяет upscale с помощью Real-ESRGAN.
- `test_style_transfer_nst_sync`: проверяет перенос стиля (NST) с VGG19.
- Все внешние зависимости мокируются с помощью `pytest-mock`, чтобы обеспечить независимость и работу.

### 🖥️ Локальная проверка

```bash
black .                                # Форматирование
black --check .                        # Проверка форматирования
flake8 . --config=.flake8              # Проверка стиля
pylint *.py                            # Анализ
mypy .                                 # Проверка типов
pytest tests/ -v --asyncio-mode=auto   # Запуск тестов
./lint.sh                              # Объединённый запуск всех линтеров
```


## Заключение

Реализован Telegram-бот на aiogram с асинхронной обработкой, позволяющий применять стили (Ван Гог, Моне, Укиё-э, Сезанн) и улучшать качество изображений.
Использованы модели CycleGAN для стилизации, VGG для нейронного переноса стиля (NST) и Real-ESRGAN для улучшения качества.
Настроен Docker-контейнер с поддержкой GPU/CPU, автоматическими тестами (pytest) и линтерами (black, flake8, pylint, mypy).

Улучшение качества с Real-ESRGAN: Дало лучшие результаты, значительно повышая детализацию и разрешение изображений.
Нейронный перенос стиля (NST): Обеспечил хорошую стилизацию с точной передачей художественного стиля.
Стилизация с CycleGAN: Эффект менее заметен, чем ожидалось, из-за ограничений готовой модели.

Проект стал отличным опытом в работе с GAN и Docker. В будущем стоит обучить собственную модель CycleGAN для более выразительной стилизации, что потребует GPU и качественного датасета.
