# Базовый образ с поддержкой CUDA (если доступно) или обычный Python
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# Аргумент для токена
ARG TELEGRAM_BOT_TOKEN

# Установка переменной окружения для токена
ENV TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости отдельно для кэширования
COPY requirements.txt .

# Установка Python-зависимостей с автоматическим определением CUDA
RUN pip install --upgrade pip && \
    if command -v nvidia-smi &> /dev/null; then \
        echo "Установка версий с поддержкой CUDA"; \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        echo "Установка CPU-версий"; \
        sed -i 's/+cu[0-9]*//g' requirements.txt; \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Копируем остальные файлы
COPY . .

# Исправление для BasicSR
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /app/BasicSR/basicsr/data/degradations.py

# Установка BasicSR
RUN pip install --no-cache-dir -e /app/BasicSR

# Запуск линтеров и тестов во время сборки
RUN python -m black --check . && \
    python -m flake8 . --config=.flake8 && \
    python -m pylint *.py && \
    python -m mypy . && \
    python -m pytest tests/ -v --asyncio-mode=auto

# Создаём entrypoint.sh
RUN echo "#!/bin/sh\n" > /entrypoint.sh && \
    echo "if command -v nvidia-smi &> /dev/null; then\n" >> /entrypoint.sh && \
    echo "  echo 'Container running with GPU acceleration'\n" >> /entrypoint.sh && \
    echo "else\n" >> /entrypoint.sh && \
    echo "  echo 'Container running on CPU only'\n" >> /entrypoint.sh && \
    echo "fi\n" >> /entrypoint.sh && \
    echo "python3 /app/app_v3.py\n" >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["python3", "app_v3.py"]
