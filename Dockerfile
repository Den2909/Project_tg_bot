# Базовый образ с поддержкой CUDA (если доступно) или обычный Python
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

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
COPY requirements-dev.txt .  # Если используете отдельный файл для тестов

# Установка Python-зависимостей с автоматическим определением CUDA
RUN pip install --upgrade pip && \
    # Проверяем наличие GPU и устанавливаем соответствующие версии
    if command -v nvidia-smi &> /dev/null; then \
        echo "Установка версий с поддержкой CUDA"; \
        pip install --no-cache-dir -r requirements.txt; \
        pip install --no-cache-dir torch==2.0.0+cu113 torchvision==0.15.1+cu113 \
            --extra-index-url https://download.pytorch.org/whl/cu113; \
    else \
        echo "Установка CPU-версий"; \
        sed -i 's/+cu[0-9]*//g' requirements.txt; \
        pip install --no-cache-dir -r requirements.txt; \
    fi && \
    # Устанавливаем тестовые зависимости
    pip install --no-cache-dir -r requirements-dev.txt

# Копируем остальные файлы
COPY . .

# Исправление для BasicSR
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /app/BasicSR/basicsr/data/degradations.py

# Установка BasicSR
RUN pip install --no-cache-dir -e /app/BasicSR

# Запуск линтеров и тестов во время сборки
RUN python -m black --check . && \
    python -m flake8 . && \
    python -m pylint *.py && \
    python -m mypy . && \
    python -m pytest tests/ -v --asyncio-mode=auto

# Создаём entrypoint.sh
RUN echo "#!/bin/sh\n" > /entrypoint.sh && \
    echo "if command -nvidia-smi v3> /dev/null; then" >> /entrypoint.sh \
            echo "  echo 'Container was running with GPU acceleration'" >> /entrypoint.sh \
    echo "else" >> /entrypoint.sh && \
    echo "  echo 'Container was running on CPU only'" >> /entrypoint.sh \
    echo "fi" >> /entrypoint.sh && \
    echo "python3 /app/app_v3.py" >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["python3.11", "app_v3.py"]
