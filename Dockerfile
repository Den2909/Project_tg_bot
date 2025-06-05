(myenv311) den@LAPTOP-ACQHB722:/mnt/d/Python/Project/tg_bot/pytorch-CycleGAN-and-pix2pix/Project_tg_bot$ cat Dockerfile
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

# Установка Python-зависимостей с автоматическим определением CUDA
RUN pip install --upgrade pip && \
    # Проверяем наличие GPU и устанавливаем соответствующие версии
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

RUN echo "#!/bin/bash\n" > /entrypoint.sh && \
    echo "python3 /app/app_v3.py" >> /entrypoint.sh && \
    echo "if command -v nvidia-smi &> /dev/null; then" >> /entrypoint.sh && \
    echo "  echo 'Container was running with GPU acceleration'" >> /entrypoint.sh && \
    echo "else" >> /entrypoint.sh && \
    echo "  echo 'Container was running on CPU only'" >> /entrypoint.sh && \
    echo "fi" >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["python3", "app_v3.py"]