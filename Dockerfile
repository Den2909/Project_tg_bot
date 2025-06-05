# Используем базовый образ с Python 3.11
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем все файлы проекта (включая BasicSR и requirements.txt)
COPY . /app

# Проверяем наличие файлов
RUN ls -la /app
RUN ls -la /app/BasicSR  # Для отладки

# Обновляем pip
RUN pip install --upgrade pip

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем модифицированный basicsr из локальной директории BasicSR
# Убедимся, что degradations.py исправлен (если не исправлен вручную)
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /app/BasicSR/basicsr/data/degradations.py
RUN pip install --no-cache-dir -e /app/BasicSR

# Устанавливаем зависимости для OpenCV и Real-ESRGAN
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && apt-get clean

# Запуск бота
CMD ["python3", "app_v3.py"]
