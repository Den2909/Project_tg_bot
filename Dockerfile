# Используем базовый образ с Python 3.11
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем все файлы проекта
COPY . /app

# Проверяем наличие файлов
RUN ls -la /app
RUN ls -la /app/BasicSR  # Добавляем для отладки

# Обновляем pip
RUN pip install --upgrade pip

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем модифицированный basicsr из локальной директории
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
