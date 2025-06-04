# Используем базовый образ с Python 3.11
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта (кроме basicsr_local)
COPY . /app

# Устанавливаем git
RUN apt-get update && apt-get install -y git


# Проверяем наличие файлов
RUN ls -la /app

# Обновляем pip
RUN pip install --upgrade pip

# Устанавливаем зависимости из requirements.txt 

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем модифицированный basicsr из локальной директории
RUN pip install --no-cache-dir -e /app/basicsr_local

# Устанавливаем зависимости для OpenCV и Real-ESRGAN, включая libGL
RUN apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && apt-get clean

# Запуск бота
CMD ["python3", "app_v3.py"]