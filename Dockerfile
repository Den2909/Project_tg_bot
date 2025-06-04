# Используем базовый образ с Python 3.11
FROM python:3.11-slim

# Устанавливаем git для клонирования
RUN apt-get update && apt-get install -y git && apt-get clean

# Устанавливаем рабочую директорию
WORKDIR /app

# Клонируем репозиторий
RUN git clone https://github.com/Den2909/Project_tg_bot.git .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем зависимости для OpenCV и Real-ESRGAN
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

# Запуск бота
CMD ["python3", "app_v3.py"]
