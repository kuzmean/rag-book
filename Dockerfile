FROM python:3.9-slim

WORKDIR /app

# Копируем список зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем весь исходный код в контейнер
COPY . .

# Запускаем приложение
CMD ["python", "src/main.py"]
