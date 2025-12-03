FROM python:3.11-slim

WORKDIR /app

# Устанавливаем зависимости для sherpa-onnx
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаём директорию для моделей
RUN mkdir -p /models

# Скачиваем модель sherpa-onnx-streaming-zipformer-en (English)
# Источник: https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26
RUN wget --progress=bar:force:noscroll \
    "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx" \
    -O /models/encoder.int8.onnx && \
    wget --progress=bar:force:noscroll \
    "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx" \
    -O /models/decoder.int8.onnx && \
    wget --progress=bar:force:noscroll \
    "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx" \
    -O /models/joiner.int8.onnx && \
    wget --progress=bar:force:noscroll \
    "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/tokens.txt" \
    -O /models/tokens.txt

# Копируем сервер
COPY asr_server.py .

# Порт
EXPOSE 2700

# Переменные по умолчанию
ENV SHERPA_MODEL_DIR=/models
ENV SHERPA_NUM_THREADS=4
ENV SHERPA_SAMPLE_RATE=16000
ENV SHERPA_HOST=0.0.0.0
ENV SHERPA_PORT=2700

CMD ["python", "asr_server.py"]

