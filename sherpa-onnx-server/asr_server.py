#!/usr/bin/env python3
"""
Sherpa-ONNX WebSocket Server - Streaming Speech Recognition
Полная совместимость с Vosk API + расширенные возможности sherpa-onnx
Модель: vosk-model-streaming-ru (Zipformer2)
"""

import asyncio
import json
import os
import logging
import time
import numpy as np
import websockets
from urllib.parse import parse_qs, urlparse

try:
    import sherpa_onnx
except ImportError:
    print("ERROR: sherpa_onnx not installed. Run: pip install sherpa-onnx")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Глобальные переменные
recognizer = None

# ====== Конфигурация из ENV ======
MODEL_DIR = os.environ.get('SHERPA_MODEL_DIR', '/models')
NUM_THREADS = int(os.environ.get('SHERPA_NUM_THREADS', 4))
SAMPLE_RATE = int(os.environ.get('SHERPA_SAMPLE_RATE', 16000))
AUTH_TOKEN = os.environ.get('SHERPA_AUTH_TOKEN', None)

# Endpoint detection (определение пауз)
ENABLE_ENDPOINT = os.environ.get('SHERPA_ENABLE_ENDPOINT', 'true').lower() == 'true'
RULE1_SILENCE = float(os.environ.get('SHERPA_RULE1_SILENCE', 2.4))  # Пауза после слов
RULE2_SILENCE = float(os.environ.get('SHERPA_RULE2_SILENCE', 1.2))  # Короткая пауза
RULE3_UTTERANCE = float(os.environ.get('SHERPA_RULE3_UTTERANCE', 20.0))  # Макс длина

# Hotwords (phrase_list)
HOTWORDS_SCORE = float(os.environ.get('SHERPA_HOTWORDS_SCORE', 1.5))

# Decoding
DECODING_METHOD = os.environ.get('SHERPA_DECODING_METHOD', 'greedy_search')
MAX_ACTIVE_PATHS = int(os.environ.get('SHERPA_MAX_ACTIVE_PATHS', 4))

# VAD (Voice Activity Detection) - фильтрация тишины
VAD_ENABLED = os.environ.get('SHERPA_VAD_ENABLED', 'true').lower() == 'true'
VAD_THRESHOLD = float(os.environ.get('SHERPA_VAD_THRESHOLD', 0.02))  # RMS порог (0.01-0.05)
VAD_MIN_SPEECH_SEC = float(os.environ.get('SHERPA_VAD_MIN_SPEECH_SEC', 0.1))  # Мин. длина речи


def is_speech(samples: np.ndarray) -> bool:
    """Проверка наличия речи в аудио чанке (простой VAD на основе RMS)"""
    if not VAD_ENABLED:
        return True
    
    # Вычисляем RMS (Root Mean Square) - среднюю энергию сигнала
    rms = np.sqrt(np.mean(samples ** 2))
    return rms > VAD_THRESHOLD


def load_model():
    """Загрузка модели Sherpa-ONNX"""
    global recognizer
    
    logger.info("=" * 50)
    logger.info("=== SHERPA-ONNX SERVER STARTING ===")
    logger.info("=" * 50)
    logger.info(f"Model dir: {MODEL_DIR}")
    logger.info(f"Threads: {NUM_THREADS}")
    logger.info(f"Sample rate: {SAMPLE_RATE}")
    
    # Проверяем наличие файлов модели (INT8 версия - быстрее!)
    encoder_path = os.path.join(MODEL_DIR, 'encoder.int8.onnx')
    decoder_path = os.path.join(MODEL_DIR, 'decoder.int8.onnx')
    joiner_path = os.path.join(MODEL_DIR, 'joiner.int8.onnx')
    tokens_path = os.path.join(MODEL_DIR, 'tokens.txt')
    
    # Альтернативные имена (float32)
    if not os.path.exists(encoder_path):
        encoder_path = os.path.join(MODEL_DIR, 'encoder.onnx')
    if not os.path.exists(decoder_path):
        decoder_path = os.path.join(MODEL_DIR, 'decoder.onnx')
    if not os.path.exists(joiner_path):
        joiner_path = os.path.join(MODEL_DIR, 'joiner.onnx')
    
    logger.info(f"Encoder: {encoder_path}")
    logger.info(f"Decoder: {decoder_path}")
    logger.info(f"Joiner: {joiner_path}")
    logger.info(f"Tokens: {tokens_path}")
    
    # Проверяем существование файлов
    for path, name in [(encoder_path, 'encoder'), (decoder_path, 'decoder'), 
                       (joiner_path, 'joiner'), (tokens_path, 'tokens')]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"{name} not found: {path}")
    
    # Создаём онлайн recognizer с полной конфигурацией
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens_path,
        encoder=encoder_path,
        decoder=decoder_path,
        joiner=joiner_path,
        num_threads=NUM_THREADS,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        # Decoding
        decoding_method=DECODING_METHOD,
        max_active_paths=MAX_ACTIVE_PATHS,
        # Endpoint detection (как у Vosk!)
        enable_endpoint_detection=ENABLE_ENDPOINT,
        rule1_min_trailing_silence=RULE1_SILENCE,
        rule2_min_trailing_silence=RULE2_SILENCE,
        rule3_min_utterance_length=RULE3_UTTERANCE,
        # Hotwords (phrase_list)
        hotwords_score=HOTWORDS_SCORE,
        # Provider
        provider="cpu",
    )
    
    logger.info("-" * 50)
    logger.info("Model loaded successfully!")
    logger.info(f"Decoding: {DECODING_METHOD}")
    logger.info(f"Endpoint detection: {'ENABLED' if ENABLE_ENDPOINT else 'DISABLED'}")
    if ENABLE_ENDPOINT:
        logger.info(f"  - rule1 (long pause): {RULE1_SILENCE}s")
        logger.info(f"  - rule2 (short pause): {RULE2_SILENCE}s")
        logger.info(f"  - rule3 (max utterance): {RULE3_UTTERANCE}s")
    logger.info(f"Hotwords score: {HOTWORDS_SCORE}")
    logger.info(f"VAD: {'ENABLED' if VAD_ENABLED else 'DISABLED'}" + 
               (f" (threshold={VAD_THRESHOLD})" if VAD_ENABLED else ""))
    logger.info("-" * 50)


async def recognize(websocket):
    """Обработчик WebSocket соединения - полная совместимость с Vosk API"""
    global recognizer
    
    path = websocket.request.path if hasattr(websocket, 'request') else '/'
    client_addr = websocket.remote_address
    
    # Авторизация
    if AUTH_TOKEN:
        parsed = urlparse(path)
        params = parse_qs(parsed.query)
        url_token = params.get('token', [None])[0]
        
        headers = websocket.request.headers if hasattr(websocket, 'request') else {}
        auth_header = headers.get('Authorization', '')
        header_token = auth_header.replace('Bearer ', '') if auth_header.startswith('Bearer ') else None
        
        token = url_token or header_token
        
        if token != AUTH_TOKEN:
            logger.warning(f"[AUTH] DENIED {client_addr}")
            await websocket.close(4001, "Unauthorized")
            return
    
    logger.info(f"[CONNECT] {client_addr}")
    
    # Состояние сессии
    sample_rate = SAMPLE_RATE
    hotwords = None
    words_enabled = False
    partial_results = True
    stream = None
    
    last_text = ""
    total_samples = 0
    start_time = time.time()
    
    # VAD state
    silence_chunks = 0
    speech_detected = False
    min_speech_samples = int(VAD_MIN_SPEECH_SEC * sample_rate)
    
    try:
        async for message in websocket:
            # Конфигурация (JSON)
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    
                    # EOF - конец потока
                    if data.get('eof') == 1 or message == '{"eof" : 1}':
                        logger.info(f"[EOF] {client_addr}")
                        
                        if stream is not None:
                            # Финализируем распознавание
                            stream.input_finished()
                            while recognizer.is_ready(stream):
                                recognizer.decode_stream(stream)
                            
                            # Нативный JSON от sherpa-onnx с timestamps!
                            result_text = recognizer.get_result(stream).strip()
                            
                            if result_text:
                                if words_enabled:
                                    # Полный формат с токенами и timestamps
                                    result_json = recognizer.get_result_as_json_string(stream)
                                    logger.info(f"[SEND FINAL+WORDS] -> {client_addr}: {result_text[:50]}...")
                                    await websocket.send(result_json)
                                else:
                                    # Простой формат как Vosk
                                    final_result = {"text": result_text}
                                    logger.info(f"[SEND FINAL] -> {client_addr}: {result_text[:50]}...")
                                    await websocket.send(json.dumps(final_result, ensure_ascii=False))
                        break
                    
                    # Конфигурация сессии
                    if 'config' in data:
                        config = data['config']
                        logger.info(f"[CONFIG] {client_addr}: {config}")
                        
                        # Sample rate
                        if 'sample_rate' in config:
                            sample_rate = int(config['sample_rate'])
                        
                        # Partial results
                        if 'partial_results' in config:
                            partial_results = bool(config['partial_results'])
                        
                        # Words (timestamps) - Vosk compatible
                        if 'words' in config:
                            words_enabled = bool(config['words'])
                        
                        # Phrase list -> Hotwords
                        if 'phrase_list' in config:
                            phrase_list = config['phrase_list']
                            if isinstance(phrase_list, list):
                                # Убираем дубликаты и соединяем пробелами
                                unique_phrases = list(dict.fromkeys(phrase_list))
                                hotwords = ' '.join(unique_phrases)
                                logger.info(f"[HOTWORDS] {client_addr}: {len(unique_phrases)} phrases")
                        
                        # Создаём stream с hotwords
                        if stream is None:
                            stream = recognizer.create_stream(hotwords)
                            logger.info(f"[STREAM] Created for {client_addr}" + 
                                       (f" with hotwords" if hotwords else ""))
                        
                        continue
                        
                except json.JSONDecodeError:
                    pass
            
            # Аудио данные (bytes)
            elif isinstance(message, bytes):
                # Создаём stream если ещё не создан
                if stream is None:
                    stream = recognizer.create_stream(hotwords)
                
                # Конвертируем в float32
                samples = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                total_samples += len(samples)
                
                # VAD: проверяем наличие речи
                if VAD_ENABLED:
                    if is_speech(samples):
                        speech_detected = True
                        silence_chunks = 0
                    else:
                        silence_chunks += 1
                        # Если речь уже была и пошла тишина - продолжаем обработку
                        # Если речи не было - игнорируем чанк (защита от галлюцинаций)
                        if not speech_detected:
                            continue
                
                # Добавляем в stream
                stream.accept_waveform(sample_rate, samples)
                
                # Декодируем
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                
                # Проверяем endpoint (пауза в речи) - как у Vosk!
                if recognizer.is_endpoint(stream):
                    result_text = recognizer.get_result(stream).strip()
                    if result_text:
                        if words_enabled:
                            # Полный формат с токенами и timestamps
                            result_json = recognizer.get_result_as_json_string(stream)
                            logger.info(f"[SEND FINAL+WORDS] -> {client_addr}: {result_text[:50]}...")
                            await websocket.send(result_json)
                        else:
                            # Простой формат как Vosk
                            final_result = {"text": result_text}
                            logger.info(f"[SEND FINAL] -> {client_addr}: {result_text[:50]}...")
                            await websocket.send(json.dumps(final_result, ensure_ascii=False))
                    
                    # Сбрасываем для нового предложения!
                    recognizer.reset(stream)
                    last_text = ""
                    speech_detected = False  # Сбрасываем VAD состояние
                    silence_chunks = 0
                
                elif partial_results:
                    # Получаем partial результат
                    current_text = recognizer.get_result(stream).strip()
                    
                    # Отправляем partial если текст изменился
                    if current_text and current_text != last_text:
                        partial_result = {"partial": current_text}
                        logger.info(f"[SEND PARTIAL] -> {client_addr}: {current_text[:50]}...")
                        await websocket.send(json.dumps(partial_result, ensure_ascii=False))
                        last_text = current_text
        
        # Статистика
        elapsed = time.time() - start_time
        audio_duration = total_samples / sample_rate if total_samples > 0 else 0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        logger.info(f"[STATS] {client_addr}: audio={audio_duration:.1f}s, time={elapsed:.1f}s, RTF={rtf:.3f}")
            
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"[DISCONNECT] {client_addr}")
    except Exception as e:
        logger.error(f"[ERROR] {client_addr}: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Запуск сервера"""
    load_model()
    
    host = os.environ.get('SHERPA_HOST', '0.0.0.0')
    port = int(os.environ.get('SHERPA_PORT', 2700))
    
    logger.info(f"[SERVER] Listening on ws://{host}:{port}")
    logger.info("=" * 50)
    
    async with websockets.serve(
        recognize,
        host,
        port,
        max_size=None,
        ping_interval=20,
        ping_timeout=60,
    ):
        await asyncio.Future()


if __name__ == '__main__':
    asyncio.run(main())
