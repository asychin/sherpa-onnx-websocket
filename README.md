# Sherpa-ONNX WebSocket Server

Real-time speech recognition server with WebSocket interface, powered by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

## 📋 Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [WebSocket API](#websocket-api)
6. [Message Formats](#message-formats)
7. [Integration Examples](#integration-examples)
8. [Models](#models)
9. [Performance](#performance)
10. [Troubleshooting](#troubleshooting)

---

## Overview

**Sherpa-ONNX** is a next-generation speech recognition engine from the k2-fsa team. It uses modern neural network architectures (Zipformer2, Conformer) in ONNX format for efficient CPU inference.

### Key Features:

| Feature | Description |
|---------|-------------|
| **Streaming** | Real-time recognition with ~50-100ms latency |
| **Low Memory** | ~70MB RAM (INT8 quantized models) |
| **Endpoint Detection** | Automatic phrase segmentation on pauses |
| **Hotwords** | Boost recognition of specific words/phrases |
| **Word Timestamps** | Per-token timing information |
| **Multi-language** | Russian, English, Chinese, and more |

---

## Features

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Sherpa-ONNX WebSocket Server               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   asr_server.py                      │    │
│  │  • Audio input (int16 PCM)                          │    │
│  │  • Session configuration                            │    │
│  │  • Hotwords support                                 │    │
│  │  • Endpoint detection                               │    │
│  │  • Partial/final results                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              sherpa_onnx.OnlineRecognizer            │    │
│  │  • Zipformer2 Encoder (encoder.int8.onnx)           │    │
│  │  • Transducer Decoder (decoder.int8.onnx)           │    │
│  │  • Joiner (joiner.int8.onnx)                        │    │
│  │  • BPE Tokenizer (tokens.txt)                       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Client** connects via WebSocket
2. **Client** sends configuration `{"config": {...}}`
3. **Client** sends audio chunks (bytes, int16 PCM, 16kHz)
4. **Server** returns `{"partial": "..."}` on text changes
5. **Server** returns `{"text": "..."}` on pause detection (endpoint)
6. **Client** sends `{"eof": 1}` to finish

---

## Quick Start

### 1. Configure

```bash
# Copy example config
cp .env.example .env

# Edit settings
nano .env
```

### 2. Run

```bash
# Build and start
docker compose up -d --build

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Project Structure

```
├── docker-compose.yml       # Services configuration
├── .env.example             # Example environment variables
├── .env                     # Your local config (not in git)
├── sherpa-onnx-server/      # Russian model server
│   ├── Dockerfile
│   ├── asr_server.py
│   └── requirements.txt
└── sherpa-onnx-server-en/   # English model server
    ├── Dockerfile
    ├── asr_server.py
    └── requirements.txt
```

---

## Configuration

All settings are configured via `.env` file. Copy `.env.example` to `.env` and adjust values.

### Environment Variables

#### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SHERPA_MODEL_DIR` | `/models` | Path to model files |
| `SHERPA_NUM_THREADS` | `4` | Number of CPU threads |
| `SHERPA_SAMPLE_RATE` | `16000` | Audio sample rate |
| `SHERPA_HOST` | `0.0.0.0` | Listen address |
| `SHERPA_PORT` | `2700` | Server port |
| `SHERPA_AUTH_TOKEN` | `null` | Authorization token (optional) |

#### Endpoint Detection (pause detection)

| Variable | Default | Description |
|----------|---------|-------------|
| `SHERPA_ENABLE_ENDPOINT` | `true` | Enable pause detection |
| `SHERPA_RULE1_SILENCE` | `2.4` | Long pause after words (sec) |
| `SHERPA_RULE2_SILENCE` | `1.2` | Short pause (sec) |
| `SHERPA_RULE3_UTTERANCE` | `20.0` | Max utterance length (sec) |

#### Hotwords (phrase boosting)

| Variable | Default | Description |
|----------|---------|-------------|
| `SHERPA_HOTWORDS_SCORE` | `1.5` | Hotwords weight (1.0-3.0) |

#### Decoding

| Variable | Default | Description |
|----------|---------|-------------|
| `SHERPA_DECODING_METHOD` | `greedy_search` | Decoding method |
| `SHERPA_MAX_ACTIVE_PATHS` | `4` | Paths for beam_search |

### Decoding Methods

| Method | Speed | Accuracy | Hotwords | Notes |
|--------|-------|----------|----------|-------|
| `greedy_search` | ⚡ Fast | Good | ❌ No | Without phrase_list |
| `modified_beam_search` | 🐢 Slower | Better | ✅ Yes | **Required for hotwords!** |

> ⚠️ **Important:** If using `phrase_list` (hotwords), you MUST use `modified_beam_search`!

### Example .env

```bash
# Ports
SHERPA_RU_PORT=2700
SHERPA_EN_PORT=2701

# CPU
SHERPA_NUM_THREADS=4

# Endpoint detection
SHERPA_ENABLE_ENDPOINT=true
SHERPA_RULE1_SILENCE=2.4
SHERPA_RULE2_SILENCE=1.2

# Decoding (use modified_beam_search for hotwords)
SHERPA_DECODING_METHOD=modified_beam_search

# Authorization (leave empty to disable)
SHERPA_AUTH_TOKEN=
```

---

## WebSocket API

### Connection

```
ws://host:port
ws://host:port?token=YOUR_AUTH_TOKEN  # With authorization
```

### Headers (alternative authorization)

```
Authorization: Bearer YOUR_AUTH_TOKEN
```

---

## Message Formats

### 1. Send Configuration (client → server)

```json
{
  "config": {
    "sample_rate": 16000,
    "words": true,
    "partial_results": true,
    "phrase_list": ["kubernetes", "microservice", "deployment"]
  }
}
```

#### Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_rate` | int | 16000 | Audio sample rate |
| `words` | bool | false | Enable token timestamps |
| `partial_results` | bool | true | Send intermediate results |
| `phrase_list` | array | [] | List of expected words (hotwords) |

### 2. Send Audio (client → server)

```
Binary data: int16 PCM, mono, 16kHz
```

**Audio Format:**
- Type: `int16` (16-bit signed integer)
- Channels: 1 (mono)
- Rate: 16000 Hz
- Chunk size: any (recommended 4000-8000 bytes)

### 3. End Stream (client → server)

```json
{"eof": 1}
```

### 4. Partial Result (server → client)

```json
{
  "partial": "hello how are"
}
```

**When sent:**
- On each recognized text change
- Only if `partial_results: true`

### 5. Final Result WITHOUT timestamps (server → client)

```json
{
  "text": "hello how are you"
}
```

**When sent:**
- On pause detection (endpoint)
- On receiving `{"eof": 1}`
- If `words: false`

### 6. Final Result WITH timestamps (server → client)

```json
{
  "text": " hello how are you",
  "tokens": [" hel", "lo", " how", " are", " you"],
  "timestamps": [0.00, 0.24, 0.48, 0.72, 0.96],
  "ys_probs": [-2.1, -1.8, -2.3, -1.9, -2.0],
  "lm_probs": [],
  "context_scores": [],
  "segment": 0,
  "start_time": 0.00,
  "is_final": false,
  "is_eof": false
}
```

**When sent:**
- On pause detection (endpoint)
- On receiving `{"eof": 1}`
- If `words: true`

#### Final Result Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Recognized text |
| `tokens` | array[string] | BPE tokens (subwords) |
| `timestamps` | array[float] | Start time of each token (sec) |
| `ys_probs` | array[float] | Log-probabilities of tokens |
| `lm_probs` | array[float] | LM probabilities (if available) |
| `context_scores` | array[float] | Hotwords scores |
| `segment` | int | Segment number |
| `start_time` | float | Segment start time |
| `is_final` | bool | Is final result |
| `is_eof` | bool | End of stream |

---

## Integration Examples

### Python Client

```python
import asyncio
import websockets
import json

async def transcribe_audio(audio_file, server_url="ws://localhost:2700"):
    async with websockets.connect(server_url) as ws:
        # Send configuration
        config = {
            "config": {
                "sample_rate": 16000,
                "words": True,
                "partial_results": True,
                "phrase_list": ["hello", "world"]
            }
        }
        await ws.send(json.dumps(config))
        
        # Read and send audio
        with open(audio_file, 'rb') as f:
            while chunk := f.read(8000):
                await ws.send(chunk)
                
                # Receive results
                try:
                    result = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    data = json.loads(result)
                    if 'partial' in data:
                        print(f"Partial: {data['partial']}")
                    elif 'text' in data:
                        print(f"Final: {data['text']}")
                except asyncio.TimeoutError:
                    pass
        
        # End stream
        await ws.send('{"eof": 1}')
        
        # Get final result
        result = await ws.recv()
        print(f"Final result: {result}")

asyncio.run(transcribe_audio("audio.raw"))
```

### JavaScript Client (Browser)

```javascript
const ws = new WebSocket('ws://localhost:2700');

ws.onopen = () => {
    // Configuration
    ws.send(JSON.stringify({
        config: {
            sample_rate: 16000,
            words: true,
            partial_results: true,
            phrase_list: ['hello', 'world']
        }
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.partial) {
        console.log('Partial:', data.partial);
    } else if (data.text) {
        console.log('Final:', data.text);
        if (data.timestamps) {
            console.log('Timestamps:', data.timestamps);
        }
    }
};

// Send audio chunks
function sendAudioChunk(audioBuffer) {
    // audioBuffer should be Int16Array
    ws.send(audioBuffer.buffer);
}

// Finish
function finish() {
    ws.send('{"eof": 1}');
}
```

### Node.js Client

```javascript
const WebSocket = require('ws');
const fs = require('fs');

const ws = new WebSocket('ws://localhost:2700');

ws.on('open', () => {
    // Configuration
    ws.send(JSON.stringify({
        config: {
            sample_rate: 16000,
            words: true,
            phrase_list: ['hello', 'world']
        }
    }));
    
    // Read and send audio
    const audioData = fs.readFileSync('audio.raw');
    const chunkSize = 8000;
    
    for (let i = 0; i < audioData.length; i += chunkSize) {
        const chunk = audioData.slice(i, i + chunkSize);
        ws.send(chunk);
    }
    
    // Finish
    ws.send('{"eof": 1}');
});

ws.on('message', (data) => {
    const result = JSON.parse(data);
    console.log(result);
});
```

---

## Models

### Available Models

| Model | Size | Streaming | Language |
|-------|------|-----------|----------|
| `vosk-model-streaming-ru` | ~70MB | ✅ | Russian |
| `sherpa-onnx-streaming-zipformer-en` | ~70MB | ✅ | English |
| `sherpa-onnx-streaming-zipformer-bilingual-zh-en` | ~80MB | ✅ | Chinese + English |

### Model Files

```
/models/
├── encoder.int8.onnx    # Encoder (~70MB)
├── decoder.int8.onnx    # Decoder (~1.3MB)
├── joiner.int8.onnx     # Joiner (~260KB)
└── tokens.txt           # BPE tokens (~6KB)
```

### Download Models

**Russian (vosk-model-streaming-ru):**

```bash
wget https://huggingface.co/alphacep/vosk-model-streaming-ru/resolve/main/am-onnx/encoder.int8.onnx
wget https://huggingface.co/alphacep/vosk-model-streaming-ru/resolve/main/am-onnx/decoder.int8.onnx
wget https://huggingface.co/alphacep/vosk-model-streaming-ru/resolve/main/am-onnx/joiner.int8.onnx
wget https://huggingface.co/alphacep/vosk-model-streaming-ru/resolve/main/lang/tokens.txt
```

**English (sherpa-onnx-streaming-zipformer-en):**

```bash
wget https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx -O encoder.int8.onnx
wget https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx -O decoder.int8.onnx
wget https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx -O joiner.int8.onnx
wget https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/tokens.txt
```

---

## Performance

### Benchmarks (8 CPU threads)

| Metric | Value |
|--------|-------|
| Latency | ~50-100ms |
| Memory | ~70MB (INT8) |
| RTF* | ~0.05 |
| CPU usage | ~20-30% |

*RTF = Real-Time Factor (less = faster, <1 = realtime)

### Thread Recommendations

| CPU Cores | SHERPA_NUM_THREADS | Notes |
|-----------|-------------------|-------|
| 4 | 2 | Minimum |
| 8 | 4 | Recommended |
| 16 | 8 | Optimal |
| 32+ | 12-16 | Diminishing returns |

---

## Troubleshooting

### No Final Results

**Symptom:** Only `{"partial": ...}`, no `{"text": ...}`

**Solution:** Check endpoint detection settings:
```bash
SHERPA_ENABLE_ENDPOINT=true
SHERPA_RULE1_SILENCE=2.4
```

### phrase_list Not Working

**Symptom:** Special words not recognized better

**Solution:** 
1. Use `modified_beam_search` decoding method
2. Check logs for `[HOTWORDS] N phrases`
3. Increase `SHERPA_HOTWORDS_SCORE` to 2.0-3.0

### High Latency

**Solution:**
1. Use INT8 model
2. Increase `SHERPA_NUM_THREADS`
3. Use `greedy_search` (if hotwords not needed)

### Connection Error

**Solution:**
```bash
# Check if container is running
docker ps | grep sherpa

# Check logs
docker compose logs

# Test port
nc -zv localhost 2700
```

---

## Links

- [sherpa-onnx GitHub](https://github.com/k2-fsa/sherpa-onnx)
- [sherpa-onnx Documentation](https://k2-fsa.github.io/sherpa/onnx/)
- [icefall (training)](https://github.com/k2-fsa/icefall)
- [Models on HuggingFace](https://huggingface.co/models?search=sherpa-onnx)

---

## License

MIT License
