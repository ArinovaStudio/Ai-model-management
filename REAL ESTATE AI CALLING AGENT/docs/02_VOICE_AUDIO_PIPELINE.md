# 02. Voice & Audio Processing Pipeline

## 1. Overview & Audio Specifications

The telephony audio pipeline in RealEstateAI is designed for full-duplex, low-latency audio transmission over standard WebSockets. Telephony carriers (such as Twilio Media Streams, Telnyx, and browser WebRTC audio worklets) transmit raw linear PCM audio at **16,000 Hz, 16-bit signed little-endian, mono (1 channel)**.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Sample Rate** | 16,000 Hz (16 kHz) | Standard wideband telephony audio rate |
| **Bit Depth** | 16-bit signed PCM | Little-endian signed 2-byte integers (`int16`) |
| **Channels** | 1 (Mono) | Single audio channel |
| **Frame Size** | 20 ms – 30 ms | 640 – 960 bytes per binary packet |
| **Transport** | WebSocket (Binary Frames) | Bidirectional streaming over `/ws/audio/{session_id}` |

---

## 2. Voice Activity Detection (VAD) & Utterance Segmentation

The entry point for inbound caller audio is implemented in `src/audio/vad.py` using **WebRTC VAD** (`webrtcvad-wheels`):

1. **Jitter Buffering & Frame Chunking**:
   Incoming WebSocket binary packets are placed into an in-memory byte buffer. The VAD consumes exact 20ms audio frames (640 bytes at 16kHz).
2. **Aggressiveness Level**:
   Configured at Mode `2` or `3` (high sensitivity to human vocal cords, robust rejection of background air conditioning and room echo).
3. **Turn Boundary Debounce**:
   - When a caller speaks, speech frames are accumulated into an utterance buffer.
   - The system tracks continuous silence frames. When silence exceeds `MAX_SILENCE_DURATION_MS` (typically 400ms – 600ms), the turn boundary triggers.
   - The accumulated byte array is forwarded to the Automated Speech Recognition (ASR) engine for transcription.

---

## 3. Automated Speech Recognition (Faster-Whisper)

Speech-to-text is powered by **Faster-Whisper** (`src/audio/stt.py`), a CTranslate2 reimplementation of OpenAI's Whisper model:

* **Model Used**: `base.en` (39 Million parameters).
* **Quantization**: `int8` running on CPU.
* **Pre-warming**: Model weights (`models/whisper/`) are loaded during FastAPI application startup (`@asynccontextmanager` lifespan in `src/main.py`), preventing first-call initialization delays.
* **Inference Speed**: Processes a typical 4-second caller utterance in **250ms – 350ms** on standard multi-core CPUs.
* **Accuracy Filters**:
  - Automatically filters empty audio chunks or ambient noise.
  - Checks transcription confidence (`avg_logprob` and `no_speech_prob`).
  - Strips hallucinated silence patterns (e.g. repeated `"you"`, `"[Music]"`).

---

## 4. Text-to-Speech (TTS) Architecture & Model Evolution

Voice synthesis determines the naturalness and believability of the agent. The system has undergone extensive research, testing, and implementation across multiple open-source and neural architectures:

### Comprehensive Overview of Tried & Evaluated TTS Models:

```
TTS ARCHITECTURAL EVOLUTION:

[ 1. Piper ONNX ] ───────────► Fast (~100ms), low memory, but sounded synthetic/robotic for real estate.
      │
[ 2. Kokoro-82M ONNX ] ──────► CURRENT ACTIVE ENGINE: Ultra-low latency (~150ms on CPU), hyper-realistic human
      │                         articulation, zero API keys, 100% offline.
      │
[ 3. OpenVoice (v2) ] ───────► TRIED & EVALUATED: Fast tone color converter. However, Python 3.14 C-extension
      │                         build conflicts (numba/whisper-timestamped) make it unviable on host Windows CPU.
      │
[ 4. XTTS-v2 (Coqui) ] ──────► TRIED & ARCHITECTED: High fidelity zero-shot cloning. Requires Microsoft C++ Build
      │                         Tools for monotonic alignment kernels on Windows; runs at 3.5s-5.0s on CPU.
      │                         Architected via isolated uv virtualenv / sidecar container for production GPU.
      │
[ 5. F5-TTS / Qwen-TTS ] ────► PRODUCTION TARGET: Pure flow-matching diffusion. Ideal for Linux virtual GPU
                                instance (<200ms on CUDA) using the 55s Simi reference audio.
```

#### Detailed Breakdown of Each Engine:

#### 1. Piper ONNX (Initial Baseline)
- **Path**: `models/piper/en_US-lessac-medium.onnx`
- **Characteristics**: Fast single-speaker model (~80ms).
- **Why It Was Replaced**: Articulation was robotic and lacked emotional warmth. Real estate buyers reacted poorly to the synthetic, flat cadence when discussing multi-crore property investments.

#### 2. Kokoro-82M ONNX (Current Active Primary Engine)
- **Path**: `models/kokoro/kokoro-v0_19.onnx`, `models/kokoro/voices.bin`
- **Characteristics**: 82-million parameter style-based TTS model running via ONNX Runtime.
- **Performance**: Synthesizes full sentences in **~120ms – 180ms on CPU** with 24,000 Hz output.
- **Why It Is the Chosen Engine**:
  - Completely local, offline, and free (0 external API calls, 0 cost).
  - High naturalness with human-like breathing, cadence, and vocal fry.
  - Runs cleanly inside Python 3.14 without requiring external C++ compilers.
  - Active voice: `af_sarah` / `af_bella` / neutral English blend, providing an articulate, warm, and professional persona for Aria.

#### 3. OpenVoice v2 (MyShell AI) — Evaluated & Tested
- **Mechanism**: Two-stage architecture using a base speaker + Tone Color Converter to copy speaker timbre from reference audio.
- **Findings & Barrier**:
  - The Python package (`openvoice-cli`) depends on legacy pinned packages (`librosa 0.9.1`, `numba`, `whisper-timestamped`, `dtw-python`).
  - In a modern Python 3.14 Windows environment, these legacy C-extensions deadlock or fail during unmanaged wheel builds.
  - Furthermore, OpenVoice only copies timbre—the underlying accent and pronunciation rhythm remain bound to the base American/British model, failing to produce a genuine Indian English cadence.

#### 4. XTTS-v2 (Coqui) — Tested & Designed
- **Mechanism**: Autoregressive transformer with HiFi-GAN vocoder, extracting speaker conditioning latents from reference audio.
- **Findings & Barrier**:
  - Coqui's monotonic alignment kernel (`monotonic_align/core.c`) requires Microsoft Visual C++ 14.0+ build tools (`cl.exe`) on Windows, preventing a clean `pip install`.
  - On CPU, sequential autoregressive decoding takes **3.5s to 5.0s per sentence**, which exceeds conversational telephony pause thresholds.
- **Solution Provided**: Designed as an isolated sidecar (Python 3.10 virtualenv via `uv` or Docker container) so it can run without modifying the main codebase.

#### 5. F5-TTS / Zero-Shot GPU Hook (Production Deployment Ready)
- **Mechanism**: Flow-matching diffusion model (`src/audio/tts.py` provider hook).
- **Reference Audio**: Integrated with the 55.84-second reference clip:
  `storage/cartesia_2026-11-03T21_59_47Z_Simi_sonic-3.6_en-IN.wav`
- **Status in Codebase**:
  - Configured via `settings.TTS_ENGINE = "zero_shot_clone"`.
  - Automatically detects hardware (`cuda` if available, else falls back to Kokoro ONNX).
  - When deployed on a Linux virtual GPU instance with CUDA, executes at **<200ms per clause**, providing the exact voice, timbre, and accent of Simi.

---

## 5. Spoken Text Verbalization & Phonetic Pre-Processing

Raw text generated by LLMs often contains numbers, currency symbols, and real estate jargon that sound unnatural when spoken by a TTS model.

`src/audio/tts.py` implements a domain-specific text normalizer before speech synthesis:

```python
# 1. Indian Currency Verbalization
"INR 48,00,000"     -> "48 lakhs"
"Rs. 85,00,000"     -> "85 lakhs"
"Rs. 1.35 Cr"       -> "1 point 35 crore"
"INR 4,000 per sqft"-> "4 thousand rupees per square foot"

# 2. Plot Dimensions & Square Footage
"1200 sqft"         -> "twelve hundred square feet"
"2400 sqft"         -> "twenty-four hundred square feet"
"30x40 ft"          -> "thirty feet by forty feet"

# 3. Regulatory Acronyms & Titles
"PRM/KA/RERA/..."   -> "R-E-R-A approved registration"
"3BHK"              -> "three B-H-K"
```

---

## 6. Incremental Audio Streaming (Minimizing TTFA)

To ensure the caller hears the agent speak immediately:

1. As the LLM produces text tokens, the `sentence_delimiter` regex splits incoming text at grammatical boundaries:
   ```python
   re.compile(r"(?<!\d)(?<!Mr)(?<!Ms)(?<!Dr)(?<!Rs)\s*([.!?;:\n]+)(?!\d)\s*")
   ```
2. The very first clause (e.g., *"Sure thing!"* or *"Green Valley Plots has two options."*) is dispatched to Kokoro immediately.
3. Kokoro synthesizes the audio at 24,000 Hz float32.
4. **Resampling Engine**: `resample_24k_to_16k()` downsamples the waveform to 16,000 Hz using high-speed linear interpolation.
5. `float32_to_int16_bytes()` converts normalized $[-1.0, 1.0]$ floats into signed 16-bit integer PCM bytes.
6. The binary PCM bytes are transmitted over the WebSocket directly to the client audio queue, achieving a Time-To-First-Audio of **~1.2 seconds**.
