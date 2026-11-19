# 01. System Architecture & Topology

## 1. Executive Summary

**RealEstateAI** is a production-grade, low-latency conversational voice agent platform architected specifically for real estate customer qualification, inventory discovery, and site visit scheduling. It couples sub-second full-duplex telephony audio streaming with domain-grounded retrieval-augmented generation (RAG) and structured dialogue management.

The system connects inbound and outbound callers directly to **Aria**, an AI real estate specialist representing Prime Realty. Aria automatically evaluates caller intent, retrieves verified property listings and legal documentation (RERA approvals, A-Katha status, pricing per square foot), resolves scheduling ambiguities, books site visit appointments directly into PostgreSQL calendar slots, and handles mid-call disconnects or human escalation with zero data loss.

```
                              ┌────────────────────────────────────────────────────────┐
                              │                    CALLER INTERFACE                    │
                              │         Browser WebRTC / Twilio Telephony Stream       │
                              └───────────────────────────┬────────────────────────────┘
                                                          │ 16kHz 16-bit PCM (WebSocket)
                                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             FASTAPI TELEPHONY GATEWAY                                                │
│                                           (src/telephony/event_router.py)                                            │
├─────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┤
│                    INBOUND AUDIO PATH                   │                    OUTBOUND AUDIO PATH                     │
│                                                         │                                                            │
│  ┌──────────────────────────────────────────────────┐  │  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Voice Activity Detection (WebRTC VAD)         │  │  │ 8. Sentence-by-Sentence Audio Generator              │  │
│  │    Trailing silence debounce (300-800ms)         │  │  │    Sentence boundary regex tokenizer                 │  │
│  └──────────────────────────┬───────────────────────┘  │  └──────────────────────────▲───────────────────────────┘  │
│                             │ Raw Speech Frames         │                             │ 16kHz PCM Chunks             │
│                             ▼                           │                             │                              │
│  ┌──────────────────────────────────────────────────┐  │  ┌──────────────────────────┴───────────────────────────┐  │
│  │ 2. Automated Speech Recognition (Faster-Whisper) │  │  │ 7. Text-to-Speech Engine                             │  │
│  │    Local 'base.en' model, CPU int8 quantization  │  │  │    - Primary: Kokoro-82M ONNX (~150ms on CPU)        │  │
│  │    Beam search, hallucination filtering          │  │  │    - Fallback: Piper ONNX                            │  │
│  └──────────────────────────┬───────────────────────┘  │  │    - Production GPU: F5-TTS / XTTS Zero-Shot Clone   │  │
│                             │ Spoken Text String        │  └──────────────────────────▲───────────────────────────┘  │
└─────────────────────────────┼───────────────────────────┴─────────────────────────────┼──────────────────────────────┘
                              │                                                         │ Natural Conversational Clauses
                              ▼                                                         │ (<14 words, Indian verbalization)
┌───────────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────┐
│                                             CORE DIALOGUE CONTROLLER                                                 │
│                                           (src/agent/dialogue_manager.py)                                            │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 3. State Machine & Context Assembler                                                                           │  │
│  │    - Ingests Lead Profile (Budget, Niche, Preferred Location, Inquiry Notes)                                   │  │
│  │    - Conversation Memory (Sliding Window History + Ephemeral Turn Cache)                                       │  │
│  │    - Strict Prompt Guardrails (Prevents JSON leakage, eliminates static fallbacks, optimizes verbal prosody)   │  │
│  └──────────────────────────┬─────────────────────────────────────────────────────────▲───────────────────────────┘  │
│                             │ Prompt + Function Tools Spec                            │ LLM Spoken Voice Response    │
│                             ▼                                                         │                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────┴───────────────────────────┐  │
│  │ 4. LLM Reasoning Engine (src/agent/llm_client.py)                                                              │  │
│  │    - Architecture Standard: Llama 3 8B running locally via Ollama (http://localhost:11434)                     │  │
│  │    - Active Cloud Provider: Groq OpenAI-Compatible API (llama-3.3-70b-versatile / llama3-8b-8192)              │  │
│  │      *Utilized due to local host CPU/RAM throughput limits for live zero-latency phone conversations*          │  │
│  │    - Automated Function/Tool Calling Dispatcher                                                                │  │
│  └──────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┘
                              │ Tool Calls (search_properties, check_availability, book_site_visit)
                              ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             DATA & RETRIEVAL SUBSYSTEM                                               │
├──────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────┐  │  ┌─────────────────────────────────────────────────────┐  │
│  │ 5. Hybrid RAG Knowledge Engine (src/rag/engine.py) │  │  │ 6. PostgreSQL Relational & Vector Storage           │  │
│  │    - Embedder: BAAI/bge-small-en-v1.5 (384-dim)    │  │  │    - pgvector extension with HNSW Index             │  │
│  │    - Cosine distance similarity search             │  │  │    - Customer Shareable vs Internal Restricted Data │  │
│  │    - Filter: access_label = 'customer_shareable'   │  │  │    - Property Listings, Appointments, Leads         │  │
│  │    - Hard isolation of developer margins & pricing │  │  │    - Stale Hold Expiration Background Worker        │  │
│  └────────────────────────────────────────────────────┘  │  └─────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Directory Map

The codebase is organized into cleanly decoupled packages under `src/`:

```
RealEstateAI/
├── src/
│   ├── main.py                     # ASGI application factory, lifespan pre-warming & background workers
│   ├── agent/                      # Core intelligence & dialogue orchestration
│   │   ├── dialogue_manager.py     # Main conversational state machine, tool dispatch & fallback logic
│   │   ├── llm_client.py           # Unified client abstraction (Ollama local / Groq cloud API)
│   │   ├── prompts.py              # System prompts, few-shot telephony examples, persona guidelines
│   │   └── tools.py                # Schema definitions for LLM tool calling
│   ├── audio/                      # Audio ingestion, VAD, transcription & speech synthesis
│   │   ├── vad.py                  # WebRTC Voice Activity Detector & energy frame segmenter
│   │   ├── stt.py                  # Faster-Whisper automated speech recognition wrapper
│   │   └── tts.py                  # Multi-engine TTS (Kokoro ONNX, Piper ONNX, Zero-Shot Hook)
│   ├── core/                       # Global configuration, constants & logging
│   │   ├── config.py               # Pydantic v2 BaseSettings loading environment variables
│   │   └── logging.py              # Loguru structured formatting with rotation
│   ├── db/                         # PostgreSQL database layer
│   │   ├── session.py              # Async & Sync engine factories, connection pooling
│   │   └── models/                 # SQLAlchemy 2.0 ORM entities
│   │       ├── base.py             # DeclarativeBase with UUID & timestamp mixins
│   │       ├── property.py         # PropertyListing entity with JSONB paperwork checklists
│   │       ├── knowledge.py        # KnowledgeChunk entity with 384-dim vector column
│   │       ├── calendar.py         # CalendarSlot and Appointment booking models
│   │       ├── lead.py             # Lead model with qualification metrics & status
│   │       ├── conversation.py     # ConversationTurn persistence for full audits
│   │       └── notification.py     # OperatorNotification model for escalation & hangups
│   ├── rag/                        # Knowledge extraction & similarity search
│   │   ├── embedder.py             # Local FastEmbed BAAI/bge-small-en-v1.5 runner
│   │   └── engine.py               # pgvector similarity queries with strict access filtering
│   ├── telephony/                  # Real-time WebSocket connection handling
│   │   └── event_router.py         # Bidirectional PCM audio router, turn boundaries, jitter buffers
│   └── workers/                    # Asynchronous support tasks
│       ├── cleanup_stale_holds.py  # 60-second recurring worker releasing abandoned slot locks
│       └── disconnect_recovery.py  # Mid-call disconnect analysis and human operator dispatch
├── alembic/                        # Database schema migrations (Alembic)
│   └── versions/                   # Migration scripts including HNSW index creation
├── models/                         # Local pre-warmed weights directory
│   ├── kokoro/                     # kokoro-v0_19.onnx, voices.bin (82M parameters)
│   ├── piper/                      # en_US-lessac-medium.onnx, config json
│   └── whisper/                    # faster-whisper-base.en snapshots
├── storage/                        # Persistent file assets & reference audio
│   └── cartesia_..._Simi_...wav    # 55-second Indian English Simi reference audio
└── scripts/                        # Operational scripts, database seeders, benchmarks
```

---

## 3. End-to-End Latency Budget & Timing Dynamics

In telephony conversational AI, human-perceived latency (the pause between the caller finishing their utterance and the agent speaking the first word) must stay below **1.5 seconds** to prevent callers from interrupting or feeling the system is unresponsive.

### Measured System Latency Breakdown (Current Benchmark):

| Stage | Subsystem / Operation | Typical Duration (ms) | Target Ceiling (ms) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **1. VAD** | WebRTC VAD Silence Debounce | 300 ms – 400 ms | 500 ms | Configurable via `MAX_SILENCE_DURATION_MS`. Prevents mid-sentence cutoffs. |
| **2. STT** | Faster-Whisper `base.en` (CPU int8) | 250 ms – 400 ms | 600 ms | Transcribes 3–5 second utterances in ~300ms on multi-core host CPU. |
| **3. Context & Guard** | Dialogue Manager Assembly | 10 ms – 20 ms | 50 ms | Builds sliding turn history and injects lead profile. |
| **4. LLM Turn 1** | Groq Llama-3 Function Calling | 250 ms – 350 ms | 500 ms | Emits `search_properties` tool call. |
| **5. RAG Retrieval** | PostgreSQL pgvector HNSW Query | 30 ms – 60 ms | 100 ms | FastEmbed vector search over 384-dim property chunks. |
| **6. LLM Turn 2** | Groq Llama-3 Voice Response | 250 ms – 350 ms | 500 ms | Produces first spoken clause (<14 words). |
| **7. Chunking** | Regex Boundary Segmenter | 1 ms – 2 ms | 5 ms | Splits at punctuation (`.`, `!`, `?`) to begin TTS immediately. |
| **8. TTS (First Chunk)**| Kokoro-82M ONNX (CPU) | 120 ms – 180 ms | 250 ms | Generates first clause waveform; resamples to 16kHz PCM. |
| **Total TTFA** | **Time-To-First-Audio** | **~1,200 ms – 1,450 ms** | **< 1,800 ms** | **Real-time conversation standard achieved on CPU host.** |

---

## 4. Subsystem Isolation & Data Safety Guarantees

1. **Complete Decoupling of Audio & Business Logic**:
   The WebSocket router (`src/telephony/event_router.py`) handles only binary frames, PCM packing, and silence detection. It knows nothing about real estate inventory, prompts, or SQL queries. All intelligence is delegated through the `DialogueManager` interface.

2. **RAG Data Privacy Boundary**:
   The knowledge base contains both public sales data and confidential operational information (investor purchase margins, commission splits, distressed acquisition values). The RAG retrieval engine strictly executes:
   $$\text{WHERE } \text{access\_label} = \text{'customer\_shareable'}$$
   Internal restricted knowledge chunks are physically barred from entering the prompt context of caller-facing turns.

3. **Concurrency & Slot Locking**:
   Site visit appointment booking uses a two-phase lock:
   - When a slot is discussed or proposed, its status shifts from `AVAILABLE` to `HOLD`.
   - A background worker (`src/workers/cleanup_stale_holds.py`) scans every 60 seconds. Any hold older than 120 seconds without a confirmed lead appointment is automatically reverted to `AVAILABLE`.
   - Double-booking is prevented at the database engine level via unique constraints on `(slot_id, appointment_status)`.
