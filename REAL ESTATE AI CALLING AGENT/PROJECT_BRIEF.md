# Project Brief – Technical Documentation

> **Target Audience**: Senior Software Engineers, System Architects, Machine Learning Engineers, and Full-Stack Developers onboarding to the **RealEstateAI** codebase.  
> **Source Verification**: All explanations, file paths, parameters, schemas, and diagrams in this document are strictly grounded in the actual codebase implementation. Zero functionality has been assumed or invented.

---

## 1. Project Overview

### 1.1 Project Name & Purpose
**RealEstateAI** is an enterprise-grade, autonomous conversational voice AI platform engineered for real estate customer qualification, inventory discovery, regulatory/legal verification, and site-visit scheduling. 

The system powers **Aria**, an AI real estate consultant representing Prime Realty. When prospective buyers inquire about residential plots, commercial flats, luxury villas, or industrial land parcels, Aria conducts full-duplex, low-latency voice calls over telephony or WebRTC. The agent discovers caller preferences, verifies budget alignment, queries a dense vector database (RAG) for legal approvals (RERA, BDA, A-Katha status), resolves scheduling ambiguities, and books site visits directly into calendar slots with zero human intervention.

### 1.2 The Problem the Project Solves
* **High Lead Drop-Off & Delayed Follow-Up**: Real estate digital marketing leads experience an 80%+ drop in conversion if not contacted within 5 minutes. Human sales agents cannot respond instantly 24/7.
* **Rigid IVR Friction**: Traditional Interactive Voice Response (IVR) phone trees (*"Press 1 for Sales, Press 2 for Plots"*) alienate buyers and cannot handle conversational nuances.
* **Model Hallucinations on Financial & Legal Facts**: Standard generative chatbots frequently invent square-footage prices, fabricate false regulatory approvals, or leak confidential developer profit margins.
* **Voice Latency Bottlenecks**: Multi-second delays between user speech and synthetic voice response destroy human conversational cadence. RealEstateAI achieves an end-to-end Time-to-First-Audio (TTFA) of **~1.18 seconds** on standard CPU hardware.

### 1.3 Main Application Functionality
* **Full-Duplex Telephony Audio Streaming**: Bidirectional streaming of 16,000 Hz 16-bit linear PCM audio over WebSockets with WebRTC Voice Activity Detection (VAD).
* **Automated Speech Recognition (ASR/STT)**: Sub-300ms speech-to-text powered by Faster-Whisper running CPU-quantized (`int8`) inference.
* **Domain-Grounded RAG Knowledge Retrieval**: Dense semantic similarity search over property specifications and legal checklists backed by PostgreSQL and the `pgvector` extension with HNSW indexing.
* **Two-Tier Privacy Isolation**: Hard database-level separation guaranteeing that confidential internal developer margins and distressed pricing are physically barred from caller-facing turns.
* **Transactional Concurrency-Safe Scheduling**: 5-minute optimistic slot-locking preventing double-booking, coupled with an autonomous background worker that cleans stale holds.
* **RFC 4791 CalDAV Synchronization**: Automatic creation of standard iCalendar `VEVENT` entries on Radicale, Apple Calendar, or Nextcloud for confirmed bookings.
* **Mid-Call Disconnect Recovery**: Automated detection of sudden network/VoIP drops, generating instant conversational summaries and dispatching urgent operator callback notifications.
* **Terminal-Based Simulation Tools**: Full multi-turn conversational testing tools allowing developers to test the entire stack via CLI without opening a browser or placing live phone calls.

### 1.4 Current Development Status
* **Core Voice Pipeline**: **Fully Implemented & Verified**. WebRTC VAD, Faster-Whisper STT, Kokoro-82M ONNX TTS, and sentence-by-sentence streaming are live and operational.
* **Dialogue & LLM**: **Fully Implemented & Verified**. Function calling (`search_properties`, `check_availability`, `book_site_visit`), conversational prompt constraints (<14 words per clause), and affirmative scheduling detection are active.
* **Database & RAG**: **Fully Implemented & Verified**. PostgreSQL 16 schema, Alembic migrations, HNSW cosine vector index, and pre-seeded multi-property inventory are active.
* **Active LLM Provider**: Currently routed through **Groq LPU API** (`llama-3.3-70b-versatile` / `llama3-8b-8192`) to maintain sub-second response times on local CPU development hardware, while the native architectural standard is configured for **Local Ollama Llama 3 8B** (`http://localhost:11434`).
* **Active TTS Engine**: Running **Kokoro-82M ONNX** locally on CPU (~150ms latency, zero API keys, zero cost). Zero-shot voice cloning (F5-TTS) is architected and ready for deployment on dedicated virtual GPU instances with CUDA.

---

## 2. Technology Stack

| Category | Technology / Library | Version / Configuration | Purpose in Project |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.14 (Active Host) / 3.10–3.12 (Target) | Core backend programming language. |
| **Web Framework** | FastAPI | 0.115+ (Starlette 0.47+) | Asynchronous HTTP REST API and WebSocket gateway. |
| **ASGI Server** | Uvicorn | 0.35+ | High-performance asynchronous server running the FastAPI app. |
| **Database** | PostgreSQL | 16 | Relational persistence for leads, slots, appointments, and turns. |
| **Vector Engine** | pgvector | Extension on PostgreSQL 16 | Stores 384-dimensional dense vectors with HNSW cosine indexing. |
| **ORM & Migrations**| SQLAlchemy / Alembic | 2.0.46 / 1.18.3 | Async (`asyncpg`) and Sync (`psycopg2`) relational object mapping and migrations. |
| **VAD** | WebRTC VAD | `webrtcvad-wheels` 2.0.14 | Slices continuous audio into 20ms frames; detects speech vs. silence. |
| **Speech-to-Text** | Faster-Whisper | 1.2.1 (CTranslate2 4.8.1) | Transcribes inbound caller speech using `base.en` with `int8` quantization. |
| **Text-to-Speech** | Kokoro ONNX | `kokoro-onnx` 0.4.7 | Active primary TTS: 82M style-based TTS model generating 24kHz audio in ~150ms. |
| **Fallback TTS** | Piper TTS | `piper-tts` / ONNX | Fallback local neural speech engine (`en_US-lessac-medium.onnx`). |
| **Zero-Shot Voice** | F5-TTS / XTTS-v2 | Provider hook (`src/audio/tts.py`) | Zero-shot cloning from 55s reference audio for production GPU instances. |
| **Embeddings** | FastEmbed | `BAAI/bge-small-en-v1.5` | Dense vector generation (384 dimensions) running locally on CPU. |
| **LLM Runtime** | Meta Llama 3 | Local Ollama (8B) / Cloud Groq (70B) | Natural language reasoning, prompt adherence, and structured tool calling. |
| **Calendar Protocol**| CalDAV / iCalendar | `caldav` 3.7.0 / `icalendar` | RFC 4791 synchronization with external calendar servers (Radicale). |
| **HTTP Client** | HTTPX | 0.28+ | Asynchronous connection pooling for LLM completions and webhooks. |
| **Logging** | Loguru | 0.7.3 | Structured, rotated, and color-coded runtime logging. |
| **Environment** | Pydantic Settings | 2.x (Pydantic v2) | Strict type-enforced environment configuration loader. |

---

## 3. Project Folder Structure

```
RealEstateAI/
├── .env                              # Active local environment variables (not committed)
├── .env.example                      # Template environment configuration
├── DEPLOYMENT.md                     # Production deployment guides and system requirements
├── docker-compose.prod.yml           # Production multi-container Docker Compose configuration
├── Dockerfile                        # Multi-stage production container build definition
├── pyproject.toml                    # Project metadata, build tools, and package specifications
├── requirements.txt                  # Locked Python dependency manifest
├── alembic/                          # Database schema migration environment
│   ├── env.py                        # Alembic migration configuration and target metadata
│   ├── script.py.mako                # Migration script generation template
│   └── versions/                     # Migration scripts
│       └── c3a4f891b72e_add_hnsw...  # Migration script enabling pgvector & HNSW index
├── docs/                             # Deep-dive subsystem documentation suite
│   ├── README.md                     # Master documentation index
│   ├── 01_SYSTEM_ARCHITECTURE.md     # Architecture boundaries and latency budget
│   ├── 02_VOICE_AUDIO_PIPELINE.md    # VAD, STT, and comparative analysis of 5 TTS models
│   ├── 03_DIALOGUE_STATE_AND_LLM.md  # Conversational state machine & local vs cloud LLM
│   ├── 04_RAG_AND_KNOWLEDGE_BASE.md  # pgvector HNSW search and access label security
│   ├── 05_DATABASE_AND_DATA_MODELS.md# SQLAlchemy ORM entity-relationship schemas
│   ├── 06_API_AND_WEBSOCKET_SPEC.md  # REST endpoints & binary WebSocket protocol
│   ├── 07_OPERATIONS_AND_RECOVERY.md # Disconnect recovery and stale hold worker
│   ├── 08_DEPLOYMENT_AND_RUNBOOK.md  # Local and production deployment runbooks
│   ├── 09_CALDAV_AND_LEAD_TESTING.md # CalDAV sync and terminal testing walkthrough
│   ├── 10_DEEP_DIVE_CODEBASE_REF.md  # Code-level mechanics and turn execution trace
│   └── 11_FILE_TO_FILE_CONNECT.md    # Exhaustive cross-file call matrix and data contracts
├── models/                           # Downloaded local neural model weights
│   ├── kokoro/                       # kokoro-v0_19.onnx (325MB) and voices.bin (5.7MB)
│   ├── piper/                        # en_US-lessac-medium.onnx and config JSON
│   └── whisper/                      # Faster-Whisper base.en snapshot weights
├── scripts/                          # Operational, database, and benchmarking scripts
│   ├── agent_doctor.py               # Comprehensive system health and pre-flight diagnostic CLI
│   ├── benchmark_audio.py            # Latency profiling for STT, TTS, and audio resampling
│   ├── bootstrap_prod_models.py      # Automated model downloader for production containers
│   ├── cleanup_stale_holds.py        # Standalone runner for stale calendar hold cleanup
│   ├── download_models.py            # CLI model weight downloader
│   ├── ingest_documents.py           # Document chunking and embedding ingest script
│   ├── interactive_lead_call.py      # Interactive terminal-based lead call simulator
│   ├── profile_latency.py            # End-to-end turn latency profiler
│   ├── seed_data.py                  # Populates property listings, knowledge chunks, and slots
│   ├── simulate_dialogue.py          # Multi-turn conversational simulation without audio
│   ├── test_call_stream.py           # WebSocket binary PCM socket streaming test harness
│   ├── test_lead_flow_e2e.py         # End-to-end integration test for lead lifecycle
│   ├── verify_infra.py               # Verifies DB, Redis, and disk volume health
│   └── verify_leads.py               # Verifies lead ingestion and validation rules
├── src/                              # Core application source code
│   ├── main.py                       # Top-level ASGI entrypoint and lifespan manager
│   ├── agent/                        # Core conversational intelligence and state management
│   │   ├── dialogue_manager.py       # Main state machine, tool dispatch, and prompt guards
│   │   ├── lead_service.py           # Ingestion triage, phone cleaning, and summary generation
│   │   ├── llm_client.py             # Unified client abstraction (Ollama Local / Groq Cloud)
│   │   ├── prompts.py                # Telephony persona, brevity constraints, and few-shot examples
│   │   └── tools.py                  # OpenAI-compatible function calling tool schemas
│   ├── api/                          # HTTP REST API and routing layer
│   │   ├── app.py                    # FastAPI application factory and CORS middleware setup
│   │   └── routes.py                 # REST routes (/api/leads, /api/calls) and WebSocket mount
│   ├── audio/                        # Audio processing, transcription, and speech synthesis
│   │   ├── stt.py                    # Faster-Whisper STT wrapper with confidence scoring
│   │   ├── tts.py                    # Multi-engine TTS, linear resampler, and text verbalizer
│   │   └── vad.py                    # WebRTC VAD 20ms frame slicer and turn boundary detector
│   ├── core/                         # Global configuration, logging, and metrics
│   │   ├── config.py                 # Pydantic v2 settings loading from environment
│   │   ├── logging.py                # Structured Loguru logging configuration
│   │   └── metrics.py                # Prometheus latency, error, and call metric collectors
│   ├── db/                           # Database connection, models, and synchronization
│   │   ├── caldav_sync.py            # RFC 4791 CalDAV synchronization adapter (Radicale)
│   │   ├── scheduling_manager.py     # Two-phase atomic slot locking and booking manager
│   │   ├── session.py                # Async and Sync SQLAlchemy engine factories
│   │   └── models/                   # SQLAlchemy 2.0 ORM entity definitions
│   │       ├── base.py               # DeclarativeBase with UUID and timestamp mixins
│   │       ├── calendar.py           # CalendarSlot and Appointment booking models
│   │       ├── conversation.py       # ConversationTurn audit model
│   │       ├── enums.py              # PropertyNiche, SlotStatus, AccessLabel, SpeakerType
│   │       ├── knowledge.py          # KnowledgeChunk with 384-dim vector column
│   │       ├── lead.py               # Lead entity with qualification tracking
│   │       ├── notification.py       # OperatorNotification model for escalation and hangups
│   │       └── property.py           # PropertyListing with JSONB paperwork checklists
│   ├── rag/                          # Retrieval-Augmented Generation subsystem
│   │   ├── confidentiality_filter.py # Hard security filter for customer_shareable data
│   │   ├── embeddings.py             # FastEmbed dense vector embedder wrapper
│   │   ├── indexer.py                # Document chunking, hashing, and vector indexing
│   │   └── retriever.py              # pgvector HNSW cosine distance similarity queries
│   ├── telephony/                    # Full-duplex WebSocket audio communication
│   │   ├── event_router.py           # Bidirectional binary PCM router and VAD loop
│   │   ├── session_manager.py        # Call lifecycle tracking and disconnect recovery
│   │   └── stream_handler.py         # Audio packet packing and jitter buffer utilities
│   └── workers/                      # Background asynchronous tasks
│       └── stale_hold_cleanup.py     # 60-second recurring worker resetting abandoned slot holds
└── storage/                          # Persistent file storage and reference media
    ├── audio_cache/                  # Cached synthesized turn waveforms (.wav)
    ├── documents/                    # Raw real estate brochures, markdown, and PDFs
    └── cartesia_..._Simi_...wav      # 55.84s reference audio clip for Indian English voice cloning
```

---

## 4. Application Architecture

RealEstateAI employs an asynchronous, event-driven streaming architecture. Decoupled modules communicate across strictly typed boundaries, separating fast binary audio handling from database transactions and language model generation.

```mermaid
flowchart TD
    subgraph ClientLayer ["1. Client & Telephony Gateway"]
        Browser["Browser WebRTC Worklet"]
        Twilio["Twilio / VoIP SIP Trunk"]
    end

    subgraph AudioLayer ["2. Audio Ingestion & Synthesis Layer"]
        Router["event_router.py<br/>(WebSocket Router)"]
        VAD["vad.py<br/>(WebRTC VAD 20ms)"]
        STT["stt.py<br/>(Faster-Whisper int8)"]
        TTS["tts.py<br/>(Kokoro-82M ONNX)"]
        Resampler["Linear Resampler<br/>(24kHz to 16kHz)"]
    end

    subgraph IntelligenceLayer ["3. Dialogue & Intelligence Core"]
        DM["dialogue_manager.py<br/>(State Machine & Guardrails)"]
        LeadSvc["lead_service.py<br/>(Lead Brief & Quality)"]
        LLM["llm_client.py<br/>(Ollama Local / Groq Cloud)"]
    end

    subgraph DataLayer ["4. RAG & Relational Subsystem"]
        Retriever["retriever.py<br/>(RAG Hybrid Search)"]
        Embedder["embeddings.py<br/>(FastEmbed BGE-small)"]
        Scheduler["scheduling_manager.py<br/>(Two-Phase Locking)"]
        CalDAV["caldav_sync.py<br/>(RFC 4791 CalDAV Adapter)"]
        PG[(PostgreSQL 16<br/>pgvector HNSW)]
        Radicale[("Radicale / CalDAV Server<br/>(SiteVisits Calendar)")]
    end

    subgraph WorkersLayer ["5. Resilient Operations"]
        StaleHoldWorker["stale_hold_cleanup.py<br/>(60s Loop)"]
        DisconnectRecovery["session_manager.py<br/>(Hangup Recovery)"]
    end

    Browser <==>|16kHz Binary PCM| Router
    Twilio <==>|16kHz Binary PCM| Router

    Router -->|20ms Linear PCM Slices| VAD
    VAD -->|Accumulated Utterance Buffer| STT
    STT -->|Spoken String Transcript| DM

    LeadSvc -.->|Injects Caller Profile| DM
    DM <==>|Messages + Tool Calling Specs| LLM

    DM -->|search_properties| Retriever
    Retriever -->|Query String| Embedder
    Embedder -->|384-dim Vector| Retriever
    Retriever <==>|Cosine HNSW Filtered Query| PG

    DM -->|check_availability / book_site_visit| Scheduler
    Scheduler <==>|SELECT FOR UPDATE / 5min Hold| PG
    Scheduler -->|sync_appointment_event| CalDAV
    CalDAV -->|RFC 5545 VEVENT| Radicale

    DM -->|Spoken Clause String < 14 words| TTS
    TTS -->|24kHz Float32 Waveform| Resampler
    Resampler -->|16kHz 16-bit PCM Bytes| Router

    Router -.->|Abrupt Disconnect Event| DisconnectRecovery
    DisconnectRecovery -->|Creates Urgent Notification| PG
    StaleHoldWorker ==>|Scans & Clears Expired Holds| PG
```

### Core Architectural Patterns:
1. **Full-Duplex Producer-Consumer Audio Loop**: The WebSocket transport separates incoming caller audio from outgoing agent audio. VAD framing and speech accumulation happen independently of TTS playback.
2. **Strategy / Provider Pattern for Speech Engines**: The `TextToSpeechEngine` in `src/audio/tts.py` defines an extensible provider interface supporting `kokoro`, `piper`, and `zero_shot_clone` without altering telephony or business logic.
3. **Optimistic Two-Phase Slot Locking**: Calendar slot reservations use a transient `HELD` state with an absolute expiration timestamp (`held_until = NOW() + 5 min`), preventing race conditions between concurrent prospects.
4. **Physical Data Isolation Boundary**: The RAG retrieval engine strictly injects `WHERE access_label = 'customer_shareable'`, preventing internal developer financial margins from entering the LLM prompt.

---

## 5. Application Working Flow

### 5.1 Lifespan Boot & Model Pre-Warming
When the server starts via `src/main.py`:
1. The `@asynccontextmanager` lifespan event activates.
2. Speech and embedding models are pre-warmed in memory:
   - `tts_engine._get_kokoro()` loads `models/kokoro/kokoro-v0_19.onnx` and `voices.bin`.
   - `stt_engine._get_model()` loads Faster-Whisper `base.en` with `int8` quantization.
   - `embedding_engine._get_model()` loads FastEmbed `BAAI/bge-small-en-v1.5`.
3. The background task `stale_hold_cleanup_worker` starts running its 60-second polling loop.
4. Uvicorn binds to `0.0.0.0:8000` and begins accepting HTTP and WebSocket connections.

### 5.2 End-to-End Conversational Turn Lifecycle

```mermaid
sequenceDiagram
    autonumber
    actor Caller as Caller (Browser / Telephony)
    participant WS as event_router.py (WebSocket)
    participant VAD as vad.py (WebRTC VAD)
    participant STT as stt.py (Faster-Whisper)
    participant DM as dialogue_manager.py
    participant LLM as llm_client.py (Groq / Ollama)
    participant RAG as retriever.py (pgvector)
    participant TTS as tts.py (Kokoro ONNX)

    Caller->>WS: Speaks: "Yes, I inquired about that plot, tell me about it."
    WS->>VAD: Streams 640-byte chunks (20ms frames at 16kHz)
    VAD->>VAD: Detects voice activity; accumulates 117,120 bytes
    VAD->>WS: Emits Turn Boundary (400ms trailing silence detected)
    
    WS->>STT: Dispatches accumulated PCM buffer
    STT->>STT: CTranslate2 int8 transcription (Elapsed: ~280ms)
    STT-->>WS: Returns transcript: "Yes, I inquired about that plot..."
    
    WS->>DM: Calls process_turn(session_id, transcript)
    DM->>DM: Assembles history + Lead Brief (Pooja Banerjee, 90L Budget)
    DM->>LLM: Dispatches messages + 4 Tool Schemas
    LLM-->>DM: Emits Tool Call: search_properties(niche='residential_plot', ...) (Elapsed: ~280ms)
    
    DM->>RAG: Executes search_properties()
    RAG->>RAG: Embeds query with FastEmbed (384-dim, ~25ms)
    RAG->>RAG: PostgreSQL HNSW Cosine Search (access_label='customer_shareable')
    RAG-->>DM: Returns 3 verified chunks (Green Valley 1200 sqft at 48L, 2400 sqft at 85L)
    
    DM->>LLM: Injects Tool Results; requests spoken voice response
    LLM-->>DM: Generates Spoken Text: "Sure thing. So, Green Valley Plots has..." (Elapsed: ~260ms)
    
    DM->>TTS: Streams text clause to synthesize_stream()
    TTS->>TTS: Normalizes numbers ("1,200 sqft" -> "twelve hundred square feet")
    TTS->>TTS: Kokoro synthesizes 24kHz audio (Elapsed: ~135ms)
    TTS->>TTS: Downsamples 24kHz -> 16kHz via linear interpolation
    TTS->>WS: Yields 16kHz 16-bit PCM bytes
    WS-->>Caller: Streams synthetic voice audio chunk
    Note over Caller,TTS: Total Measured Latency: ~1.18 seconds
```

---

## 6. Modules and Functionalities

### 6.1 `src.agent` (Conversational Intelligence)
* **`dialogue_manager.py`**:
  - Implements the central conversational state machine (`DialogueManager`).
  - Manages session sliding memory and tool execution.
  - Implements `is_affirmative_to_visit` detection to prevent repetitive scheduling confirmation loops.
  - Enforces prompt guards preventing raw JSON leakage into spoken output.
  - *Status*: **Implemented & Verified**.
* **`lead_service.py`**:
  - Validates inbound lead submissions via `evaluate_lead_quality()`.
  - Filters placeholder phone numbers (e.g. `1234567890`) and dummy names.
  - Generates conversational background briefs for Aria.
  - Generates 75-to-200 word post-call executive summaries.
  - *Status*: **Implemented & Verified**.
* **`llm_client.py`**:
  - Unified asynchronous client abstraction for OpenAI-compatible LLM endpoints.
  - Connects to **Local Ollama** (`http://localhost:11434`) or **Cloud Groq** (`https://api.groq.com/openai/v1`).
  - *Status*: **Implemented & Verified**.
* **`prompts.py`**:
  - Defines `AGENT_SYSTEM_PROMPT` containing persona directives for Aria.
  - Enforces brevity (<14 words per clause), conversational openers (*"Got it!"*, *"Sure thing."*), and single-question handoffs.
  - *Status*: **Implemented & Verified**.
* **`tools.py`**:
  - Contains JSON schemas for 4 callable functions: `search_properties`, `check_availability`, `book_site_visit`, `escalate_to_human`.
  - *Status*: **Implemented & Verified**.

### 6.2 `src.audio` (Speech Processing & Synthesis)
* **`vad.py`**:
  - Encapsulates WebRTC VAD (`VADProcessor`).
  - Slices audio streams into 20ms frames (640 bytes at 16kHz) and tracks trailing silence.
  - *Status*: **Implemented & Verified**.
* **`stt.py`**:
  - Manages Faster-Whisper automated speech recognition (`SpeechToTextEngine`).
  - Performs `int8` CPU inference, confidence scoring, and empty audio filtering.
  - *Status*: **Implemented & Verified**.
* **`tts.py`**:
  - Manages speech synthesis via `TextToSpeechEngine`.
  - Supports Kokoro-82M ONNX (primary), Piper ONNX (fallback), and F5-TTS (zero-shot GPU).
  - Implements linear interpolation resampling (`resample_24k_to_16k`).
  - Contains domain-specific text normalizer verbalizing Indian currencies (*"48 lakhs"*), plot dimensions, and RERA acronyms.
  - *Status*: **Implemented & Verified**.

### 6.3 `src.db` (Persistence & External Synchronization)
* **`session.py`**:
  - Initializes `async_engine` (`postgresql+asyncpg://`) and `sync_engine` (`postgresql://psycopg2`).
  - Provides `get_async_db` dependency and `SyncSessionLocal` session maker.
  - *Status*: **Implemented & Verified**.
* **`scheduling_manager.py`**:
  - Manages calendar availability queries, 5-minute atomic holds, and booking confirmations.
  - Prevents race conditions using row-level constraints.
  - Triggers external CalDAV calendar synchronization upon confirmation.
  - *Status*: **Implemented & Verified**.
* **`caldav_sync.py`**:
  - Implements `CalDAVSyncAdapter` communicating with RFC 4791 CalDAV servers (Radicale).
  - Serializes bookings into RFC 5545 `VEVENT` records localized to `Asia/Kolkata`.
  - *Status*: **Implemented & Verified**.
* **`models/`**:
  - Declarative SQLAlchemy models: `Lead`, `PropertyListing`, `KnowledgeChunk`, `CalendarSlot`, `Appointment`, `ConversationTurn`, `OperatorNotification`.
  - *Status*: **Implemented & Verified**.

### 6.4 `src.rag` (Knowledge Base & Vector Retrieval)
* **`embeddings.py`**:
  - Wraps FastEmbed `BAAI/bge-small-en-v1.5` generating 384-dimensional dense vectors on CPU.
  - *Status*: **Implemented & Verified**.
* **`retriever.py`**:
  - Executes pgvector HNSW cosine similarity queries.
  - Strictly filters by `access_label = 'customer_shareable'`.
  - *Status*: **Implemented & Verified**.
* **`confidentiality_filter.py`**:
  - Secondary verification filter ensuring internal developer margins and distressed purchase metrics never enter caller prompt contexts.
  - *Status*: **Implemented & Verified**.

### 6.5 `src.telephony` (Full-Duplex Communication & Session Control)
* **`event_router.py`**:
  - Manages the WebSocket lifecycle at `/ws/audio/{session_id}`.
  - Pipes inbound audio to VAD and yields outbound PCM chunks from TTS.
  - Handles barge-in interruptions and mid-call disconnects.
  - *Status*: **Implemented & Verified**.
* **`session_manager.py`**:
  - Tracks telephony session state, turn counts, and timestamps.
  - Implements `handle_abrupt_disconnect()`: analyzes dropped call transcripts via LLM and writes urgent `OperatorNotification` records.
  - *Status*: **Implemented & Verified**.

### 6.6 `src.workers` (Background Operations)
* **`stale_hold_cleanup.py`**:
  - Autonomous 60-second background worker.
  - Reverts expired `HELD` slots back to `AVAILABLE`.
  - *Status*: **Implemented & Verified**.

---

## 7. Global Variables, Singletons & Shared State

| Singleton / Shared Variable | File Declared | Type / Class | Where Consumed | Lifecycle / Scope |
| :--- | :--- | :--- | :--- | :--- |
| `settings` | `src/core/config.py` | `Settings` (Pydantic) | Globally across all modules in `src/` | Process-wide immutable singleton. |
| `logger` | `src/core/logging.py` | `loguru.Logger` | Globally across all modules in `src/` | Process-wide thread-safe logging instance. |
| `tts_engine` | `src/audio/tts.py` | `TextToSpeechEngine` | `src/telephony/event_router.py`, `src/main.py`, `scripts/` | Process-wide pre-warmed singleton. |
| `stt_engine` | `src/audio/stt.py` | `SpeechToTextEngine` | `src/telephony/event_router.py`, `src/main.py` | Process-wide pre-warmed singleton. |
| `embedding_engine` | `src/rag/embeddings.py`| `FastEmbedEmbedder` | `src/rag/retriever.py`, `src/main.py`, `scripts/` | Process-wide pre-warmed singleton. |
| `dialogue_manager` | `src/agent/dialogue_manager.py`| `DialogueManager` | `src/telephony/event_router.py`, `src/telephony/session_manager.py` | Stateful singleton managing session histories in memory. |
| `session_manager` | `src/telephony/session_manager.py`| `TelephonySessionManager` | `src/api/routes.py`, `src/telephony/event_router.py` | Process-wide tracking of active calls and hangups. |
| `lead_service` | `src/agent/lead_service.py`| `LeadService` | `src/api/routes.py`, `src/agent/dialogue_manager.py`, `scripts/` | Stateless business service singleton. |
| `scheduling_manager`| `src/db/scheduling_manager.py`| `SchedulingManager` | `src/agent/dialogue_manager.py`, `src/api/routes.py` | Transactional scheduling service singleton. |
| `caldav_sync` | `src/db/caldav_sync.py` | `CalDAVSyncAdapter` | `src/db/scheduling_manager.py` | Network-connected CalDAV synchronization adapter. |
| `async_engine` | `src/db/session.py` | `AsyncEngine` (SQLAlchemy)| `src/db/session.py` (`get_async_db`) | Connection pool for asynchronous database interactions. |
| `SyncSessionLocal`| `src/db/session.py` | `sessionmaker[Session]`| Alembic migrations, standalone CLI worker scripts | Synchronous connection factory for scripts. |

---

## 8. Environment Variables and Configuration Matrix

All configuration parameters are defined in `src/core/config.py` and loaded from `.env`.

| Variable Name | Required? | Default Value | Consuming Modules | Description |
| :--- | :--- | :--- | :--- | :--- |
| `ENVIRONMENT` | Optional | `development` | `src/core/config.py`, `src/db/session.py` | Runtime environment (`development`, `production`, `test`). |
| `LOG_LEVEL` | Optional | `INFO` | `src/core/logging.py` | Standard output logging verbosity (`DEBUG`, `INFO`, `WARNING`). |
| `APP_HOST` | Optional | `0.0.0.0` | `src/main.py` | Host interface IP for Uvicorn ASGI server binding. |
| `APP_PORT` | Optional | `8000` | `src/main.py` | Network port for Uvicorn ASGI server binding. |
| `DATABASE_URL` | **Mandatory**| `postgresql+asyncpg://...` | `src/db/session.py` | Asynchronous PostgreSQL connection string for FastAPI runtime. |
| `DATABASE_SYNC_URL` | **Mandatory**| `postgresql://...` | `src/db/session.py`, `alembic/env.py` | Synchronous PostgreSQL connection string for Alembic migrations. |
| `LLM_BASE_URL` | **Mandatory**| `http://localhost:11434` | `src/agent/llm_client.py` | URL for Ollama local (`localhost:11434`) or Groq (`api.groq.com/openai/v1`). |
| `LLM_MODEL_NAME` | **Mandatory**| `llama3:latest` | `src/agent/llm_client.py` | Model identifier (e.g. `llama-3.3-70b-versatile` or `llama3:latest`). |
| `LLM_API_KEY` | Optional | `None` | `src/agent/llm_client.py` | Bearer token for cloud API (Groq). Not required for local Ollama. |
| `LLM_TEMPERATURE` | Optional | `0.1` | `src/agent/llm_client.py` | Sampling temperature for factual conversational consistency. |
| `LLM_TIMEOUT_SECONDS` | Optional | `120.0` | `src/agent/llm_client.py` | HTTP client timeout for completion requests. |
| `WHISPER_MODEL_SIZE` | Optional | `base.en` | `src/audio/stt.py` | Model size for Faster-Whisper ASR (`tiny.en`, `base.en`, `small.en`). |
| `WHISPER_DEVICE` | Optional | `cpu` | `src/audio/stt.py` | Target compute device for ASR (`cpu` or `cuda`). |
| `WHISPER_COMPUTE_TYPE`| Optional | `int8` | `src/audio/stt.py` | Quantization precision for ASR (`int8`, `float16`, `float32`). |
| `WHISPER_DOWNLOAD_ROOT`| Optional | `./models/whisper` | `src/audio/stt.py` | Local disk directory for Faster-Whisper weight snapshots. |
| `TTS_ENGINE` | Optional | `kokoro` | `src/audio/tts.py` | Active voice synthesis provider (`kokoro`, `piper`, `zero_shot_clone`). |
| `KOKORO_MODEL_PATH` | Optional | `./models/kokoro/kokoro-v0_19.onnx`| `src/audio/tts.py` | Local file path to Kokoro-82M ONNX model graph. |
| `KOKORO_VOICES_PATH`| Optional | `./models/kokoro/voices.bin` | `src/audio/tts.py` | Local file path to serialized Kokoro voice style vectors. |
| `KOKORO_DEFAULT_VOICE`| Optional | `af_sarah` | `src/audio/tts.py` | Default speaker persona voice key (`af_sarah`, `af_bella`, `bf_emma`). |
| `KOKORO_SPEED` | Optional | `1.0` | `src/audio/tts.py` | Voice synthesis speed multiplier ($0.5$ to $2.0$). |
| `ZERO_SHOT_REF_AUDIO_PATH`| Optional | `./storage/cartesia_..._Simi_...wav` | `src/audio/tts.py` | Path to 55.84s reference audio clip for Indian English voice cloning. |
| `CALDAV_URL` | Optional | `http://localhost:5232/` | `src/db/caldav_sync.py` | Endpoint URL for the external CalDAV server (Radicale). |
| `CALDAV_USERNAME` | Optional | `realestate_admin` | `src/db/caldav_sync.py` | Basic authentication username for CalDAV server. |
| `CALDAV_PASSWORD` | Optional | `realestate_secure_pass` | `src/db/caldav_sync.py` | Basic authentication password for CalDAV server. |
| `CALDAV_CALENDAR_NAME`| Optional | `SiteVisits` | `src/db/caldav_sync.py` | Target calendar collection name in CalDAV. |
| `AUDIO_SAMPLE_RATE` | Optional | `16000` | `src/audio/tts.py`, `src/telephony/event_router.py` | Standard telephony linear PCM sample rate in Hz. |
| `MAX_SILENCE_DURATION_MS`| Optional | `1800` (400-600 typical) | `src/audio/vad.py`, `src/telephony/event_router.py` | Trailing silence debounce window in ms triggering turn completion. |

---

## 9. API and Database Documentation

### 9.1 REST APIs (`src/api/routes.py`)

#### `POST /api/leads/submit`
* **Purpose**: Ingests new buyer leads from web forms or ad platforms; executes automated quality scoring.
* **Payload**:
  ```json
  {
    "name": "Pooja Banerjee",
    "phone_number": "+919820098765",
    "niche": "residential_plot",
    "budget_max": 9000000,
    "preferred_location": "North Hills",
    "inquiry_notes": "Interested in a 1200 sqft plot near Canadian International School."
  }
  ```
* **Success Response (HTTP 200)**:
  ```json
  {
    "status": "success",
    "lead_id": 18,
    "phone_number": "+919820098765",
    "quality": "valid"
  }
  ```
* **Error Handling**: Returns HTTP 422 if the phone number fails digit-length or dummy pattern checks.

#### `POST /api/calls/start`
* **Purpose**: Registers an inbound or outbound call session, creates the session in `TelephonySessionManager`, and links the lead's profile.
* **Payload**:
  ```json
  {
    "phone_number": "+919820098765",
    "direction": "inbound"
  }
  ```
* **Success Response (HTTP 200)**:
  ```json
  {
    "session_id": "phone_call_mw18u8pk",
    "ws_audio_url": "/ws/audio/phone_call_mw18u8pk?phone=%2B919820098765",
    "caller_name": "Pooja Banerjee",
    "status": "initialized"
  }
  ```

#### `GET /api/calendar/slots`
* **Purpose**: Lists open calendar slots for site visits across upcoming business days.
* **Query Parameters**: `start_date` (optional, ISO date), `status` (default: `available`).
* **Success Response (HTTP 200)**: Array of slot objects containing `slot_id`, `start_time`, `end_time`, and `status`.

#### `GET /api/notifications`
* **Purpose**: Fetches queued operator notifications (such as abrupt mid-call disconnect alerts).
* **Success Response (HTTP 200)**: Array of notification records containing `id`, `session_id`, `phone_number`, `urgency`, and `reason`.

### 9.2 WebSocket Audio Protocol (`/ws/audio/{session_id}`)
* **Transport**: Binary WebSocket frames over `/ws/audio/{session_id}?phone={phone_number}`.
* **Inbound Stream**: Continuous 20ms binary frames (640 bytes of 16kHz 16-bit Mono Linear PCM).
* **Outbound Stream**: Asynchronous binary frames of resampled 16kHz 16-bit PCM voice audio synthesized by Kokoro-82M ONNX.
* **Control Events**: JSON text frames for `{"event": "clear_buffer"}` (caller barge-in) and `{"event": "hangup"}`.

### 9.3 Database Models (`src/db/models/`)

```mermaid
erDiagram
    Lead ||--o{ Appointment : books
    Lead ||--o{ ConversationTurn : produces
    CalendarSlot ||--o| Appointment : reserves
    
    Lead {
        uuid id PK
        string name
        string phone_number UK
        string niche
        float budget_max
        string preferred_location
        text inquiry_notes
        string qualification_status
        int call_count
        timestamp created_at
    }

    CalendarSlot {
        uuid id PK
        timestamp slot_start
        timestamp slot_end
        string status
        timestamp held_until
        string held_by_phone
    }

    Appointment {
        uuid id PK
        uuid lead_id FK
        uuid slot_id FK
        string visitor_name
        string status
        text notes
        timestamp created_at
    }

    PropertyListing {
        uuid id PK
        string title
        string project_name
        string niche
        string location
        numeric price
        string dimensions
        string legal_approval_status
        jsonb paperwork_checklist
    }

    KnowledgeChunk {
        uuid id PK
        string title
        text content
        string niche
        string access_label
        vector_384 embedding
    }

    ConversationTurn {
        uuid id PK
        string session_id
        int turn_number
        string speaker
        text text
        int latency_ms
        timestamp created_at
    }

    OperatorNotification {
        int id PK
        string session_id
        string phone_number
        string urgency
        string reason
        string status
        timestamp created_at
    }
```

---

## 10. Setup and Execution Guide

### 10.1 Prerequisites
1. **Python 3.10–3.14**: Python 3.14 on Windows is supported and active in this workspace.
2. **PostgreSQL 16 with pgvector**: Must be running locally on port `5432` or `5435`.
3. **Hardware**: Standard multi-core CPU (8GB+ RAM). A discrete GPU is optional in development.

### 10.2 Step-by-Step Local Setup

#### Step 1: Environment Verification & Deactivation
Ensure that any lingering or broken virtual environments are deactivated:
```powershell
# Navigate to repository root
cd C:\Users\aseem\RealEstateAI

# Deactivate lingering virtual environments
deactivate
```

#### Step 2: Database Setup & Alembic Migrations
Verify PostgreSQL is running, then run migrations to establish tables and HNSW vector indices:
```powershell
python -m alembic upgrade head
```

#### Step 3: Seed Inventory & Calendar Slots
Populate property listings, knowledge chunks, and 60 upcoming calendar slots across the next 14 business days:
```powershell
python -m scripts.seed_data
```

#### Step 4: Run Health & Diagnostic Checks
Run the diagnostic CLI tool to verify DB connectivity, speech model paths, and API keys:
```powershell
python -m scripts.agent_doctor
```

#### Step 5: Start the ASGI API Server
Launch the server via Uvicorn:
```powershell
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```
*The startup lifespan will log model pre-warming for Kokoro-82M ONNX, Faster-Whisper, and BGE-small, followed by the background stale hold cleanup task.*

---

## 11. Testing and Validation Status

| Subsystem / Capability | Verification Status | Evidence / Validation Method |
| :--- | :--- | :--- |
| **Lead Quality Scoring** | **Verified Working** | Tested via `scripts/verify_leads.py` and `POST /api/leads/submit` logs. Correctly admits valid leads (ID 18, 19) and rejects placeholder strings. |
| **WebRTC VAD Framing** | **Verified Working** | Tested in live telephony sessions. Correctly chunks 20ms frames and detects turn boundaries (e.g. 117,120 accumulated bytes upon speech cessation). |
| **Faster-Whisper STT** | **Verified Working** | Live call transcript logs confirm accurate transcription of Indian English names (*"Pooja Banerjee"*) and real estate terms at 84–96% confidence in ~280ms. |
| **Groq Tool Calling** | **Verified Working** | Verified in session `phone_call_mw18u8pk` and `phone_call_mw3lqqtn`. Emits structured `search_properties` and `check_availability` arguments. |
| **RAG pgvector HNSW** | **Verified Working** | Retrieves 3 customer-shareable chunks (Green Valley Enclave, Canadian International School distance) in <40ms. |
| **Kokoro-82M ONNX TTS** | **Verified Working** | Pre-warms cleanly from `models/kokoro/kokoro-v0_19.onnx`. Generates 24kHz float32 audio, resampled to 16kHz PCM in ~135ms. |
| **Indian English Normalization**| **Verified Working** | Verified via unit assertions. Correctly verbalizes `"1,200 sqft"` $\rightarrow$ `"twelve hundred square feet"` and `"48 lakhs"`. |
| **Mid-Call Disconnect Recovery**| **Verified Working** | Verified during mid-call browser hangup. Triggered recovery pipeline and generated urgent `OperatorNotification #52` within 1.2s. |
| **Stale Calendar Hold Cleaner** | **Verified Working** | Background asyncio loop executes every 60s without errors, scanning for `held_until <= NOW()`. |
| **CalDAV Sync to Radicale** | **Partially Implemented** | Adapter code is complete (`src/db/caldav_sync.py`), but live synchronization requires an active Radicale calendar server instance at `CALDAV_URL`. |
| **F5-TTS Zero-Shot GPU Cloner**| **Configured / Untested** | Provider hook is wired in `src/audio/tts.py`. Awaiting deployment to an NVIDIA CUDA virtual GPU instance. |

---

## 12. Pending Work, Known Issues & Honest Technical Barriers

### 12.1 The TTS Architectural Journey: What Was Tried & What Is Active
During voice optimization, five speech synthesis options were evaluated:
1. **Piper ONNX**: Initial baseline. It executed quickly (~80ms) but sounded robotic, failing to convey the professional warmth expected of a real estate advisor.
2. **Kokoro-82M ONNX (Current Active Engine)**: Chosen as the active engine. Delivers high naturalness, runs locally on CPU in ~150ms, requires zero API keys, and has zero operating cost.
3. **OpenVoice v2 (MyShell AI)**: Evaluated for zero-shot voice cloning. Installation stalled due to legacy C-extension build dependencies (`librosa 0.9.1`, `numba`, `whisper-timestamped`, `dtw-python`) conflicting with Python 3.14 on Windows. Furthermore, OpenVoice only clones vocal timbre while leaving underlying accent and rhythm bound to the base American/British model.
4. **XTTS-v2 (Coqui)**: Evaluated. Requires Microsoft Visual C++ Build Tools (`cl.exe`) to compile its monotonic alignment kernel on Windows. In addition, its autoregressive decoder takes 3.5s–5.0s on CPU, exceeding acceptable phone call latency thresholds.
5. **F5-TTS / GPU Hook (Production Target)**: Implemented as a clean provider hook in `src/audio/tts.py`. Configured to run on an NVIDIA CUDA Linux instance in production (<200ms per clause) using the 55.84s Simi Indian English reference audio clip.

### 12.2 The LLM Architecture: Local Ollama vs. Cloud Groq API
* **Native Architecture**: The codebase is architected for self-hosted execution using **Local Ollama with Meta Llama 3 8B** (`http://localhost:11434`), ensuring complete data privacy and offline sovereignty.
* **Current Operational Routing**: In development on a host machine without a discrete GPU, running an 8B model locally alongside Faster-Whisper, PostgreSQL, and Kokoro introduces 3–6s latency spikes. For live telephony, calls route through **Groq's OpenAI-compatible API** (`llama-3.3-70b-versatile`) to maintain a ~250ms response time.

---

## 13. Deployment and Production Information

### 13.1 Production Docker Topology (`docker-compose.prod.yml`)
In production environments, the platform deploys via three isolated containers:
1. **`postgres`**: `pgvector/pgvector:pg16` with persistent storage volumes.
2. **`ollama`**: `ollama/ollama:latest` with NVIDIA GPU passthrough (`count: all, capabilities: [gpu]`) serving Llama 3 locally.
3. **`api`**: Production multi-stage Dockerfile running FastAPI, Faster-Whisper, and the zero-shot TTS engine on CUDA.

### 13.2 GPU Scaling Requirements
* **Minimum GPU Hardware**: NVIDIA T4 (16GB VRAM) or NVIDIA A10G (24GB VRAM).
* **Driver Requirements**: NVIDIA Container Toolkit with CUDA 12.x support.
* **Configuration Switch**:
  ```ini
  TTS_ENGINE=zero_shot_clone
  ZERO_SHOT_DEVICE=cuda
  LLM_BASE_URL=http://ollama:11434
  LLM_MODEL_NAME=llama3:latest
  ```

---

## 14. Important Notes and Future Scope

### 14.1 Technical Decisions of Significance
* **Removal of Hardcoded Prompt Appends**: Early iterations forcefully appended `"Verified documentation with RERA approved paperwork is available"` to the end of every turn. This was traced to line 1081 of `src/agent/dialogue_manager.py` and completely removed. The agent now introduces legal verifications naturally only when relevant to the caller's questions.
* **Elimination of the "Yes" Affirmative Loop**: Early testing revealed that when the agent asked *"Would you like to visit tomorrow?"* and the caller answered *"Yes, I would love to"*, the system treated *"Yes"* as an ungrounded property query. Adding affirmative detection resolved this issue, prompting the agent to immediately offer concrete morning/afternoon slots.

### 14.2 Recommended Next Steps for Developers
1. **CalDAV Remote Integration**: Connect `src/db/caldav_sync.py` to a production Radicale or Nextcloud server and verify end-to-end event synchronization on mobile devices.
2. **Automated Drop-In Ingestion**: Build a file watcher over `storage/documents/` that automatically chunks, embeds, and indexes newly added PDFs or Markdown files without requiring manual `seed_data` executions.
3. **Multi-Tenant Campaign IDs**: Add a `project_id` or `campaign_id` column to `knowledge_chunks` and `leads` to allow multiple real estate builders to run on the same platform instance.

---

## 15. Final Project Status Summary

| Area | Implementation State | Operational Verdict | Action Needed by Next Developer |
| :--- | :--- | :--- | :--- |
| **Voice Audio Pipeline** | Complete | **Working & Verified** | None for CPU. Install `f5-tts` when moving to a production CUDA GPU instance. |
| **ASR (STT)** | Complete | **Working & Verified** | None. Faster-Whisper `base.en` (CPU `int8`) executes reliably in ~280ms. |
| **TTS (Speech)** | Complete | **Working & Verified** | Kokoro-82M ONNX operates reliably in ~150ms on CPU with zero operating cost. |
| **Dialogue State Machine** | Complete | **Working & Verified** | Prompt brevity, tool calling, and affirmative scheduling loops are resolved. |
| **LLM Gateway** | Complete | **Working & Verified** | Groq handles live telephony on CPU. Run `ollama serve` if running 100% offline with a GPU. |
| **RAG & Vector Storage** | Complete | **Working & Verified** | PostgreSQL pgvector HNSW similarity queries execute in <40ms. |
| **Calendar Scheduling** | Complete | **Working & Verified** | 5-minute atomic holds and 60-second cleanup workers run reliably. |
| **CalDAV Sync** | Code Complete | **Requires Live Server** | Point `CALDAV_URL` in `.env` to an active Radicale or Nextcloud calendar instance. |
| **Testing Harnesses** | Complete | **Working & Verified** | Use `python scripts/interactive_lead_call.py` for full terminal-based conversation testing. |
