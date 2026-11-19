# 11. Complete File-to-File Call Graph & Import Lineage Reference

## 1. Architectural Import Lineage Map

This document establishes the exact **file-to-file connection matrix** across the entire `src/` codebase. It details:
- **Who imports whom**: Exact imports and function-call sites.
- **Data contracts**: The exact objects, datatypes, and payloads crossing file boundaries.
- **Runtime execution sequence**: Which module initializes first, which instances are singletons, and how events propagate from the socket boundary down to the database and back.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                           ENTRYPOINT                                                             │
│                                                          src/main.py                                                             │
└────────────────────────────────┬────────────────────────────────────────────────────────────────┬────────────────────────────────┘
                                 │ imports app                                                    │ imports pre-warmers & workers
                                 ▼                                                                ▼
                 ┌───────────────────────────────┐                                ┌───────────────────────────────┐
                 │       src/api/app.py          │                                │ src/workers/                  │
                 │   (FastAPI App Factory)       │                                │   stale_hold_cleanup.py       │
                 └───────────────┬───────────────┘                                └───────────────────────────────┘
                                 │ mounts router
                                 ▼
                 ┌───────────────────────────────┐
                 │      src/api/routes.py        │
                 └───────┬───────────────┬───────┘
                         │               │
       HTTP REST Calls   │               │ WebSocket Upgrades (/ws/audio/{session_id})
                         ▼               ▼
┌───────────────────────────────┐ ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│   src/agent/lead_service.py   │ │                                src/telephony/event_router.py                                   │
└───────────────┬───────────────┘ └───────┬───────────────────────────────┬───────────────────────────────┬────────────────────────┘
                │                         │                               │                               │
                │ LeadBrief               │ Inbound PCM Bytes             │ Disconnect Event              │ Text to Synthesize
                ▼                         ▼                               ▼                               ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────────┐
│ src/agent/dialogue_manager.py │ │       src/audio/vad.py        │ │ src/telephony/            │ │       src/audio/tts.py        │
└───────┬───────────────┬───────┘ │  (WebRTC VAD Frame Slicer)   │ │   session_manager.py      │ │  (Kokoro ONNX / Resampler)    │
        │               │         └───────────────┬───────────────┘ └─────────────┬─────────────┘ └───────────────────────────────┘
        │               │                         │ Speech Frames                 │ Disconnect Handler
        │               │                         ▼                               ▼
        │               │         ┌───────────────────────────────┐ ┌───────────────────────────┐
        │               │         │       src/audio/stt.py        │ │ src/agent/llm_client.py   │
        │               │         │   (Faster-Whisper int8)       │ │   (Post-Call Recovery)    │
        │               │         └───────────────┬───────────────┘ └─────────────┬─────────────┘
        │               │                         │ Transcription Text            │
        │               │                         └──────────────────────────────►│
        │ LLM Calls     │ Tool Dispatches
        ▼               ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐ ┌───────────────────────────┐
│   src/agent/llm_client.py     │ │ src/db/scheduling_manager.py  │ │   src/rag/retriever.py      │
│  (Groq Cloud / Ollama Local)  │ └───────────────┬───────────────┘ └─────────────┬─────────────┘
└───────────────────────────────┘                 │                               │
                                                  │ CalDAV VEVENT Sync            │ FastEmbed Dense Vector
                                                  ▼                               ▼
                                  ┌───────────────────────────────┐ ┌───────────────────────────┐
                                  │    src/db/caldav_sync.py      │ │   src/rag/embeddings.py   │
                                  └───────────────┬───────────────┘ └─────────────┬─────────────┘
                                                  │                               │
                                                  ▼                               ▼
                                  ┌─────────────────────────────────────────────────────────────┐
                                  │            src/db/session.py & src/db/models/               │
                                  │           (PostgreSQL 16 + pgvector HNSW DB)                │
                                  └─────────────────────────────────────────────────────────────┘
```

---

## 2. File-by-File Cross-Reference & Call Matrix

### 1. `src/main.py`
The top-level ASGI server entrypoint executed by Uvicorn.
* **Imports From**:
  - `src.api.app` $\rightarrow$ imports `app` (FastAPI instance).
  - `src.core.config` $\rightarrow$ imports `settings` (validates runtime environment).
  - `src.core.logging` $\rightarrow$ imports `logger`.
  - `src.audio.tts` $\rightarrow$ imports `tts_engine` (triggers TTS pre-warming).
  - `src.audio.stt` $\rightarrow$ imports `stt_engine` (triggers Faster-Whisper pre-warming).
  - `src.rag.embeddings` $\rightarrow$ imports `embedding_engine` (pre-warms FastEmbed BGE-small).
  - `src.workers.stale_hold_cleanup` $\rightarrow$ imports `start_cleanup_worker()` (launches background asyncio loop).
* **Exported To**:
  - `uvicorn src.main:app` (ASGI application server).

---

### 2. `src/api/app.py`
The application factory configuring middleware, CORS, and routing.
* **Imports From**:
  - `src.api.routes` $\rightarrow$ imports `api_router` (mounts all `/api/*` and `/ws/*` routes).
  - `src.core.config` $\rightarrow$ imports `settings`.
  - `src.core.logging` $\rightarrow$ imports `logger`.
* **Exported To**:
  - `src/main.py` (instantiates the lifespan context manager).

---

### 3. `src/api/routes.py`
The HTTP and WebSocket routing gateway.
* **Imports From**:
  - `src.agent.lead_service` $\rightarrow$ imports `lead_service` (calls `lead_service.evaluate_lead_quality()`, `lead_service.ingest_lead()`).
  - `src.telephony.session_manager` $\rightarrow$ imports `session_manager` (calls `session_manager.create_session()`).
  - `src.telephony.event_router` $\rightarrow$ imports `event_router` (routes live WebSocket stream `handle_audio_websocket()`).
  - `src.db.scheduling_manager` $\rightarrow$ imports `scheduling_manager` (serves `GET /api/calendar/slots`).
  - `src.db.session` $\rightarrow$ imports `get_async_db` (FastAPI DB session dependency).
* **Exported To**:
  - `src/api/app.py` (included via `app.include_router(api_router)`).

---

### 4. `src/telephony/event_router.py`
The full-duplex WebSocket binary PCM audio router.
* **Imports From**:
  - `src.audio.vad` $\rightarrow$ imports `VADProcessor` (slices 20ms frames and computes turn boundaries).
  - `src.audio.stt` $\rightarrow$ imports `stt_engine` (transcribes caller speech byte buffers).
  - `src.agent.dialogue_manager` $\rightarrow$ imports `dialogue_manager` (dispatches transcribed text: `dialogue_manager.process_turn()`).
  - `src.audio.tts` $\rightarrow$ imports `tts_engine` (streams synthesized speech chunks: `tts_engine.synthesize_stream()`).
  - `src.telephony.session_manager` $\rightarrow$ imports `session_manager` (notifies mid-call hangups: `session_manager.handle_abrupt_disconnect()`).
  - `src.core.logging` $\rightarrow$ imports `logger`.
* **Exported To**:
  - `src/api/routes.py` (handles incoming `/ws/audio/{session_id}`).

---

### 5. `src/audio/vad.py`
Voice Activity Detection and frame segmentation.
* **Imports From**:
  - `src.core.config` $\rightarrow$ imports `settings` (reads `AUDIO_SAMPLE_RATE=16000`, `JITTER_BUFFER_MS=20`, `MAX_SILENCE_DURATION_MS=400`).
  - External: `webrtcvad` (`webrtcvad-wheels`).
* **Exported To**:
  - `src/telephony/event_router.py` (instantiated per active WebSocket session).

---

### 6. `src/audio/stt.py`
Speech-to-Text inference wrapper.
* **Imports From**:
  - `src.core.config` $\rightarrow$ imports `settings` (`WHISPER_MODEL_SIZE`, `WHISPER_DOWNLOAD_ROOT`, `WHISPER_COMPUTE_TYPE`).
  - `src.core.logging` $\rightarrow$ imports `logger`.
  - External: `faster_whisper` (`WhisperModel`).
* **Exported To**:
  - `src/telephony/event_router.py` (invoked via `stt_engine.transcribe(audio_bytes)`).
  - `src/main.py` (pre-warmed during server boot).

---

### 7. `src/audio/tts.py`
Multi-engine Text-to-Speech synthesis and linear audio resampling.
* **Imports From**:
  - `src.core.config` $\rightarrow$ imports `settings` (`TTS_ENGINE`, `KOKORO_MODEL_PATH`, `KOKORO_VOICES_PATH`, `ZERO_SHOT_REF_AUDIO_PATH`).
  - `src.core.logging` $\rightarrow$ imports `logger`.
  - External: `kokoro_onnx`, `piper`, `torch`, `f5_tts`.
* **Exported To**:
  - `src/telephony/event_router.py` (calls `tts_engine.synthesize_stream(text)`).
  - `src/main.py` (pre-warmed during server boot).
  - `scripts/interactive_lead_call.py` (used for terminal CLI testing).

---

### 8. `src/agent/dialogue_manager.py`
The stateful conversation orchestrator, prompt guard, and tool dispatcher.
* **Imports From**:
  - `src.agent.llm_client` $\rightarrow$ imports `llm_client` (calls `llm_client.chat_completion()`).
  - `src.agent.prompts` $\rightarrow$ imports `AGENT_SYSTEM_PROMPT`.
  - `src.agent.tools` $\rightarrow$ imports `TOOL_DEFINITIONS` (schemas for LLM function calling).
  - `src.agent.lead_service` $\rightarrow$ imports `lead_service` (fetches lead context brief).
  - `src.rag.retriever` $\rightarrow$ imports `knowledge_retriever` (dispatches `search_properties`).
  - `src.db.scheduling_manager` $\rightarrow$ imports `scheduling_manager` (dispatches `check_availability` and `book_site_visit`).
  - `src.db.session` $\rightarrow$ imports `AsyncSessionLocal`.
  - `src.db.models` $\rightarrow$ imports `ConversationTurn`, `Lead`, `Appointment`.
* **Exported To**:
  - `src/telephony/event_router.py` (executes conversational turns).
  - `src/telephony/session_manager.py` (loads turn history during disconnect recovery).

---

### 9. `src/agent/llm_client.py`
Unified LLM client communicating with Local Ollama or Cloud Groq API.
* **Imports From**:
  - `src.core.config` $\rightarrow$ imports `settings` (`LLM_BASE_URL`, `LLM_MODEL_NAME`, `LLM_API_KEY`, `LLM_TIMEOUT_SECONDS`).
  - `src.core.logging` $\rightarrow$ imports `logger`.
  - External: `httpx` (async HTTP connection pooling).
* **Exported To**:
  - `src/agent/dialogue_manager.py` (primary conversational turns and tool calls).
  - `src/telephony/session_manager.py` (mid-call disconnect summarization).
  - `src/agent/lead_service.py` (post-call executive report generator).

---

### 10. `src/agent/lead_service.py`
Lead qualification triage, brief creation, and post-call analytics.
* **Imports From**:
  - `src.db.session` $\rightarrow$ imports `AsyncSessionLocal`.
  - `src.db.models.lead` $\rightarrow$ imports `Lead`.
  - `src.core.logging` $\rightarrow$ imports `logger`.
* **Exported To**:
  - `src/api/routes.py` (`POST /api/leads/submit`).
  - `src/agent/dialogue_manager.py` (injects lead context into Aria's system prompt).
  - `scripts/interactive_lead_call.py` (terminal lead ingestion).

---

### 11. `src/rag/retriever.py` & `src/rag/embeddings.py`
RAG dense retrieval and vector similarity matching.
* **Imports From**:
  - `src.rag.embeddings` $\rightarrow$ imports `embed_text` (FastEmbed `BAAI/bge-small-en-v1.5`).
  - `src.db.session` $\rightarrow$ imports `AsyncSessionLocal`.
  - `src.db.models.knowledge` $\rightarrow$ imports `KnowledgeChunk`.
  - `src.core.config` $\rightarrow$ imports `settings`.
* **Exported To**:
  - `src/agent/dialogue_manager.py` (invoked when LLM executes `search_properties`).
  - `scripts/ingest_documents.py` (embeds raw markdown into pgvector).

---

### 12. `src/db/scheduling_manager.py`
Concurrency-safe calendar booking and slot locking.
* **Imports From**:
  - `src.db.session` $\rightarrow$ imports `AsyncSessionLocal`.
  - `src.db.models.calendar` $\rightarrow$ imports `CalendarSlot`, `Appointment`.
  - `src.db.models.enums` $\rightarrow$ imports `SlotStatus`.
  - `src.db.caldav_sync` $\rightarrow$ imports `caldav_sync` (triggers background CalDAV `VEVENT` write).
  - `src.core.logging` $\rightarrow$ imports `logger`.
* **Exported To**:
  - `src/agent/dialogue_manager.py` (invoked when LLM executes `check_availability` or `book_site_visit`).
  - `src/api/routes.py` (serves `GET /api/calendar/slots`).
  - `src/workers/stale_hold_cleanup.py` (scans and resets expired holds).

---

### 13. `src/db/caldav_sync.py`
RFC 4791 CalDAV external calendar synchronization adapter.
* **Imports From**:
  - `src.core.config` $\rightarrow$ imports `settings` (`CALDAV_URL`, `CALDAV_USERNAME`, `CALDAV_PASSWORD`, `CALDAV_CALENDAR_NAME`).
  - `src.core.logging` $\rightarrow$ imports `logger`.
  - External: `caldav` (`DAVClient`), `icalendar` (`Calendar`, `Event`).
* **Exported To**:
  - `src/db/scheduling_manager.py` (called via `caldav_sync.sync_appointment_event(appointment)`).

---

### 14. `src/telephony/session_manager.py`
VoIP session tracking, lifecycle audits, and abrupt disconnect recovery.
* **Imports From**:
  - `src.db.session` $\rightarrow$ imports `AsyncSessionLocal`.
  - `src.db.models` $\rightarrow$ imports `ConversationTurn`, `OperatorNotification`, `Lead`.
  - `src.agent.llm_client` $\rightarrow$ imports `llm_client`.
  - `src.agent.dialogue_manager` $\rightarrow$ imports `dialogue_manager`.
  - `src.core.logging` $\rightarrow$ imports `logger`.
* **Exported To**:
  - `src/api/routes.py` (`POST /api/calls/start`).
  - `src/telephony/event_router.py` (detects abrupt WebSocket drops and initiates recovery).

---

### 15. `src/workers/stale_hold_cleanup.py`
Autonomous 60-second background garbage collector for abandoned calendar holds.
* **Imports From**:
  - `src.db.session` $\rightarrow$ imports `AsyncSessionLocal`.
  - `src.db.models.calendar` $\rightarrow$ imports `CalendarSlot`.
  - `src.db.models.enums` $\rightarrow$ imports `SlotStatus`.
  - `src.core.logging` $\rightarrow$ imports `logger`.
* **Exported To**:
  - `src/main.py` (spawned as an `asyncio.create_task` during application lifespan startup).

---

## 3. Data Contract & Object Flow Table

| Source File | Destination File | Calling Method / Function | Payload / Object Passed |
| :--- | :--- | :--- | :--- |
| `event_router.py` | `vad.py` | `vad.process_frame(chunk)` | 640-byte raw linear PCM slice |
| `event_router.py` | `stt.py` | `stt.transcribe(audio_bytes)` | Complete accumulated speech byte buffer |
| `event_router.py` | `dialogue_manager.py` | `dialogue.process_turn(session_id, text)` | Transcribed caller utterance string |
| `dialogue_manager.py` | `llm_client.py` | `llm.chat_completion(messages, tools)` | OpenAI-compatible message list + tool JSON schemas |
| `dialogue_manager.py` | `retriever.py` | `retriever.search_properties(query, niche)` | Semantic search parameters dict |
| `retriever.py` | `embeddings.py` | `embeddings.embed_text(query)` | Query string $\rightarrow$ Returns 384-dim float list |
| `dialogue_manager.py` | `scheduling_manager.py` | `scheduler.book_slot(slot_id, lead_phone)` | UUID slot string + normalized caller phone |
| `scheduling_manager.py` | `caldav_sync.py` | `caldav.sync_appointment_event(appt)` | Confirmed SQLAlchemy `Appointment` ORM instance |
| `dialogue_manager.py` | `tts.py` | `tts.synthesize_stream(text)` | Conversational sentence clause string |
| `tts.py` | `event_router.py` | `yield pcm_16k_bytes` | Downsampled 16kHz 16-bit PCM binary chunk |
| `event_router.py` | `session_manager.py` | `session.handle_abrupt_disconnect(id)` | Dropped session identifier string |
