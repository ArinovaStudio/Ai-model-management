# RealEstateAI Comprehensive Technical Documentation Suite

Welcome to the definitive, deep-dive technical documentation suite for **RealEstateAI**, an enterprise-grade, low-latency conversational AI voice agent platform engineered for real estate customer qualification, inventory discovery, and site visit scheduling.

---

## Complete Documentation Index

| Module / Document | Purpose & Depth |
| :--- | :--- |
| **[01. System Architecture & Topology](file:///c:/Users/aseem/RealEstateAI/docs/01_SYSTEM_ARCHITECTURE.md)** | Subsystem boundaries, end-to-end data flow diagrams (Mermaid), package directory map, and ~1.1s latency budget. |
| **[02. Voice & Audio Pipeline](file:///c:/Users/aseem/RealEstateAI/docs/02_VOICE_AUDIO_PIPELINE.md)** | WebRTC VAD, Faster-Whisper STT (int8), audio resampling (24k to 16k), phonetic verbalization, **and exhaustive comparison of all tested TTS engines (Piper, Kokoro ONNX, OpenVoice, XTTS-v2, F5-TTS)**. |
| **[03. Dialogue Management & LLM](file:///c:/Users/aseem/RealEstateAI/docs/03_DIALOGUE_STATE_AND_LLM.md)** | Conversational state machine, prompt engineering (<14 words), tool calling schemas, elimination of robotic fallback loops, **and local Ollama standard vs. cloud Groq API explanation**. |
| **[04. RAG & Vector Database](file:///c:/Users/aseem/RealEstateAI/docs/04_RAG_AND_KNOWLEDGE_BASE.md)** | FastEmbed BGE-small-en-v1.5 (384-dim), PostgreSQL pgvector HNSW index, access control (`customer_shareable` vs `internal_restricted`), and pre-seeded inventory. |
| **[05. Database & Data Models](file:///c:/Users/aseem/RealEstateAI/docs/05_DATABASE_AND_DATA_MODELS.md)** | SQLAlchemy 2.0 ORM schemas, Entity-Relationship diagrams, foreign keys, unique slot locks, and Alembic migration history. |
| **[06. API & WebSocket Specifications](file:///c:/Users/aseem/RealEstateAI/docs/06_API_AND_WEBSOCKET_SPEC.md)** | REST endpoints (`/api/leads/submit`, `/api/calls/start`), bidirectional WebSocket audio protocol, JSON control frames, and payload schemas. |
| **[07. Operations & Failure Recovery](file:///c:/Users/aseem/RealEstateAI/docs/07_OPERATIONS_AND_RECOVERY.md)** | Background worker architecture, 60s stale calendar hold cleanup, abrupt mid-call disconnect recovery, and operator escalation dispatch. |
| **[08. Deployment, GPU & Runbook](file:///c:/Users/aseem/RealEstateAI/docs/08_DEPLOYMENT_AND_RUNBOOK.md)** | Local development environment setup, production Docker Compose configuration, virtual GPU (CUDA) scaling, and diagnostic utilities (`agent_doctor.py`). |
| **[09. CalDAV Sync & Terminal Lead Testing](file:///c:/Users/aseem/RealEstateAI/docs/09_CALDAV_AND_LEAD_TESTING.md)** | **RFC 4791 CalDAV synchronization adapter (Radicale/Apple/Nextcloud)** and step-by-step developer guide for **testing lead-based voice calls directly in the terminal** (`interactive_lead_call.py`). |
| **[10. Deep-Dive Codebase Reference](file:///c:/Users/aseem/RealEstateAI/docs/10_DEEP_DIVE_CODEBASE_REFERENCE.md)** | **Source-level mechanics**: Lead quality triage algorithms, optimistic two-phase slot locking, step-by-step turn execution trace (VAD $\rightarrow$ STT $\rightarrow$ LLM $\rightarrow$ RAG $\rightarrow$ TTS), and abrupt disconnect diagnosis. |
| **[11. File-to-File Call Graph & Import Lineage](file:///c:/Users/aseem/RealEstateAI/docs/11_FILE_TO_FILE_CONNECTIONS.md)** | **Exhaustive file-to-file wiring matrix**: Details exact imports, call sites, data contracts, and cross-module payload transitions across all 15 source files in `src/`. |

---

## Architectural Summary

1. **Local Architecture Standard with Pragmatic Cloud Routing**:
   * The platform's standard implementation is **Meta Llama 3 8B running locally via Ollama** (`http://localhost:11434`) for full self-hosted operation.
   * On local CPU development environments where running an 8B LLM concurrently with STT, PostgreSQL, and TTS introduces conversational pauses, the system routes seamlessly to **Groq's OpenAI-compatible API** to maintain a ~250ms LLM response time during live calls.
2. **Current Active Speech Engine**:
   * Runs **Kokoro-82M ONNX** locally on CPU in **~150ms** with zero API keys and zero cost.
   * Evaluated **OpenVoice v2** and **XTTS-v2**; both encountered build-tool and latency barriers on Windows host CPU.
   * Ready for **F5-TTS / Zero-Shot Cloning** with the 55.84s Simi Indian English reference audio on a production virtual GPU (CUDA) instance.
3. **Enterprise Integration & Terminal Testing**:
   * Site visits sync seamlessly with external calendar apps via CalDAV `VEVENT` records.
   * Any engineer can run and test multi-turn lead conversations directly via `python scripts/interactive_lead_call.py` without needing a browser or active telephony gateway.
4. **Complete Codebase Traceability**:
   * Every file in `src/` is explicitly mapped with its callers, callees, and data contracts documented in `docs/11_FILE_TO_FILE_CONNECTIONS.md`.
