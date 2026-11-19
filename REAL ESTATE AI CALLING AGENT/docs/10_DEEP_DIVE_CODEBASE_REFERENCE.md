# 10. Deep-Dive Codebase Reference: Subsystems, Mechanics & Lifecycle Traces

## 1. Complete Package & Subsystem Dependency Graph

This document provides an exhaustive, code-level technical breakdown of how all modules in `src/` instantiate, interact, pass data, and recover from failures.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             FASTAPI APP & ROUTER                                       │
│                                            (src/api/routes.py)                                         │
└──────────────┬──────────────────────────────────────────┬──────────────────────────────────────────────┘
               │ HTTP REST Requests                       │ WebSocket Upgrades (/ws/audio/{session_id})
               ▼                                          ▼
┌───────────────────────────────┐        ┌───────────────────────────────────────────────────────────────┐
│         LeadService           │        │                         event_router                          │
│  (src/agent/lead_service.py)  │        │                 (src/telephony/event_router.py)               │
├───────────────────────────────┤        ├───────────────────────────────┬───────────────────────────────┤
│ - evaluate_lead_quality()     │        │ Inbound Binary PCM (16kHz)    │ Outbound Binary PCM (16kHz)   │
│ - generate_lead_brief()       │        │   │                           │   ▲                           │
│ - generate_call_summary()     │        │   ▼                           │   │                           │
└──────────────┬────────────────┘        │ WebRTC VAD (src/audio/vad.py) │ TextToSpeechEngine (tts.py)   │
               │                         │   │                           │   ▲                           │
               │                         │   ▼                           │   │ Downsampled 16kHz PCM     │
               │                         │ Faster-Whisper (audio/stt.py) │ Sentence Chunking Buffer      │
               │                         └───────────────┬───────────────┴───┼───────────────────────────┘
               │                                         │ Spoken Text       │ Spoken Sentence Clauses
               │                                         ▼                   │ (<14 words, natural verbal)
               │                         ┌───────────────────────────────────┴───────────────────────────┐
               │                         │                       DialogueManager                         │
               │                         │              (src/agent/dialogue_manager.py)                  │
               │                         ├───────────────────────────────────────────────────────────────┤
               │ Context Brief           │ - Ephemeral Session State (turn history, slot holds)          │
               └────────────────────────►│ - Function/Tool Dispatcher                                    │
                                         │ - Fallback & Affirmative Decision Logic                       │
                                         └───────────────┬───────────────────────────────────▲───────────┘
                                                         │ LLM Tool Call                     │ Tool Return Data
                                                         ▼                                   │
                                         ┌───────────────────────────────────────────────────┴───────────┐
                                         │                      DISPATCHED SERVICES                      │
                                         ├─────────────────────────────────┬─────────────────────────────┤
                                         │ SchedulingManager               │ Hybrid RAG Engine           │
                                         │ (src/db/scheduling_manager.py)  │ (src/rag/retriever.py)      │
                                         ├─────────────────────────────────┼─────────────────────────────┤
                                         │ - check_availability()          │ - Dense Embedder (FastEmbed)│
                                         │ - acquire_5min_hold()           │ - PostgreSQL pgvector       │
                                         │ - confirm_booking()             │   (HNSW Cosine Search)      │
                                         │ - CalDAVSyncAdapter (syncs RFC) │ - access_label Filter       │
                                         └────────────────┬────────────────┴──────────────┬──────────────┘
                                                          │                               │
                                                          ▼                               ▼
                                         ┌───────────────────────────────────────────────────────────────┐
                                         │                       POSTGRESQL 16                           │
                                         │   (leads, calendar_slots, appointments, knowledge_chunks)     │
                                         └───────────────────────────────────────────────────────────────┘
```

---

## 2. Inbound Lead Validation Mechanics (`src/agent/lead_service.py`)

When a lead enters via `POST /api/leads/submit`, `LeadService.evaluate_lead_quality()` executes strict triage rules before the voice agent ever places or accepts a call:

### 1. E.164 Phone Sanitization & Anti-Dummy Heuristic:
```python
# Strip all non-digit characters
clean_phone = re.sub(r"[^\d+]", "", phone_number or "")
digits_only = re.sub(r"\D", "", clean_phone)

# Reject if under 10 digits
if len(digits_only) < 10:
    return False, "Invalid phone number: Must contain at least 10 digits."

# Strip Indian country code prefix 91 if present
clean_digits = re.sub(r"^91", "", digits_only) if len(digits_only) > 10 and digits_only.startswith("91") else digits_only

# Reject repetitive patterns (e.g., 9999999999, 0000000000) or obvious sequential dummy numbers
if len(set(clean_digits)) <= 2 or clean_digits in ["1234567890", "0123456789", "9876543210"]:
    return False, "Invalid phone number: Identified as placeholder or dummy digits."
```

### 2. Name & Spam Filtering:
Rejects test inputs matching:
```python
re.match(r"^(test|test\s+user|asdf|qwerty|n/a|na|none|unknown|foo|bar)$", clean_name, re.IGNORECASE)
```

### 3. Substantive Discovery Signals:
A lead is rejected as junk unless it contains at least **one verifiable buying signal**:
- Target property niche (`residential_plot`, `commercial_flat`, etc.)
- Stated budget > 0 INR
- Preferred geographic corridor (e.g. `"North Hills"`, `"Whitefield"`)
- Descriptive inquiry text with at least 5 characters

---

## 3. Concurrency & Optimistic Hold Mechanics (`src/db/scheduling_manager.py`)

To prevent two concurrent callers from booking the same site visit slot, `SchedulingManager` implements a **two-phase reservation protocol**:

```
Caller A (Phone Session 1)                                       Caller B (Phone Session 2)
           │                                                                 │
           │ 1. Requests Friday 10 AM                                        │
           ├────────────────────────────┐                                    │
           │                            ▼                                    │
           │               Acquire 5-Minute Hold                             │
           │               - UPDATE calendar_slots                           │
           │                 SET status = 'held',                            │
           │                     held_by_phone = '+9198200...',              │
           │                     hold_expires_at = NOW() + 5 min             │
           │                 WHERE id = :slot_id                             │
           │                   AND (status = 'available' OR expired)         │
           │                            │                                    │
           │◄───────────────────────────┤                                    │
           │ 2. Slot held for Caller A  │                                    │
           │                            │  3. Inquires for Friday 10 AM      │
           │                            │◄───────────────────────────────────┤
           │                            │                                    │
           │                            │  4. System rejects slot for B:     │
           │                            │     "Friday 10 AM is currently     │
           │                            │      held. We have 11:30 AM."      │
           │                            ├───────────────────────────────────►│
           │ 5. Confirms booking        │
           ├───────────────────────────►│
           │  Commit Appointment:       │
           │  - status -> 'booked'      │
           │  - CalDAV Event Created    │
           │◄───────────────────────────┤
```

### Automatic Hold Expiration:
If Caller A disconnects without confirming, `src/workers/stale_hold_cleanup.py` runs every 60 seconds:
```sql
UPDATE calendar_slots
SET status = 'available',
    held_by_phone = NULL,
    hold_expires_at = NULL
WHERE status = 'held'
  AND hold_expires_at <= NOW();
```
The slot instantly becomes available to subsequent callers without manual operator intervention.

---

## 4. End-to-End Turn Execution Trace

Here is the exact step-by-step trace of what happens under the hood during a single conversational turn:

### Step 1: WebSocket Packet Ingestion & WebRTC VAD
1. The caller speaks: *"Yes, I inquired about that plot, so tell me about it."*
2. The client transmits binary frames over `/ws/audio/{session_id}`.
3. `src/telephony/event_router.py` receives 640-byte chunks (20ms at 16kHz PCM).
4. `src/audio/vad.py` evaluates speech energy. When continuous silence exceeds 400ms (`MAX_SILENCE_DURATION_MS`), VAD emits a turn boundary event with **117,120 accumulated bytes**.

### Step 2: Speech-to-Text (Faster-Whisper)
1. `src/audio/stt.py` receives the raw audio buffer.
2. Faster-Whisper `base.en` (CPU `int8`) processes the waveform in **280ms**.
3. Emits transcription:
   ```json
   {
     "text": "Yes, I inquired about that plot, so tell me about it.",
     "confidence": 0.841,
     "requires_confirm": false
   }
   ```

### Step 3: Dialogue Context Assembly & LLM Tool Dispatch
1. `src/agent/dialogue_manager.py` fetches the session state:
   - Caller Phone: `+919820098765`
   - Injected Lead Brief: Name = Pooja Banerjee, Stated Budget = 90 Lakhs, Niche = residential_plot, Location = North Hills.
2. Injects system prompt (`AGENT_SYSTEM_PROMPT`) and tool schemas (`tools.py`).
3. Sends payload to **Groq Llama-3 API** (`src/agent/llm_client.py`).
4. **Groq emits tool call in 280ms**:
   ```json
   {
     "name": "search_properties",
     "arguments": {
       "niche": "residential_plot",
       "budget_max": 9000000.0,
       "location": "North Hills",
       "query": "North Hills residential plot plot sizes inventory pricing specifications"
     }
   }
   ```

### Step 4: Vector Similarity Search (FastEmbed + pgvector)
1. `src/rag/retriever.py` passes the query string to FastEmbed `BAAI/bge-small-en-v1.5`.
2. Generates 384-dimensional dense vector in **25ms**.
3. Executes HNSW cosine search on PostgreSQL:
   ```sql
   SELECT content, legal_status FROM knowledge_chunks
   WHERE access_label = 'customer_shareable'
     AND niche = 'residential_plot'
   ORDER BY embedding <=> :vec ASC
   LIMIT 3;
   ```
4. Returns 3 verified chunks:
   - Green Valley Enclave: Standard plots 1,200 sqft at 48 Lakhs; Corner plots 2,400 sqft at 85 Lakhs.
   - Clear BDA & RERA approvals, immediate registration.

### Step 5: Second LLM Pass (Voice Articulation)
1. Tool return data is appended to messages list.
2. LLM produces natural voice response in **260ms**:
   > *"Sure thing. So, Green Valley Plots has a few options. Standard plots are 1,200 sqft at 48 lakhs. We also have larger corner plots at 85 lakhs. All clear BDA and RERA approvals. Does that size work for you?"*

### Step 6: Text Verbalization & Resampling
1. Text is normalized in `src/audio/tts.py`:
   - `"1,200 sqft"` $\rightarrow$ `"twelve hundred square feet"`
   - `"48 lakhs"` $\rightarrow$ `"48 lakhs"`
2. Sentence chunking splits at punctuation boundaries.

### Step 7: Kokoro-82M ONNX Synthesis & Audio Streaming
1. Kokoro synthesizes the first clause (*"Sure thing."*) in **135ms** at 24,000 Hz.
2. `resample_24k_to_16k()` downsamples to 16,000 Hz using linear interpolation.
3. `float32_to_int16_bytes()` converts floats to signed 16-bit integer PCM.
4. Binary audio bytes stream over the WebSocket directly to the caller's audio worklet.
5. **Total elapsed latency from caller silence to first spoken word: ~1.18 seconds.**

---

## 5. Mid-Call Disconnect Diagnostics (`src/telephony/session_manager.py`)

If a caller drops abruptly mid-conversation:

```python
# src/telephony/event_router.py
except WebSocketDisconnect:
    logger.warning(f"Active session {session_id} closed abruptly! Triggering recovery handler...")
    await session_manager.handle_abrupt_disconnect(session_id)
```

1. `session_manager.py` queries `conversation_turns` for all exchanges up to that second.
2. Sends the transcript to LLM with the prompt:
   *"Analyze this interrupted call. Identify if the lead showed positive intent, what property they wanted, and write a 2-sentence urgent action note for the human sales team."*
3. Generates an `OperatorNotification` record:
   - `phone_number`: `+919811223344`
   - `urgency`: `urgent`
   - `reason`: `abrupt_midcall_disconnect`
   - `recommended_action`: `Caller was reviewing 3BHK flats in Whitefield before signal drop. Follow up immediately via WhatsApp with floor plan PDF.`
4. The CRM dashboard surfaces the alert within **1.2 seconds** of the call dropping.
