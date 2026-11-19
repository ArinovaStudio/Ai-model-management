# 07. Operations, Background Workers & Failure Recovery

## 1. Resilience Philosophy

In live telephony, networks drop, cellular connections cut out abruptly, and API services can encounter intermittent rate limits. A production voice agent must be fault-tolerant:
1. **Never drop lead context**: An interrupted call must be diagnosed instantly and surfaced to human staff.
2. **Never leave inventory in an invalid state**: Abandoned appointment holds must expire automatically.
3. **Never expose internal stack traces over voice**: Graceful conversational fallbacks must handle system errors naturally.

---

## 2. Background Workers Architecture

The FastAPI lifespan manager (`src/main.py`) launches concurrent asynchronous background workers during startup:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI LIFESPAN MANAGER                        │
│                             (src/main.py)                              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
│    STALE HOLD CLEANUP WORKER        │   │     DISCONNECT RECOVERY PIPELINE    │
│  (src/workers/cleanup_stale_holds)  │   │  (src/workers/disconnect_recovery)  │
├─────────────────────────────────────┤   ├─────────────────────────────────────┤
│ - Frequency: Every 60 seconds       │   │ - Trigger: Abrupt WebSocket Close   │
│ - Target: CalendarSlots with        │   │ - Analyzes: Transcript up to drop   │
│   status = 'held'                   │   │ - Determines: Lead intent & urgency │
│ - Condition: held_until < NOW()     │   │ - Creates: OperatorNotification     │
│ - Action: Reverts slot to           │   │ - Status: Surfaced on CRM dashboard │
│   status = 'available'              │   │   for immediate human callback      │
└─────────────────────────────────────┘   └─────────────────────────────────────┘
```

---

## 3. Stale Hold Cleanup Engine (`src/workers/cleanup_stale_holds.py`)

When a caller inquires about booking a site visit, the agent places an ephemeral hold on the requested slot to prevent other callers from booking it concurrently.

### The Problem:
If the caller's phone battery dies, their signal drops, or they decide not to proceed, the slot would remain permanently blocked if not released.

### The Implementation:
* Runs continuously as a background `asyncio` task with a 60-second sleep interval.
* Queries:
  ```sql
  UPDATE calendar_slots
  SET status = 'available',
      held_until = NULL,
      held_by_phone = NULL
  WHERE status = 'held'
    AND held_until <= NOW();
  ```
* Ensures high calendar availability without manual database maintenance.

---

## 4. Abrupt Mid-Call Disconnect Recovery Pipeline

In actual operations, callers frequently disconnect unexpectedly (e.g. driving through a tunnel, bad reception, accidental hangup).

When a WebSocket connection drops before an orderly conversation close:
1. **Detection** (`src/telephony/event_router.py`):
   ```python
   except WebSocketDisconnect:
       logger.warning(f"Active session {session_id} closed abruptly! Triggering recovery handler...")
   ```
2. **Conversation Analysis** (`src/workers/disconnect_recovery.py`):
   - Loads all `ConversationTurn` records up to the exact point of disconnection.
   - Evaluates whether the lead was in the middle of scheduling or had unresolved inquiries.
3. **Dispatching `OperatorNotification`**:
   - Automatically writes a record to `operator_notifications` with `urgency = 'urgent'`.
   - Records the caller's phone number, name, last topic discussed, and the recommended follow-up action for human sales reps.
4. **Result in Logs**:
   ```text
   WARNING  | Mid-call disconnect detected for session phone_call_mw3lqqtn!
   INFO     | Created urgent OperatorNotification #52 for session phone_call_mw3lqqtn
              (Status: pending, Urgency: urgent, Phone: +919811223344)
   ```

---

## 5. Rate Limit & Network Circuit Breakers

When external cloud endpoints (such as Groq or external speech services) hit rolling per-minute token or request ceilings (HTTP 429):

1. **Automatic Retry with Exponential Backoff**:
   `src/agent/llm_client.py` catches `httpx.HTTPStatusError` and checks `Retry-After` headers.
2. **Graceful Voice Recovery**:
   If an LLM call fails or times out, the agent does not go silent or read raw JSON. It speaks a brief conversational bridging phrase:
   > *"I'm pulling up those exact documentation details right now. Just a quick moment."*
3. **Fallback to Kokoro ONNX**:
   If any zero-shot voice cloning engine fails, the system falls back to the local Kokoro ONNX engine without interrupting the phone call.
