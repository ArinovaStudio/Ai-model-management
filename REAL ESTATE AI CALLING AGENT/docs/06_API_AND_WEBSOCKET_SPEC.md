# 06. API & WebSocket Interface Specifications

## 1. REST API Endpoints

The FastAPI HTTP gateway is exposed on port `8000`.

### 1. Ingest New Lead
* **Route**: `POST /api/leads/submit`
* **Description**: Ingests prospective buyer leads from landing pages, Facebook Lead Ads, or CRM webhooks.
* **Request Payload**:
  ```json
  {
    "name": "Pooja Banerjee",
    "phone_number": "+919820098765",
    "niche": "residential_plot",
    "budget_max": 9000000,
    "preferred_location": "North Hills",
    "inquiry_notes": "Looking for a villa plot in Green Valley North Hills close to Canadian International School under 90 Lakhs."
  }
  ```
* **Response (HTTP 200 OK)**:
  ```json
  {
    "status": "success",
    "lead_id": 18,
    "phone_number": "+919820098765",
    "quality": "valid"
  }
  ```

---

### 2. Initiate Inbound/Outbound Telephony Session
* **Route**: `POST /api/calls/start`
* **Description**: Initializes a telephony session, looks up the registered lead profile, and returns the unique session ID.
* **Request Payload**:
  ```json
  {
    "phone_number": "+919820098765",
    "direction": "inbound"
  }
  ```
* **Response (HTTP 200 OK)**:
  ```json
  {
    "session_id": "phone_call_mw18u8pk",
    "ws_audio_url": "/ws/audio/phone_call_mw18u8pk?phone=%2B919820098765",
    "caller_name": "Pooja Banerjee",
    "status": "initialized"
  }
  ```

---

### 3. List Open Calendar Slots
* **Route**: `GET /api/calendar/slots`
* **Query Parameters**:
  - `start_date` (optional): `YYYY-MM-DD`
  - `status` (optional): `available` (default)
* **Response (HTTP 200 OK)**:
  ```json
  [
    {
      "slot_id": "7f3b5a12-89cd-4e56-bc78-90abcdef1234",
      "start_time": "2026-11-17T10:00:00+05:30",
      "end_time": "2026-11-17T11:00:00+05:30",
      "status": "available"
    },
    {
      "slot_id": "8a4c6b23-90de-5f67-cd89-01bcdefa2345",
      "start_time": "2026-11-17T14:30:00+05:30",
      "end_time": "2026-11-17T15:30:00+05:30",
      "status": "available"
    }
  ]
  ```

---

### 4. Fetch Operator Notifications
* **Route**: `GET /api/notifications`
* **Query Parameters**:
  - `status` (optional): `pending`, `acknowledged`, `resolved`
* **Response (HTTP 200 OK)**:
  ```json
  [
    {
      "id": 52,
      "session_id": "phone_call_mw3lqqtn",
      "phone_number": "+919811223344",
      "urgency": "urgent",
      "reason": "abrupt_midcall_disconnect",
      "created_at": "2026-11-05T19:20:12+05:30"
    }
  ]
  ```

---

## 2. Telephony WebSocket Audio Protocol

* **WebSocket Route**: `ws://<host>:8000/ws/audio/{session_id}?phone={phone_number}`
* **Protocol Handler**: `src/telephony/event_router.py`

```
CALLER CLIENT                                                    FASTAPI WEBSOCKET
     │                                                                   │
     │ 1. Connects: /ws/audio/phone_call_abc123?phone=+9198200...        │
     ├──────────────────────────────────────────────────────────────────►│
     │ 2. Handshake Accepted                                             │
     │◄──────────────────────────────────────────────────────────────────┤
     │                                                                   │
     │ 3. Agent Initial Greeting (Binary 16kHz PCM audio stream)         │
     │◄──────────────────────────────────────────────────────────────────┤
     │                                                                   │
     │ 4. Caller Utterance (Continuous Binary 16kHz PCM chunks)          │
     │──────────────────────────────────────────────────────────────────►│
     │    [WebRTC VAD accumulates speech until 400ms silence detected]   │
     │    [Faster-Whisper transcribes -> Dialogue Manager -> LLM / RAG]  │
     │                                                                   │
     │ 5. Agent Response Audio (Streaming Binary 16kHz PCM chunks)       │
     │◄──────────────────────────────────────────────────────────────────┤
     │                                                                   │
     │ 6. Heartbeat / Keep-Alive (Text JSON: {"type": "ping"})           │
     │◄─────────────────────────────────────────────────────────────────►│
```

### Binary Frame Format:
- Every binary frame sent or received is raw **16,000 Hz, 16-bit Mono Little-Endian PCM**.
- No WAV header encapsulation is used inside the stream; audio is piped directly into audio renderers/buffers.

### Text Control Messages (JSON):
While audio is transmitted as binary frames, out-of-band call events can be transmitted as JSON text frames:
* **Caller Interruption / Barge-In**:
  ```json
  {"event": "clear_buffer"}
  ```
  Signals the server to halt immediate downstream TTS playback if the caller starts speaking over the agent.
* **Call Termination**:
  ```json
  {"event": "hangup", "reason": "call_completed"}
  ```
