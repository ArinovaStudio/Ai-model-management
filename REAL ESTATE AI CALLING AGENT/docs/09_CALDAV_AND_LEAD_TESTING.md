# 09. Advanced Features: CalDAV Calendar Sync & Lead-Based Interactive Testing

## 1. Overview

This document details two core enterprise features of the **RealEstateAI** platform:
1. **CalDAV Calendar Integration**: Automatic, RFC 4791-compliant calendar synchronization for site visit appointments with self-hosted (Radicale) or enterprise calendar servers (Apple Calendar, Nextcloud, Google Calendar via CalDAV).
2. **Terminal-Based Interactive Lead Calling**: The developer workflow to test end-to-end voice conversations, RAG accuracy, and appointment bookings from PowerShell or bash without needing a web browser or real Twilio telephony provider.

---

## 2. CalDAV Calendar Synchronization (`src/db/caldav_sync.py`)

When Aria confirms a site visit appointment with a caller, the booking must be persisted in two locations:
1. **Internal PostgreSQL Database**: The `appointments` and `calendar_slots` tables (enforcing ACID transactions, foreign keys, and unique slot-locking).
2. **External CalDAV Server (Radicale / Apple Calendar / Nextcloud)**: Propagated as a standard iCalendar `VEVENT` object so sales executives and property managers see appointments on their mobile phones and laptops in real time.

```
┌─────────────────────────────────┐
│     Caller Confirms Visit       │
│  "Yes, tomorrow at 10 AM works" │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Agent Calls Function:       │
│       book_site_visit()         │
└────────────────┬────────────────┘
                 │
                 ├───────────────────────────────────────────────┐
                 │ 1. Transactional DB Write                     │ 2. Background CalDAV Sync
                 ▼                                               ▼
┌─────────────────────────────────┐             ┌─────────────────────────────────┐
│    PostgreSQL Database          │             │  CalDAVSyncAdapter              │
│    - Status -> 'booked'         │             │  (src/db/caldav_sync.py)        │
│    - Foreign Key -> Lead.id     │             └────────────────┬────────────────┘
└─────────────────────────────────┘                              │ RFC 4791 VEVENT
                                                                 ▼
                                                ┌─────────────────────────────────┐
                                                │   Radicale / Nextcloud / CalDAV │
                                                │   Sales Team Mobile Calendars   │
                                                └─────────────────────────────────┘
```

### CalDAV Architecture & Configuration:
* **Protocol**: RFC 4791 / RFC 5545 (iCalendar `VEVENT`).
* **Client Library**: `caldav` (Python) with `icalendar` event serialization.
* **Configuration Parameters (`src/core/config.py`)**:
  ```ini
  CALDAV_URL=http://localhost:5232/
  CALDAV_USERNAME=realestate_admin
  CALDAV_PASSWORD=realestate_secure_pass
  CALDAV_CALENDAR_NAME=SiteVisits
  ```

### Synchronization Method: `sync_appointment_event()`
When invoked, the adapter:
1. Connects to the CalDAV principal URL and retrieves (or auto-creates) the target `SiteVisits` calendar.
2. Constructs a standard RFC 5545 `Event` component:
   - **`SUMMARY`**: E.g., `Site Visit: Pooja Banerjee - Green Valley Enclave Plot 42`
   - **`DTSTART` / `DTEND`**: Slot timestamps localized to `Asia/Kolkata` (UTC+5:30).
   - **`DESCRIPTION`**: Caller phone number, stated budget, verified legal checklist, and special visitor notes.
   - **`LOCATION`**: Exact geo-coordinates or project address (e.g., `North Hills, Bengaluru`).
3. Saves the event with a unique `UID` (`f"site-visit-{appointment_id}"`), enabling idempotent updates and cancellations.

---

## 3. Lead-Based Interactive Testing from the Terminal

Developers can test full telephony calls without opening a browser or placing a real phone call. The repository includes two dedicated CLI testing utilities:

1. **`scripts/interactive_lead_call.py`**: Interactive CLI testing tool for end-to-end lead qualification, live voice generation, and appointment booking.
2. **`scripts/test_call_stream.py`**: Audio socket simulation tool testing the full WebSocket binary PCM pipeline.

---

## 4. Step-by-Step Developer Workflow for Terminal Testing

Follow this guide to simulate a lead and conduct a full multi-turn conversational call directly in your terminal:

### Step 1: Ensure Prerequisites Are Running
Ensure PostgreSQL is active and your environment is ready:
```powershell
# Ensure you are in the project root
cd C:\Users\aseem\RealEstateAI

# Make sure any broken virtualenv is deactivated
deactivate
```

### Step 2: Seed Properties & Upcoming Calendar Slots
If testing on a fresh database, populate the inventory and 60 upcoming calendar slots:
```powershell
python -m scripts.seed_data
```

### Step 3: Run the Interactive Lead Call Utility
Launch the interactive terminal testing tool:
```powershell
python scripts/interactive_lead_call.py
```

### Step 4: Follow the Terminal Prompts
The utility guides you through the complete lead lifecycle:

```text
======================================================================
  REALESTATE.AI - INTERACTIVE LEAD INGESTION & VOICE CALL SIMULATOR
======================================================================
1. INGEST NEW LEAD:
   Enter Caller Name       : Pooja Banerjee
   Enter Phone (+E.164)    : +919820098765
   Stated Budget (INR)     : 9000000
   Target Niche            : residential_plot
   Preferred Location      : North Hills
   Inquiry Notes           : Interested in a 1200 sqft plot near Canadian International School.

[LEAD INGESTED] Lead ID: 18 | Quality: VALID | Status: NEW
----------------------------------------------------------------------
2. STARTING CONVERSATIONAL CALL SESSION:
   Telephony Session ID: phone_call_sim_01
   Pre-warming Kokoro-82M ONNX Voice Engine... Ready!

Aria (Agent): "Hello Pooja Banerjee! This is Aria from Prime Realty.
              I see you inquired about a residential plot in North Hills.
              How can I assist you today?"
[TTS Synthesizing -> storage/audio_cache/turn_0_agent.wav]

Caller: Yes, tell me what options you have under 90 lakhs.

Aria (Agent): "Sure thing! Green Valley Plots has 1200 square feet plots
              at 48 lakhs and larger corner plots at 85 lakhs.
              Both have clear BDA and RERA approvals.
              Which of those sizes sounds best to you?"
[TTS Synthesizing -> storage/audio_cache/turn_1_agent.wav]

Caller: The 1200 sqft plot sounds good. Can we see it tomorrow?

Aria (Agent): "Tomorrow sounds great. We have an open slot at 10 AM
              and another at 2:30 PM. Which works better for you?"
```

---

## 5. Simulating via REST & Browser WebSocket (Manual Webflow)

To test the actual HTTP API and browser WebSocket exactly like a production Twilio/WebRTC client:

### 1. Ingest a Lead via PowerShell:
```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/leads/submit" -ContentType "application/json" -Body (@{
    name = "Vikramaditya Rao"
    phone_number = "+919988776655"
    niche = "luxury_villa"
    budget_max = 35000000
    preferred_location = "Airport Road"
    inquiry_notes = "Interested in a 4BHK luxury gated community villa near Devanahalli Airport Road within 3.5 Crore."
} | ConvertTo-Json)
```

### 2. Initiate Call Session:
```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/calls/start" -ContentType "application/json" -Body (@{
    phone_number = "+919988776655"
    direction = "inbound"
} | ConvertTo-Json)
```
Output returns:
```json
{
  "session_id": "phone_call_mw98x1",
  "ws_audio_url": "/ws/audio/phone_call_mw98x1?phone=%2B919988776655"
}
```

### 3. Stream Audio via Terminal Socket Simulation:
Run the stream test script to feed raw PCM audio into the WebSocket:
```powershell
python scripts/test_call_stream.py --session phone_call_mw98x1 --phone "+919988776655"
```
The script streams 20ms frames, exercises WebRTC VAD, receives Kokoro ONNX speech chunks, and verifies RAG retrieval in real-time.
