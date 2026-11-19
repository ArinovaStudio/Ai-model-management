# 03. Dialogue Management & LLM Reasoning Engine

## 1. Dialogue Architecture Overview

The intelligence and conversational control of RealEstateAI reside in `src/agent/dialogue_manager.py` and `src/agent/llm_client.py`. Rather than relying on rigid, pre-scripted decision trees or unconstrained chatbot prompts, the system employs an **augmented state machine** that couples:

1. **Lead State & Turn Memory**: Tracks caller intent, lead budget, niche, and conversational history across the lifecycle of the call.
2. **Dynamic Tool Calling**: Uses function calling to retrieve real estate data and calendar availability directly from PostgreSQL.
3. **Conversational Prompt Constraints**: Enforces spoken-language rules (short sentences, natural conversational openers, conversational recovery) designed specifically for phone calls.

---

## 2. LLM Engine: Local Ollama Standard vs. Cloud Groq API

The codebase features a unified LLM client interface (`src/agent/llm_client.py`) that abstracts model interactions:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        LLM CLIENT ABSTRACTION                          │
│                       (src/agent/llm_client.py)                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
│       ARCHITECTURAL STANDARD:       │   │        ACTIVE RUNTIME ENGINE:       │
│           LOCAL OLLAMA              │   │              CLOUD GROQ             │
├─────────────────────────────────────┤   ├─────────────────────────────────────┤
│ Model: Meta Llama 3 8B              │   │ Model: llama-3.3-70b-versatile /    │
│ Endpoint: http://localhost:11434    │   │        llama3-8b-8192               │
│ Runtime: 100% Local, Offline        │   │ Endpoint: api.groq.com/openai/v1    │
│ Function Calling: Native tools API  │   │ Latency: ~250ms – 350ms             │
│ Target Host: Machine with 16GB+ RAM │   │ Purpose: Live Zero-Latency Telephony│
│              or Dedicated GPU       │   │          on CPU Development Machine │
└─────────────────────────────────────┘   └─────────────────────────────────────┘
```

### Why Both Exist & How They Are Used:

1. **The Native Architecture is Local Ollama (`Llama 3 8B`)**:
   - The platform is designed to be fully self-hosted, sovereign, and offline.
   - Configured via `LLM_BASE_URL="http://localhost:11434"` and `LLM_MODEL_NAME="llama3:latest"`.
   - In production environments equipped with a GPU or high-throughput multi-core RAM, Ollama serves Llama 3 locally with zero external API dependencies.

2. **Active Telephony Execution with Groq API**:
   - In development and testing on the local machine (running on host CPU without a discrete GPU), running a local 8B LLM alongside Faster-Whisper, PostgreSQL, and Kokoro TTS introduces inference latency spikes (3–6 seconds per LLM response).
   - In live voice calls, a 4-second pause breaks the conversational flow.
   - To solve this constraint without altering system architecture, the client routes requests through **Groq's OpenAI-compatible LPUs**, achieving sub-second token generation (**~250ms per turn**) while maintaining identical function calling schemas and system prompts.

---

## 3. Function & Tool Calling Specifications

The LLM is provided with 4 structured tool definitions (`src/agent/tools.py`) during every turn:

### 1. `search_properties`
* **Purpose**: Searches property listings and verified legal documentation in the RAG knowledge base.
* **Parameters**:
  - `query` (str): Search keywords (e.g. *"Green Valley 1200 sqft pricing"*).
  - `niche` (str): `residential_plot`, `commercial_flat`, `luxury_villa`, `land_parcel`.
  - `location` (str): Desired geography (e.g. *"North Hills"*, *"Whitefield"*).
  - `budget_max` (float): Maximum ceiling price in INR.

### 2. `check_availability`
* **Purpose**: Queries real-time open calendar slots for site visits across upcoming business days.
* **Parameters**:
  - `start_date` (str, optional): ISO start date.
  - `end_date` (str, optional): ISO end date.

### 3. `book_site_visit`
* **Purpose**: Reserves an open slot and creates an official appointment linked to the caller's lead profile.
* **Parameters**:
  - `slot_id` (str): UUID of the selected `CalendarSlot`.
  - `lead_phone` (str): Caller's phone number.
  - `visitor_name` (str): Confirmed visitor name.
  - `notes` (str): Special instructions or companion details.

### 4. `escalate_to_human`
* **Purpose**: Immediately flags the call for a human sales director or legal specialist when complex legal exceptions or distressed financial negotiations occur.
* **Parameters**:
  - `reason` (str): Detailed reason for transfer.
  - `urgency` (str): `low`, `medium`, `urgent`.

---

## 4. Prompt Engineering & Spoken Voice Persona

Telephone dialogue requires fundamentally different prompt engineering than text-based chat. Reading long paragraphs of text over a phone connection causes caller fatigue.

### Core Persona Rules in `src/agent/prompts.py`:
1. **Clause Brevity Constraint**:
   Spoken sentences must be concise—strictly **under 14 words per clause**.
2. **Conversational Rhythm**:
   The agent must use natural conversational openers:
   - *"Got it!"*, *"Sure thing."*, *"I completely understand."*, *"That makes sense."*
3. **Single Question Rule**:
   The agent must never ask two questions in the same turn. Ask one clear question at the end to pass conversational initiative back to the caller.
4. **Number Formatting**:
   Never spell out raw numbers like `"4800000"` or `"1.35 Cr"` directly; use clean conversational phrases like *"48 lakhs"* or *"one point three five crore"*.

---

## 5. Elimination of Fallback Loops & Hallucinations

During real-world telephony stress testing, two critical bugs were diagnosed and eliminated in `src/agent/dialogue_manager.py`:

### Issue A: The Robotic Legal Append Bug
* **Symptom**: Earlier versions forcefully appended `"Verified documentation with RERA approved paperwork is available"` to the end of turns.
* **Root Cause**: An outdated heuristic in `dialogue_manager.py` (line 1081) checked if the word `"paperwork"` was present in the LLM response, and if not, concatenated the static string.
* **Resolution**: Completely eliminated the forced concatenation block. RERA and legal verifications are now naturally integrated by the LLM only when relevant to the caller's inquiry.

### Issue B: The Affirmative Scheduling Loop
* **Symptom**: When the agent asked *"Would you like to visit tomorrow?"* and the caller responded *"Yes, I would love to"*, the system previously failed to recognize the affirmation, falling back to re-reading the property overview.
* **Root Cause**: The slot selection handler required an explicit date string (e.g. `"November 6th"`) and treated pure affirmative words (`"Yes"`, `"Sure"`, `"Definitely"`) as ungrounded property queries.
* **Resolution**: Added `is_affirmative_to_visit` detection logic. When the caller agrees to an offer, the system immediately presents the next 2 concrete morning/afternoon slots from the database.
