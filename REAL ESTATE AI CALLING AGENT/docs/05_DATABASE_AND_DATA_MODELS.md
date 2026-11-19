# 05. Database Architecture & Data Models

## 1. Relational Schema & Entity-Relationship Model

The RealEstateAI data persistence layer is built on **PostgreSQL 16** with the **pgvector** extension. All database entities are managed via **SQLAlchemy 2.0** models in `src/db/models/`.

```
┌─────────────────────────┐               ┌─────────────────────────┐
│          Lead           │               │     PropertyListing     │
├─────────────────────────┤               ├─────────────────────────┤
│ id (PK, UUID)           │               │ id (PK, UUID)           │
│ name (String)           │               │ title (String)          │
│ phone_number (Unique)   │               │ project_name (String)   │
│ niche (Enum)            │               │ niche (Enum)            │
│ budget_max (Float)      │               │ location (String)       │
│ preferred_location (Str)│               │ price (Numeric)         │
│ qualification_status    │               │ dimensions (String)     │
│ created_at (Timestamp)  │               │ legal_approval_status   │
└───────────┬─────────────┘               │ paperwork_checklistJSONB│
            │ 1                           └─────────────────────────┘
            │
            │ N
┌───────────▼─────────────┐               ┌─────────────────────────┐
│       Appointment       │       N     1 │      CalendarSlot       │
├─────────────────────────┤◄──────────────┤─────────────────────────┤
│ id (PK, UUID)           │               │ id (PK, UUID)           │
│ lead_id (FK -> Lead.id) │               │ start_time (Timestamp)  │
│ slot_id (FK -> Slot.id) │               │ end_time (Timestamp)    │
│ status (Enum)           │               │ status (Enum)           │
│ visitor_name (String)   │               │ held_until (Timestamp)  │
│ notes (Text)            │               │ held_by_phone (String)  │
└─────────────────────────┘               └─────────────────────────┘
            │
            │
┌───────────▼─────────────┐               ┌─────────────────────────┐
│    ConversationTurn     │               │     KnowledgeChunk      │
├─────────────────────────┤               ├─────────────────────────┤
│ id (PK, UUID)           │               │ id (PK, UUID)           │
│ session_id (String)     │               │ title (String)          │
│ turn_number (Integer)   │               │ content (Text)          │
│ speaker (Enum: USER/AI) │               │ niche (Enum)            │
│ text (Text)             │               │ access_label (Enum)     │
│ latency_ms (Integer)    │               │ embedding (vector(384)) │
│ created_at (Timestamp)  │               └─────────────────────────┘
└─────────────────────────┘
```

---

## 2. Core Entities & Table Specifications

### 1. `leads` Table (`src/db/models/lead.py`)
Tracks customer identity, qualification attributes, and pipeline progress:
* `id` (`UUID`, Primary Key, default `gen_random_uuid()`)
* `name` (`VARCHAR(255)`, Not Null): Caller full name.
* `phone_number` (`VARCHAR(32)`, Unique, Indexed): Normalized E.164 phone number (e.g. `+919820098765`).
* `niche` (`VARCHAR(64)`): Target real estate segment (`residential_plot`, `commercial_flat`, `luxury_villa`, `land_parcel`).
* `budget_max` (`DOUBLE PRECISION`): Ceiling budget in INR.
* `preferred_location` (`VARCHAR(255)`): Target geographic corridor.
* `inquiry_notes` (`TEXT`): Initial digital ad / web form inquiry context.
* `qualification_status` (`VARCHAR(32)`): `new`, `qualified`, `disqualified`, `converted`.
* `call_count` (`INTEGER`, Default `0`): Total phone conversations completed.

### 2. `property_listings` Table (`src/db/models/property.py`)
Official property inventory catalog:
* `id` (`UUID`, Primary Key)
* `title` (`VARCHAR(255)`): Public marketing title.
* `project_name` (`VARCHAR(255)`, Indexed): E.g. *"Green Valley Enclave"*.
* `niche` (`VARCHAR(64)`, Indexed): Property type.
* `location` (`VARCHAR(255)`): Geographic corridor and municipality.
* `price` (`NUMERIC(14, 2)`): Base listing price.
* `dimensions` (`VARCHAR(128)`): Measured boundary and square footage.
* `legal_approval_status` (`VARCHAR(255)`): E.g. *"RERA & BDA Approved"*.
* `paperwork_checklist` (`JSONB`): Array of verified legal documents available for inspection.

### 3. `calendar_slots` Table (`src/db/models/calendar.py`)
Manages site visit scheduling inventory:
* `id` (`UUID`, Primary Key)
* `start_time` (`TIMESTAMP WITH TIME ZONE`, Indexed): Slot commencement.
* `end_time` (`TIMESTAMP WITH TIME ZONE`, Indexed): Slot termination.
* `status` (`VARCHAR(32)`): Slot state (`available`, `held`, `booked`, `cancelled`).
* `held_until` (`TIMESTAMP WITH TIME ZONE`, Nullable): Expiry timestamp for temporary holds.
* `held_by_phone` (`VARCHAR(32)`, Nullable): Phone number of caller holding the slot.

### 4. `appointments` Table (`src/db/models/calendar.py`)
Confirmed on-site visitor bookings:
* `id` (`UUID`, Primary Key)
* `lead_id` (`UUID`, Foreign Key -> `leads.id`, Indexed)
* `slot_id` (`UUID`, Foreign Key -> `calendar_slots.id`, Unique Constraint)
* `visitor_name` (`VARCHAR(255)`)
* `status` (`VARCHAR(32)`): `confirmed`, `completed`, `cancelled`, `no_show`.
* `notes` (`TEXT`)

### 5. `knowledge_chunks` Table (`src/db/models/knowledge.py`)
Vector repository for RAG:
* `id` (`UUID`, Primary Key)
* `title` (`VARCHAR(255)`)
* `content` (`TEXT`, Not Null): Factual descriptive paragraph.
* `niche` (`VARCHAR(64)`, Indexed)
* `access_label` (`VARCHAR(64)`, Indexed): Enforces security (`customer_shareable` vs `internal_restricted`).
* `embedding` (`vector(384)`): Dense embedding indexed via HNSW.

### 6. `conversation_turns` Table (`src/db/models/conversation.py`)
Per-turn conversation persistence:
* `id` (`UUID`, Primary Key)
* `session_id` (`VARCHAR(64)`, Indexed): Telephony call identifier.
* `turn_number` (`INTEGER`, Not Null)
* `speaker` (`VARCHAR(16)`): `caller` or `agent`.
* `text` (`TEXT`, Not Null): Spoken transcript.
* `latency_ms` (`INTEGER`): Processing latency for audit.

### 7. `operator_notifications` Table (`src/db/models/notification.py`)
Dispatched alerts for human intervention:
* `id` (`INTEGER`, Primary Key, Autoincrement)
* `session_id` (`VARCHAR(64)`, Indexed)
* `phone_number` (`VARCHAR(32)`)
* `urgency` (`VARCHAR(32)`): `low`, `medium`, `urgent`.
* `reason` (`VARCHAR(255)`): E.g. *"abrupt_midcall_disconnect"*, *"human_escalation_requested"*.
* `status` (`VARCHAR(32)`): `pending`, `acknowledged`, `resolved`.

---

## 3. Database Migration History (Alembic)

The schema is versioned using Alembic (`alembic/versions/`):

1. **`c3a4f891b72e_add_hnsw_vector_index.py`**:
   - Enables `pgvector` extension (`CREATE EXTENSION IF NOT EXISTS vector;`).
   - Creates `idx_knowledge_chunks_embedding_hnsw` on `knowledge_chunks.embedding` using `vector_cosine_ops`.
2. **`init_schema.py`**:
   - Creates core relational tables (`leads`, `property_listings`, `calendar_slots`, `appointments`, `conversation_turns`, `operator_notifications`).
   - Establishes foreign keys, unique slot-locking constraints, and cascading rules.
