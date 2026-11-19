# 04. Retrieval-Augmented Generation (RAG) & Vector Database

## 1. Overview & Information Grounding

In real estate voice transactions, hallucinated information (e.g. inventing false square-foot prices, promising unverified BDA approvals, or quoting inaccurate dimensions) carries direct legal and financial liability.

The **RealEstateAI RAG Subsystem** (`src/rag/`) enforces strict factual grounding. Every property specification, price range, documentation requirement, and connectivity distance quoted by the agent is retrieved in real-time from a vector-indexed knowledge repository backed by **PostgreSQL** and **pgvector**.

---

## 2. Dense Vector Embeddings (FastEmbed / BAAI BGE-Small)

Text vectorization is handled by `src/rag/embedder.py` using **FastEmbed**:

* **Model**: `BAAI/bge-small-en-v1.5`
* **Dimension**: **384 dimensions** (dense floating-point vector).
* **Execution**: Runs locally on CPU via ONNX Runtime without external network calls.
* **Normalization**: Vectors are normalized to unit length ($\|v\|_2 = 1.0$), allowing cosine similarity to be computed via dot products.
* **Pre-warming**: The embedder is instantiated during application startup, ensuring that RAG search queries execute in **< 30 ms**.

---

## 3. PostgreSQL & pgvector HNSW Indexing

Vectors are stored directly in PostgreSQL within the `knowledge_chunks` table using the `vector(384)` data type provided by the `pgvector` extension.

### Database Index Schema (Alembic Migration `c3a4f891b72e`):
To support sub-millisecond retrieval across tens of thousands of document chunks, an **Hierarchical Navigable Small World (HNSW)** index is maintained:

```sql
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding_hnsw
ON knowledge_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

* `m = 16`: Number of bidirectional links per vector node in the graph.
* `ef_construction = 64`: Size of the dynamic candidate list during index construction.
* `vector_cosine_ops`: Optimized for cosine distance calculations.

### Retrieval Query Anatomy (`src/rag/engine.py`):
```sql
SELECT id, title, content, niche, legal_status, access_label,
       1 - (embedding <=> :query_vector) AS similarity_score
FROM knowledge_chunks
WHERE access_label = 'customer_shareable'
  AND (niche = :niche OR :niche IS NULL)
ORDER BY embedding <=> :query_vector ASC
LIMIT 3;
```

---

## 4. Multi-Tenant Access Control & Privacy Isolation

Real estate developers maintain sensitive internal documents (e.g. minimum acceptable distressed sale prices, land acquisition costs, sales agent commission structures, and legal dispute risk logs) that must never be revealed to a prospective customer over the phone.

The system enforces a **strict physical access boundary**:

```
                               ┌────────────────────────────────────────────────────────┐
                               │             REAL ESTATE KNOWLEDGE DOCUMENTS            │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                                      ┌──────────────────────────────────────────┐
                                      │        DOCUMENT CHUNKER & TAGGER         │
                                      │         (scripts/seed_data.py)           │
                                      └────────────┬─────────────────────────────┘
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
┌─────────────────────────────────────────────────┐ ┌─────────────────────────────────────────────────┐
│     ACCESS LABEL: 'customer_shareable'          │ │     ACCESS LABEL: 'internal_restricted'         │
├─────────────────────────────────────────────────┤ ├─────────────────────────────────────────────────┤
│ - Approved Pricing & Payment Milestones         │ │ - Minimum Internal Bottom-Line Distress Pricing │
│ - Verified Plot Dimensions (30x40, 40x60)       │ │ - Land Acquisition Margins & Developer Net      │
│ - RERA Certificates & A-Katha Documentation     │ │ - Sales Agent Commission Splits & Incentive Tiers│
│ - Canadian International School Proximity (3 km)│ │ - Internal Legal Risk Summaries & Caveats       │
└────────────────────────┬────────────────────────┘ └────────────────────────┬────────────────────────┘
                         │                                                   │
                         ▼                                                   ▼
┌─────────────────────────────────────────────────┐ ┌─────────────────────────────────────────────────┐
│ ACCESSIBLE TO CALLER-FACING VOICE AGENT         │ │ STRICTLY ISOLATED FROM CALLER-FACING AGENT      │
│ RAG Engine strictly filters:                    │ │ Filter:                                         │
│   WHERE access_label = 'customer_shareable'     │ │   access_label != 'customer_shareable'          │
│ Injected into conversational LLM context.       │ │ NEVER returned to the dialogue manager.         │
└─────────────────────────────────────────────────┘ └─────────────────────────────────────────────────┘
```

---

## 5. Seed Knowledge Base Inventory

The system includes pre-seeded property listings and knowledge chunks (`scripts/seed_data.py`):

1. **Green Valley Enclave (North Hills)**:
   - **Niche**: Residential Plots.
   - **Units**: Standard Plots (1,200 sqft / 30x40 ft at **48 Lakhs**); Corner Plots (2,400 sqft / 40x60 ft at **85 Lakhs**).
   - **Approvals**: BDA & RERA Approved (`PRM/KA/RERA/1251/309/PR/190525`), Clear A-Katha title deeds ready for immediate registration.
   - **Landmarks**: 3 km from Canadian International School, 20 minutes from Bangalore International Airport.
2. **Skyline Heights (Whitefield Tech Corridor)**:
   - **Niche**: Commercial / Luxury Flats.
   - **Units**: 3BHK Luxury Flat (1,650 sqft Super Built-up at **1.35 Crore**).
   - **Approvals**: RERA Approved, Occupancy Certificate (OC) and Commencement Certificate (CC) issued.
3. **Palm Meadows Estates (Devanahalli Airport Road)**:
   - **Niche**: Luxury Villas.
   - **Units**: Signature 4BHK Villa (3,800 sqft on 4,000 sqft plot at **3.20 Crore**).
   - **Approvals**: BIAAPA Approved with Airport Authority NOC.
4. **Logistics Corridor Phase 1 (Dabaspete Highway)**:
   - **Niche**: Industrial Land Parcels.
   - **Units**: 2.5 Acres Commercial Zone at **4.50 Crore**.
   - **Approvals**: KIADB Industrial Zoned.


Right now, the system operates on one central knowledge base:

Stored in the PostgreSQL knowledge_chunks table (indexed with pgvector HNSW).
Seeded from the real estate dataset we created (covering Green Valley Enclave plots, Skyline Heights flats, Palm Meadows villas, and Logistics Corridor parcels, along with their verified RERA numbers, pricing, and Canadian International School distances).
How Flexible Is It:

While it is currently populated with this single knowledge base:

The RAG engine itself is fully generic.
If you ever drop in a new project's documents or run scripts/ingest_documents.py, it automatically chunks, embeds, and indexes them into that same table without needing code changes.



How to Make THIS Completely Plug-and-Play:
1. A Drop-In "Documents Inbox" Folder (data/inbox/)

Instead of hardcoding seed data or writing database scripts:

Any developer, real estate firm, or agency simply drops their property brochures, floor plans, legal FAQs, or pricing sheets (.pdf, .docx, .txt, .md) into a data/inbox/ folder.
An automatic watcher / ingestion command:
Detects new files.
Parses and chunks the text.
Tags metadata (project_name, niche, access_label).
Automatically computes embeddings with the local FastEmbed model and inserts them directly into PostgreSQL.
2. REST API for Dynamic Document Ingestion (POST /api/knowledge/upload)

Add a simple endpoint where users or CRM systems can upload documents dynamically:

Send a PDF or Markdown file via an API call.
The system indexes it in seconds.
Aria immediately gains knowledge of the new project on the very next phone call, with zero restarts and zero server downtime.
3. Multi-Tenant Project / Campaign Separation

Right now, the database queries by niche and access_label.

By adding a single metadata tag like project_id or campaign_id:
You can run 10 different real estate companies or builders on the exact same voice platform.
When an incoming call hits a specific phone number or lead campaign, Aria automatically binds to only that company’s specific properties, brochures, and calendar slots.
4. Automated Persona & Company Configuration

Right now, Aria introduces herself as "Aria from Prime Realty".

By pulling the company name, agent persona, and greeting from the lead’s campaign record:
Caller A hears: "Hello Pooja! This is Aria from Prime Realty..."
Caller B hears: "Hello Vikram! This is Maya from Prestige Builders..."
All with zero code changes—purely driven by the ingested project profile.
Summary

The hardest, most complex parts are already built and working:

The sub-second telephony streaming audio engine.
WebRTC VAD and Faster-Whisper STT.
Kokoro-82M ONNX local synthesis.
The pgvector HNSW vector similarity search.
The structured appointment locking and CalDAV calendar sync.

Because those foundations are solid, turning this into a pure plug-and-play, multi-project SaaS platform where anyone can drop in their files and start taking calls is completely achievable and straightforward.