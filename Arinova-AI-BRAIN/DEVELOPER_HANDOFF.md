# Arinova AI Brain Microservice & Autonomous Supervisor
> **Developer Handoff, Architecture & Deployment Guide**  
> *Author / Architectural Reference: AseemPrasad*  
> *Last Updated: November 2026*

---

## 1. Executive Summary & Architecture Overview

`Arinova-AI-BRAIN` is a dedicated, distributed microservice and autonomous supervisor daemon extracted from the original `managment-system` modular monolith. 

### Why This Extraction Was Done:
In the monolithic setup, heavy AI tasks (SDLC Phase-Gate reasoning, topological candidate scoring, dynamic workload density calculations, and LLM text generation) were running in-process inside Next.js API routes. Under concurrent team usage, this overloaded the Node.js event loop, causing HTTP timeouts, high memory pressure, and sluggish response times.

### Architectural Decoupling:
```
                      +------------------------------------------+
                      |         Management System (Next.js)      |
                      |        Branch: feat/microservice-ai-brain|
                      +--------------------+---------------------+
                                           |
                    REST Requests          |          Real-time WebSockets
                  (via ai-brain-client)    |       (via socket-server:3001)
                                           v
+---------------------------------------------------------------------------------------+
|  Arinova-AI-BRAIN Service (Docker / Standalone Node.js on port 5005)                  |
|                                                                                       |
|  [brain-api]      Express REST API (Stealth Policy, Decisions, Summaries, Dataset)     |
|  [brain-daemon]   Autonomous AI Supervisor (Postgres LISTEN / Event Observer)         |
+-------------------+-----------------------------------------------+-------------------+
                    |                                               |
                    | (http://ollama:11434)                         | (Shared PG LISTEN/NOTIFY)
                    v                                               v
+------------------------------------+             +------------------------------------+
|  Unified Ollama LLM Engine         |             |  Shared Neon PostgreSQL DB         |
|  Container: arinova_unified_ollama |             |  (ep-lingering-glade-a1srlv0e)     |
|  Models: llama3:8b, nomic-embed    |<------------|  Tasks, Projects, AgentMemories    |
+------------------------------------+             +-----------------+------------------+
                    ^                                                ^
                    | (http://arinova_unified_ollama:11434)          |
+-------------------+------------------------------------------------+------------------+
|  Headless AI Planning Agent (Python FastAPI compute worker on port 8000)              |
|  Crunches 8-stage WBS breakdown, estimations, and dependency math on demand           |
+---------------------------------------------------------------------------------------+
```

---

## 2. Directory Structure & Key Modules

```
Arinova-AI-BRAIN/
├── Dockerfile                   # Production container definition (multi-stage build)
├── docker-compose.yml           # Unified orchestration (Ollama + Sidecar + API + Daemon)
├── package.json                 # Node.js dependencies (Express, Prisma, Winston, Bcryptjs)
├── tsconfig.json                # TypeScript compiler config
├── prisma/
│   └── schema.prisma            # Mirror schema for shared Neon PostgreSQL database
├── src/
│   ├── server.ts                # Main Express REST Microservice (Port 5005)
│   ├── lib/
│   │   ├── client.ts            # Neon PostgreSQL Prisma Client
│   │   ├── socket.ts            # Outbound WebSocket event dispatcher to socket server
│   │   ├── projectGroups.ts     # Topological SDLC phase slotting math
│   │   ├── jwt.ts               # Token verification and signing helpers
│   │   └── agent-api.ts         # Policy error classes & authentication guards
│   └── ai/
│       ├── server.ts            # Autonomous background supervisor daemon entrypoint
│       ├── events.ts            # PostgreSQL LISTEN/NOTIFY event observer & dispatcher
│       ├── audit.ts             # Tamper-evident AgentActionLog & AgentMemory recorder
│       ├── circuit-breaker.ts   # Error-rate & latency circuit breaker
│       ├── dataset-formatter.ts # SFT fine-tuning dataset builder (JSONL exporter)
│       ├── scheduler.ts         # Distributed workflow locks and cron routines
│       ├── logger.ts            # Winston structured logging
│       ├── context/             # Dynamic Workload Density & team capacity calculations
│       ├── executor/            # Safe task mutations, rollbacks, and verifications
│       ├── ingestion/           # Deep project cloning & topological dependency remapping
│       ├── nlu/                 # Natural language intent recognition
│       ├── policy/              # SDLC Phase-Gate rules, Universal Stealth, and confidence bounds
│       └── reasoner/            # Decision scoring matrix, candidate ranking, Ollama inference
```

---

## 3. The Unified Open Ollama Setup

Previously, both systems had isolated or fragmented Ollama setups. In `Arinova-AI-BRAIN`, there is **one single, unified Ollama setup** shared across all services:

* **Container Name**: `arinova_unified_ollama`
* **Docker Network**: `arinova_shared_net` (external: true)
* **Aliases**: `arinova_unified_ollama`, `ollama`
* **Sidecar Loader (`ollama-init`)**: Automatically pulls `llama3:8b` and `nomic-embed-text` into the shared volume `ollama_models`.
* **Access Points**:
  * Internal (Docker bridge for Brain): `http://ollama:11434`
  * Internal (Docker bridge for Planning Agent): `http://arinova_unified_ollama:11434`
  * External (Host machine / dev): `http://localhost:11434`

---

## 4. Endpoints & REST Contract Reference

The microservice runs on port `5005` (configured via `PORT` or `BRAIN_PORT`).

| Method | Endpoint | Description | Request Payload / Params |
|---|---|---|---|
| `GET` | `/health` | Healthcheck & uptime | None |
| `GET` | `/api/agent/stealth-policy` | Fetch project stealth policy | `?projectId=<id>` |
| `POST` | `/api/agent/stealth-policy` | Update stealth policy & kick off allocations | `{ projectId, isActive, minConfidenceScore, modeType, selectedTaskIds }` |
| `GET` | `/api/agent/stealth-select-queue` | Get candidate tasks with AI match scores | `?projectId=<id>` |
| `GET` | `/api/agent/universal-stealth` | Get global admin stealth config | None |
| `POST` | `/api/agent/universal-stealth` | Update global stealth config | `{ isUniversalActive, globalMinConfidence, exceptionProjectIds }` |
| `GET` | `/api/decisions` | Fetch pending decisions (with hydrated task/project) | `?projectId=<optional_id>` |
| `POST` | `/api/decisions/approve` | Approve and execute decision in database | `{ id \| decisionId, overrideUserId?, approvedBy? }` |
| `POST` | `/api/decisions/reject` | Dismiss decision | `{ id \| decisionId, rejectedBy?, reason? }` |
| `POST` | `/api/project/ingest` | Deep-clone project template & remap graph | `{ source_project_id, new_name, members? }` |
| `GET` | `/api/agent/summary` | Generate project health narrative & risk analysis | `?projectId=<id>&includeRisks=true` |
| `POST` | `/api/agent/blocker` | Record risk blockage via AI executor | `{ projectId, taskId?, blockerTitle, blockerDescription?, ... }` |
| `POST` | `/api/agent/request-update` | Queue status update requests to assignees | `{ projectId, userIds: string[], agentId? }` |
| `GET` | `/api/agent/export-dataset` | Stream fine-tuning dataset as `.jsonl` | `?projectId=<optional_id>` |

---

## 5. Environment Variables (`.env`)

Create a `.env` file in `Arinova-AI-BRAIN/`:

```env
NODE_ENV=production
BRAIN_PORT=5005
PORT=5005

# Shared Neon PostgreSQL Database URL
DATABASE_URL=postgresql://neondb_owner:npg_NtvVI2fjC6XH@ep-lingering-glade-a1srlv0e.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&connection_limit=25&pool_timeout=30
DB_URI=postgresql://neondb_owner:npg_NtvVI2fjC6XH@ep-lingering-glade-a1srlv0e.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&connection_limit=25&pool_timeout=30

# Unified Ollama Host (Use http://ollama:11434 in Docker, http://localhost:11434 for local dev)
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3:8b

# Management System Socket Server (For Outbound Realtime Events)
SOCKET_URL=http://arinova_socket_server:3001
SOCKET_WEBHOOK_URL=http://arinova_socket_server:3001/api/webhook/ai-event

# Agent Identity
AGENT_ID=agent-service-001
AGENT_NAME=Arinova Autonomous Supervisor
JWT_SECRET=supersecretjwtkey_arinova_ai
```

---

## 6. How to Set Up & Run

### Step 1: Network Bootstrapping
Because Docker Compose uses an external shared network across repositories, create the network once:
```bash
docker network create arinova_shared_net
```

### Step 2: Running via Docker (Recommended for Production)
```bash
cd Arinova-AI-BRAIN
docker-compose up -d --build
```
This will launch:
1. `arinova_unified_ollama`: Unified Ollama model server on `11434`.
2. `arinova_ollama_loader`: One-shot sidecar pulling `llama3:8b` and `nomic-embed-text`.
3. `arinova_ai_brain_api`: Express REST API on port `5005`.
4. `arinova_ai_brain_daemon`: Background supervisor daemon running `LISTEN agent_task_completed`.

### Step 3: Running for Local Development (Without Docker)
Make sure you have a local Ollama running on `localhost:11434` with `llama3:8b`.
```bash
cd Arinova-AI-BRAIN
npm install
npm run generate
npm run dev      # Starts Express REST API on port 5005
npm run daemon   # Starts the background supervisor in a second terminal
```

---

## 7. Troubleshooting & FAQs

* **Q: Why does the daemon not pick up task completions?**
  * Check that `DB_URI` points to the exact same Neon database as the Management System. PostgreSQL `NOTIFY` is database-instance specific.
* **Q: How does Stealth Mode pre-approval work across processes?**
  * Stealth policies are written to the database `AgentMemory` table with type `stealth_policy`. The daemon reads directly using `getStealthPolicyAsync(projectId)` to bypass local in-memory cache desynchronization.
* **Q: Why are decisions showing up in `/decisions`?**
  * When actions have a confidence score below the threshold (default: $0.75$) or require human confirmation, they are placed in `AgentDecision`. Approving them executes the real DB mutation and emits PostgreSQL `NOTIFY` to trigger the next dependent tasks.
