# Production Deployment Guide: Sovereign RealEstateAI Voice Agent

This manual provides complete, production-grade instructions for deploying the **RealEstateAI Autonomous Voice Agent** on a dedicated, on-premises or co-located bare-metal server with an internal NVIDIA GPU. 

The deployment operates under a **100% self-hosted, local-sovereign architecture**: zero cloud API dependencies, zero external LLM egress, local speech processing (STT & TTS), carrier-grade SIP trunking, and automated self-healing reliability.

---

## 1. Architectural Topology

```
                   ┌──────────────────────────────────────────────┐
                   │            Inbound PSTN / Telephony          │
                   │     (Twilio Elastic SIP / Telnyx / PBX)      │
                   └──────────────────────┬───────────────────────┘
                                          │ SIP (5060/5080) + RTP (16384-16484)
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │           FreeSWITCH VoIP Container          │
                   │         (Dialplan / SIP Profile Edge)        │
                   └──────────────────────┬───────────────────────┘
                                          │ Full-Duplex Audio (16kHz PCM / WebRTC)
                                          │ ws://voice-agent-app:8000/ws/audio/<uuid>
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │    FastAPI Unified Voice Agent Application   │
                   │         (Uvicorn Multi-Worker Cluster)       │
                   ├──────────────────────┬───────────────────────┤
                   │  Speech Pipeline     │  Reasoning Engine     │
                   │  - Silero VAD        │  - Dialogue Manager   │
                   │  - Faster-Whisper    │  - RAG Firewall       │
                   │  - Piper ONNX TTS    │  - Pydantic Tools     │
                   └───────┬──────────────┴───────────────┬───────┘
                           │                              │
          Internal Ollama  │ Local Inference              │ Row-Level Lock & CalDAV
          HTTP (Port 11434)│                              │
                           ▼                              ▼
        ┌────────────────────────────┐    ┌─────────────────────────────────┐
        │       Ollama Engine        │    │ PostgreSQL 16 + pgvector        │
        │   (NVIDIA GPU Accelerated) │    │ - Relational Schema & Locks     │
        │   Llama 3 8B / Mistral 13B │    │ - Semantic Vector Embeddings    │
        └────────────────────────────┘    ├─────────────────────────────────┤
                                          │ Radicale CalDAV (Port 5232)     │
                                          │ - Site Visit Calendar Ledger    │
                                          ├─────────────────────────────────┤
                                          │ Redis 7 Alpine (Port 6379)      │
                                          │ - Ephemeral Cache & State       │
                                          └─────────────────────────────────┘
```

---

## 2. Server Hardware & Sizing Requirements

The system utilizes internal hardware acceleration for all models. Select your hardware configuration based on model size and expected concurrent call concurrency:

| Parameter | Standard (8B Llama 3) | Enterprise (13B Llama / Mistral) |
| :--- | :--- | :--- |
| **Target Concurrency** | 5 – 15 simultaneous active calls | 15 – 35 simultaneous active calls |
| **GPU Model** | NVIDIA RTX 3080 / 4070 / RTX A4000 | NVIDIA RTX 3090 / 4090 / RTX A5000 / A10G |
| **GPU VRAM** | **12 GB VRAM** (minimum 8 GB for Q4) | **24 GB VRAM** |
| **Host CPU** | 8 cores / 16 threads (AVX2 support) | 16 cores / 32 threads (AVX-512 recommended) |
| **Host System RAM** | 32 GB DDR4 / DDR5 | 64 GB DDR5 ECC |
| **Storage (NVMe SSD)**| 500 GB NVMe PCIe 4.0 | 1 TB NVMe PCIe 4.0 |
| **Operating System** | Ubuntu 22.04 LTS / 24.04 LTS x86_64 | Ubuntu 22.04 LTS / 24.04 LTS x86_64 |
| **Docker Engine** | Docker 24.0+ with Compose v2 | Docker 24.0+ with Compose v2 |
| **NVIDIA Driver** | 535.xx or newer + Container Toolkit | 535.xx or newer + Container Toolkit |

> [!IMPORTANT]
> Faster-Whisper and Piper ONNX run with CPU multithreading (`int8` quantization), requiring AVX2 CPU instruction support. Ollama LLM execution is strictly offloaded to the internal NVIDIA GPU via the NVIDIA Container Toolkit.

---

## 3. Host System Pre-Flight Setup

### Step 3.1: Install NVIDIA Container Toolkit (GPU Pass-Through)
Ensure the host server has official NVIDIA drivers installed, then configure Docker GPU acceleration:

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker daemon and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU detection inside Docker
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Step 3.2: Configure Linux Kernel & VoIP Network Tuning
VoIP and WebSockets require low latency and sufficient UDP socket buffer sizes for RTP media streams:

```bash
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216
sudo sysctl -w net.core.netdev_max_backlog=10000

# Persist settings in /etc/sysctl.d/99-voip.conf
cat << 'EOF' | sudo tee /etc/sysctl.d/99-voip.conf
net.core.rmem_max=16777216
net.core.wmem_max=16777216
net.core.netdev_max_backlog=10000
net.ipv4.udp_rmem_min=8192
net.ipv4.udp_wmem_min=8192
EOF
sudo sysctl -p /etc/sysctl.d/99-voip.conf
```

---

## 4. Single-Step Launch & Production Deployment

### Step 4.1: Clone & Configure Production Environment
1. Clone the repository into `/opt/realestate-ai`:
   ```bash
   cd /opt
   git clone https://github.com/your-org/RealEstateAI.git realestate-ai
   cd realestate-ai
   ```

2. Review and adjust secrets in [config/production.env](file:///c:/Users/aseem/RealEstateAI/config/production.env):
   - Set a strong `POSTGRES_PASSWORD`.
   - Update `CALDAV_PASSWORD` and carrier SIP credentials.

### Step 4.2: Pre-Pull Models & Initialize Caches
Execute the model bootstrap script to cache the LLM, Piper TTS voice weights, and Faster-Whisper models:

```bash
chmod +x scripts/bootstrap_prod_models.sh
./scripts/bootstrap_prod_models.sh
```

*(On Windows or cross-platform environments, execute `python scripts/bootstrap_prod_models.py`)*.

### Step 4.3: Launch All Production Containers
Run the single-command production deployment:

```bash
docker compose -f docker-compose.prod.yml up -d
```

### Step 4.4: Verify Service Health
Check that all 6 containers are running and healthy:

```bash
docker compose -f docker-compose.prod.yml ps
```

Expected output:
```
NAME                      IMAGE                              STATUS                    PORTS
realestate_postgres       pgvector/pgvector:pg16             Up (healthy)              0.0.0.0:5432->5432/tcp
realestate_caldav         tomsquest/docker-radicale:latest   Up (healthy)              0.0.0.0:5232->5232/tcp
realestate_redis          redis:7-alpine                     Up (healthy)              0.0.0.0:6379->6379/tcp
realestate_ollama         ollama/ollama:latest               Up (healthy)              0.0.0.0:11434->11434/tcp
realestate_freeswitch     realestateai-freeswitch:latest     Up (healthy)              0.0.0.0:5060->5060/tcp, 0.0.0.0:5060->5060/udp, ...
realestate_voice_agent    realestateai-voice-agent:latest    Up (healthy)              0.0.0.0:8000->8000/tcp
```

---

## 5. SIP Trunking & Carrier Setup

The FreeSWITCH edge container handles inbound carrier SIP signaling and bridges raw audio into the FastAPI WebSocket endpoint `ws://voice-agent-app:8000/ws/audio/{uuid}?phone={caller_id_number}`.

### Option A: Twilio Elastic SIP Trunking
1. In the **Twilio Console**, navigate to **Voice** > **SIP Trunking** > **Trunks** > **Create New Trunk**.
2. Under **Origination**, add an Origination SIP URI pointing to your server's public IP:
   `sip:+15551234567@<YOUR_SERVER_PUBLIC_IP>:5080;transport=udp`
3. Under **Termination**, configure your IP Access Control List (ACL) with `<YOUR_SERVER_PUBLIC_IP>`.
4. In [docker/freeswitch/conf/sip_profiles/external.xml](file:///c:/Users/aseem/RealEstateAI/docker/freeswitch/conf/sip_profiles/external.xml), update the gateway realm:
   ```xml
   <gateway name="twilio_trunk">
     <param name="realm" value="your-company.pstn.twilio.com"/>
     <param name="register" value="false"/>
     <param name="caller-id-in-from" value="true"/>
     <param name="ping" value="25"/>
   </gateway>
   ```

### Option B: Telnyx SIP Trunking
1. In the **Telnyx Mission Control Portal**, create a SIP Connection with **IP Authentication**.
2. Add your server's public IP address and set Port to `5080`.
3. Set preferred audio codec to **OPUS** (16000Hz) or **G.711u (PCMU)**.
4. Assign your purchased Telnyx DID phone numbers to this SIP Connection.

### Option C: Local On-Premise PBX / Asterisk Bridge
If you have an existing on-premise PBX (e.g. FreePBX, Asterisk, Grandstream UCM), create a SIP peer pointing to FreeSWITCH:
- Host: `<SERVER_IP>`
- Port: `5080`
- Codecs: `opus,ulaw`
- Dialplan: route customer service extension (e.g., `800`) directly to FreeSWITCH trunk.

---

## 6. Automated Reliability & Maintenance Workers

### 6.1: Stale Slot Hold Cleanup Worker
Prospective buyers receive a **5-minute reservation hold** during appointment negotiation. If the prospect disconnects or never verbally confirms, the hold must be reclaimed so the slot becomes available to other callers.

The cleanup worker runs automatically in the background via two redundant mechanisms:
1. **Integrated FastAPI Lifespan Loop:** Runs every 60 seconds inside `voice-agent-app`.
2. **Dedicated CLI Script (Cron fallback):**
   ```bash
   # Add to system crontab: sudo crontab -e
   * * * * * cd /opt/realestate-ai && python scripts/cleanup_stale_holds.py >> /var/log/stale_holds.log 2>&1
   ```

### 6.2: Automated PostgreSQL Database Backups & pgvector Snapshots
The system includes an automated backup utility in [scripts/backup_db.sh](file:///c:/Users/aseem/RealEstateAI/scripts/backup_db.sh) and [scripts/backup_db.py](file:///c:/Users/aseem/RealEstateAI/scripts/backup_db.py).
It performs a complete snapshot of relational tables, vector embeddings, alembic versions, and applies a **7-day retention policy**.

Configure a daily cron job at 02:00 AM:
```bash
# Add to crontab: sudo crontab -e
0 2 * * * /opt/realestate-ai/scripts/backup_db.sh >> /var/log/postgres_backup.log 2>&1
```

To run a manual on-demand backup:
```bash
./scripts/backup_db.sh
# Output saved to: storage/backups/backup_realestate_ai_YYYYMMDD_HHMMSS.sql.gz
```

To restore from a backup snapshot:
```bash
gunzip -c storage/backups/backup_realestate_ai_YYYYMMDD_HHMMSS.sql.gz | \
  docker exec -i realestate_postgres psql -U postgres -d realestate_ai
```

---

## 7. Observability & Prometheus Metrics Exporter

The FastAPI application natively exposes standard OpenMetrics on `GET /metrics`.

### Monitored Metrics:
| Metric Name | Type | Description |
| :--- | :--- | :--- |
| `realestate_active_holds` | Gauge | Number of active 5-minute slot reservation holds |
| `realestate_confirmed_visits` | Gauge | Total physical site visits confirmed in database |
| `realestate_pending_callbacks` | Gauge | Operator callback notifications pending human triage |
| `realestate_calls_total` | Counter | Total inbound calls handled |
| `realestate_calls_active` | Gauge | Number of calls currently in progress |
| `realestate_calls_dropped_total` | Counter | Number of unexpected telephony disconnections |
| `realestate_pipeline_latency_seconds_bucket` | Histogram | Latency distributions for STT, Dialogue Reasoning, and TTS |

### Prometheus Configuration (`prometheus.yml`):
```yaml
scrape_configs:
  - job_name: "realestate_ai_voice_agent"
    scrape_interval: 10s
    metrics_path: "/metrics"
    static_configs:
      - targets: ["voice-agent-app:8000"]
```

Test the metrics endpoint locally:
```bash
curl -s http://localhost:8000/metrics
```

---

## 8. Verification & Diagnostic Testing Runbook

Execute the complete diagnostic suite to verify production readiness:

### 1. Comprehensive System Diagnostic (`agent_doctor.py`)
Run the 9-point comprehensive diagnostic health check:
```bash
python scripts/agent_doctor.py
```
*Verifies: Python version, PostgreSQL connection, pgvector extension, Radicale CalDAV, Redis, Ollama LLM, Faster-Whisper, Piper TTS, and storage directories.*

### 2. End-to-End Voice Latency Profiling (`profile_latency.py`)
Profile audio turn turnaround time across VAD -> STT -> Reasoning -> TTS:
```bash
python scripts/profile_latency.py --runs 3
```

### 3. Automated Test Suite Execution
Execute the automated test suite verifying row-level locks, firewall boundaries, and telephony bridging:
```bash
pytest tests/ -v
```

---

## 9. Firewall & Network Hardening

Configure Host UFW firewall rules to safeguard internal services while opening required telephony ports:

```bash
# Allow SSH access
sudo ufw allow 22/tcp

# Allow FreeSWITCH SIP & RTP audio ports
sudo ufw allow 5060/udp comment "SIP Signaling"
sudo ufw allow 5060/tcp comment "SIP Signaling TCP"
sudo ufw allow 5080/udp comment "External Carrier Trunk"
sudo ufw allow 5080/tcp comment "External Carrier Trunk TCP"
sudo ufw allow 16384:16484/udp comment "RTP Media Audio Range"

# Protect Internal Management Ports (Bind only to internal subnet / VPN)
# Ports 5432 (Postgres), 5232 (CalDAV), 6379 (Redis), 11434 (Ollama), 8000 (App)
# are isolated within Docker's bridge network 'realestate_net'.
sudo ufw enable
```

---

## 10. Summary of Production CLI Commands

| Operation | Command |
| :--- | :--- |
| **Start Stack** | `docker compose -f docker-compose.prod.yml up -d` |
| **Stop Stack** | `docker compose -f docker-compose.prod.yml down` |
| **Check Logs** | `docker compose -f docker-compose.prod.yml logs -f voice-agent-app` |
| **Run Migrations** | `docker compose -f docker-compose.prod.yml exec voice-agent-app alembic upgrade head` |
| **Run Backup** | `./scripts/backup_db.sh` |
| **Scrape Metrics** | `curl -s http://localhost:8000/metrics` |
| **Doctor Health** | `python scripts/agent_doctor.py` |
| **Reclaim Holds** | `python scripts/cleanup_stale_holds.py` |
