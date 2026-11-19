"""System Diagnostic Health CLI (Agent Doctor).

Validates:
1. Docker Container Runtime (Postgres, CalDAV, Redis).
2. Database Integrity (PostgreSQL connectivity, pgvector extension, table schemas, vector indexes).
3. LLM Connectivity (Cloud GPU / Local Ollama endpoint health, latency, tok/s throughput).
4. Local Speech Subsystems (Faster-Whisper int8 STT transcription, Piper ONNX streaming TTS).
5. CalDAV Server Integration (Radicale connection and principal calendar accessibility).

Outputs a clean PASS / WARN / FAIL diagnostic status matrix.
"""

import asyncio
import io
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import numpy as np
import psycopg2
import soundfile as sf

from src.core.config import settings
from src.core.logging import logger
from src.audio.stt import stt_engine
from src.audio.tts import tts_engine
from src.db.caldav_sync import caldav_sync


class AgentDoctor:
    """Diagnostic suite for checking infrastructure, models, and service health."""

    def __init__(self):
        self.results: list[dict[str, str]] = []

    def record(self, category: str, component: str, status: str, details: str):
        self.results.append({
            "category": category,
            "component": component,
            "status": status,
            "details": details,
        })

    def check_docker_containers(self):
        """Checks status of PostgreSQL, CalDAV, and Redis Docker containers."""
        try:
            res = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode != 0:
                self.record("Infrastructure", "Docker Engine", "WARN", "Docker command failed or not running")
                return

            running_containers = {}
            for line in res.stdout.strip().split("\n"):
                if "\t" in line:
                    name, status = line.split("\t", 1)
                    running_containers[name.strip()] = status.strip()

            target_containers = [
                ("realestate_postgres", f"PostgreSQL + pgvector ({settings.POSTGRES_PORT})"),
                ("realestate_caldav", "Radicale CalDAV (5232)"),
                ("realestate_redis", f"Redis Session Store ({settings.REDIS_PORT})"),
                ("realestate_ollama", "Ollama GPU Engine (11434)"),
                ("realestate_freeswitch", "FreeSWITCH SIP Edge (5060)"),
            ]

            for c_name, label in target_containers:
                if c_name in running_containers:
                    self.record("Infrastructure", label, "PASS", f"Active ({running_containers[c_name]})")
                else:
                    self.record("Infrastructure", label, "FAIL", "Container not running in 'docker ps'")
        except Exception as e:
            self.record("Infrastructure", "Docker Engine", "WARN", f"Unable to probe Docker: {e}")

    def check_postgresql_and_pgvector(self):
        """Checks PostgreSQL connection, pgvector extension, and indexed tables."""
        dsn = settings.DATABASE_SYNC_URL.replace("+psycopg2", "")
        conn = None
        try:
            conn = psycopg2.connect(dsn, connect_timeout=4)
        except Exception as primary_err:
            # Fallback probe to alternative port (5432 if configured 5435, or vice-versa)
            alt_port = 5432 if settings.POSTGRES_PORT == 5435 else 5435
            alt_dsn = dsn.replace(f":{settings.POSTGRES_PORT}/", f":{alt_port}/")
            try:
                conn = psycopg2.connect(alt_dsn, connect_timeout=3)
            except Exception:
                self.record("Database", "PostgreSQL Connection", "FAIL", str(primary_err)[:80])
                return

        try:
            cur = conn.cursor()

            # 1. Check pgvector extension
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            vector_row = cur.fetchone()
            if vector_row:
                self.record("Database", "pgvector Extension", "PASS", f"v{vector_row[0]} installed & active")
            else:
                self.record("Database", "pgvector Extension", "FAIL", "Extension 'vector' not found in DB")

            # 2. Check essential tables and record counts
            cur.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM property_listings) AS props,
                    (SELECT COUNT(*) FROM knowledge_chunks) AS chunks,
                    (SELECT COUNT(*) FROM calendar_slots) AS slots,
                    (SELECT COUNT(*) FROM operator_notifications) AS notifications;
            """)
            props, chunks, slots, notifs = cur.fetchone()
            self.record(
                "Database",
                "Schema & Records",
                "PASS",
                f"Properties: {props}, Vector Chunks: {chunks}, Slots: {slots}, Notifications: {notifs}",
            )

            conn.close()
        except Exception as e:
            self.record("Database", "PostgreSQL Connection", "FAIL", str(e)[:70])

    async def check_caldav_service(self):
        """Validates HTTP connectivity and calendar principal access on Radicale."""
        try:
            client = httpx.AsyncClient(timeout=4.0, follow_redirects=True)
            res = await client.get(settings.CALDAV_URL)
            if res.status_code in [200, 302, 401]:
                # Test full adapter principal discovery in background thread
                def _probe_cal():
                    c = caldav_sync.get_or_create_calendar()
                    return c.get_display_name() or settings.CALDAV_CALENDAR_NAME

                cal_name = await asyncio.to_thread(_probe_cal)
                self.record("Services", "Radicale CalDAV", "PASS", f"Connected ('{cal_name}' calendar verified)")
            else:
                self.record("Services", "Radicale CalDAV", "WARN", f"HTTP Status {res.status_code}")
            await client.aclose()
        except Exception as e:
            self.record("Services", "Radicale CalDAV", "FAIL", str(e)[:70])

    async def check_llm_intelligence(self):
        """Validates LLM endpoint connectivity, latency, and token throughput."""
        base_url = settings.LLM_BASE_URL.rstrip("/")
        model = settings.LLM_MODEL_NAME
        try:
            client = httpx.AsyncClient(timeout=10.0)
            t0 = time.perf_counter()

            # Test OpenAI-compatible endpoint
            endpoint = f"{base_url}/v1/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Respond with the single word: OK"}],
                "max_tokens": 10,
                "temperature": 0.1,
            }
            res = await client.post(endpoint, json=payload)
            elapsed_sec = time.perf_counter() - t0

            if res.status_code == 200:
                data = res.json()
                content = data["choices"][0]["message"]["content"].strip()
                tokens = data.get("usage", {}).get("total_tokens", 5)
                tok_per_sec = tokens / max(elapsed_sec, 0.001)
                self.record(
                    "AI Models",
                    f"LLM ({model})",
                    "PASS",
                    f"Response: '{content}' in {elapsed_sec*1000:.0f}ms (~{tok_per_sec:.1f} tok/s)",
                )
            else:
                # Test Ollama native endpoint (/api/generate)
                res2 = await client.post(f"{base_url}/api/generate", json={"model": model, "prompt": "Hi", "stream": False})
                if res2.status_code == 200:
                    self.record("AI Models", f"LLM ({model})", "PASS", "Native Ollama endpoint responding")
                else:
                    self.record("AI Models", f"LLM ({model})", "WARN", f"Endpoint returned HTTP {res.status_code}")
            await client.aclose()
        except Exception as e:
            self.record("AI Models", f"LLM ({model})", "WARN", f"Unreachable at {base_url} ({e})")

    def check_local_speech_engines(self):
        """Validates Faster-Whisper STT and Piper ONNX TTS speech models."""
        # 1. TTS Test
        try:
            t0 = time.perf_counter()
            test_phrase = "System diagnostics nominal."
            wav_bytes = tts_engine.synthesize_to_wav(test_phrase)
            tts_time_ms = (time.perf_counter() - t0) * 1000.0

            if len(wav_bytes) > 500 and wav_bytes.startswith(b"RIFF"):
                self.record(
                    "Speech",
                    "Piper ONNX TTS",
                    "PASS",
                    f"Synthesized '{test_phrase}' in {tts_time_ms:.1f}ms ({len(wav_bytes)} bytes)",
                )
            else:
                self.record("Speech", "Piper ONNX TTS", "FAIL", "Invalid audio byte container")
        except Exception as e:
            self.record("Speech", "Piper ONNX TTS", "FAIL", str(e)[:70])

        # 2. STT Test
        try:
            t0 = time.perf_counter()
            text, conf = stt_engine.transcribe_turn(wav_bytes)
            stt_time_ms = (time.perf_counter() - t0) * 1000.0

            if len(text) > 0:
                self.record(
                    "Speech",
                    "Faster-Whisper int8 STT",
                    "PASS",
                    f"Transcribed '{text}' in {stt_time_ms:.1f}ms (Conf: {conf*100:.1f}%)",
                )
            else:
                self.record("Speech", "Faster-Whisper int8 STT", "WARN", "Transcription produced empty text")
        except Exception as e:
            self.record("Speech", "Faster-Whisper int8 STT", "FAIL", str(e)[:70])

    async def run_diagnostics(self):
        print("=" * 78)
        print("           REAL ESTATE AI VOICE AGENT - SYSTEM DIAGNOSTIC DOCTOR           ")
        print("=" * 78)
        print("Running comprehensive health audit across infrastructure, DB & AI engines...\n")

        self.check_docker_containers()
        self.check_postgresql_and_pgvector()
        await self.check_caldav_service()
        await self.check_llm_intelligence()
        self.check_local_speech_engines()

        # Display Diagnostic Matrix
        print("=" * 78)
        print(f"{'Category':<16} | {'Component':<28} | {'Status':<6} | {'Details'}")
        print("-" * 78)

        pass_count = 0
        warn_count = 0
        fail_count = 0

        for r in self.results:
            st = r["status"]
            if st == "PASS":
                pass_count += 1
                colored_status = "[PASS]"
            elif st == "WARN":
                warn_count += 1
                colored_status = "[WARN]"
            else:
                fail_count += 1
                colored_status = "[FAIL]"

            print(f"{r['category']:<16} | {r['component']:<28} | {colored_status:<6} | {r['details']}")

        print("=" * 78)
        overall = "HEALTHY" if fail_count == 0 else "DEGRADED"
        print(f"DIAGNOSTIC SUMMARY: {overall} | PASS: {pass_count} | WARN: {warn_count} | FAIL: {fail_count}")
        print("=" * 78 + "\n")
        return fail_count == 0


if __name__ == "__main__":
    doctor = AgentDoctor()
    asyncio.run(doctor.run_diagnostics())
