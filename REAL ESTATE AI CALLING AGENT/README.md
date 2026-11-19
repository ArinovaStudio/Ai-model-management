# RealEstateAI: Local-First AI Voice Agent

Production-grade, self-hosted, local-sovereign AI voice agent for real estate lead qualification, inventory discovery, and deterministic site-visit booking.

## Prerequisites
- Docker & Docker Compose
- Python 3.11
- Ollama or remote Cloud GPU endpoint

## Quick Start
1. Copy `.env.example` to `.env`
2. Start local infrastructure:
   ```bash
   docker compose up -d
   ```
3. Verify infrastructure health:
   ```bash
   python -m scripts.verify_infra
   ```
