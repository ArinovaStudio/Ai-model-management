# 01. Project Overview

The **AI Planning Agent** is an intelligent assistant designed to ingest initial project parameters (e.g. project name, type, description, targeted timeline, and requirements) and automatically orchestrate a feasibility-validated development plan. 

Unlike basic template generators, this system runs a mathematical and LLM-assisted execution pipeline to resolve feature boundaries, build a Work Breakdown Structure (WBS), estimate development effort hours, map resource constraints, generate chronological schedules, and run feasibility checks.

---

## 1. High-Level Problem Statement
Traditional project planning relies on spreadsheets and human estimates that frequently suffer from:
1. **Scope and Requirement Gaps:** Vague specs lead to missed edge cases.
2. **Scheduling Feasibility Issues:** Overestimating developer velocity or underestimating dependencies.
3. **Inconsistent Resource Allocations:** Under-allocating critical roles or over-allocating personnel.
4. **Poor State Invalidation:** Changing WBS tasks manual updates to schedules, sprint allocations, and dependency chains.

The AI Planning Agent resolves this by using an **automated 8-stage pipeline** where the output of each stage acts as the input to the next, and changes can be incrementally updated through a **Natural Language Section Enhancer** with transaction rollback mechanics.

---

## 2. Core Features
* **Sequential Planner Pipeline:** Executes requirements analysis, technical architecture extraction, WBS group generation, hourly effort estimation, headcount planning, sprint scheduling, timeline check, and gap reports.
* **Targeted Section Enhancer:** Allows natural language modification of any plan section (e.g., "Add security testing to all milestones") while keeping unaffected sections and database IDs intact.
* **State Rollback & Deduplication:** Reverts the database cleanly if enhancements fail or are rejected by the user. Automatically deduplicates database keys when importing complex plans.
* **Monkeypatched Telemetry Profiler:** Measures runtime resource usage (CPU/memory), database queries, and LLM token costs, generating detailed reports on request completion.

---

## 3. Technology Stack & Core Boundaries
The agent is built using the following stack:

| Component | Technology | Role |
|---|---|---|
| **Backend Framework** | FastAPI (Python) | API routing, request validation, and orchestrator execution. |
| **ORM & Database** | Prisma Client (Python) + PostgreSQL / SQLite | Schema management, migrations, and relationship mapping. |
| **AI LLM Engine** | Ollama Client (Local/Tunnel) | Model generation for project specs, WBS structures, and enhancements. |
| **Telemetry** | Custom monkeypatched middleware (`psutil`) | Runtime tracking of database latency and LLM token stats. |
| **Frontend UI** | HTML5, Vanilla CSS, Vanilla JavaScript | Workspace stepper rendering, api bridge integrations. |
