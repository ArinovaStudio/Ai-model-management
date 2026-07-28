# AI Planning Agent Documentation

Welcome to the comprehensive technical documentation for the **AI Planning Agent**. This system uses local Large Language Models (Ollama) and database orchestration to generate feasibility-validated, resource-allocated, and scheduled project plans.

---

## Documentation Directory

### 1. [Overview & Architecture](01_Project_Overview.md)
* High-level project purpose and features.
* Tech stack and core dependencies.
* High-level architectural pattern.

### 2. [System Architecture & Flow](02_System_Architecture.md)
* Component boundary and engine coordination diagrams.
* In-memory database caching mechanism.
* External LLM communications structure.

### 3. [Directory Structure](03_Directory_and_Modules.md)
* Codebase folder map.
* File-by-file explanations.

### 4. Backend Modules & Core Engines
* **[Pipeline Stages](04_Backend_Engines/04a_Pipeline_Stages.md):** The 8 sequential planning engines (`analysis`, `intelligence`, `wbs`, `estimation`, `resource`, `schedule`, `validation`, `gap`).
* **[Enhancement Engine](04_Backend_Engines/04b_Enhancement_Engine.md):** Targeted Natural Language updates, database rollbacks, unique key deduplication, and dependency mapping.
* **[Caching Layer](04_Backend_Engines/04c_Caching_Layer.md):** Prisma client in-memory cache coordination.

### 5. Frontend UI Architecture
* **[UI State Controllers](05_Frontend_Architecture/05a_UI_State_Controllers.md):** Workspace step controllers and DOM updates.
* **[API Bridge](05_Frontend_Architecture/05b_API_Bridge.md):** Request mapping and error handling layers.

### 6. [Core Workflows](06_Core_Workflows.md)
* Lifecycle of project plan generation.
* Targeted editing and transaction rollback sequence.

### 7. [Database & LLM Contracts](07_Database_and_LLM_Contracts.md)
* Prisma database schema and relational topology.
* LLM prompts, constraints, and JSON response models.

### 8. [Operations & Telemetry Profiling](08_Operations_and_Profiling.md)
* Run-time monkeypatching system (`profile_launcher.py`).
* Resource monitoring thread (`psutil`).
* CSV/Markdown performance report builders.

### 9. [Setup Guide](09_Setup_Guide.md)
* Local environment variables, Python 3.14 requirements, Neon DB, and local Ollama setup.

### 10. [Configuration Reference](10_Configuration_Reference.md)
* Detailed environment variable matrix (.env mapping).

### 11. [Developer Guide](11_Developer_Guide.md)
* Design patterns, coding standards, adding new pipeline engines, and schema updates.

### 12. [Troubleshooting Diagnostics](12_Troubleshooting.md)
* Handling 422 errors, unique constraint collisions, and LLM JSON parsing failures.
