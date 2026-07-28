# 10. Configuration Reference

The AI Planning Agent is configured primarily via environment variables defined in the `.env` file located in the project root.

---

## 1. Environment Variable Matrix

| Variable | Description | Default Value | Production Recommendation |
|---|---|---|---|
| **`DATABASE_URL`** | The connection string for the database (PostgreSQL, Neon, or SQLite). | N/A | Must point to a highly available PostgreSQL/Neon instance. |
| **`OLLAMA_BASE_URL`** | The API endpoint for the Ollama server. | `http://localhost:11434` | Point to a dedicated remote/local server containing high-performance GPUs. |
| **`OLLAMA_MODEL`** | The LLM model name. | `llama3` | Use high-capacity models (e.g. `llama3`, `mistral`, or custom fine-tunes) that reliably follow JSON schemas. |
| **`LOG_LEVEL`** | The logging verbosity level. | `INFO` | Set to `INFO` or `WARNING` in production to reduce log disk usage. |
| **`PORT`** | The port for the backend FastAPI application. | `8000` | Bind to standard HTTP/HTTPS proxy ports behind Nginx. |

---

## 2. Default Calculation Parameters
The planning engines use the following default parameters defined inside `app/services/schedule_engine.py`:
* **Sprint Duration:** 14 days (2 weeks).
* **Productive Hours per Developer per Day:** 6.0 hours (accounts for administrative overhead).
* **Default Developer Headcount:** 2 developers (assigned if the user doesn't specify a project deadline).
