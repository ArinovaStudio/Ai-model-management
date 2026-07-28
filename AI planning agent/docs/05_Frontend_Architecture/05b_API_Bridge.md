# 05b. API Bridge Layer

The frontend interface communicates with the FastAPI backend through a unified API client wrapper.

* **File:** `frontend/js/api.js`
* **Purpose:** Abstracts network communication, maps requests, and handles error formatting for all FastAPI endpoints.

---

## 1. Key API Methods

### `async runPipelineStage(projectId, stageName)`
* **Description:** Triggers specific sequential engine stages (e.g. `wbs`, `estimate`, `schedule`) inside the backend.
* **Payload Mapping:** Passes the target `projectId` and the endpoint route mapping.

### `async enhanceSection(projectId, payload)`
* **Description:** POSTs targeted edit instructions to the `/enhance` route.
* **Arguments:** 
  * `projectId`
  * `payload`: `{ section_type, section_key, user_instruction, current_content }`

### `async revertSection(projectId, sectionType, sectionKey)`
* **Description:** Dispatches a POST request to discard recent edits, reverting the database to the saved snapshot.

---

## 2. Global Request Exception Interceptors
The `api.js` client intercepts errors from network responses:
* **HTTP 422 Catching:** If the server returns a 422 error, `api.js` extracts the `detail` parameter from the JSON error payload (which includes detailed LLM exception tracebacks) and bubbles it up to the UI modal alerts.
* **Connection Failure Handling:** Catches Ollama offline timeouts or HTTP exceptions and raises clean user notifications instead of silent console crashes.
