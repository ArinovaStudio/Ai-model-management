# 08. Operations & Telemetry Profiling

The AI Planning Agent includes a custom telemetry and performance profiling system that monitors runtime resources and database query performance.

---

## 1. Runtime Telemetry Middleware (`profile_launcher.py`)
At startup, `profile_launcher.py` loads the application and runs monkeypatches to hook into key system operations:

* **Prisma Query Hook:** Hooks into `Prisma._execute` to record the type of query, database model target, duration, and number of returned rows.
* **Ollama chat Interceptor:** Hooks into `ollama_service.client.chat` to measure execution time, input prompt characters, output response characters, and estimate API token usage and costs.
* **Resource Monitoring Thread:** Spawns a background thread using `psutil` to log process CPU and memory utilization every second.

---

## 2. Telemetry Triggers & Phase Identification
The tracking loop begins when the profiler middleware intercepts a request to the requirements analysis endpoint (`POST /analyze`):
1. **Trace ID Initialization:** Generates a unique UUID and resets telemetry lists.
2. **Metrics Collection:** Records metrics for all database queries and LLM chat requests made during subsequent stages (e.g. WBS, Estimation, Scheduling).
3. **Trace Completion:** When the final stage endpoint (`POST /gap-analysis`) completes, the tracker stops collection and compiles the reports.

---

## 3. Compiled Telemetry Outputs
After completing a pipeline run, the telemetry launcher writes the following profiling reports to the project root:

1. **`timeline.json`:** JSON structure containing time statistics for each planning stage and their percentage of the total runtime.
2. **`function_profile.csv`:** Lists calling frequencies, average durations, and maximum/minimum latency times for core backend functions.
3. **`resource_usage.csv`:** Contains CPU percentage and memory usage (in MB) recorded during execution.
4. **`database_report.md`:** Markdown table detailing the slowest queries, models targeted, and rows returned.
5. **`llm_report.md`:** Tracks LLM calls, durations, input/output token usage, and cost estimates.
