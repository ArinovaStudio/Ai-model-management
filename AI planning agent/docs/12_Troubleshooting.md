# 12. Troubleshooting Diagnostics

A reference of common errors, causes, and steps to resolve them.

---

## 1. 422 Unprocessable Content (Section Enhancement Failures)

### Symptom:
A `POST /enhance` request returns a `422` error, and the UI alerts with *"WBS enhancement reconstruction failed..."* or *"The editing engine failed to output valid JSON"*.

### Diagnostics:
1. **Malformed LLM Output:** Check the terminal console logs. The router prints the raw text received from Ollama when parsing fails. Often, Ollama prepends intro/outro text (e.g. *"Here is your updated JSON:"*) which causes json-parser crashes.
2. **Missing Dependencies/Ollama Offline:** Verify that the Ollama port (`11434` or custom tunnel) is active and accessible.
3. **Database unique key constraint violation:** If the LLM generates tasks with colliding IDs, the database throws unique violations. The deduplication module in `app/routers/enhancement.py` handles this, but verify that task ID formats match UUID patterns.

---

## 2. Python 3.14 Compatibility Warnings
```
UserWarning: Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.
  from pydantic.v1 import Extra as Extra
```
### Symptom:
Console prints warnings from `prisma._compat` at startup.

### Diagnostics:
These warnings are generated because Python 3.14 deprecates older Pydantic V1 APIs used inside Prisma Client Python bindings. These are **non-blocking warnings** and do not affect the application's runtime stability. You can safely ignore them.

---

## 3. Milestones Render Out of Order in the UI

### Symptom:
The project workspace shows `Milestone 5`, followed by `Milestone 3`, `Milestone 2`, etc.

### Diagnostics:
1. **Prisma database query sorting:** Ensure that the database queries specify `order={"order": "asc"}` when fetching groups:
   ```python
   groups = await db.groupingtask.find_many(where={"projectId": project_id}, order={"order": "asc"})
   ```
2. **Missing Order Parameter in Backups:** If a project was reverted, check if the backup content database insertion logic has the `"order": g.order` field.
