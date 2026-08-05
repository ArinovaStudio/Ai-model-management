# 04b. Section Enhancement Engine

The **Section Enhancement Router** (`app/routers/enhancement.py`) provides targeted modification of project planning sections (WBS, Resources, Intelligence, Risks, and Docs) via natural language user instructions. 

It ensures that only the requested changes are applied, preserving the integrity, structure, and unique database identifiers of the rest of the plan.

---

## 1. Unified WBS Edit Flow
To prevent task duplication across multiple milestones, the WBS enhancement is structured as a **single-call pipeline**:
1. The backend gathers the entire WBS (all milestone groups and their tasks) and serializes it to a single JSON array.
2. It queries the `ProjectTechnology` table to fetch the project's actual database (e.g. `PostgreSQL`).
3. It sends this combined state to Ollama with instructions to modify ONLY what was asked, strictly forbidding introducing foreign tech stacks (e.g. MongoDB in a PostgreSQL project).
4. The parsed JSON response is compared against the database records to preserve untouched tasks.

---

## 2. Collision Deduplication & Dependency Repair
When a modified WBS is written back to the database, a cascade of `delete_many` and `create_many` runs. To prevent database crashes due to key violations, the engine implements a deduplication middleware block before saving:

```python
# 1. Compare incoming task and group IDs against the DB keys
existing_tasks = await db.task.find_many()
db_task_ids = {et.id for et in existing_tasks}

# 2. Map colliding IDs to a new unique UUID
for tr in task_records:
    old_tid = tr["id"]
    if old_tid in seen_task_ids or old_tid in db_task_ids:
        new_tid = str(uuid.uuid4())
        task_id_mapping[old_tid] = new_tid
        tr["id"] = new_tid
```

### Dependency Repair:
If any task ID changes to avoid a database collision, the engine automatically parses task dependency fields (which store JSON arrays of task IDs) and rewrites them in-flight:
```python
# 3. Rewrite dependency references in-flight
for tr in task_records:
    deps_str = tr.get("dependencies")
    if deps_str:
        deps_list = json.loads(deps_str)
        updated_deps = [task_id_mapping.get(d, d) for d in deps_list]
        tr["dependencies"] = json.dumps(updated_deps)
```

---

## 3. Transaction State Rollbacks
Every enhancement request makes a temporary snapshot backup in `ProjectBackup` before applying changes:
* **Rollback Trigger:** If the LLM generates invalid JSON, times out, or fails database constraints, the router catches the exception, calls `_rollback_backup()`, and restores the exact state (including the milestone `order` indices) before raising a `422 Unprocessable Content` response.
* **Manual Reversion:** Users can discard saved edits in the frontend UI. This calls `revert_section()`, which retrieves the snapshot from `ProjectBackup` and restores it, preserving all original IDs and sequences.
