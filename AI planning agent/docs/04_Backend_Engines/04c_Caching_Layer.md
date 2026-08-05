# 04c. Caching Layer

The AI Planning Agent pipeline performs complex multi-stage computations where downstream engines rely on database values written by upstream engines. 

To reduce database query overhead and prevent Neon/PostgreSQL latency from blocking the pipeline, the system uses an in-memory caching wrapper.

---

## 1. Caching File
* **Location:** `app/db/prisma.py`
* **Mechanism:** A global dictionary `_project_cache` stores serialized Prisma project representations, keyed by `project_id`.

---

## 2. API Reference

### `get_cached_project(project_id: str) -> Optional[Project]`
* **Purpose:** Retrieves a cached project representation.
* **Usage:** Used at the beginning of each engine's execution (e.g. `schedule_engine`, `estimation_engine`) to fetch project info, tasks, feature requests, and resource plans without query overhead.

### `set_cached_project(project_id: str, project: Project) -> None`
* **Purpose:** Caches the project model state in-memory.
* **Usage:** Invoked immediately after database queries fetch a project's full relation graph:
  ```python
  project = await db.project.find_unique(
      where={"id": project_id},
      include={
          "featureRequests": True,
          "projectInfo": True,
          "groupingTasks": {"include": {"tasks": True}},
          # ...
      }
  )
  if project:
      set_cached_project(project_id, project)
  ```

### `clear_project_cache(project_id: str) -> None`
* **Purpose:** Invalidates and purges the cached record.
* **Usage:** Called whenever a database record is created, modified, or enhanced (such as inside the WBS enhancement route). This forces subsequent engine steps to reload the fresh data from the database.
