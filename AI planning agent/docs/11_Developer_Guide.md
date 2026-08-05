# 11. Developer Guide

This guide describes development workflows, patterns, and conventions for adding features or modifying planning engines.

---

## 1. Plan Generation Pipeline Architecture
The generation engines inside `app/services/` are decoupled and execute in sequence:
* **The Rule of Persistence:** Every engine step **must write** its computed properties to the database models. Downstream engines should query the database (or cache) to obtain these values rather than receiving them as in-memory function parameters.
* **Cache Invalidation:** When modifying records within a route, always call `clear_project_cache(project_id)` to invalidate stale project states.

---

## 2. Modifying Database Schemas
The database schema is defined in `prisma/schema.prisma`. To update the schema:

1. Modify `prisma/schema.prisma` using standard Prisma format syntax.
2. Regenerate the Python client and push changes directly during local development:
   ```bash
   prisma db push
   prisma generate
   ```
3. Update Pydantic response models inside `app/models/schemas.py` to match the new database fields.

---

## 3. Adding a New Pipeline Stage
To insert a new stage into the planning wizard:

1. Create a service file `app/services/new_stage_engine.py` housing the generation logic.
2. Register the stage endpoint in `app/routers/` (e.g. `app/routers/new_stage.py`).
3. Import and mount the router in `app/main.py`.
4. Wrap the engine function in `profile_launcher.py` to trace execution telemetry:
   ```python
   import app.services.new_stage_engine as new_stage_engine
   instrument_function(new_stage_engine, "run_new_stage")
   ```
5. Update `frontend/js/workspace.js` stepper sequence to render the new wizard view.
