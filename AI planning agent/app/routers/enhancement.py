import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.db.prisma import db, clear_project_cache
from app.services.ollama_service import ollama_service
from prisma import Json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/projects/{project_id}/enhance")


def get_backup_path(project_id: str) -> str:
    backup_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backups")
    os.makedirs(backup_dir, exist_ok=True)
    return os.path.join(backup_dir, f"project_{project_id}.json")


def deep_merge(dict1: dict, dict2: dict) -> dict:
    merged = dict1.copy()
    for key, val in dict2.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return default
            import re
            match = re.search(r"[-+]?\d*\.\d+|\d+", val)
            if match:
                return float(match.group())
            return default
        return float(val)
    except Exception:
        return default


def safe_int(val, default=0):
    if val is None:
        return default
    try:
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return default
            import re
            match = re.search(r"[-+]?\d+", val)
            if match:
                return int(match.group())
            return default
        return int(val)
    except Exception:
        return default


def extract_milestone_prefix(title: str) -> tuple[str, str]:
    import re
    match = re.match(r"^(\[(?:Milestone|Feature)\s+[^\]]+\]\s*(?:Task\s*\d+(?:\.\d+)*:\s*)?)(.*)$", title, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    match_just_prefix = re.match(r"^(\[(?:Milestone|Feature)\s+[^\]]+\]\s*)(.*)$", title, re.IGNORECASE)
    if match_just_prefix:
        return match_just_prefix.group(1), match_just_prefix.group(2)
    return "", title


async def _rollback_backup(project_id: str, section: str, key: str):
    logger.warning("Enhancement failed. Rolling back section %s for project %s", section, project_id)
    backup = await db.projectbackup.find_unique(
        where={
            "projectId_section_key": {
                "projectId": project_id,
                "section": section,
                "key": key
            }
        }
    )
    if not backup or not backup.content:
        logger.error("No backup found to rollback for %s / %s", section, key)
        return

    original_val = backup.content
    if section == "wbs":
        parsed = json.loads(original_val)
        await db.task.delete_many(where={"projectId": project_id})
        await db.groupingtask.delete_many(where={"projectId": project_id})
        
        # Restore Groups
        for g in parsed.get("groups", []):
            await db.groupingtask.create(
                data={
                    "id": g["id"],
                    "name": g.get("name") or "Unnamed Phase",
                    "order": int(g.get("order", 0)),
                    "totalEffortHours": float(g.get("totalEffortHours", g.get("total_effort_hours", g.get("effort_hours", 0.0)))),
                    "totalDurationDays": float(g.get("totalDurationDays", g.get("total_duration_days", 0.0))),
                    "description": g.get("description", ""),
                    "projectId": project_id
                }
            )
            
        # Restore Tasks
        for t in parsed.get("tasks", []):
            due_dt = None
            if t.get("dueDate"):
                try:
                    due_dt = datetime.fromisoformat(t["dueDate"])
                except Exception:
                    pass
            if not due_dt:
                due_dt = datetime.now(timezone.utc)

            start_dt = None
            if t.get("scheduledStart"):
                try:
                    start_dt = datetime.fromisoformat(t["scheduledStart"])
                except Exception:
                    pass

            end_dt = None
            if t.get("scheduledEnd"):
                try:
                    end_dt = datetime.fromisoformat(t["scheduledEnd"])
                except Exception:
                    pass
            
            await db.task.create(
                data={
                    "id": t["id"],
                    "title": t.get("title") or t.get("name") or t.get("task_title") or "Unnamed Task",
                    "description": t.get("description") or t.get("content") or t.get("task_description") or "",
                    "assignee": t.get("assignee", "Developer"),
                    "assigneeAvatar": t.get("assigneeAvatar"),
                    "priority": t.get("priority", "medium"),
                    "dueDate": due_dt,
                    "employeeId": t.get("employeeId"),
                    "tags": t.get("tags", []),
                    "status": t.get("status", "pending"),
                    "projectId": project_id,
                    "groupId": t.get("groupId"),
                    "featureRequestId": t.get("featureRequestId"),
                    "order": int(t.get("order", 0)),
                    "dependencies": json.dumps(t.get("dependencies")) if isinstance(t.get("dependencies"), list) else t.get("dependencies"),
                    "estimatedHours": float(t.get("estimatedHours", t.get("estimated_hours", 0.0))),
                    "complexityScore": float(t.get("complexityScore", 0.0)),
                    "effortHours": float(t.get("effortHours", t.get("estimatedHours", 0.0))),
                    "durationDays": float(t.get("durationDays", 0.0)),
                    "scheduledStart": start_dt,
                    "scheduledEnd": end_dt
                }
            )
        try:
            from app.services.schedule_engine import generate_schedule
            await generate_schedule(project_id, reindex=True)
        except Exception:
            pass

    elif section == "resources":
        parsed = json.loads(original_val)
        await db.resourceallocation.delete_many(where={"projectId": project_id})
        for a in parsed:
            await db.resourceallocation.create(
                data={
                    "id": a["id"],
                    "projectId": project_id,
                    "role": a["role"],
                    "requiredCount": int(a.get("requiredCount", a.get("required_count", 1))),
                    "allocatedHours": float(a.get("allocatedHours", a.get("allocated_hours", 0.0))),
                    "workloadPercentage": float(a.get("workloadPercentage", a.get("workload_percentage", 0.0))),
                    "recommendedCount": float(a.get("recommendedCount", a.get("recommended_count", 0.0))),
                    "metadata": a.get("metadata")
                }
            )
        try:
            from app.services.schedule_engine import generate_schedule
            await generate_schedule(project_id)
        except Exception:
            pass



async def ensure_backup(project_id: str, section_type: str, section_key: str):
    # Check if backup already exists in database
    existing = await db.projectbackup.find_unique(
        where={
            "projectId_section_key": {
                "projectId": project_id,
                "section": section_type,
                "key": section_key
            }
        }
    )
    if existing:
        return

    original_val = None
    if section_type == "intelligence":
        intel = await db.projectintelligence.find_unique(where={"projectId": project_id})
        if intel and hasattr(intel, section_key):
            val = getattr(intel, section_key)
            original_val = json.dumps(val, ensure_ascii=False) if val is not None else None
    elif section_type == "documentation":
        try:
            pdoc = await db.plandoc.find_unique(where={"id": section_key})
            if pdoc:
                original_val = pdoc.content
        except Exception:
            pass
        if original_val is None:
            try:
                doc = await db.doc.find_unique(where={"id": section_key})
                if doc:
                    original_val = doc.content
            except Exception:
                pass
    elif section_type == "wbs":
        tasks = await db.task.find_many(where={"projectId": project_id}, order={"order": "asc"})
        groups = await db.groupingtask.find_many(where={"projectId": project_id}, order={"order": "asc"})
        
        tasks_list = []
        for t in tasks:
            t_dict = {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "assignee": t.assignee,
                "assigneeAvatar": t.assigneeAvatar,
                "priority": t.priority,
                "dueDate": t.dueDate.isoformat() if t.dueDate else None,
                "employeeId": t.employeeId,
                "tags": t.tags,
                "status": t.status,
                "groupId": t.groupId,
                "order": t.order,
                "dependencies": t.dependencies,
                "estimatedHours": t.estimatedHours,
                "complexityScore": t.complexityScore,
                "effortHours": t.effortHours,
                "durationDays": t.durationDays,
                "scheduledStart": t.scheduledStart.isoformat() if t.scheduledStart else None,
                "scheduledEnd": t.scheduledEnd.isoformat() if t.scheduledEnd else None,
                "featureRequestId": t.featureRequestId,
            }
            tasks_list.append(t_dict)
            
        groups_list = []
        for g in groups:
            g_dict = {
                "id": g.id,
                "name": g.name,
                "order": g.order,
                "totalEffortHours": g.totalEffortHours,
                "description": g.description,
                "projectId": g.projectId,
            }
            groups_list.append(g_dict)
            
        original_val = json.dumps({"tasks": tasks_list, "groups": groups_list}, ensure_ascii=False)
        
    elif section_type == "resources":
        allocs = await db.resourceallocation.find_many(where={"projectId": project_id})
        allocs_list = []
        for a in allocs:
            a_dict = {
                "id": a.id,
                "role": a.role,
                "requiredCount": a.requiredCount,
                "allocatedHours": a.allocatedHours,
                "workloadPercentage": a.workloadPercentage,
                "recommendedCount": a.recommendedCount,
                "metadata": a.metadata
            }
            allocs_list.append(a_dict)
        original_val = json.dumps(allocs_list, ensure_ascii=False)

    if original_val is not None:
        try:
            await db.projectbackup.create(
                data={
                    "projectId": project_id,
                    "section": section_type,
                    "key": section_key,
                    "content": original_val
                }
            )
        except Exception as e:
            logger.error("Failed to write database backup: %s", e)


class SectionEnhanceRequest(BaseModel):
    section_type: str  # 'intelligence', 'wbs', 'schedule', 'resources', 'risks', 'documentation'
    section_key: str   # field name or database row ID
    user_instruction: str
    current_content: str


@router.post("")
async def enhance_section(project_id: str, body: SectionEnhanceRequest):
    try:
        # 1. Verify project exists
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        # Create backup if not already present
        await ensure_backup(project_id, body.section_type, body.section_key)

        current_content = body.current_content
        if body.section_type == "wbs" and body.section_key == "tasks_and_groups":
            tasks = await db.task.find_many(where={"projectId": project_id}, order={"order": "asc"})
            groups = await db.groupingtask.find_many(where={"projectId": project_id}, order={"order": "asc"})
            grouped_data = []
            for g in groups:
                g_tasks = [t for t in tasks if t.groupId == g.id]
                tasks_list = []
                for t in g_tasks:
                    tasks_list.append({
                        "id": t.id,
                        "title": t.title,
                        "description": t.description,
                        "assignee": t.assignee,
                        "assigneeAvatar": t.assigneeAvatar,
                        "priority": t.priority,
                        "dueDate": t.dueDate.isoformat() if t.dueDate else None,
                        "employeeId": t.employeeId,
                        "tags": t.tags,
                        "status": t.status,
                        "order": t.order,
                        "dependencies": t.dependencies,
                        "estimatedHours": t.estimatedHours,
                        "complexityScore": t.complexityScore,
                        "effortHours": t.effortHours,
                        "durationDays": t.durationDays,
                        "scheduledStart": t.scheduledStart.isoformat() if t.scheduledStart else None,
                        "scheduledEnd": t.scheduledEnd.isoformat() if t.scheduledEnd else None,
                        "featureRequestId": t.featureRequestId
                    })
                grouped_data.append({
                    "id": g.id,
                    "name": g.name,
                    "description": g.description,
                    "totalEffortHours": g.totalEffortHours,
                    "featureIds": g.featureIds,
                    "tasks": tasks_list
                })
            if grouped_data:
                current_content = json.dumps(grouped_data, ensure_ascii=False, indent=2)

        # 2. Intent Analysis & Prompt Enrichment Layer
        enrichment_prompt = (
            f"You are a Prompt Enrichment Assistant. A user wants to edit a section of their project planning report.\n"
            f"Section Type: {body.section_type}\n"
            f"Current Content of Section:\n{current_content}\n\n"
            f"User's Natural Language Instruction: '{body.user_instruction}'\n\n"
            f"Analyze the user's intent. Ambiguous or simple instructions (like 'make it better', 'expand', 'wrap up', 'extend') "
            f"should be translated into concrete, detailed, structured editing objectives (e.g. specific metrics/details to add, "
            f"tone changes, structural adjustments, or timeline expansions/compressions).\n"
            f"Output only the enriched, highly descriptive instructions for the editing model. Do not write the edited content itself."
        )

        enriched_instruction = await ollama_service.generate_document_style(
            system_prompt="You are a professional prompt engineer. You output only the enriched, detailed editing instructions.",
            context=enrichment_prompt
        )
        enriched_instruction = enriched_instruction.strip()

        # 3. Targeted Section Regeneration
        editing_prompt = (
            f"You are an Expert Technical Writer and Project Architect.\n"
            f"You must edit the following section content based on the provided enriched editing instructions.\n\n"
            f"Current Content:\n{current_content}\n\n"
            f"Editing Instructions:\n{enriched_instruction}\n\n"
            f"CRITICAL DIRECTIVES:\n"
            f"- Regenerate ONLY the updated section content. Do not add intro/outro comments or reference the instructions.\n"
            f"- If the original content is JSON, output valid JSON matching the schema.\n"
            f"- Preserve the original formatting style (e.g. Markdown or JSON structure).\n"
        )
        if body.section_type == "wbs" and body.section_key == "tasks_and_groups":
            editing_prompt += (
                "\nCRITICAL: You MUST preserve all existing groups (phases) and their tasks. Do NOT discard, delete, or replace the existing project structure. Instead, augment or modify the tasks within each group to incorporate the changes requested in the editing instructions. Keep all original task structures and IDs intact where possible, and only update the titles, descriptions, and estimated hours to reflect the requested edits.\n"
                "\nJSON SCHEMA REQUIREMENT:\n"
                "You MUST output a strictly valid JSON array of objects representing groups, where each object has this format:\n"
                "[\n"
                "  {\n"
                "    \"id\": \"(string) Keep existing ID if group is unmodified, or generate a new UUID for new groups\",\n"
                "    \"name\": \"(string) Group/Epic Name\",\n"
                "    \"description\": \"(string) Group Description\",\n"
                "    \"tasks\": [\n"
                "      {\n"
                "        \"id\": \"(string) Keep existing ID if task is unmodified, or generate a new UUID for new tasks\",\n"
                "        \"title\": \"(string) Task Title\",\n"
                "        \"description\": \"(string) Task Description\",\n"
                "        \"estimatedHours\": (float) Effort hours,\n"
                "        \"priority\": \"(string) high/medium/low\",\n"
                "        \"status\": \"(string) pending/completed/in_progress\"\n"
                "      }\n"
                "    ]\n"
                "  }\n"
                "]\n"
            )
        elif body.section_type == "resources" and body.section_key == "allocations":
            editing_prompt += (
                "\nJSON SCHEMA REQUIREMENT:\n"
                "You MUST output a strictly valid JSON array of objects representing allocations in this format:\n"
                "[\n"
                "  {\n"
                "    \"id\": \"(string) Keep existing ID if unmodified, or generate a new UUID\",\n"
                "    \"role\": \"(string) Role Name\",\n"
                "    \"requiredCount\": (int) Devs count,\n"
                "    \"allocatedHours\": (float) Total hours,\n"
                "    \"workloadPercentage\": (float) Workload percentage,\n"
                "    \"recommendedCount\": (float) Recommended devs count,\n"
                "    \"metadata\": \"(string) Optional metadata\"\n"
                "  }\n"
                "]\n"
            )
        editing_prompt += "Updated Content:"

        if body.section_type == "wbs" and body.section_key == "tasks_and_groups":
            updated_content = ""
        else:
            updated_content = await ollama_service.generate_document_style(
                system_prompt="You are an expert editor. You only output the updated section content without any conversational chatter.",
                context=editing_prompt
            )
            updated_content = updated_content.strip()

        # 4. Database In-Place Replacement
        if body.section_type == "intelligence":
            # For JSON fields inside ProjectIntelligence
            try:
                json_data = json.loads(updated_content, strict=False)
            except Exception as e:
                logger.error("Failed to parse LLM output as JSON: %s. Output was: %s", e, updated_content)
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="The editing engine failed to output valid JSON. Please try again.",
                )

            # Get current database record to deep merge
            intel = await db.projectintelligence.find_unique(where={"projectId": project_id})
            existing_val = getattr(intel, body.section_key) if intel else None
            if not isinstance(existing_val, dict):
                existing_val = {}

            if isinstance(json_data, dict):
                json_data = deep_merge(existing_val, json_data)

            await db.projectintelligence.update(
                where={"projectId": project_id},
                data={body.section_key: Json(json_data)}
            )

        elif body.section_type == "wbs":
            if body.section_key == "tasks_and_groups":
                try:
                    tasks = await db.task.find_many(where={"projectId": project_id}, order={"order": "asc"})
                    groups = await db.groupingtask.find_many(where={"projectId": project_id}, order={"order": "asc"})
                    
                    # 1. Build WBS context representation for AI analysis
                    wbs_context = {
                        "groups": [
                            {
                                "id": eg.id,
                                "name": eg.name,
                                "description": eg.description,
                                "order": eg.order,
                                "tasks": [
                                    {
                                        "id": t.id,
                                        "title": t.title,
                                        "description": t.description,
                                        "estimatedHours": t.estimatedHours,
                                        "priority": t.priority,
                                        "dependencies": json.loads(t.dependencies) if t.dependencies else [],
                                    }
                                    for t in tasks if t.groupId == eg.id
                                ]
                            }
                            for eg in groups
                        ]
                    }

                    # Ensure backup exists before mutating anything
                    await ensure_backup(project_id, body.section_type, body.section_key)

                    # 2. Query AI to generate selective mutations
                    wbs_diff_prompt = (
                        f"You are an Expert Project Architect and Technical Editor.\n"
                        f"Analyze the current WBS structure and the user's instructions, then output a JSON object describing selective mutations (ADD/UPDATE/DELETE) to perform.\n\n"
                        f"Current WBS Context:\n{json.dumps(wbs_context, indent=2)}\n\n"
                        f"User's Editing Instructions:\n{enriched_instruction}\n\n"
                        f"CRITICAL DIRECTIVES:\n"
                        f"- Output ONLY a valid JSON object matching the schema below. Do not include markdown code blocks or conversational chatter:\n"
                        f"{{\n"
                        f"  \"group_mutations\": [\n"
                        f"    {{ \"action\": \"ADD\"|\"UPDATE\"|\"DELETE\", \"id\": \"group_id\", \"name\": \"...\", \"description\": \"...\", \"order\": 1 }}\n"
                        f"  ],\n"
                        f"  \"task_mutations\": [\n"
                        f"    {{ \"action\": \"ADD\"|\"UPDATE\"|\"DELETE\", \"id\": \"task_id\", \"group_id\": \"group_id\", \"title\": \"...\", \"description\": \"...\", \"estimatedHours\": 4.0, \"priority\": \"medium\", \"dependencies\": [\"dep_task_id\"] }}\n"
                        f"  ]\n"
                        f"}}\n"
                        f"- Group Creation Restraints:\n"
                        f"  - DO NOT add new groups in 'group_mutations' unless the user's instruction explicitly requests to create a new phase/milestone.\n"
                        f"  - All new tasks MUST be mapped to the 'group_id' of the most relevant existing milestone group (for example, if adding logging or deployment tasks, put them under the existing 'Milestone 5: Deployment & Go-Live' group ID).\n"
                        f"  - Do not create custom or descriptive milestone group names like 'Deployment & Go-Live Part: ...'. Reuse the exact existing IDs.\n"
                        f"- Actions:\n"
                        f"  - ADD: Create a new task. Generate a valid random UUID for the new 'id'.\n"
                        f"  - UPDATE: Modify an existing group or task. Specify 'id' and only the fields to change.\n"
                        f"  - DELETE: Remove an existing group or task. Specify only 'id'.\n"
                        f"- DO NOT touch, modify, or output mutations for groups or tasks that are unrelated to the instruction.\n"
                        f"JSON Mutation Output:"
                    )

                    ai_output = await ollama_service.generate_document_style(
                        system_prompt="You are an expert editor. You only output WBS mutation JSON matching the schema.",
                        context=wbs_diff_prompt
                    )
                    ai_output = ai_output.strip()

                    # Extract JSON substring robustly to strip conversational prefixes/suffixes
                    def extract_json_substring(text: str) -> str:
                        start_idx_obj = text.find("{")
                        start_idx_arr = text.find("[")
                        if start_idx_obj != -1 and (start_idx_arr == -1 or start_idx_obj < start_idx_arr):
                            start = start_idx_obj
                            end = text.rfind("}")
                        elif start_idx_arr != -1:
                            start = start_idx_arr
                            end = text.rfind("]")
                        else:
                            start = -1
                            end = -1
                        if start != -1 and end != -1 and end > start:
                            return text[start:end+1]
                        return text

                    cleaned_output = extract_json_substring(ai_output)
                    mutations = json.loads(cleaned_output, strict=False)

                    # Import estimation metrics helpers
                    from app.services.estimation_engine import _compute_task_complexity, _compute_effort

                    # Keep track of group ID mappings to resolve duplicate milestone/group name additions
                    group_id_mapping = {}
                    existing_group_names = {g.name.lower().strip(): g.id for g in groups}

                    # --- Process Group Mutations ---
                    group_muts = mutations.get("group_mutations", [])
                    for gm in group_muts:
                        action = gm.get("action", "").upper()
                        g_id = gm.get("id")
                        if not g_id:
                            continue

                        if action == "DELETE":
                            await db.task.delete_many(where={"groupId": g_id})
                            await db.groupingtask.delete(where={"id": g_id})
                        elif action == "ADD":
                            g_name = gm.get("name") or "New Milestone"
                            normalized_name = g_name.lower().strip()
                            if normalized_name in existing_group_names:
                                # Deduplicate: map the new group ID to the existing group ID
                                group_id_mapping[g_id] = existing_group_names[normalized_name]
                                logger.info("Deduplicated milestone group name '%s' from mutation ADD: mapped %s to existing %s", g_name, g_id, existing_group_names[normalized_name])
                            else:
                                await db.groupingtask.create(
                                    data={
                                        "id": g_id,
                                        "name": g_name,
                                        "description": gm.get("description", ""),
                                        "order": int(gm.get("order", 0)),
                                        "projectId": project_id,
                                    }
                                )
                                existing_group_names[normalized_name] = g_id
                        elif action == "UPDATE":
                            update_data = {}
                            if "name" in gm:
                                update_data["name"] = gm["name"]
                            if "description" in gm:
                                update_data["description"] = gm["description"]
                            if "order" in gm:
                                update_data["order"] = int(gm["order"])

                            if update_data:
                                await db.groupingtask.update(
                                    where={"id": g_id},
                                    data=update_data,
                                )

                    # --- Process Task Mutations ---
                    task_muts = mutations.get("task_mutations", [])
                    for tm in task_muts:
                        action = tm.get("action", "").upper()
                        t_id = tm.get("id")
                        if not t_id:
                            continue

                        if action == "DELETE":
                            await db.task.delete(where={"id": t_id})
                        elif action == "ADD":
                            g_id = tm.get("group_id") or tm.get("groupId")
                            # Map group ID if we deduplicated the parent group name
                            if g_id in group_id_mapping:
                                g_id = group_id_mapping[g_id]

                            if not g_id:
                                if groups:
                                    g_id = groups[0].id
                                else:
                                    continue

                            title = tm.get("title") or "New Task"
                            desc = tm.get("description") or ""
                            est_hours = float(tm.get("estimatedHours") or tm.get("estimated_hours") or 4.0)
                            deps_list = tm.get("dependencies", [])
                            if not isinstance(deps_list, list):
                                deps_list = [deps_list] if deps_list else []

                            complexity = _compute_task_complexity(title, desc, len(deps_list))
                            effort, duration = _compute_effort(est_hours, complexity, 6.0)

                            due_dt = None
                            if tm.get("dueDate"):
                                try:
                                    due_dt = datetime.fromisoformat(tm["dueDate"])
                                except Exception:
                                    pass
                            if not due_dt:
                                due_dt = datetime.now(timezone.utc)

                            await db.task.create(
                                data={
                                    "id": t_id,
                                    "title": title,
                                    "description": desc,
                                    "order": int(tm.get("order", 0)),
                                    "projectId": project_id,
                                    "groupId": g_id,
                                    "priority": tm.get("priority", "medium"),
                                    "status": "pending",
                                    "assignee": tm.get("assignee") or "Developer",
                                    "assigneeAvatar": tm.get("assigneeAvatar"),
                                    "dueDate": due_dt,
                                    "estimatedHours": float(est_hours),
                                    "complexityScore": float(complexity),
                                    "effortHours": float(effort),
                                    "durationDays": float(duration),
                                    "dependencies": json.dumps(deps_list),
                                }
                            )
                        elif action == "UPDATE":
                            update_data = {}
                            if "title" in tm:
                                update_data["title"] = tm["title"]
                            if "description" in tm:
                                update_data["description"] = tm["description"]
                            if "priority" in tm:
                                update_data["priority"] = tm["priority"]
                            if "assignee" in tm:
                                update_data["assignee"] = tm["assignee"]
                            if "assigneeAvatar" in tm:
                                update_data["assigneeAvatar"] = tm["assigneeAvatar"]
                            if "order" in tm:
                                update_data["order"] = int(tm["order"])
                            
                            target_gid = tm.get("group_id") or tm.get("groupId")
                            if target_gid:
                                if target_gid in group_id_mapping:
                                    target_gid = group_id_mapping[target_gid]
                                update_data["groupId"] = target_gid

                            if "dependencies" in tm:
                                deps_list = tm["dependencies"]
                                if not isinstance(deps_list, list):
                                    deps_list = [deps_list] if deps_list else []
                                update_data["dependencies"] = json.dumps(deps_list)

                            existing_task = await db.task.find_unique(where={"id": t_id})
                            if existing_task:
                                title_val = tm.get("title", existing_task.title)
                                desc_val = tm.get("description", existing_task.description or "")
                                hours_val = float(tm.get("estimatedHours", tm.get("estimated_hours", existing_task.estimatedHours or 4.0)))
                                
                                deps_val = tm.get("dependencies", json.loads(existing_task.dependencies) if existing_task.dependencies else [])
                                if not isinstance(deps_val, list):
                                    deps_val = [deps_val] if deps_val else []

                                complexity = _compute_task_complexity(title_val, desc_val, len(deps_val))
                                effort, duration = _compute_effort(hours_val, complexity, 6.0)

                                update_data["estimatedHours"] = hours_val
                                update_data["complexityScore"] = float(complexity)
                                update_data["effortHours"] = float(effort)
                                update_data["durationDays"] = float(duration)

                            if update_data:
                                await db.task.update(
                                    where={"id": t_id},
                                    data=update_data,
                                )

                    # Recalculate total effort and duration for all groups in the project
                    all_project_tasks = await db.task.find_many(where={"projectId": project_id})
                    updated_groups = await db.groupingtask.find_many(where={"projectId": project_id})
                    for g in updated_groups:
                        g_tasks = [t for t in all_project_tasks if t.groupId == g.id]
                        effort_sum = sum(t.effortHours or 0.0 for t in g_tasks)
                        duration_max = max((t.durationDays or 0.0) for t in g_tasks) if g_tasks else 0.0
                        await db.groupingtask.update(
                            where={"id": g.id},
                            data={
                                "totalEffortHours": effort_sum,
                                "totalDurationDays": duration_max
                            }
                        )

                    # Downstream schedule realignment
                    try:
                        from app.services.schedule_engine import generate_schedule
                        await generate_schedule(project_id, reindex=True)
                    except Exception as se:
                        logger.error("Failed to automatically reschedule during WBS diff enhancement: %s", se)

                except Exception as loop_err:
                    import traceback
                    tb_str = traceback.format_exc()
                    logger.error("Error during WBS diff reconstruction: %s\n%s", loop_err, tb_str)
                    await _rollback_backup(project_id, body.section_type, body.section_key)
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"WBS diff enhancement reconstruction failed: {loop_err}. Traceback: {tb_str}. Original WBS has been restored.",
                    )

            else:
                # Update specific Task
                try:
                    # Try to parse if LLM updated multiple task fields in JSON
                    task_data = json.loads(updated_content, strict=False)
                    await db.task.update(
                        where={"id": body.section_key},
                        data={
                            "title": task_data.get("title", task_data.get("name")),
                            "description": task_data.get("description", task_data.get("content")),
                            "estimatedHours": float(task_data.get("estimatedHours", task_data.get("hours", 0))),
                        }
                    )
                except Exception:
                    # Fallback to updating description
                    await db.task.update(
                        where={"id": body.section_key},
                        data={"description": updated_content}
                    )

        elif body.section_type == "schedule":
            # Update sprint GroupingTask
            try:
                group_data = json.loads(updated_content)
                await db.groupingtask.update(
                    where={"id": body.section_key},
                    data={
                        "name": group_data.get("name"),
                        "description": group_data.get("description"),
                        "totalDurationDays": float(group_data.get("totalDurationDays", group_data.get("duration", 0))),
                    }
                )
            except Exception:
                await db.groupingtask.update(
                    where={"id": body.section_key},
                    data={"description": updated_content}
                )

        elif body.section_type == "resources":
            if body.section_key == "allocations":
                try:
                    try:
                        allocs_data = json.loads(updated_content, strict=False)
                    except Exception as e:
                        logger.error("Failed to parse LLM resources output as JSON: %s. Output was: %s", e, updated_content)
                        raise ValueError("The editing engine failed to output valid Resources JSON.")

                    # Purge and recreate allocations
                    await db.resourceallocation.delete_many(where={"projectId": project_id})
                    for a in allocs_data:
                        alloc_id = a.get("id") or str(uuid.uuid4())
                        await db.resourceallocation.create(
                            data={
                                "id": alloc_id,
                                "projectId": project_id,
                                "role": a["role"],
                                "requiredCount": int(a.get("requiredCount", a.get("required_count", 1))),
                                "allocatedHours": float(a.get("allocatedHours", a.get("allocated_hours", 0.0))),
                                "workloadPercentage": float(a.get("workloadPercentage", a.get("workload_percentage", 0.0))),
                                "recommendedCount": float(a.get("recommendedCount", a.get("recommended_count", 0.0))),
                                "metadata": a.get("metadata")
                            }
                        )

                    # Recalculate schedule
                    from app.services.schedule_engine import generate_schedule
                    await generate_schedule(project_id)

                except Exception as loop_err:
                    logger.error("Error during Resources reconstruction: %s. Initiating auto-rollback.", loop_err)
                    await _rollback_backup(project_id, body.section_type, body.section_key)
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Resource allocation enhancement failed: {loop_err}. Original resources have been restored.",
                    )

            else:
                # Update specific ResourceAllocation
                try:
                    res_data = json.loads(updated_content)
                    await db.resourceallocation.update(
                        where={"id": body.section_key},
                        data={
                            "role": res_data.get("role"),
                            "requiredCount": int(res_data.get("requiredCount", res_data.get("count", 1))),
                        }
                    )
                except Exception:
                    await db.resourceallocation.update(
                        where={"id": body.section_key},
                        data={"role": updated_content}
                    )

        elif body.section_type == "risks":
            # Update RiskBlockage
            try:
                risk_data = json.loads(updated_content)
                await db.riskblockage.update(
                    where={"id": body.section_key},
                    data={
                        "title": risk_data.get("title"),
                        "description": risk_data.get("description"),
                        "severity": risk_data.get("severity"),
                    }
                )
            except Exception:
                await db.riskblockage.update(
                    where={"id": body.section_key},
                    data={"description": updated_content}
                )

        elif body.section_type == "documentation":
            # Update PlanDoc or Doc
            try:
                # Check PlanDoc first
                await db.plandoc.update(
                    where={"id": body.section_key},
                    data={"content": updated_content}
                )
            except Exception:
                try:
                    await db.doc.update(
                        where={"id": body.section_key},
                        data={"content": updated_content}
                    )
                except Exception:
                    pass

        # 5. Clear project cache to invalidate stale in-memory states
        clear_project_cache(project_id)
        try:
            from app.db.redis import invalidate_project_cache
            await invalidate_project_cache(project_id)
        except Exception as cache_err:
            logger.warning("Failed to invalidate Redis cache during WBS enhancement: %s", cache_err)

        return {
            "status": "success",
            "enriched_instruction": enriched_instruction,
            "updated_content": updated_content
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Section enhancement failed for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Section enhancement failed: {exc}",
        )


class RevertSectionRequest(BaseModel):
    section_type: str
    section_key: str


@router.post("/revert")
async def revert_section(project_id: str, body: RevertSectionRequest):
    try:
        # Verify project exists
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        backup = await db.projectbackup.find_unique(
            where={
                "projectId_section_key": {
                    "projectId": project_id,
                    "section": body.section_type,
                    "key": body.section_key
                }
            }
        )
        if not backup or not backup.content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No backup found for this specific section.",
            )

        original_val = backup.content

        if body.section_type == "intelligence":
            parsed_json = json.loads(original_val)
            await db.projectintelligence.update(
                where={"projectId": project_id},
                data={body.section_key: Json(parsed_json)}
            )
        elif body.section_type == "documentation":
            try:
                await db.plandoc.update(
                    where={"id": body.section_key},
                    data={"content": original_val}
                )
            except Exception:
                await db.doc.update(
                    where={"id": body.section_key},
                    data={"content": original_val}
                )
        elif body.section_type == "wbs":
            parsed = json.loads(original_val)
            
            # Delete tasks first to prevent foreign key issues
            await db.task.delete_many(where={"projectId": project_id})
            await db.groupingtask.delete_many(where={"projectId": project_id})
            
            # Restore Groups
            for g in parsed.get("groups", []):
                await db.groupingtask.create(
                    data={
                        "id": g["id"],
                        "name": g.get("name") or "Unnamed Phase",
                        "order": int(g.get("order", 0)),
                        "totalEffortHours": float(g.get("totalEffortHours", g.get("total_effort_hours", g.get("effort_hours", 0.0)))),
                        "totalDurationDays": float(g.get("totalDurationDays", g.get("total_duration_days", 0.0))),
                        "description": g.get("description", ""),
                        "projectId": project_id
                    }
                )
                
            # Restore Tasks
            for t in parsed.get("tasks", []):
                due_dt = None
                if t.get("dueDate"):
                    try:
                        due_dt = datetime.fromisoformat(t["dueDate"])
                    except Exception:
                        pass
                if not due_dt:
                    due_dt = datetime.now(timezone.utc)

                start_dt = None
                if t.get("scheduledStart"):
                    try:
                        start_dt = datetime.fromisoformat(t["scheduledStart"])
                    except Exception:
                        pass

                end_dt = None
                if t.get("scheduledEnd"):
                    try:
                        end_dt = datetime.fromisoformat(t["scheduledEnd"])
                    except Exception:
                        pass
                
                await db.task.create(
                    data={
                        "id": t["id"],
                        "title": t.get("title") or t.get("name") or t.get("task_title") or "Unnamed Task",
                        "description": t.get("description") or t.get("content") or t.get("task_description") or "",
                        "assignee": t.get("assignee", "Developer"),
                        "assigneeAvatar": t.get("assigneeAvatar"),
                        "priority": t.get("priority", "medium"),
                        "dueDate": due_dt,
                        "employeeId": t.get("employeeId"),
                        "tags": t.get("tags", []),
                        "status": t.get("status", "pending"),
                        "projectId": project_id,
                        "groupId": t.get("groupId"),
                        "featureRequestId": t.get("featureRequestId"),
                        "order": int(t.get("order", 0)),
                        "dependencies": json.dumps(t.get("dependencies")) if isinstance(t.get("dependencies"), list) else t.get("dependencies"),
                        "estimatedHours": float(t.get("estimatedHours", t.get("estimated_hours", 0.0))),
                        "complexityScore": float(t.get("complexityScore", 0.0)),
                        "effortHours": float(t.get("effortHours", t.get("estimatedHours", 0.0))),
                        "durationDays": float(t.get("durationDays", 0.0)),
                        "scheduledStart": start_dt,
                        "scheduledEnd": end_dt
                    }
                )
                
            # Trigger dynamic reschedule calculation after restoring tasks
            try:
                from app.services.schedule_engine import generate_schedule
                await generate_schedule(project_id, reindex=True)
            except Exception as se:
                logger.error("Failed to automatically reschedule during WBS revert: %s", se)

        elif body.section_type == "resources":
            parsed = json.loads(original_val)
            await db.resourceallocation.delete_many(where={"projectId": project_id})
            for a in parsed:
                await db.resourceallocation.create(
                    data={
                        "id": a["id"],
                        "projectId": project_id,
                        "role": a["role"],
                        "requiredCount": a["requiredCount"],
                        "allocatedHours": a["allocatedHours"],
                        "workloadPercentage": a["workloadPercentage"],
                        "recommendedCount": a["recommendedCount"],
                        "metadata": a["metadata"]
                    }
                )
                
            # Trigger dynamic reschedule calculation after restoring resources
            try:
                from app.services.schedule_engine import generate_schedule
                await generate_schedule(project_id)
            except Exception as se:
                logger.error("Failed to automatically reschedule during resources revert: %s", se)

        clear_project_cache(project_id)
        try:
            from app.db.redis import invalidate_project_cache
            await invalidate_project_cache(project_id)
        except Exception as cache_err:
            logger.warning("Failed to invalidate Redis cache during WBS revert: %s", cache_err)
        return {"status": "success", "message": "Section reverted to original state."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to revert section for project %s: %s", project_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reversion failed: {e}"
        )
