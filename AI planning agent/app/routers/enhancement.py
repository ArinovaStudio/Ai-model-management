import json
import logging
import os
from datetime import datetime
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
        tasks = await db.task.find_many(where={"projectId": project_id})
        groups = await db.groupingtask.find_many(where={"projectId": project_id})
        
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
            }
            tasks_list.append(t_dict)
            
        groups_list = []
        for g in groups:
            g_dict = {
                "id": g.id,
                "name": g.name,
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

        # 2. Intent Analysis & Prompt Enrichment Layer
        enrichment_prompt = (
            f"You are a Prompt Enrichment Assistant. A user wants to edit a section of their project planning report.\n"
            f"Section Type: {body.section_type}\n"
            f"Current Content of Section:\n{body.current_content}\n\n"
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
            f"Current Content:\n{body.current_content}\n\n"
            f"Editing Instructions:\n{enriched_instruction}\n\n"
            f"CRITICAL DIRECTIVES:\n"
            f"- Regenerate ONLY the updated section content. Do not add intro/outro comments or reference the instructions.\n"
            f"- If the original content is JSON, output valid JSON matching the schema.\n"
            f"- Preserve the original formatting style (e.g. Markdown or JSON structure).\n"
            f"Updated Content:"
        )

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
            # Update ResourceAllocation
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
                        "name": g["name"],
                        "totalEffortHours": g.get("totalEffortHours", g.get("effort_hours", 0.0)),
                        "description": g.get("description", ""),
                        "projectId": project_id
                    }
                )
            
            # Restore Tasks
            for t in parsed.get("tasks", []):
                due_dt = datetime.fromisoformat(t["dueDate"]) if t["dueDate"] else None
                start_dt = datetime.fromisoformat(t["scheduledStart"]) if t["scheduledStart"] else None
                end_dt = datetime.fromisoformat(t["scheduledEnd"]) if t["scheduledEnd"] else None
                
                await db.task.create(
                    data={
                        "id": t["id"],
                        "title": t["title"],
                        "description": t["description"],
                        "assignee": t["assignee"],
                        "assigneeAvatar": t.get("assigneeAvatar"),
                        "priority": t["priority"],
                        "dueDate": due_dt,
                        "employeeId": t.get("employeeId"),
                        "tags": t.get("tags", []),
                        "status": t["status"],
                        "projectId": project_id,
                        "groupId": t.get("groupId"),
                        "order": t.get("order", 0),
                        "dependencies": t.get("dependencies"),
                        "estimatedHours": t.get("estimatedHours"),
                        "complexityScore": t.get("complexityScore", 0.0),
                        "effortHours": t.get("effortHours", 0.0),
                        "durationDays": t.get("durationDays", 0.0),
                        "scheduledStart": start_dt,
                        "scheduledEnd": end_dt
                    }
                )
                
            # Trigger dynamic reschedule calculation after restoring tasks
            try:
                from app.services.schedule_engine import generate_schedule
                await generate_schedule(project_id)
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
        return {"status": "success", "message": "Section reverted to original state."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to revert section for project %s: %s", project_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reversion failed: {e}"
        )
