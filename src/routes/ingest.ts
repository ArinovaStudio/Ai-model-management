import { NextRequest } from "next/server";
import { z } from "zod";
import db from "@/lib/client";
import { logAction } from "@/ai/audit";
import { ingestProject } from "@/ai/ingestion";
import {
  ApiError,
  authedUser,
  errorResponse,
  formatError,
  ok,
  requireAdmin,
} from "@/lib/agent-api";

const bodySchema = z.object({
  source_project_id: z.string().trim().min(1),
  new_name: z.string().trim().min(1).max(120),
  members: z.array(z.string()).optional(),
});

export async function POST(req: NextRequest) {
  try {
    const auth = await authedUser(req);
    requireAdmin(auth.role);

    let raw: unknown;
    try {
      raw = await req.json();
    } catch {
      throw new ApiError(400, "INVALID_JSON", "Request body must be valid JSON.");
    }

    const parsed = bodySchema.safeParse(raw);
    if (!parsed.success) {
      throw new ApiError(
        400,
        "VALIDATION_ERROR",
        "Invalid request body.",
        parsed.error.issues.map((issue) => `${issue.path.join(".")}: ${issue.message}`)
      );
    }
    const { source_project_id, new_name, members } = parsed.data;

    const source = await db.project.findUnique({
      where: { id: source_project_id },
      select: { id: true, name: true },
    });
    if (!source) {
      throw new ApiError(404, "PROJECT_NOT_FOUND", `Source project "${source_project_id}" does not exist.`);
    }
    if (source.name === new_name) {
      throw new ApiError(400, "NAME_UNCHANGED", "new_name must differ from the source project name.");
    }

    let ingested;
    try {
      ingested = await ingestProject(source_project_id, new_name);
      
      // Associate selected members with the new project
      if (parsed.data.members && parsed.data.members.length > 0) {
        for (const userId of parsed.data.members) {
          const exists = await db.projectMember.findFirst({
            where: { projectId: ingested.targetProjectId, userId }
          });
          if (!exists) {
            await db.projectMember.create({
              data: {
                projectId: ingested.targetProjectId,
                userId,
                role: "Developer",
                isLeader: false
              }
            });
          }
        }
      }
    } catch (error) {
      throw new ApiError(
        500,
        "INGESTION_FAILED",
        `Project ingest failed and was rolled back: ${formatError(error)}`
      );
    }

    await logAction(
      auth.id,
      "project_ingested",
      source_project_id,
      {
        sourceProjectId: source_project_id,
        targetProjectId: ingested.targetProjectId,
        newName: new_name,
        requestedBy: auth.name,
      },
      true,
      { records: ingested.records, sections: ingested.sections },
      `Project "${source.name}" ingested as "${new_name}" (${ingested.records.tasks} tasks, ${ingested.records.taskDependencies} dependencies, ${ingested.records.planDocs} plan docs).`,
      ingested.targetProjectId
    );

    // Notify the AI daemon via native Postgres NOTIFY
    await db.$executeRawUnsafe(
      `NOTIFY agent_project_ingested, '${JSON.stringify({
        id: `evt-ingest-${Date.now()}`,
        projectId: ingested.targetProjectId,
      })}'`
    ).catch(err => console.error("Failed to notify agent_project_ingested:", err));

    // Direct fallback trigger for guaranteed allocation recommendations
    if (ingested.targetProjectId) {
      import("@/ai/events").then(m => m.triggerActiveAllocations(ingested.targetProjectId)).catch(err => {
        console.error("Failed to trigger allocations after project ingestion:", err);
      });
    }

    return ok({
      data: {
        target_project_id: ingested.targetProjectId,
        status: "cloned",
        clonedSections: {
          wbsSchedule: ingested.sections.wbsSchedule,
          timeline: ingested.sections.timeline,
          resources: ingested.sections.resources,
          gaps: ingested.sections.gaps,
          productStrategy: ingested.sections.productStrategy,
          technicalArchitecture: ingested.sections.technicalArchitecture,
          documentation: ingested.sections.documentation,
        },
        records: ingested.records,
        durationMs: ingested.durationMs,
        warnings: ingested.warnings,
      },
    });
  } catch (error) {
    if (error instanceof ApiError) {
      await logAction(
        "agent-api",
        "project_ingested",
        null,
        {},
        false,
        { error: error.code },
        error.message,
        null
      );
      return errorResponse(error);
    }
    await logAction(
      "agent-api",
      "project_ingested",
      null,
      {},
      false,
      { error: formatError(error) },
      "Project ingest failed.",
      null
    );
    return errorResponse(error);
  }
}
