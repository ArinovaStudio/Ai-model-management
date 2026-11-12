import express, { Request, Response } from "express";
import cors from "cors";
import "dotenv/config";
import db from "./lib/client";
import { logger } from "./ai/logger";
import {
  getStealthPolicyAsync,
  setStealthPolicy,
  getUniversalConfig,
  setUniversalConfig,
  initStealthPoliciesFromDb,
} from "./ai/policy";
import { ingestProject } from "./ai/ingestion";
import { logAction } from "./ai/audit";
import { buildProjectContext } from "./ai/context";
import { evaluate } from "./ai/reasoner";
import { execute } from "./ai/executor";
import { LOW_RISK_POLICY } from "./lib/agent-api";
import { getProjectGroups } from "./lib/projectGroups";
import { formatTelemetryEpisode, exportSftDataset } from "./ai/dataset-formatter";
import { triggerActiveAllocations } from "./ai/events";

const app = express();
const PORT = process.env.PORT || process.env.BRAIN_PORT || 5005;

app.use(cors());
app.use(express.json());

// Health Check
app.get("/health", (req: Request, res: Response) => {
  res.json({
    status: "healthy",
    service: "Arinova-AI-BRAIN",
    timestamp: new Date().toISOString(),
  });
});

// ── Stealth Policy Routes ──────────────────────────────────────────────────
app.get("/api/agent/stealth-policy", async (req: Request, res: Response) => {
  try {
    const projectId = req.query.projectId as string;
    const policy = await getStealthPolicyAsync(projectId || null);
    const resolvedPolicy = policy || {
      projectId: projectId || "*",
      isActive: false,
      minConfidenceScore: 0.75,
      modeType: "AUTO",
      selectedTaskIds: [],
      isException: false,
    };
    res.json({
      success: true,
      policy: resolvedPolicy,
      data: resolvedPolicy,
    });
  } catch (err: any) {
    logger.error("[BRAIN-API] Error fetching stealth policy:", err);
    res.status(500).json({ error: err.message || "Failed to fetch stealth policy" });
  }
});

app.post("/api/agent/stealth-policy", async (req: Request, res: Response) => {
  try {
    const { projectId, isActive, minConfidenceScore, modeType, selectedTaskIds } = req.body;
    if (!projectId) {
      return res.status(400).json({ error: "Missing projectId" });
    }

    setStealthPolicy({
      projectId,
      isActive: Boolean(isActive),
      minConfidenceScore: typeof minConfidenceScore === "number" ? minConfidenceScore : 0.75,
      modeType: modeType ?? "AUTO",
      selectedTaskIds: Array.isArray(selectedTaskIds) ? selectedTaskIds : [],
    });

    if (Boolean(isActive)) {
      void triggerActiveAllocations(projectId).catch((err) => {
        logger.warn(`[BRAIN-API] Background active allocation kickoff failed for ${projectId}: ${err?.message}`);
      });
    }

    res.json({ success: true, message: "Stealth policy updated successfully" });
  } catch (err: any) {
    logger.error("[BRAIN-API] Error setting stealth policy:", err);
    res.status(500).json({ error: err.message || "Failed to update stealth policy" });
  }
});

// ── Stealth Select Queue (Manual Candidate Pre-Approval) ───────────────────
app.get("/api/agent/stealth-select-queue", async (req: Request, res: Response) => {
  try {
    const projectId = req.query.projectId as string;
    if (!projectId) {
      return res.status(400).json({ success: false, error: "projectId parameter is required" });
    }

    const allTasks = await db.task.findMany({
      where: { projectId },
      select: {
        id: true,
        title: true,
        priority: true,
        status: true,
        dueDate: true,
        tags: true,
        userId: true,
        createdAt: true,
      },
    });

    if (allTasks.length === 0) {
      return res.json({ success: true, tasks: [] });
    }

    const allDeps = await db.taskDependency.findMany({
      where: { projectId },
      select: {
        taskId: true,
        dependsOnTaskId: true,
      },
    });

    const groups = getProjectGroups(allTasks, allDeps);
    const flatSdlcTasks: any[] = [];
    for (const group of groups) {
      for (const t of group.tasks) {
        flatSdlcTasks.push({ task: t, slotName: group.name });
      }
    }

    const rawCandidates: any[] = [];
    for (const item of flatSdlcTasks) {
      const t = item.task;
      const isUnassigned = !t.userId && t.status !== "completed" && t.status !== "done" && t.status !== "assigned";
      if (isUnassigned) {
        rawCandidates.push(item);
        if (rawCandidates.length >= 8) break;
      }
    }

    if (rawCandidates.length === 0) {
      return res.json({ success: true, tasks: [] });
    }

    const context = await buildProjectContext(projectId);
    const candidateTasks = [];

    for (const item of rawCandidates) {
      const t = item.task;
      let score = 0.85;
      let recommendedAssignee = null;
      try {
        const round = await evaluate(context, { taskId: t.id, runLlm: false });
        const topRec = round.assignment?.ranked[0] ?? null;
        if (topRec) {
          score = topRec.score;
          recommendedAssignee = topRec.name;
        }
      } catch {
        // Fallback to default score if context is transiently sparse
      }

      candidateTasks.push({
        id: t.id,
        title: t.title,
        priority: t.priority,
        dueDate: t.dueDate ? (t.dueDate instanceof Date ? t.dueDate.toISOString() : t.dueDate) : null,
        tags: t.tags ?? [],
        isReady: true,
        slotName: item.slotName,
        score,
        recommendedAssignee,
      });
    }

    res.json({ success: true, tasks: candidateTasks });
  } catch (error: any) {
    logger.error("[BRAIN-API] Error fetching stealth select queue:", error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Failed to fetch select queue",
    });
  }
});

// ── Universal Stealth Mode Routes ───────────────────────────────────────────
app.get("/api/agent/universal-stealth", (req: Request, res: Response) => {
  try {
    const config = getUniversalConfig();
    res.json({ success: true, config, data: config });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/agent/universal-stealth", (req: Request, res: Response) => {
  try {
    const { isUniversalActive, globalMinConfidence, exceptionProjectIds } = req.body;
    setUniversalConfig({
      isUniversalActive: Boolean(isUniversalActive),
      globalMinConfidence: typeof globalMinConfidence === "number" ? globalMinConfidence : 0.75,
      exceptionProjectIds: Array.isArray(exceptionProjectIds) ? exceptionProjectIds : [],
    });
    res.json({ success: true, message: "Universal stealth configuration updated" });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Decisions Endpoints ─────────────────────────────────────────────────────
app.get("/api/decisions", async (req: Request, res: Response) => {
  try {
    const projectId = req.query.projectId as string | undefined;
    const where: any = {};
    if (projectId) where.projectId = projectId;

    const decisions = await db.agentDecision.findMany({
      where,
      orderBy: { createdAt: "desc" },
      take: 100,
    });

    const decisionsWithTaskInfo = await Promise.all(
      decisions.map(async (dec) => {
        const basisObj = (typeof dec.basis === "string" ? JSON.parse(dec.basis) : dec.basis) as any;
        let taskTitle = "";
        let projectName = "";

        const targetTaskId = basisObj?.taskId ?? basisObj?.evidence?.[0]?.id;
        if (targetTaskId) {
          const task = await db.task.findUnique({
            where: { id: targetTaskId },
            select: { title: true, Project: { select: { name: true } } },
          });
          if (task) {
            taskTitle = task.title;
            projectName = task.Project?.name || "";
          }
        }

        if (!projectName && dec.projectId) {
          const proj = await db.project.findUnique({
            where: { id: dec.projectId },
            select: { name: true },
          });
          if (proj) projectName = proj.name;
        }

        return {
          ...dec,
          taskTitle: taskTitle || "Task Assignment Recommendation",
          projectName,
        };
      })
    );

    res.json({ success: true, decisions: decisionsWithTaskInfo, data: decisionsWithTaskInfo });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/decisions/approve", async (req: Request, res: Response) => {
  try {
    const decisionId = req.body.decisionId || req.body.id;
    const overrideUserId = req.body.overrideUserId;
    const approvedBy = req.body.approvedBy || "Admin";

    if (!decisionId) return res.status(400).json({ error: "Decision ID is required" });

    const decision = await db.agentDecision.findUnique({ where: { id: decisionId } });
    if (!decision) return res.status(404).json({ error: "Decision not found" });

    const basis = (typeof decision.basis === "string" ? JSON.parse(decision.basis) : decision.basis) as any;
    const action = basis?.action;
    const taskId = basis?.taskId;

    if (taskId) {
      if (action === "update_task_status") {
        await db.task.update({
          where: { id: taskId },
          data: { status: basis.status },
        });
      } else if (action === "update_task_progress") {
        await db.task.update({
          where: { id: taskId },
          data: { progress: basis.progress } as any,
        });
      } else if (action === "assign_task") {
        let assigneeName = "Project Agent";
        let targetUserId = "agent-service-001";

        if (overrideUserId) {
          const user = await db.user.findUnique({
            where: { id: overrideUserId },
          });
          if (user) {
            targetUserId = user.id;
            assigneeName = user.name;
          }
        } else {
          const match = decision.decision.match(/to (.*?) \(score/);
          if (match && match[1]) {
            assigneeName = match[1];
            const user = await db.user.findFirst({
              where: { name: { equals: assigneeName, mode: "insensitive" } },
            });
            if (user) {
              targetUserId = user.id;
            }
          }
        }

        await db.task.update({
          where: { id: taskId },
          data: {
            userId: targetUserId,
            assignee: assigneeName,
            status: "assigned",
          },
        });
      }

      const updatedTask = await db.task.findUnique({
        where: { id: taskId },
        select: { id: true, status: true, projectId: true },
      });

      if (updatedTask && updatedTask.status) {
        await db.$executeRawUnsafe(
          `NOTIFY agent_task_completed, '${JSON.stringify({
            id: `evt-task-${Date.now()}`,
            taskId: updatedTask.id,
            projectId: updatedTask.projectId,
          })}'`
        ).catch((err) => logger.error("Failed to notify agent_task_completed:", err));
      }
    }

    await db.agentDecision.delete({
      where: { id: decisionId },
    }).catch(async () => {
      await db.agentDecision.update({
        where: { id: decisionId },
        data: { status: "APPROVED", approvedBy },
      });
    });

    await logAction(
      "DECISION_APPROVED",
      `Decision ${decisionId} approved by ${approvedBy}`,
      decision.projectId
    );

    res.json({ success: true, message: "Decision approved and executed successfully" });
  } catch (err: any) {
    logger.error("[BRAIN-API] Error approving decision:", err);
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/decisions/reject", async (req: Request, res: Response) => {
  try {
    const decisionId = req.body.decisionId || req.body.id;
    const rejectedBy = req.body.rejectedBy || "Admin";

    if (!decisionId) return res.status(400).json({ error: "Decision ID is required" });

    await db.agentDecision.delete({
      where: { id: decisionId },
    }).catch(async () => {
      await db.agentDecision.update({
        where: { id: decisionId },
        data: { status: "REJECTED", rejectedBy },
      });
    });

    res.json({ success: true, message: "Decision rejected successfully" });
  } catch (err: any) {
    logger.error("[BRAIN-API] Error rejecting decision:", err);
    res.status(500).json({ error: err.message });
  }
});

// ── Project Ingestion Pipeline ──────────────────────────────────────────────
app.post("/api/project/ingest", async (req: Request, res: Response) => {
  try {
    const { source_project_id, new_name, members } = req.body;
    if (!source_project_id || !new_name) {
      return res.status(400).json({ error: "Missing source_project_id or new_name" });
    }

    const source = await db.project.findUnique({
      where: { id: source_project_id },
      select: { id: true, name: true },
    });
    if (!source) {
      return res.status(404).json({ error: `Source project "${source_project_id}" does not exist.` });
    }

    const ingested = await ingestProject(source_project_id, new_name);

    if (Array.isArray(members) && members.length > 0) {
      for (const userId of members) {
        const exists = await db.projectMember.findFirst({
          where: { projectId: ingested.targetProjectId, userId },
        });
        if (!exists) {
          await db.projectMember.create({
            data: {
              projectId: ingested.targetProjectId,
              userId,
              role: "Developer",
            },
          });
        }
      }
    }

    await logAction(
      "PROJECT_INGESTED",
      `Project "${source.name}" ingested as "${new_name}" (${ingested.records.tasks} tasks cloned).`,
      ingested.targetProjectId
    );

    void triggerActiveAllocations(ingested.targetProjectId).catch((err) => {
      logger.warn(`[BRAIN-API] Background active allocation kickoff failed for ingested project ${ingested.targetProjectId}: ${err?.message}`);
    });

    res.json({
      success: true,
      message: `Project cloned successfully into target ID: ${ingested.targetProjectId}`,
      data: ingested,
    });
  } catch (err: any) {
    logger.error("[BRAIN-API] Ingest failed:", err);
    res.status(500).json({ error: err.message || "Failed to ingest project" });
  }
});

// ── Agent Summary, Blocker & Update Endpoints ───────────────────────────────
app.get("/api/agent/summary", async (req: Request, res: Response) => {
  try {
    const projectId = req.query.projectId as string;
    const includeRisks = req.query.includeRisks === "true";

    if (!projectId) {
      return res.status(400).json({ success: false, error: "projectId parameter is required" });
    }

    const context = await buildProjectContext(projectId);
    const evaluation = await evaluate(context, { runLlm: includeRisks });

    const risk = evaluation.risk;
    const project = context.project;
    const deadline = project.deadline;
    const deadlineOverdue = project.deadlineOverdue;
    const overdueTasks = context.risks.overdueTaskCount;
    const blockedTasks = context.risks.blockedTaskCount;
    const activeTaskCount = context.risks.activeTaskCount;

    let overallStatus: "on_track" | "at_risk" | "delayed";
    if (deadlineOverdue) {
      overallStatus = "delayed";
    } else if (risk.level === "HIGH" || overdueTasks > 0 || blockedTasks > 0) {
      overallStatus = "at_risk";
    } else {
      overallStatus = "on_track";
    }

    const concerns = risk.concerns.map((concern) => ({
      id: concern.id,
      label: concern.label,
      severity: concern.severity,
    }));
    const activeWork = context.activeTasks.map((task) => ({
      id: task.id,
      title: task.title,
      status: task.status,
      priority: task.priority,
      assignee: task.assignee,
      dueDate: task.dueDate,
      isBlocked: task.isBlocked,
    }));
    const risks = {
      level: risk.level,
      confidence: risk.confidence.score,
      progress: risk.progress,
      concerns,
      activeTaskCount,
      blockedTaskCount: blockedTasks,
      overdueTaskCount: overdueTasks,
      topBlockers: context.risks.topRiskBlockers.map((blocker) => ({
        id: blocker.id,
        title: blocker.title,
        severity: blocker.severity,
        status: blocker.status,
      })),
    };
    const recommendations =
      evaluation.riskNarrative?.recommendations ??
      (risk.concerns.length > 0
        ? risk.concerns.map((concern) => `Address ${concern.label} (${concern.severity} severity).`)
        : ["No immediate concerns; continue current execution plan."]);

    const summary =
      `"${project.name}" is ${overallStatus === "on_track" ? "on track" : overallStatus === "at_risk" ? "at risk" : "delayed"}: ` +
      `${activeTaskCount} active task(s), ${blockedTasks} blocked, ${overdueTasks} overdue, ` +
      `${project.progress !== null ? `${project.progress}%` : "n/a"} progress` +
      `${deadline ? ` against a deadline of ${deadline}` : ""}. ` +
      `Risk is ${risk.level}. ${risk.explanation}`;

    res.json({
      success: true,
      data: {
        projectName: project.name,
        overallStatus,
        progress: project.progress,
        deadline,
        summary,
        activeWork,
        risks,
        recommendations,
      },
    });
  } catch (error: any) {
    logger.error("[BRAIN-API] Summary error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to generate summary" });
  }
});

app.post("/api/agent/blocker", async (req: Request, res: Response) => {
  try {
    const { projectId, taskId, blockerTitle, blockerDescription, blockedByTaskId, blockedByUserId, agentId } = req.body;
    if (!projectId || !blockerTitle) {
      return res.status(400).json({ success: false, error: "Missing projectId or blockerTitle" });
    }

    const payload: Record<string, unknown> = {
      clientId: "agent-api",
      title: blockerTitle,
      riskTitle: blockerTitle,
      severity: "medium",
      description: blockerDescription || "Blocker reported via API.",
      category: "manual",
    };
    if (blockedByTaskId) payload.blockedByTaskId = blockedByTaskId;
    if (blockedByUserId) payload.blockedByUserId = blockedByUserId;

    const result = await execute("record_blocker", {
      agentId: agentId || "agent-service-001",
      projectId,
      taskId,
      payload,
      policy: LOW_RISK_POLICY,
    });

    res.json({ success: result.success, data: result });
  } catch (error: any) {
    logger.error("[BRAIN-API] Blocker error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to record blocker" });
  }
});

app.post("/api/agent/request-update", async (req: Request, res: Response) => {
  try {
    const { projectId, userIds, agentId } = req.body;
    if (!projectId || !Array.isArray(userIds) || userIds.length === 0) {
      return res.status(400).json({ success: false, error: "Missing projectId or userIds" });
    }

    const project = await db.project.findUnique({ where: { id: projectId }, select: { name: true } });
    let queued = 0;
    for (const userId of userIds) {
      const title = `Status update requested: ${project?.name ?? "project"}`;
      const message = `Please reply with an update on your tasks in "${project?.name ?? "project"}": progress %, status, blockers, and next steps.`;
      const result = await execute("send_notification", {
        agentId: agentId || "agent-service-001",
        projectId,
        targetUserId: userId,
        payload: { title, message },
        policy: LOW_RISK_POLICY,
      });
      if (result.success) queued++;
    }

    res.json({ success: true, data: { messagesQueued: queued } });
  } catch (error: any) {
    logger.error("[BRAIN-API] Request-update error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to request update" });
  }
});

// ── SFT Telemetry Dataset Export ────────────────────────────────────────────
app.get("/api/agent/export-dataset", async (req: Request, res: Response) => {
  try {
    const dataset = await exportSftDataset();
    res.setHeader("Content-Type", "application/x-jsonlines");
    res.setHeader("Content-Disposition", `attachment; filename=sft_dataset_${Date.now()}.jsonl`);
    res.send(dataset);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// Initialize persisted policies from DB and start server
void initStealthPoliciesFromDb().then(() => {
  app.listen(PORT, () => {
    logger.info(`🚀 [Arinova-AI-BRAIN] Autonomous Intelligence Microservice running on port ${PORT}`);
  });
});

