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
import { getProjectGroups } from "./lib/projectGroups";
import { formatTelemetryEpisode, exportSftDataset } from "./ai/dataset-formatter";

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
    res.json({ success: true, data: decisions });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/decisions/approve", async (req: Request, res: Response) => {
  try {
    const { decisionId, approvedBy } = req.body;
    if (!decisionId) return res.status(400).json({ error: "Missing decisionId" });

    const decision = await db.agentDecision.findUnique({ where: { id: decisionId } });
    if (!decision) return res.status(404).json({ error: "Decision not found" });

    await db.agentDecision.update({
      where: { id: decisionId },
      data: { status: "APPROVED", approvedBy: approvedBy || "Admin" },
    });

    await logAction(
      "DECISION_APPROVED",
      `Decision ${decisionId} approved by ${approvedBy || "Admin"}`,
      decision.projectId
    );

    res.json({ success: true, message: "Decision approved and executed" });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/decisions/reject", async (req: Request, res: Response) => {
  try {
    const { decisionId, rejectedBy, reason } = req.body;
    if (!decisionId) return res.status(400).json({ error: "Missing decisionId" });

    await db.agentDecision.update({
      where: { id: decisionId },
      data: { status: "REJECTED", rejectedBy: rejectedBy || "Admin" },
    });

    res.json({ success: true, message: "Decision rejected" });
  } catch (err: any) {
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

