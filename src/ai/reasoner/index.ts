import db from "@/lib/client";
import { logger } from "@/ai/logger";
import { circuitBreaker } from "@/ai/circuit-breaker";
import type { ProjectContext } from "@/ai/context/types";
import { startSpan } from "@/ai/telemetry";
import { taskReadiness, riskScore, assignmentScore } from "./scoring";
import { assessRiskWithLlm } from "./ollama";
import { getProjectGroups } from "../../lib/projectGroups";
import type {
  AssignmentCandidate,
  AssignmentScore,
  ReasonerEvaluation,
  ReasonerOptions,
  TaskReadiness,
  RiskNarrative,
} from "./types";

async function determineTaskReadiness(
  taskId: string,
  taskStatus: string,
  projectId: string
): Promise<TaskReadiness> {
  // 1. Fetch tasks and dependencies in parallel
  const [allProjectTasks, edges] = await Promise.all([
    db.task.findMany({
      where: { projectId },
      select: { id: true, status: true, title: true }
    }),
    db.taskDependency.findMany({
      where: { projectId },
      select: { taskId: true, dependsOnTaskId: true, relationType: true }
    })
  ]);

  const getTaskPhase = (tTitle: string) => {
    const tTitleLower = tTitle.toLowerCase();
    const isQA = /(deploy|go-live|qa|testing|verification|audit|monitoring)/i.test(tTitleLower);
    if (isQA) return 3;
    const isBase = /(infrastructure|baseline|setup|initialize)/i.test(tTitleLower);
    if (isBase) return 1;
    return 2;
  };

  const taskMap = new Map(allProjectTasks.map(t => [t.id, t]));
  // Filter out reversed dependencies (e.g. Feature task depending on a Testing/Deployment task)
  const cleanEdges = edges.filter(edge => {
    const task = taskMap.get(edge.taskId);
    const parent = taskMap.get(edge.dependsOnTaskId);
    if (task && parent) {
      const taskPhase = getTaskPhase(task.title);
      const parentPhase = getTaskPhase(parent.title);
      if (taskPhase === 2 && parentPhase === 3) {
        logger.info(`[REASONER] Ignoring reversed dependency: Feature "${task.title}" depends on QA/Testing "${parent.title}"`);
        return false;
      }
    }
    return true;
  });

  // 2. Enforce Active Group Chunk constraint
  const groups = getProjectGroups(allProjectTasks, cleanEdges);
  const activeGroup = groups.find(g => g.status === "ACTIVE") || groups[0];
  if (activeGroup) {
    const isTaskInActiveGroup = activeGroup.tasks.some(t => t.id === taskId);
    if (!isTaskInActiveGroup) {
      return {
        taskId,
        ready: false,
        reason: `blocked because task does not belong to the currently active logical group: "${activeGroup.name}".`,
        unresolvedDependencies: [],
        hasCycle: false
      };
    }
  }

  // 3. Enforce SDLC Phase-Gate constraints dynamically in software (using task titles/groups)
  const targetTask = allProjectTasks.find(t => t.id === taskId);
  if (targetTask) {
    const title = targetTask.title;
    const groupMatch = title.match(/^\[(.*?)\]/);
    const taskGroup = groupMatch ? groupMatch[1].toLowerCase() : "";

    const TERMINAL_STATUSES = new Set(["completed", "done", "cancelled", "canceled"]);
    const isTerminalStatus = (s: string) => TERMINAL_STATUSES.has(s.toLowerCase());

    // Phase 3: QA/Deployment/Go-Live
    const isPhase3 = /(deploy|go-live|qa|testing|verification|audit|monitoring)/i.test(taskGroup) || 
                     /(deploy|go-live|qa|testing|verification|audit|monitoring)/i.test(title);
                     
    if (isPhase3) {
      // Check if there are any incomplete Phase 2 (Feature Development) tasks
      const incompleteFeatures = allProjectTasks.filter(t => {
        const tGroupMatch = t.title.match(/^\[(.*?)\]/);
        const tGroup = tGroupMatch ? tGroupMatch[1].toLowerCase() : "";
        const isFeature = /feature/i.test(tGroup) || /feature/i.test(t.title);
        const isQAOrTesting = /(deploy|go-live|qa|testing|verification|audit|monitoring)/i.test(t.title) ||
                              /(deploy|go-live|qa|testing|verification|audit|monitoring)/i.test(tGroup);
        return isFeature && !isQAOrTesting && !isTerminalStatus(t.status);
      });

      if (incompleteFeatures.length > 0) {
        return {
          taskId,
          ready: false,
          reason: `blocked by SDLC Phase-Gate constraint: ${incompleteFeatures.length} feature development task(s) are incomplete.`,
          unresolvedDependencies: incompleteFeatures.map(f => f.id),
          hasCycle: false
        };
      }
    }

    // Phase 2: Feature Development
    const isPhase2 = /feature/i.test(taskGroup) || /feature/i.test(title);
    if (isPhase2) {
      // Check if there are any incomplete Phase 1 (Baseline/Infrastructure) tasks
      const incompleteBaselines = allProjectTasks.filter(t => {
        const tGroupMatch = t.title.match(/^\[(.*?)\]/);
        const tGroup = tGroupMatch ? tGroupMatch[1].toLowerCase() : "";
        const isBaseline = /(infrastructure|baseline|setup|initialize)/i.test(tGroup) || 
                           /(infrastructure|baseline|setup|initialize)/i.test(t.title);
        const isQAOrDeploy = /(deploy|go-live|qa|testing|verification|audit|monitoring)/i.test(t.title);
        return isBaseline && !isQAOrDeploy && !isTerminalStatus(t.status);
      });

      if (incompleteBaselines.length > 0) {
        return {
          taskId,
          ready: false,
          reason: `blocked by SDLC Phase-Gate constraint: ${incompleteBaselines.length} infrastructure/baseline task(s) are incomplete.`,
          unresolvedDependencies: incompleteBaselines.map(b => b.id),
          hasCycle: false
        };
      }
    }
  }

  // 4. Fallback to standard task-to-task dependency check
  const targetIds = new Set<string>([taskId]);
  for (const edge of cleanEdges) {
    targetIds.add(edge.taskId);
    targetIds.add(edge.dependsOnTaskId);
  }

  const statusRows = await db.task.findMany({
    where: { id: { in: [...targetIds] } },
    select: { id: true, status: true },
  });

  const statusById = new Map<string, string>();
  for (const status of statusRows) statusById.set(status.id, status.status);
  statusById.set(taskId, taskStatus);

  return taskReadiness(taskId, cleanEdges, statusById);
}

async function scoreCandidates(
  taskId: string,
  taskTitle: string,
  taskPriority: string,
  taskTags: string[],
  projectId: string,
  context: ProjectContext,
  candidateIds: string[]
): Promise<AssignmentScore[]> {
  const [memberships, users] = await Promise.all([
    db.projectMember.findMany({
      where: { projectId },
      select: { userId: true, role: true, isLeader: true },
    }),
    db.user.findMany({
      where: { id: { in: candidateIds } },
      select: {
        id: true,
        name: true,
        workingAs: true,
        department: true,
        bio: true,
        subRole: { select: { name: true } },
      },
    }),
  ]);

  const memberIds = new Set(memberships.map((member) => member.userId));
  const roleById = new Map<string, { role: string | null; isLeader: boolean }>();
  for (const member of memberships) {
    roleById.set(member.userId, { role: member.role, isLeader: member.isLeader });
  }
  const capacityByUserId = new Map<string, { workloadPercent: number; pendingEffortHours?: number; workloadDensity?: number }>();
  for (const entry of context.teamCapacity) {
    capacityByUserId.set(entry.userId, {
      workloadPercent: entry.workloadPercent,
      pendingEffortHours: entry.pendingEffortHours,
      workloadDensity: entry.workloadDensity,
    });
  }

  const ranked: AssignmentScore[] = users
    .map((user) => {
      const role = roleById.get(user.id);
      const cap = capacityByUserId.get(user.id);
      const candidate: AssignmentCandidate = {
        userId: user.id,
        name: user.name,
        membership: memberIds.has(user.id),
        isLeader: role?.isLeader ?? false,
        role: role?.role ?? null,
        workloadPercent: cap?.workloadPercent ?? null,
        pendingEffortHours: cap?.pendingEffortHours ?? null,
        workloadDensity: cap?.workloadDensity ?? null,
        workingAs: user.workingAs,
        department: user.department,
        subRole: user.subRole?.name ?? null,
        bio: user.bio,
      };
      return assignmentScore(taskTags, taskTitle, taskPriority, candidate);
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  logger.info(
    `[REASONER] Assignment ranking produced for task "${taskId}": top candidate ${ranked[0]?.name ?? "none"} (score ${ranked[0]?.score ?? 0}).`
  );
  return ranked;
}

export async function evaluate(
  context: ProjectContext,
  options?: ReasonerOptions
): Promise<ReasonerEvaluation> {
  const startedAt = Date.now();
  const runLlm = options?.runLlm !== false;

  const circuitOk = await circuitBreaker.shouldAllow("read");
  if (!circuitOk) {
    const snapshot = await circuitBreaker.state;
    logger.warn(
      `[REASONER] Skipped evaluation because the circuit is ${snapshot.state.toUpperCase()}; reads are paused.`
    );
    return {
      task: null,
      risk: {
        level: "LOW",
        confidence: { score: 0, dataSourcesUsed: [], dataSourcesTotal: 0 },
        explanation: `Evaluation skipped: circuit is ${snapshot.state.toUpperCase()}.`,
        concerns: [],
        progress: 0,
        progressClamped: false,
      },
      riskNarrative: null,
      assignment: null,
      createdAt: new Date().toISOString(),
      llmUsed: false,
    };
  }

  const risk = riskScore(context);

  let riskNarrative: RiskNarrative | null = null;
  if (runLlm) {
    riskNarrative = await assessRiskWithLlm(context, risk);
  }

  let task: TaskReadiness | null = null;
  let assignment: ReasonerEvaluation["assignment"] = null;

  if (options?.taskId) {
    const taskRow = await db.task.findUnique({
      where: { id: options.taskId },
      select: { id: true, title: true, priority: true, status: true, tags: true },
    });

    if (!taskRow) {
      logger.warn(
        `[REASONER] Task "${options.taskId}" not found; skipping task readiness and assignment scoring.`
      );
    } else {
      task = await determineTaskReadiness(
        taskRow.id,
        taskRow.status,
        context.project.id
      );

      const memberRows = await db.projectMember.findMany({
        where: { projectId: context.project.id },
        select: { userId: true },
      });
      const memberIds = memberRows.map((member) => member.userId);
      const candidateIds =
        options.candidates && options.candidates.length > 0
          ? options.candidates
          : memberIds;

      if (candidateIds.length === 0) {
        logger.warn(
          `[REASONER] No candidates available for assignment scoring on task "${taskRow.id}".`
        );
      } else if (task.ready) {
        const ranked = await scoreCandidates(
          taskRow.id,
          taskRow.title,
          taskRow.priority,
          taskRow.tags,
          context.project.id,
          context,
          candidateIds
        );
        assignment = { taskId: taskRow.id, ranked };
      } else {
        logger.info(
          `[REASONER] Skipping assignment scoring for task "${taskRow.id}" because it is not ready.`
        );
      }
    }
  }

  const span = startSpan("evaluate_reasoner", "reasoner", {
    projectId: context.project.id,
    taskId: options?.taskId ?? null,
  });

  const evaluation: ReasonerEvaluation = {
    task,
    risk,
    riskNarrative,
    assignment,
    createdAt: new Date().toISOString(),
    llmUsed: runLlm && riskNarrative !== null,
  };

  await circuitBreaker.recordSuccess();

  span.end("OK", null, {
    riskLevel: risk.level,
    taskReady: task?.ready ?? null,
    topCandidate: assignment?.ranked[0]?.name ?? null,
    topCandidateScore: assignment?.ranked[0]?.score ?? null,
    llmUsed: evaluation.llmUsed,
  });

  logger.info(
    `[REASONER] Evaluation complete for project "${context.project.id}" in ${Date.now() - startedAt}ms ` +
      `(risk=${risk.level}, task=${task ? (task.ready ? "ready" : `not-ready`) : "skipped"}, llmUsed=${evaluation.llmUsed}).`
  );
  return evaluation;
}