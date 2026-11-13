import db from "@/lib/client";
import { logger } from "@/ai/logger";
import { startSpan } from "@/ai/telemetry";
import type {
  DependencyEdge,
  ProjectContext,
  ProjectSnapshot,
  RiskBlocker,
  RiskIndicators,
  TaskSnapshot,
  TeamCapacityEntry,
} from "./types";

const ACTIVE_TASKS_LIMIT = 50;
const WORK_HOURS_WINDOW_DAYS = 7;
const WEEKLY_CAPACITY_HOURS = 40;
const TOP_RISK_BLOCKERS = 3;

const TERMINAL_STATUSES = new Set([
  "completed",
  "cancelled",
  "canceled",
  "done",
]);

const SEVERITY_WEIGHT: Record<string, number> = {
  critical: 4,
  high: 3,
  medium: 2,
  low: 1,
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round2(value: number): number {
  return Math.round(value * 100) / 100;
}

function toIso(value: Date | null | undefined): string | null {
  return value instanceof Date && !Number.isNaN(value.getTime())
    ? value.toISOString()
    : null;
}

function isTerminalStatus(status: string | null): boolean {
  return status !== null && TERMINAL_STATUSES.has(status.toLowerCase());
}

function severityWeight(severity: string | null): number {
  if (!severity) return 0;
  return SEVERITY_WEIGHT[severity.toLowerCase()] ?? 0;
}

function rankRiskBlockers(blockers: RiskBlocker[]): RiskBlocker[] {
  return [...blockers]
    .sort((a, b) => {
      const bySeverity = severityWeight(b.severity) - severityWeight(a.severity);
      if (bySeverity !== 0) return bySeverity;
      return (Date.parse(b.createdAt ?? "") || 0) - (Date.parse(a.createdAt ?? "") || 0);
    })
    .slice(0, TOP_RISK_BLOCKERS);
}

export async function buildProjectContext(
  projectId: string
): Promise<ProjectContext> {
  const startedAt = Date.now();
  const now = new Date();

  if (!projectId || projectId.trim() === "") {
    throw new Error("[CONTEXT] buildProjectContext requires a non-empty projectId.");
  }

  const span = startSpan("build_project_context", "context", { projectId });

  const [project, activeTasks, dependencies, members, riskBlockages] =
    await Promise.all([
      db.project.findUnique({
        where: { id: projectId },
        select: {
          id: true,
          name: true,
          summary: true,
          priority: true,
          status: true,
          progress: true,
          currentPhase: true,
          projectInfo: { select: { deadline: true } },
        },
      }),
      db.task.findMany({
        where: { projectId, status: { notIn: [...TERMINAL_STATUSES] } },
        orderBy: { dueDate: "asc" },
        take: ACTIVE_TASKS_LIMIT,
        select: {
          id: true,
          title: true,
          status: true,
          priority: true,
          assignee: true,
          userId: true,
          dueDate: true,
          createdAt: true,
        },
      }),
      db.taskDependency.findMany({
        where: { projectId },
        select: {
          taskId: true,
          dependsOnTaskId: true,
          relationType: true,
          source: true,
        },
      }),
      db.projectMember.findMany({
        where: { projectId },
        select: {
          userId: true,
          role: true,
          isLeader: true,
          user: { select: { id: true, name: true } },
        },
      }).then(async (pmList) => {
        if (pmList.length > 0) return pmList;
        // Fallback: If no project members explicitly linked, query active developers
        const fallbackUsers = await db.user.findMany({
          where: { isDev: true, workingAs: { not: "AI Agent" } },
          select: { id: true, name: true },
          take: 20,
        });
        return fallbackUsers.map(u => ({
          userId: u.id,
          role: "Developer",
          isLeader: false,
          user: { id: u.id, name: u.name }
        }));
      }),
      db.riskBlockage.findMany({
        where: { projectId },
        select: {
          id: true,
          title: true,
          severity: true,
          status: true,
          description: true,
          createdAt: true,
        },
      }),
    ]);

  if (!project) {
    throw new Error(`[CONTEXT] Project not found: "${projectId}".`);
  }

  const memberIds = members.map((member) => member.userId);
  const sevenDaysAgo = new Date(now.getTime() - WORK_HOURS_WINDOW_DAYS * 24 * 60 * 60 * 1000);

  const workHours = await db.workHours.groupBy({
    by: ["userId"],
    where: {
      userId: { in: memberIds },
      createdAt: { gte: sevenDaysAgo },
    },
    _sum: { hours: true },
  });

  const dependencyTargetIds = new Set<string>();
  for (const edge of dependencies) {
    dependencyTargetIds.add(edge.taskId);
    dependencyTargetIds.add(edge.dependsOnTaskId);
  }

  const dependencyStatuses = await db.task.findMany({
    where: { id: { in: [...dependencyTargetIds] } },
    select: { id: true, status: true },
  });

  const statusById = new Map<string, string>();
  for (const task of activeTasks) statusById.set(task.id, task.status ?? "");
  for (const task of dependencyStatuses) statusById.set(task.id, task.status ?? "");

  const outgoing = new Map<string, DependencyEdge[]>();
  for (const edge of dependencies) {
    const list = outgoing.get(edge.taskId);
    if (list) list.push(edge);
    else outgoing.set(edge.taskId, [edge]);
  }

  const blockedCache = new Map<string, boolean>();
  function isBlocked(taskId: string): boolean {
    const cached = blockedCache.get(taskId);
    if (cached !== undefined) return cached;
    const edges = outgoing.get(taskId) ?? [];
    let blocked = false;
    for (const edge of edges) {
      if ((edge.relationType ?? "").toLowerCase() !== "hard") continue;
      const dependencyStatus = statusById.get(edge.dependsOnTaskId);
      if (dependencyStatus === undefined) {
        logger.warn(
          `[CONTEXT] TaskDependency references missing task "${edge.dependsOnTaskId}" (depends-on edge from "${edge.taskId}"); skipping edge.`
        );
        continue;
      }
      if (!isTerminalStatus(dependencyStatus)) {
        blocked = true;
        break;
      }
    }
    blockedCache.set(taskId, blocked);
    return blocked;
  }

  const hardEdges = dependencies.filter(
    (edge) => (edge.relationType ?? "").toLowerCase() === "hard"
  );

  const PRIORITY_HOURS: Record<string, number> = {
    urgent: 8,
    high: 6,
    medium: 4,
    low: 2,
  };

  const taskSnapshots: TaskSnapshot[] = activeTasks.map((task) => {
    const p = (task.priority ?? "medium").toLowerCase();
    const est = PRIORITY_HOURS[p] ?? 4;
    return {
      id: task.id,
      title: task.title,
      status: task.status,
      priority: task.priority,
      assignee: task.assignee,
      userId: task.userId,
      dueDate: toIso(task.dueDate),
      createdAt: toIso(task.createdAt),
      isBlocked: isBlocked(task.id),
      estimatedHours: est,
    };
  });

  const hoursByUserId = new Map<string, number>();
  for (const entry of workHours) {
    hoursByUserId.set(entry.userId, round2(entry._sum.hours ?? 0));
  }

  // Calculate pending effort and work density index per developer
  const memberAssignedTasks = new Map<string, TaskSnapshot[]>();
  for (const task of taskSnapshots) {
    const assignedUserOrName = (task.userId ?? task.assignee ?? "").trim();
    if (assignedUserOrName !== "" && !/^unassigned$/i.test(assignedUserOrName)) {
      const rawLower = assignedUserOrName.toLowerCase();
      // Find matching member by user ID, task.userId, or sanitized display name
      const matchingMember = members.find(
        (m) => m.userId === task.userId || m.userId === task.assignee || (m.user?.name && m.user.name.trim().toLowerCase() === rawLower)
      );
      if (matchingMember) {
        const list = memberAssignedTasks.get(matchingMember.userId) ?? [];
        list.push(task);
        memberAssignedTasks.set(matchingMember.userId, list);
      }
    }
  }

  const memberUserIds = Array.from(new Set(members.map((m) => m.userId).filter(Boolean)));
  const performanceRatings = memberUserIds.length > 0
    ? await db.performanceRating.findMany({
        where: { userId: { in: memberUserIds } },
        orderBy: { createdAt: "desc" },
      }).catch(() => [])
    : [];

  const ratingsByUserId = new Map<string, { average: number; breakdown: Record<string, number> }>();
  for (const r of performanceRatings) {
    if (!ratingsByUserId.has(r.userId)) {
      const breakdown = {
        newLearnings: r.newLearnings,
        speed: r.speed,
        workQuality: r.workQuality,
        leaves: r.leaves,
        communication: r.communication,
        feedback: r.feedback,
      };
      const sum = r.newLearnings + r.speed + r.workQuality + r.leaves + r.communication + r.feedback;
      const average = round2(sum / 6);
      ratingsByUserId.set(r.userId, { average, breakdown });
    }
  }

  const teamCapacity: TeamCapacityEntry[] = members.map((member) => {
    const totalHours = hoursByUserId.get(member.userId) ?? 0;
    const assigned = memberAssignedTasks.get(member.userId) ?? [];
    const pendingEffortHours = assigned.reduce((sum, t) => sum + (t.estimatedHours ?? 4), 0);
    
    // Work Density = Sum( (Effort / Max(Days to due date, 0.5)) * WeightFactor )
    let workloadDensity = 0;
    for (const t of assigned) {
      const est = t.estimatedHours ?? 4;
      let daysRemaining = 3; // Default 3 days if no due date
      let weightFactor = 1.0;
      if (t.dueDate) {
        const parsedMs = Date.parse(t.dueDate);
        if (!Number.isNaN(parsedMs)) {
          const diffMs = parsedMs - now.getTime();
          const daysDiff = diffMs / (1000 * 60 * 60 * 24);
          if (daysDiff < -14) {
            // Overdue by > 14 days (stale/forgotten task): decay contribution factor
            daysRemaining = 3;
            weightFactor = 0.25;
          } else {
            daysRemaining = Math.max(0.5, daysDiff);
          }
        }
      }
      workloadDensity += (est / daysRemaining) * weightFactor;
    }

    const ratingObj = ratingsByUserId.get(member.userId);

    return {
      userId: member.userId,
      name: member.user?.name ?? "Unknown",
      role: member.role,
      isLeader: member.isLeader,
      totalHours7d: totalHours,
      workloadPercent: clamp(
        Math.round((totalHours / WEEKLY_CAPACITY_HOURS) * 100),
        0,
        100
      ),
      pendingEffortHours: round2(pendingEffortHours),
      workloadDensity: round2(workloadDensity),
      performanceRating: ratingObj ? ratingObj.average : null,
      ratingBreakdown: ratingObj ? ratingObj.breakdown : null,
    };
  });

  const deadline = toIso(project.projectInfo?.deadline);
  const deadlineMs = project.projectInfo?.deadline
    ? project.projectInfo.deadline.getTime()
    : null;
  const projectSnapshot: ProjectSnapshot = {
    id: project.id,
    name: project.name,
    summary: project.summary,
    priority: project.priority,
    status: project.status,
    progress: clamp(project.progress, 0, 100),
    currentPhase: project.currentPhase,
    deadline,
    deadlineOverdue:
      deadlineMs !== null && deadlineMs < now.getTime() && project.progress < 100,
    memberCount: members.length,
  };

  const blockedTaskCount = taskSnapshots.filter((task) => task.isBlocked).length;
  const overdueTaskCount = taskSnapshots.filter(
    (task) =>
      task.dueDate !== null &&
      Date.parse(task.dueDate) < now.getTime() &&
      !isTerminalStatus(task.status)
  ).length;

  const topRiskBlockers: RiskBlocker[] = riskBlockages.map((blocker) => ({
    id: blocker.id,
    title: blocker.title,
    severity: blocker.severity,
    status: blocker.status,
    description: blocker.description,
    createdAt: toIso(blocker.createdAt),
  }));

  const risks: RiskIndicators = {
    activeTaskCount: taskSnapshots.length,
    blockedTaskCount,
    overdueTaskCount,
    topRiskBlockers: rankRiskBlockers(topRiskBlockers),
  };

  const dependencyGraph: DependencyEdge[] = dependencies.map((edge) => ({
    taskId: edge.taskId,
    dependsOnTaskId: edge.dependsOnTaskId,
    relationType: edge.relationType,
    source: edge.source,
  }));

  const context: ProjectContext = {
    project: projectSnapshot,
    activeTasks: taskSnapshots,
    dependencyGraph,
    teamCapacity,
    risks,
    createdAt: now.toISOString(),
  };

  span.end("OK", null, {
    activeTasks: taskSnapshots.length,
    edges: dependencyGraph.length,
    members: members.length,
    blockers: riskBlockages.length,
  });

  logger.info(
    `[CONTEXT] Built project context for "${project.id}" in ${Date.now() - startedAt}ms ` +
      `(activeTasks=${taskSnapshots.length}, edges=${dependencyGraph.length}, ` +
      `hardEdges=${hardEdges.length}, members=${members.length}, blockers=${riskBlockages.length}).`
  );

  return context;
}