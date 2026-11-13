import { logger } from "@/ai/logger";
import type { ProjectContext } from "@/ai/context/types";
import type {
  AssignmentCandidate,
  AssignmentScore,
  AssignmentWeights,
  DependencyInfo,
  RiskAssessment,
  RiskConcern,
  TaskReadiness,
} from "./types";

const TERMINAL_STATUSES = new Set([
  "completed",
  "done",
  "cancelled",
  "canceled",
]);

const PRIORITY_WEIGHT: Record<string, number> = {
  low: 0.4,
  medium: 0.6,
  high: 0.9,
  urgent: 1.0,
};

export const ASSIGNMENT_WEIGHTS: AssignmentWeights = {
  workload: 0.4,
  skill: 0.35,
  complexity: 0.25,
};

export const WORKLOAD_OVERLOAD_THRESHOLD = 90;
export const CRITICAL_BLOCKER_SEVERITIES = new Set(["high", "critical"]);

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function isTerminal(status: string): boolean {
  return TERMINAL_STATUSES.has(status.toLowerCase());
}

function priorityWeight(priority: string): number {
  return PRIORITY_WEIGHT[priority.toLowerCase()] ?? 0.6;
}

function tokenize(value: string | null | undefined): Set<string> {
  if (!value) return new Set();
  const words = value.toLowerCase().match(/[a-z0-9]+/g);
  return new Set(words ?? []);
}

function hasCycleFrom(taskId: string, edges: DependencyInfo[]): boolean {
  const outgoing = new Map<string, DependencyInfo[]>();
  for (const edge of edges) {
    const list = outgoing.get(edge.taskId);
    if (list) list.push(edge);
    else outgoing.set(edge.taskId, [edge]);
  }
  const visiting = new Set<string>();
  const visited = new Set<string>();
  const visit = (node: string): boolean => {
    if (visiting.has(node)) return true;
    if (visited.has(node)) return false;
    visiting.add(node);
    const next = outgoing.get(node) ?? [];
    for (const edge of next) {
      if (visit(edge.dependsOnTaskId)) return true;
    }
    visiting.delete(node);
    visited.add(node);
    return false;
  };
  return visit(taskId);
}

export function taskReadiness(
  taskId: string,
  edges: DependencyInfo[],
  statusById: Map<string, string>
): TaskReadiness {
  const hardEdges = edges.filter(
    (edge) =>
      edge.taskId === taskId &&
      (edge.relationType ?? "").toLowerCase() === "hard"
  );

  const unresolvedDependencies: string[] = [];
  for (const edge of hardEdges) {
    const status = statusById.get(edge.dependsOnTaskId);
    if (status === undefined || !isTerminal(status)) {
      unresolvedDependencies.push(edge.dependsOnTaskId);
    }
  }

  if (hasCycleFrom(taskId, edges)) {
    logger.warn(
      `[REASONER] Circular dependency detected involving task "${taskId}"; marking it unactionable.`
    );
    return {
      taskId,
      ready: false,
      reason: "circular dependency detected; task is not actionable",
      unresolvedDependencies,
      hasCycle: true,
    };
  }

  if (unresolvedDependencies.length > 0) {
    return {
      taskId,
      ready: false,
      reason: `has ${unresolvedDependencies.length} unsatisfied hard dependency(ies)`,
      unresolvedDependencies,
      hasCycle: false,
    };
  }

  return {
    taskId,
    ready: true,
    reason: "all hard dependencies are satisfied",
    unresolvedDependencies: [],
    hasCycle: false,
  };
}

export function riskScore(context: ProjectContext): RiskAssessment {
  const rawProgress = context.project.progress;
  const progressClamped = rawProgress < 0 || rawProgress > 100;
  const progress = clamp(rawProgress, 0, 100);
  if (progressClamped) {
    logger.warn(
      `[REASONER] Project "${context.project.id}" progress ${rawProgress} is outside [0,100]; clamped to ${progress}.`
    );
  }

  const activeCount = context.risks.activeTaskCount;
  const blockedCount = context.risks.blockedTaskCount;
  const overdueCount = context.risks.overdueTaskCount;
  const unassignedCount = context.activeTasks.filter(
    (task) => !task.assignee || task.assignee.trim() === "" || /^unassigned$/i.test(task.assignee.trim())
  ).length;

  const capacity = context.teamCapacity;
  const overloadedCount = capacity.filter(
    (member) => member.workloadPercent >= WORKLOAD_OVERLOAD_THRESHOLD
  ).length;
  const averageWorkload =
    capacity.length > 0
      ? capacity.reduce((sum, member) => sum + member.workloadPercent, 0) /
        capacity.length
      : null;

  const criticalBlockers = context.risks.topRiskBlockers.filter(
    (blocker) =>
      CRITICAL_BLOCKER_SEVERITIES.has(blocker.severity.toLowerCase()) &&
      !/^(closed|resolved)$/i.test(blocker.status)
  );

  const concerns: RiskConcern[] = [];

  const deadlineOverdue = context.project.deadlineOverdue;
  if (deadlineOverdue && progress < 100) {
    concerns.push({
      id: "deadline_overdue",
      label: "project deadline is in the past with work remaining",
      severity: "high",
    });
  }
  if (overdueCount > 0) {
    concerns.push({
      id: "overdue_tasks",
      label: `${overdueCount} active task(s) are past due`,
      severity: "high",
    });
  }
  if (blockedCount > 0) {
    concerns.push({
      id: "blocked_tasks",
      label: `${blockedCount} active task(s) are blocked by dependencies`,
      severity: "medium",
    });
  }
  if (unassignedCount > 0) {
    concerns.push({
      id: "unassigned_tasks",
      label: `${unassignedCount} active task(s) have no assignee`,
      severity: "medium",
    });
  }
  if (overloadedCount > 0 || (averageWorkload !== null && averageWorkload >= 85)) {
    concerns.push({
      id: "team_capacity",
      label: `${overloadedCount} team member(s) are at or above ${WORKLOAD_OVERLOAD_THRESHOLD}% workload`,
      severity: "medium",
    });
  }
  if (criticalBlockers.length > 0) {
    concerns.push({
      id: "critical_blockers",
      label: `${criticalBlockers.length} critical risk blocker(s) are active`,
      severity: "high",
    });
  }

  let level: RiskAssessment["level"];
  let explanation: string;

  const criticalPathBlocked =
    blockedCount > 0 && criticalBlockers.length > 0;
  const highSeverityCount = concerns.filter((concern) => concern.severity === "high").length;

  if (deadlineOverdue && progress < 100) {
    level = "HIGH";
    explanation =
      "HIGH: the deadline is in the past with work remaining (deterministic hard rule).";
  } else if (criticalPathBlocked || concerns.length >= 3 || highSeverityCount >= 2) {
    level = "HIGH";
    explanation =
      "HIGH: multiple risk concerns or a blocked critical path were detected.";
  } else if (concerns.length === 1 || concerns.length === 2) {
    level = "MEDIUM";
    explanation = `MEDIUM: ${concerns.length} concern(s) were detected.`;
  } else {
    level = "LOW";
    explanation = "LOW: the project appears to be on track.";
  }

  const dataSources: string[] = [];
  dataSources.push("project");
  if (activeCount > 0) dataSources.push("task");
  if (capacity.length > 0) dataSources.push("capacity");
  if (context.risks.topRiskBlockers.length > 0) dataSources.push("risk_blockers");
  const dataSourcesTotal = 4;
  const confidenceScore = clamp(0.6 + 0.1 * dataSources.length, 0, 1);
  const finalConfidence =
    level === "HIGH" && deadlineOverdue
      ? Math.max(confidenceScore, 0.9)
      : confidenceScore;

  explanation += ` Confidence ${finalConfidence} is based on data from ${dataSources.length}/${dataSourcesTotal} signal sources (${dataSources.join(", ") || "none"}).`;

  return {
    level,
    confidence: {
      score: round3(finalConfidence),
      dataSourcesUsed: dataSources,
      dataSourcesTotal,
    },
    explanation,
    concerns,
    progress,
    progressClamped,
  };
}

function normalizeWeights(
  weights: AssignmentWeights,
  available: (keyof AssignmentWeights)[]
): AssignmentWeights {
  const total = available.reduce((sum, key) => sum + weights[key], 0);
  const normalized: AssignmentWeights = {
    workload: 0,
    skill: 0,
    complexity: 0,
  };
  for (const key of available) {
    normalized[key] = total > 0 ? weights[key] / total : 0;
  }
  return normalized;
}

export function assignmentScore(
  taskTags: string[],
  taskTitle: string,
  taskPriority: string,
  candidate: AssignmentCandidate
): AssignmentScore {
  const basis: string[] = [];

  if (!candidate.membership) {
    basis.push("candidate is not a project member; score is 0");
    return {
      userId: candidate.userId,
      name: candidate.name,
      score: 0,
      components: {
        workloadScore: null,
        skillScore: null,
        complexityFit: null,
        membershipScore: 0,
      },
      weights: { ...ASSIGNMENT_WEIGHTS },
      basis,
    };
  }

  let workloadScore: number | null = null;
  if (candidate.workloadPercent !== null && candidate.workloadPercent !== undefined) {
    const baseScore = 1 - clamp(candidate.workloadPercent / 100, 0, 1);
    if (candidate.workloadDensity !== null && candidate.workloadDensity !== undefined && candidate.workloadDensity > 0) {
      const densityScore = 1 - clamp(candidate.workloadDensity / 10, 0, 1);
      workloadScore = round3(0.4 * baseScore + 0.6 * densityScore);
      basis.push(`composite capacity (logged ${candidate.workloadPercent}%, density index ${candidate.workloadDensity}) -> ${workloadScore}`);
    } else {
      workloadScore = round3(baseScore);
      basis.push(`workload ${candidate.workloadPercent}% -> ${workloadScore}`);
    }
  } else {
    basis.push("workload data unavailable; component excluded");
  }

  const requiredTokens = tokenize(taskTitle);
  for (const tag of taskTags) {
    for (const token of tokenize(tag)) requiredTokens.add(token);
  }

  const profileParts = [candidate.workingAs, candidate.department, candidate.subRole, candidate.bio];
  const profileTokens = new Set<string>();
  for (const part of profileParts) {
    for (const token of tokenize(part)) profileTokens.add(token);
  }

  let skillScore: number | null;
  if (requiredTokens.size === 0 || profileTokens.size === 0) {
    skillScore = null;
    basis.push(
      requiredTokens.size === 0
        ? "task has no skills/tags to match against; component excluded"
        : "candidate has no skill profile data; component excluded"
    );
  } else {
    let intersection = 0;
    for (const token of requiredTokens) {
      if (profileTokens.has(token)) intersection += 1;
    }
    skillScore = round3(clamp(intersection / requiredTokens.size, 0, 1));
    basis.push(`skill coverage ${intersection}/${requiredTokens.size} -> ${skillScore}`);
  }

  const weight = priorityWeight(taskPriority);
  let complexityFit: number | null;
  if (candidate.workloadPercent === null || candidate.workloadPercent === undefined) {
    complexityFit = null;
    basis.push("complexity fit unavailable (no workload data); component excluded");
  } else {
    complexityFit = round3(
      clamp(1 - (candidate.workloadPercent / 100) * weight, 0, 1)
    );
    basis.push(
      `complexity fit (priority ${taskPriority} weight ${weight}, workload ${candidate.workloadPercent}%) -> ${complexityFit}`
    );
  }

  const availableKeys: (keyof AssignmentWeights)[] = [];
  if (workloadScore !== null) availableKeys.push("workload");
  if (skillScore !== null) availableKeys.push("skill");
  if (complexityFit !== null) availableKeys.push("complexity");

  const weights = normalizeWeights(ASSIGNMENT_WEIGHTS, availableKeys);
  const componentValues = {
    workload: workloadScore,
    skill: skillScore,
    complexity: complexityFit,
  };

  let score = 0;
  for (const key of availableKeys) {
    const value = componentValues[key];
    if (value !== null) score += weights[key] * value;
  }
  score = round3(clamp(score, 0, 1));

  basis.push(
    `score ${score} = weighted sum over [${availableKeys.join(", ")}] with normalized weights ` +
      `workload ${weights.workload.toFixed(2)}, skill ${weights.skill.toFixed(2)}, complexity ${weights.complexity.toFixed(2)}`
  );

  return {
    userId: candidate.userId,
    name: candidate.name,
    score,
    components: {
      workloadScore,
      skillScore,
      complexityFit,
      membershipScore: 1,
    },
    weights,
    basis,
  };
}