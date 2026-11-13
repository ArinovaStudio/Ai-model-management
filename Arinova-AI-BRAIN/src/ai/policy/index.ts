import db from "@/lib/client";
import { logger } from "@/ai/logger";
import { verifyToken } from "@/lib/jwt";
import { circuitBreaker } from "@/ai/circuit-breaker";
import { getActionRule, DEFAULT_UNKNOWN_RISK } from "./rules";
import type {
  ActionType,
  PolicyCheck,
  PolicyContext,
  PolicyDecision,
} from "./types";

const AGENT_ROLE_CLAIM = "AGENT";
const AGENT_SUBROLE_NAMES = new Set(["SYSTEM_AGENT"]);

import type { StealthPolicy, UniversalStealthConfig } from "./types";

import { storeMemory } from "@/ai/audit";

const stealthPolicies = new Map<string, StealthPolicy>();
let universalConfig: UniversalStealthConfig = {
  isUniversalActive: false,
  globalMinConfidence: 0.75,
  exceptionProjectIds: [],
};

export function setUniversalConfig(config: UniversalStealthConfig): void {
  universalConfig = { ...config };
  logger.info(`[POLICY] Universal Stealth mode configured (isUniversalActive=${config.isUniversalActive}, globalMinConfidence=${config.globalMinConfidence}, exceptions=${config.exceptionProjectIds?.length ?? 0}).`);

  void storeMemory(
    "GLOBAL",
    "universal_stealth_config",
    `Universal Stealth active=${config.isUniversalActive}, minConfidence=${config.globalMinConfidence}`,
    {
      isUniversalActive: config.isUniversalActive,
      globalMinConfidence: config.globalMinConfidence,
      exceptionProjectIds: config.exceptionProjectIds ?? [],
    }
  ).catch((err) => {
    logger.warn(`[POLICY] Failed to persist universal stealth config in AgentMemory: ${err instanceof Error ? err.message : String(err)}`);
  });
}

export function getUniversalConfig(): UniversalStealthConfig {
  return universalConfig;
}

export function setStealthPolicy(policy: StealthPolicy): void {
  const modeType = policy.modeType ?? "AUTO";
  const selectedTaskIds = policy.selectedTaskIds ?? [];
  let isException = Boolean(policy.isException);

  // Sync with Universal Stealth Exception List:
  // If Universal Stealth is active globally and admin toggles this project OFF, automatically mark as Exception.
  if (universalConfig.isUniversalActive) {
    if (!policy.isActive) {
      isException = true;
      if (!universalConfig.exceptionProjectIds?.includes(policy.projectId)) {
        universalConfig.exceptionProjectIds = [...(universalConfig.exceptionProjectIds || []), policy.projectId];
        setUniversalConfig(universalConfig);
      }
    } else if (policy.isActive && universalConfig.exceptionProjectIds?.includes(policy.projectId)) {
      isException = false;
      universalConfig.exceptionProjectIds = universalConfig.exceptionProjectIds.filter((id) => id !== policy.projectId);
      setUniversalConfig(universalConfig);
    }
  }

  const updatedPolicy: StealthPolicy = {
    ...policy,
    modeType,
    selectedTaskIds,
    isException,
  };

  stealthPolicies.set(policy.projectId, updatedPolicy);
  logger.info(`[POLICY] Stealth mode policy configured for project "${policy.projectId}" (mode=${modeType}, active=${policy.isActive}, selectedTasks=${selectedTaskIds.length}, exception=${isException}).`);
  
  // Persist to AgentMemory table asynchronously so it survives daemon restarts
  void storeMemory(
    policy.projectId,
    "stealth_policy",
    `Stealth policy mode=${modeType}, active=${policy.isActive}, minConfidence=${policy.minConfidenceScore}`,
    {
      projectId: policy.projectId,
      startDate: policy.startDate ?? null,
      endDate: policy.endDate ?? null,
      minConfidenceScore: policy.minConfidenceScore,
      isActive: policy.isActive,
      modeType,
      selectedTaskIds,
      isException,
    }
  ).catch((err) => {
    logger.warn(`[POLICY] Failed to persist stealth policy in AgentMemory: ${err instanceof Error ? err.message : String(err)}`);
  });
}

function parseEndDateMs(endDateStr: string | null | undefined): number | null {
  if (!endDateStr) return null;
  const s = endDateStr.trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return Date.parse(`${s}T23:59:59.999Z`);
  return Date.parse(s);
}

export async function getStealthPolicyAsync(projectId: string | null): Promise<StealthPolicy | null> {
  if (typeof window === "undefined" && projectId) {
    try {
      const latestMem = await db.agentMemory.findFirst({
        where: { projectId, type: "stealth_policy" },
        orderBy: { createdAt: "desc" },
      });
      if (latestMem) {
        let ctx = latestMem.meta as Record<string, unknown> | null;
        if (typeof ctx === "string") {
          try { ctx = JSON.parse(ctx); } catch { ctx = null; }
        }
        if (ctx && typeof ctx.projectId === "string") {
          const freshPolicy: StealthPolicy = {
            projectId: ctx.projectId,
            startDate: (ctx.startDate as string) ?? null,
            endDate: (ctx.endDate as string) ?? null,
            minConfidenceScore: typeof ctx.minConfidenceScore === "number" ? ctx.minConfidenceScore : 0.75,
            isActive: Boolean(ctx.isActive),
            modeType: (ctx.modeType as "AUTO" | "SELECT") ?? "AUTO",
            selectedTaskIds: Array.isArray(ctx.selectedTaskIds) ? (ctx.selectedTaskIds as string[]) : [],
            isException: Boolean(ctx.isException),
          };
          stealthPolicies.set(projectId, freshPolicy);
        }
      }
    } catch {
      // Fallback to local map if db read encounters transient connection error
    }
  } else if (stealthPolicies.size === 0 && typeof window === "undefined") {
    await initStealthPoliciesFromDb();
  }
  return getStealthPolicy(projectId);
}

export function getStealthPolicy(projectId: string | null): StealthPolicy | null {
  const now = Date.now();
  const sevenDaysMs = 7 * 24 * 60 * 60 * 1000;
  const checkAndClean = (p: StealthPolicy | undefined): StealthPolicy | null => {
    if (!p) return null;
    const endMs = parseEndDateMs(p.endDate);
    if (endMs !== null && !Number.isNaN(endMs) && now - endMs > sevenDaysMs) {
      stealthPolicies.delete(p.projectId);
      return null;
    }
    return p;
  };

  let localPolicy = projectId ? checkAndClean(stealthPolicies.get(projectId)) : null;
  if (!localPolicy && !projectId) localPolicy = checkAndClean(stealthPolicies.get("*"));

  // Evaluate Universal Stealth override if active
  if (universalConfig.isUniversalActive) {
    const isException = localPolicy?.isException || (projectId && universalConfig.exceptionProjectIds?.includes(projectId));
    if (!isException) {
      return {
        projectId: projectId ?? "*",
        isActive: true,
        minConfidenceScore: localPolicy?.minConfidenceScore ?? universalConfig.globalMinConfidence,
        startDate: localPolicy?.startDate ?? null,
        endDate: localPolicy?.endDate ?? null,
        modeType: localPolicy?.modeType ?? "AUTO",
        selectedTaskIds: localPolicy?.selectedTaskIds ?? [],
        isException: false,
      };
    }
  }

  return localPolicy;
}

export async function initStealthPoliciesFromDb(): Promise<void> {
  try {
    const memories = await db.agentMemory.findMany({
      where: { type: { in: ["stealth_policy", "universal_stealth_config"] } },
      orderBy: { createdAt: "desc" },
    });
    const now = Date.now();
    const sevenDaysMs = 7 * 24 * 60 * 60 * 1000;
    for (const mem of memories) {
      let ctx = mem.meta as Record<string, unknown> | null;
      if (typeof ctx === "string") {
        try {
          ctx = JSON.parse(ctx) as Record<string, unknown>;
        } catch {
          ctx = null;
        }
      }

      if (mem.type === "universal_stealth_config" && ctx) {
        universalConfig = {
          isUniversalActive: Boolean(ctx.isUniversalActive),
          globalMinConfidence: typeof ctx.globalMinConfidence === "number" ? ctx.globalMinConfidence : 0.75,
          exceptionProjectIds: Array.isArray(ctx.exceptionProjectIds) ? (ctx.exceptionProjectIds as string[]) : [],
        };
      } else if (mem.type === "stealth_policy" && ctx && typeof ctx.projectId === "string" && typeof ctx.minConfidenceScore === "number") {
        const endDateStr = (ctx.endDate as string) ?? null;
        const endMs = parseEndDateMs(endDateStr);
        if (endMs !== null && !Number.isNaN(endMs) && now - endMs > sevenDaysMs) {
          continue;
        }
        if (!stealthPolicies.has(ctx.projectId)) {
          stealthPolicies.set(ctx.projectId, {
            projectId: ctx.projectId,
            startDate: (ctx.startDate as string) ?? null,
            endDate: endDateStr,
            minConfidenceScore: ctx.minConfidenceScore,
            isActive: Boolean(ctx.isActive),
            modeType: (ctx.modeType as "AUTO" | "SELECT") ?? "AUTO",
            selectedTaskIds: Array.isArray(ctx.selectedTaskIds) ? (ctx.selectedTaskIds as string[]) : [],
            isException: Boolean(ctx.isException),
          });
        }
      }
    }
    logger.info(`[POLICY] Hydrated ${stealthPolicies.size} stealth policy/policies and universal config from AgentMemory.`);
  } catch (err) {
    logger.warn(`[POLICY] Could not hydrate stealth policies on boot: ${err instanceof Error ? err.message : String(err)}`);
  }
}

function formatError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

function makeDecision(
  action: string,
  projectId: string | null,
  riskLevel: PolicyDecision["riskLevel"],
  status: PolicyDecision["status"],
  reason: string,
  checks: PolicyCheck[],
  allowed: boolean,
  requiresApproval: boolean
): PolicyDecision {
  return {
    status,
    allowed,
    requiresApproval,
    riskLevel,
    action,
    projectId,
    reason,
    checks,
  };
}

function sanitizeJson(value: unknown): unknown {
  if (value === null || value === undefined) return null;
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (typeof value === "object") {
    if (Array.isArray(value)) return value.map(sanitizeJson);
    const result: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) {
      result[key] = sanitizeJson(item);
    }
    return result;
  }
  return String(value);
}

async function logActionCheck(
  agentId: string,
  action: string,
  decision: PolicyDecision,
  target: string | null,
  payload: unknown
): Promise<void> {
  try {
    await db.agentActionLog.create({
      data: {
        agentId,
        action,
        target,
        payload: sanitizeJson({ projectId: decision.projectId, ...(payload as object) }) as object,
        result: sanitizeJson({
          status: decision.status,
          riskLevel: decision.riskLevel,
          requiresApproval: decision.requiresApproval,
          reason: decision.reason,
        }) as object,
        success: decision.allowed,
        reason: decision.reason,
      },
    });
  } catch (error) {
    logger.warn(
      `[POLICY] Failed to write AgentActionLog for "${action}": ${formatError(error)}`
    );
  }
}

export async function canPerformAction(
  agentId: string,
  action: string,
  context: PolicyContext = {}
): Promise<PolicyDecision> {
  const checks: PolicyCheck[] = [];
  const projectId = context.projectId ?? null;

  const record = async (decision: PolicyDecision): Promise<PolicyDecision> => {
    logger.info(
      `[POLICY] action="${action}" roleCheck=${
        checks.find((check) => check.name === "role")?.passed ?? null
      } membershipCheck=${
        checks.find((check) => check.name === "project_membership")?.passed ?? null
      } -> ${decision.status} (${decision.riskLevel}).`
    );
    if (
      decision.status === "denied" &&
      (checks.some((check) => check.name === "agent_exists" && check.passed === false) ||
        checks.some((check) => check.name === "role" && check.passed === false))
    ) {
      // Identity/role-layer failures (agent missing, role unverifiable) indicate an
      // authorization-layer problem and feed the circuit breaker's authorization
      // counter. Business denials (membership, task scope, approval) are expected and
      // are intentionally excluded.
      await circuitBreaker.recordFailure("authorization", {
        reason: `action "${action}" denied: ${decision.reason}`,
      });
    }
    await logActionCheck(agentId, action, decision, context.taskId ?? context.targetId ?? projectId, context.payload);
    return decision;
  };

  checks.push({
    name: "action",
    passed: action !== "" && action.trim() !== "",
    detail: action === "" ? "action name is empty" : `action "${action}" is defined`,
  });

  const rule = getActionRule(action);
  if (!action || action.trim() === "" || rule === null) {
    return record(
      makeDecision(
        action,
        projectId,
        DEFAULT_UNKNOWN_RISK,
        "denied",
        rule === null
          ? `Unknown action "${action}"; defaulting to HIGH risk and denying (fail-safe).`
          : "Action name is empty; cannot authorize (fail-safe).",
        checks,
        false,
        false
      )
    );
  }

  const user = await db.user.findUnique({
    where: { id: agentId },
    select: {
      id: true,
      email: true,
      workingAs: true,
      isDev: true,
      subRole: {
        select: {
          name: true,
          role: { select: { name: true } },
        },
      },
    },
  });

  checks.push({
    name: "agent_exists",
    passed: user !== null,
    detail: user
      ? `agent user "${agentId}" exists ("${user.email}")`
      : `agent user "${agentId}" does not exist in the User table`,
  });
  if (user === null) {
    return record(
      makeDecision(
        action,
        projectId,
        rule.riskLevel,
        "denied",
        `Authorization failed: agent user "${agentId}" does not exist.`,
        checks,
        false,
        false
      )
    );
  }

  let roleVerified = false;
  let roleSource: "jwt" | "db_subrole" | null = null;
  if (context.token) {
    try {
      const payload = verifyToken(context.token);
      roleVerified = payload.role === AGENT_ROLE_CLAIM;
      roleSource = roleVerified ? "jwt" : null;
    } catch {
      roleVerified = false;
    }
  } else if (
    user.subRole?.role?.name === AGENT_ROLE_CLAIM ||
    (user.subRole && AGENT_SUBROLE_NAMES.has(user.subRole.name))
  ) {
    roleVerified = true;
    roleSource = "db_subrole";
  }

  checks.push({
    name: "role",
    passed: roleVerified,
    detail: roleVerified
      ? `agent role verified (source=${roleSource})`
      : context.token
        ? "JWT was provided but its role claim is not AGENT (or the token is invalid)"
        : "no AGENT role could be verified (no valid JWT and no subRole/role linkage)",
  });
  if (!roleVerified) {
    return record(
      makeDecision(
        action,
        projectId,
        rule.riskLevel,
        "denied",
        `Authorization failed: the agent's role could not be verified as ${AGENT_ROLE_CLAIM}. ${
          context.token
            ? "The provided token is invalid or does not carry the AGENT role claim."
            : "Provide a valid AGENT JWT or link the agent's subRole to a SYSTEM_AGENT/AGENT role."
        }`,
        checks,
        false,
        false
      )
    );
  }

  checks.push({
    name: "agent_marker",
    passed: user.workingAs === "AI Agent" || user.isDev === true,
    detail:
      user.workingAs === "AI Agent" || user.isDev === true
        ? `agent marker present (workingAs="${user.workingAs ?? "n/a"}", isDev=${user.isDev ?? false})`
        : "user does not carry the agent marker (workingAs/isDev)",
  });

  if (rule.requiresProject) {
    if (!projectId) {
      return record(
        makeDecision(
          action,
          projectId,
          rule.riskLevel,
          "denied",
          `Authorization failed: action "${action}" requires a project context, but none was provided.`,
          [
            ...checks,
            {
              name: "project_exists",
              passed: false,
              detail: "projectId is missing",
            },
          ],
          false,
          false
        )
      );
    }

    const project = await db.project.findUnique({
      where: { id: projectId },
      select: { id: true },
    });
    checks.push({
      name: "project_exists",
      passed: project !== null,
      detail: project ? `project "${projectId}" exists` : `project "${projectId}" does not exist`,
    });
    if (project === null) {
      return record(
        makeDecision(
          action,
          projectId,
          rule.riskLevel,
          "denied",
          `Authorization failed: project "${projectId}" does not exist.`,
          checks,
          false,
          false
        )
      );
    }

    const membership = await db.projectMember.findFirst({
      where: { projectId, userId: user.id },
      select: { id: true, role: true, isLeader: true },
    });
    checks.push({
      name: "project_membership",
      passed: membership !== null,
      detail: membership
        ? `agent is a member of project "${projectId}" (role="${membership.role ?? "n/a"}")`
        : `agent is not a member of project "${projectId}"`,
    });
    if (membership === null) {
      return record(
        makeDecision(
          action,
          projectId,
          rule.riskLevel,
          "denied",
          `Authorization failed: the agent is not a member of project "${projectId}". Only project members may act on a project.`,
          checks,
          false,
          false
        )
      );
    }
  }

  if (rule.requiresTask) {
    if (!context.taskId) {
      return record(
        makeDecision(
          action,
          projectId,
          rule.riskLevel,
          "denied",
          `Authorization failed: action "${action}" requires a task context, but none was provided.`,
          [
            ...checks,
            {
              name: "task_scope",
              passed: false,
              detail: "taskId is missing",
            },
          ],
          false,
          false
        )
      );
    }

    const task = await db.task.findUnique({
      where: { id: context.taskId },
      select: { id: true, projectId: true, status: true },
    });
    const taskInProject = task !== null && projectId !== null && task.projectId === projectId;
    checks.push({
      name: "task_scope",
      passed: taskInProject,
      detail: taskInProject
        ? `task "${context.taskId}" exists and belongs to project "${projectId}" (status="${task.status}")`
        : task === null
          ? `task "${context.taskId}" does not exist`
          : `task "${context.taskId}" belongs to project "${task.projectId}", not "${projectId}"`,
    });
    if (!taskInProject) {
      return record(
        makeDecision(
          action,
          projectId,
          rule.riskLevel,
          "denied",
          `Authorization failed: task "${context.taskId}" does not exist or is not part of project "${projectId}".`,
          checks,
          false,
          false
        )
      );
    }
  }

  let decision: PolicyDecision;
  if (rule.riskLevel === "LOW") {
    decision = makeDecision(
      action,
      projectId,
      rule.riskLevel,
      "allowed",
      `Allowed: "${rule.label}" is a LOW-risk action and all authorization checks passed.`,
      checks,
      true,
      false
    );
  } else if (rule.riskLevel === "MEDIUM") {
    // Check if Stealth Mode Auto-Pilot applies for this project
    const stealth = await getStealthPolicyAsync(projectId);
    const payload = (context.payload ?? {}) as Record<string, unknown>;
    const score = typeof payload.confidence === "number" ? payload.confidence : (typeof payload.score === "number" ? payload.score : 0.85);
    const nowMs = Date.now();
    const parseEndDate = (dStr?: string | null): number | null => {
      if (!dStr) return null;
      const s = dStr.trim();
      if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return Date.parse(`${s}T23:59:59.999Z`);
      return Date.parse(s);
    };
    const endMs = parseEndDate(stealth?.endDate);
    const inDateRange = (!stealth?.startDate || Date.parse(stealth.startDate) <= nowMs) &&
                        (endMs === null || !Number.isNaN(endMs) && endMs >= nowMs);

    if (stealth && stealth.isActive && inDateRange && score >= stealth.minConfidenceScore) {
      checks.push({
        name: "stealth_mode_auto_pilot",
        passed: true,
        detail: `stealth auto-pilot active for project "${projectId}" (confidence ${score} >= min ${stealth.minConfidenceScore})`,
      });
      decision = makeDecision(
        action,
        projectId,
        rule.riskLevel,
        "allowed",
        `Allowed via Stealth Mode Auto-Pilot: candidate score ${score} meets admin pre-approved threshold (${stealth.minConfidenceScore}).`,
        checks,
        true,
        false
      );
    } else {
      decision = makeDecision(
        action,
        projectId,
        rule.riskLevel,
        "requires_approval",
        `Requires approval: "${rule.label}" is a MEDIUM-risk action. The agent may prepare and recommend it, but a recommendation record (AgentDecision) must be created and a human must confirm before it is executed.`,
        checks,
        true,
        true
      );
    }
  } else {
    const approval = context.approval;
    const hasApproval = approval !== undefined && Boolean(approval.by) && approval.by.trim() !== "";
    decision = hasApproval
      ? makeDecision(
          action,
          projectId,
          rule.riskLevel,
          "allowed",
          `Allowed: "${rule.label}" is a HIGH-risk action but explicit human approval was provided (by ${approval.by}).`,
          [
            ...checks,
            {
              name: "explicit_approval",
              passed: true,
              detail: `approval by "${approval.by}"${approval.at ? ` at ${approval.at}` : ""}`,
            },
          ],
          true,
          true
        )
      : makeDecision(
          action,
          projectId,
          rule.riskLevel,
          "denied",
          `Denied: "${rule.label}" is a HIGH-risk action and requires explicit human approval. No approval was provided, so the action is not permitted.`,
          [
            ...checks,
            {
              name: "explicit_approval",
              passed: false,
              detail: "no approval information provided",
            },
          ],
          false,
          true
        );
  }

  return record(decision);
}

export type { ActionType };

export function classifyRisk(action: string): PolicyDecision["riskLevel"] {
  return getActionRule(action)?.riskLevel ?? DEFAULT_UNKNOWN_RISK;
}