import type { Prisma } from "@prisma/client";
import db from "@/lib/client";
import { logger } from "@/ai/logger";
import { canPerformAction } from "@/ai/policy";
import { circuitBreaker } from "@/ai/circuit-breaker";
import { startSpan } from "@/ai/telemetry";
import type { PolicyDecision } from "@/ai/policy/types";
import { compareFields, detectConflict, mergeVerification } from "./verifier";
import type {
  ExecutionContext,
  ExecutionResult,
  VerificationResult,
} from "./types";

const MAX_ATTEMPTS = 2;
const RETRY_DELAY_MS = 500;
const ALLOWED_TASK_STATUSES = new Set([
  "pending",
  "assigned",
  "in-progress",
  "completed",
]);

const TRANSIENT_ERROR_CODES = new Set([
  "P1001",
  "P1002",
  "P1017",
  "P2024",
]);

function formatError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isTransientPrismaError(error: unknown): boolean {
  const code =
    typeof error === "object" && error !== null && "code" in error
      ? (error as { code?: string }).code
      : undefined;
  return code !== undefined && TRANSIENT_ERROR_CODES.has(code);
}

function sanitizeJson(value: unknown): unknown {
  if (value === null || value === undefined) return null;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
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

async function createActionLog(
  context: ExecutionContext,
  action: string,
  phase: string,
  target: string | null,
  payload: unknown,
  reason: string
): Promise<string | null> {
  try {
    const row = await db.agentActionLog.create({
      data: {
        agentId: context.agentId,
        action,
        target,
        payload: sanitizeJson(payload) as object,
        result: sanitizeJson({ status: phase }) as object,
        success: false,
        reason,
      },
    });
    return row.id;
  } catch (error) {
    logger.warn(
      `[EXECUTOR] Failed to write PENDING AgentActionLog for "${action}": ${formatError(error)}`
    );
    return null;
  }
}

async function updateActionLog(
  logId: string | null,
  action: string,
  phase: string,
  reason: string,
  success: boolean,
  verification: VerificationResult | null
): Promise<void> {
  if (logId === null) return;
  try {
    await db.agentActionLog.update({
      where: { id: logId },
      data: {
        result: sanitizeJson({
          status: phase,
          verification: verification
            ? {
                outcome: verification.outcome,
                passed: verification.passed,
                detail: verification.detail,
              }
            : null,
        }) as object,
        success,
        reason,
      },
    });
  } catch (error) {
    logger.warn(
      `[EXECUTOR] Failed to update AgentActionLog "${logId}" (${action}): ${formatError(error)}`
    );
  }
}

interface MutationOutcome {
  ok: boolean;
  createdId: string | null;
  expected: Record<string, unknown> | null;
  prior: Record<string, unknown> | null;
  rollbackAvailable: boolean;
  transient: boolean;
  message: string;
  rollback?: () => Promise<void>;
}

interface PreconditionOutcome {
  ok: boolean;
  message: string;
}

async function taskExists(taskId: string): Promise<boolean> {
  const task = await db.task.findUnique({ where: { id: taskId }, select: { id: true } });
  return task !== null;
}

async function projectExists(projectId: string): Promise<boolean> {
  const project = await db.project.findUnique({ where: { id: projectId }, select: { id: true } });
  return project !== null;
}

async function userExists(userId: string): Promise<boolean> {
  const user = await db.user.findUnique({ where: { id: userId }, select: { id: true } });
  return user !== null;
}

async function tableHasColumn(table: string, column: string): Promise<boolean> {
  const rows = await db.$queryRaw<{ column_name: string }[]>`
    SELECT column_name FROM information_schema.columns
    WHERE table_name = ${table} AND column_name = ${column}`;
  return rows.length > 0;
}

async function runWithRetry(
  action: string,
  mutate: () => Promise<MutationOutcome>
): Promise<MutationOutcome> {
  let lastMessage = "";
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt += 1) {
    try {
      return await mutate();
    } catch (error) {
      lastMessage = formatError(error);
      const transient = isTransientPrismaError(error);
      if (attempt < MAX_ATTEMPTS && transient) {
        logger.warn(
          `[EXECUTOR] "${action}" attempt ${attempt}/${MAX_ATTEMPTS} hit a transient failure (${lastMessage}); retrying in ${RETRY_DELAY_MS}ms.`
        );
        await sleep(RETRY_DELAY_MS);
        continue;
      }
      return {
        ok: false,
        createdId: null,
        expected: null,
        prior: null,
        rollbackAvailable: false,
        transient,
        message: lastMessage,
      };
    }
  }
  return {
    ok: false,
    createdId: null,
    expected: null,
    prior: null,
    rollbackAvailable: false,
    transient: false,
    message: lastMessage,
  };
}

async function precondition(
  action: string,
  context: ExecutionContext
): Promise<PreconditionOutcome> {
  const taskId = context.taskId;
  const projectId = context.projectId;
  const targetUserId = context.targetUserId;

  if (action === "add_comment" || action === "update_task_status" ||
      action === "update_task_progress" || action === "assign_task") {
    if (!taskId) {
      return { ok: false, message: `taskId is required for ${action}` };
    }
    if (!(await taskExists(taskId))) {
      return { ok: false, message: `task "${taskId}" does not exist` };
    }
  }

  if (action === "record_blocker" || action === "create_task") {
    if (!projectId) {
      return { ok: false, message: `projectId is required for ${action}` };
    }
    if (!(await projectExists(projectId))) {
      return { ok: false, message: `project "${projectId}" does not exist` };
    }
  }

  if (action === "assign_task" || action === "send_notification") {
    if (!targetUserId) {
      return { ok: false, message: `targetUserId is required for ${action}` };
    }
    if (!(await userExists(targetUserId))) {
      return { ok: false, message: `target user "${targetUserId}" does not exist` };
    }
  }

  if (action === "create_task") {
    const dueDate = context.payload.dueDate;
    if (typeof dueDate !== "string" || dueDate.trim() === "") {
      return { ok: false, message: "create_task requires a dueDate (ISO string) because Task.dueDate is mandatory" };
    }
  }

  return { ok: true, message: "preconditions satisfied" };
}

async function mutate(
  action: string,
  context: ExecutionContext
): Promise<MutationOutcome> {
  const payload = context.payload;
  const taskId = context.taskId;
  const projectId = context.projectId;
  const targetUserId = context.targetUserId;

  switch (action) {
    case "add_comment": {
      const content = String(payload.content ?? "");
      if (!content.trim()) {
        return { ok: false, createdId: null, expected: null, prior: null, rollbackAvailable: false, transient: false, message: "add_comment requires a non-empty content" };
      }
      const commentUserId = String(payload.userId ?? context.agentId);
      return runWithRetry("add_comment", async () => {
        const comment = await db.comment.create({
          data: { content, taskId: taskId!, userId: commentUserId },
          select: { id: true, content: true, taskId: true, userId: true },
        });
        return {
          ok: true,
          createdId: comment.id,
          expected: { taskId: taskId!, content, userId: commentUserId },
          prior: null,
          rollbackAvailable: true,
          transient: false,
          message: `comment "${comment.id}" created`,
          rollback: async () => {
            await db.comment.delete({ where: { id: comment.id } });
            logger.warn(`[EXECUTOR][ROLLBACK] Deleted comment "${comment.id}".`);
          },
        };
      });
    }

    case "record_blocker": {
      const title = String(payload.title ?? payload.riskTitle ?? "Agent-recognized blocker");
      const severity = String(payload.severity ?? "medium");
      const description = String(payload.description ?? "");
      const riskTitle = String(payload.riskTitle ?? title);
      const clientId = String(payload.clientId ?? "");
      return runWithRetry("record_blocker", async () => {
        const blocker = await db.riskBlockage.create({
          data: {
            projectId: projectId!,
            clientId,
            riskTitle,
            title,
            severity,
            description,
            reportType: "risk",
            type: "risk",
            status: "open",
            category: String(payload.category ?? "general"),
          },
          select: { id: true, projectId: true, title: true, severity: true, description: true },
        });
        return {
          ok: true,
          createdId: blocker.id,
          expected: { projectId: projectId!, title, severity, description },
          prior: null,
          rollbackAvailable: true,
          transient: false,
          message: `blocker "${blocker.id}" recorded`,
          rollback: async () => {
            await db.riskBlockage.delete({ where: { id: blocker.id } });
            logger.warn(`[EXECUTOR][ROLLBACK] Deleted risk blocker "${blocker.id}".`);
          },
        };
      });
    }

    case "update_task_status": {
      const status = String(payload.status ?? "");
      if (!ALLOWED_TASK_STATUSES.has(status)) {
        return { ok: false, createdId: null, expected: null, prior: null, rollbackAvailable: false, transient: false, message: `update_task_status requires a valid status, got "${status}"` };
      }
      return runWithRetry("update_task_status", async () => {
        const prior = await db.task.findUnique({
          where: { id: taskId! },
          select: { status: true },
        });
        const updated = await db.task.update({
          where: { id: taskId! },
          data: { status },
          select: { id: true, status: true },
        });
        return {
          ok: true,
          createdId: null,
          expected: { status: updated.status },
          prior: { status: prior?.status ?? null },
          rollbackAvailable: true,
          transient: false,
          message: `task "${taskId}" status -> "${status}"`,
          rollback: async () => {
            if (prior) {
              await db.task.update({ where: { id: taskId! }, data: { status: prior.status } });
              logger.warn(`[EXECUTOR][ROLLBACK] Restored task "${taskId}" status to "${prior.status}".`);
            }
          },
        };
      });
    }

    case "update_task_progress": {
      if (!(await tableHasColumn("Task", "progress"))) {
        return {
          ok: false,
          createdId: null,
          expected: null,
          prior: null,
          rollbackAvailable: false,
          transient: false,
          message: "Task has no progress column in the current schema; refusing to guess a mapping (field_missing)",
        };
      }
      const progressNode = payload.progress;
      const progressValue = typeof progressNode === "number" ? progressNode : Number(progressNode ?? NaN);
      if (!Number.isFinite(progressValue)) {
        return { ok: false, createdId: null, expected: null, prior: null, rollbackAvailable: false, transient: false, message: "update_task_progress requires a numeric progress value" };
      }
      const clamped = Math.min(100, Math.max(0, Math.round(progressValue)));
      return runWithRetry("update_task_progress", async () => {
        const updated = await db.task.update({
          where: { id: taskId! },
          data: { progress: clamped } as unknown as Prisma.TaskUncheckedUpdateInput,
          select: { id: true, progress: true } as unknown as Prisma.TaskSelect,
        });
        return {
          ok: true,
          createdId: null,
          expected: { progress: (updated as unknown as { progress: number }).progress },
          prior: null,
          rollbackAvailable: false,
          transient: false,
          message: `task "${taskId}" progress -> ${clamped}`,
        };
      });
    }

    case "assign_task": {
      return runWithRetry("assign_task", async () => {
        const task = await db.task.findUnique({
          where: { id: taskId! },
          select: { userId: true, assignee: true },
        });
        const user = await db.user.findUnique({
          where: { id: targetUserId! },
          select: { id: true, name: true },
        });
        const assigneeName = user?.name ?? String(payload.assignee ?? targetUserId);
        await db.task.update({
          where: { id: taskId! },
          data: { userId: targetUserId, assignee: assigneeName, status: "assigned" },
          select: { id: true, userId: true, assignee: true },
        });
        return {
          ok: true,
          createdId: null,
          expected: { userId: targetUserId, assignee: assigneeName },
          prior: { userId: task?.userId ?? null, assignee: task?.assignee ?? null },
          rollbackAvailable: true,
          transient: false,
          message: `task "${taskId}" assigned to "${assigneeName}"`,
          rollback: async () => {
            await db.task.update({
              where: { id: taskId! },
              data: { userId: task?.userId ?? null, assignee: task?.assignee ?? "unassigned" },
            });
            logger.warn(`[EXECUTOR][ROLLBACK] Restored assignee of task "${taskId}".`);
          },
        };
      });
    }

    case "create_task": {
      const title = String(payload.title ?? "");
      if (!title.trim()) {
        return { ok: false, createdId: null, expected: null, prior: null, rollbackAvailable: false, transient: false, message: "create_task requires a title" };
      }
      const dueDate = new Date(String(payload.dueDate));
      return runWithRetry("create_task", async () => {
        const task = await db.task.create({
          data: {
            title,
            description: String(payload.description ?? ""),
            priority: String(payload.priority ?? "medium"),
            status: String(payload.status ?? "pending"),
            assignee: String(payload.assignee ?? "unassigned"),
            tags: Array.isArray(payload.tags) ? payload.tags.map(String) : [],
            projectId: projectId!,
            dueDate,
            userId: payload.userId ? String(payload.userId) : null,
          },
          select: { id: true, title: true, projectId: true, dueDate: true },
        });
        return {
          ok: true,
          createdId: task.id,
          expected: { projectId: projectId!, title },
          prior: null,
          rollbackAvailable: true,
          transient: false,
          message: `task "${task.id}" created in project "${projectId}"`,
          rollback: async () => {
            await db.task.delete({ where: { id: task.id } });
            logger.warn(`[EXECUTOR][ROLLBACK] Deleted created task "${task.id}".`);
          },
        };
      });
    }

    case "send_notification": {
      const title = String(payload.title ?? "");
      const message = String(payload.message ?? "");
      if (!title.trim() || !message.trim()) {
        return { ok: false, createdId: null, expected: null, prior: null, rollbackAvailable: false, transient: false, message: "send_notification requires a title and message" };
      }
      return runWithRetry("send_notification", async () => {
        const notification = await db.notification.create({
          data: { userId: targetUserId!, title, message },
          select: { id: true, userId: true, title: true, message: true },
        });
        return {
          ok: true,
          createdId: notification.id,
          expected: { userId: targetUserId, title, message },
          prior: null,
          rollbackAvailable: true,
          transient: false,
          message: `notification "${notification.id}" created`,
          rollback: async () => {
            await db.notification.delete({ where: { id: notification.id } });
            logger.warn(`[EXECUTOR][ROLLBACK] Deleted notification "${notification.id}".`);
          },
        };
      });
    }

    default:
      return {
        ok: false,
        createdId: null,
        expected: null,
        prior: null,
        rollbackAvailable: false,
        transient: false,
        message: `unknown executor action "${action}"`,
      };
  }
}

async function verify(
  action: string,
  context: ExecutionContext,
  outcome: MutationOutcome
): Promise<VerificationResult | null> {
  if (!outcome.ok || outcome.expected === null) return null;

  const taskId = context.taskId;
  const createdId = outcome.createdId;

  switch (action) {
    case "add_comment": {
      const row = await db.comment.findUnique({ where: { id: createdId! } });
      if (!row) {
        return { passed: false, outcome: "field_missing", detail: `comment "${createdId}" not found after creation`, actual: null, expected: outcome.expected };
      }
      return mergeVerification(compareFields({ taskId: row.taskId, content: row.content, userId: row.userId }, outcome.expected));
    }
    case "record_blocker": {
      const row = await db.riskBlockage.findUnique({ where: { id: createdId! } });
      if (!row) {
        return { passed: false, outcome: "field_missing", detail: `blocker "${createdId}" not found after creation`, actual: null, expected: outcome.expected };
      }
      return mergeVerification(compareFields({ projectId: row.projectId, title: row.title, severity: row.severity, description: row.description ?? "" }, outcome.expected));
    }
    case "update_task_status": {
      const row = await db.task.findUnique({ where: { id: taskId! }, select: { status: true } });
      if (!row) return null;
      return detectConflict(row.status, outcome.expected.status, outcome.prior?.status ?? null, "status");
    }
    case "update_task_progress": {
      const row = (await db.task.findUnique({
        where: { id: taskId! },
        select: { progress: true } as unknown as Prisma.TaskSelect,
      })) as unknown as { progress: number } | null;
      if (!row) return null;
      return mergeVerification(compareFields({ progress: row.progress }, outcome.expected));
    }
    case "assign_task": {
      const row = await db.task.findUnique({ where: { id: taskId! }, select: { userId: true, assignee: true } });
      if (!row) return null;
      const results = [
        detectConflict(row.userId, outcome.expected.userId, outcome.prior?.userId ?? null, "userId"),
        detectConflict(row.assignee, outcome.expected.assignee, outcome.prior?.assignee ?? null, "assignee"),
      ];
      return mergeVerification(results);
    }
    case "create_task": {
      const row = await db.task.findUnique({ where: { id: createdId! } });
      if (!row) {
        return { passed: false, outcome: "field_missing", detail: `task "${createdId}" not found after creation`, actual: null, expected: outcome.expected };
      }
      return mergeVerification(compareFields({ projectId: row.projectId ?? null, title: row.title }, outcome.expected));
    }
    case "send_notification": {
      const row = await db.notification.findUnique({ where: { id: createdId! } });
      if (!row) {
        return { passed: false, outcome: "field_missing", detail: `notification "${createdId}" not found after creation`, actual: null, expected: outcome.expected };
      }
      return mergeVerification(compareFields({ userId: row.userId, title: row.title, message: row.message }, outcome.expected));
    }
    default:
      return null;
  }
}

export async function execute(
  action: string,
  context: ExecutionContext
): Promise<ExecutionResult> {
  const startedAt = Date.now();
  const target = context.taskId ?? context.projectId ?? context.targetUserId ?? null;
  const span = startSpan(`execute_action:${action}`, "executor", {
    projectId: context.projectId ?? null,
    taskId: context.taskId ?? null,
    attributes: { action },
  });

  const plan: ExecutionResult = {
    action,
    status: "REFUSED_POLICY",
    success: false,
    policy: { allowed: false, requiresApproval: false, status: "denied", riskLevel: "HIGH" },
    verification: null,
    logId: null,
    reason: "",
    rolledBack: false,
    executionMs: 0,
    createdAt: new Date().toISOString(),
  };

  try {
    const circuitOk = await circuitBreaker.shouldAllow("execute");
    if (!circuitOk) {
      const snapshot = await circuitBreaker.state;
      plan.status = "CIRCUIT_OPEN";
      plan.reason = `Blocked by circuit breaker: circuit is ${snapshot.state.toUpperCase()}; writes are not permitted until it recovers. No mutation performed.`;
      plan.logId = await createActionLog(context, action, "CIRCUIT_OPEN", target, context.payload, plan.reason);
      logger.warn(`[EXECUTOR] ${plan.reason}`);
      plan.executionMs = Date.now() - startedAt;
      return plan;
    }

  let policy: PolicyDecision;
  if (context.policy) {
    policy = context.policy;
  } else {
    policy = await canPerformAction(context.agentId, action, {
      projectId: context.projectId,
      taskId: context.taskId,
      token: context.agentToken,
      payload: context.payload,
    });
  }

  plan.policy = {
    allowed: policy.allowed,
    requiresApproval: policy.requiresApproval,
    status: policy.status,
    riskLevel: policy.riskLevel,
  };

  const grant = context.executionGrant;

  if (!policy.allowed) {
    plan.status = "REFUSED_POLICY";
    plan.reason = `Refused: policy decision "${policy.status}" for "${action}" (${policy.reason}). No mutation performed.`;
    plan.logId = await createActionLog(context, action, "REFUSED_POLICY", target, context.payload, plan.reason);
    logger.warn(`[EXECUTOR] ${plan.reason}`);
    plan.executionMs = Date.now() - startedAt;
    return plan;
  }

  if (policy.requiresApproval && !(grant && grant.by && grant.by.trim() !== "")) {
    try {
      await db.agentDecision.create({
        data: {
          agentId: context.agentId,
          decision: JSON.stringify({
            action,
            projectId: context.projectId,
            taskId: context.taskId,
            targetUserId: context.targetUserId,
            payload: sanitizeJson(context.payload),
          }),
          basis: {
            action,
            riskLevel: policy.riskLevel,
            reason: policy.reason,
          },
        },
      });
      plan.status = "SKIPPED_RECOMMENDED";
      plan.reason = `Skipped execution: "${action}" is a ${policy.riskLevel}-risk action pending human approval. An AgentDecision recommendation was recorded instead of mutating state.`;
      plan.logId = await createActionLog(context, action, "SKIPPED_RECOMMENDED", target, context.payload, plan.reason);
      logger.info(`[EXECUTOR] ${plan.reason} (AgentDecision recorded).`);
      plan.executionMs = Date.now() - startedAt;
      return plan;
    } catch (error) {
      plan.status = "FAILED";
      plan.reason = `Failed to record AgentDecision recommendation: ${formatError(error)}`;
      plan.logId = await createActionLog(context, action, "FAILED", target, context.payload, plan.reason);
      plan.executionMs = Date.now() - startedAt;
      return plan;
    }
  }

  plan.logId = await createActionLog(
    context,
    action,
    "PENDING",
    target,
    context.payload,
    grant && grant.by ? `execution granted by ${grant.by}` : "pre-execution"
  );

  const preconditionResult = await precondition(action, context);
  if (!preconditionResult.ok) {
    plan.status = "PRECONDITION_FAILED";
    plan.reason = `Precondition failed: ${preconditionResult.message}. State was not modified.`;
    await updateActionLog(plan.logId, action, "PRECONDITION_FAILED", plan.reason, false, null);
    logger.error(`[EXECUTOR] ${plan.reason}`);
    plan.executionMs = Date.now() - startedAt;
    return plan;
  }

  let outcome: MutationOutcome;
  try {
    outcome = await mutate(action, context);
  } catch (error) {
    outcome = {
      ok: false,
      createdId: null,
      expected: null,
      prior: null,
      rollbackAvailable: false,
      transient: false,
      message: formatError(error),
    };
  }

  if (!outcome.ok) {
    const fieldMissing = outcome.message.includes("field_missing");
    plan.status = fieldMissing ? "FAILED" : "FAILED";
    plan.reason = fieldMissing
      ? `Execution failed: ${outcome.message}.`
      : `Execution failed (${outcome.transient ? "after 1 retry" : "permanent error"}): ${outcome.message}.`;
    plan.verification = fieldMissing
      ? { passed: false, outcome: "field_missing", detail: outcome.message, actual: null, expected: null }
      : null;
    await circuitBreaker.recordFailure("execution", { reason: outcome.message });
    await updateActionLog(plan.logId, action, "FAILED", plan.reason, false, plan.verification);
    logger.error(`[EXECUTOR] ${plan.reason}`);
    plan.executionMs = Date.now() - startedAt;
    return plan;
  }

  plan.reason = outcome.message;

  const verification = await verify(action, context, outcome);
  if (verification === null) {
    plan.status = "FAILED";
    plan.reason = `${outcome.message}; verification could not be completed.`;
    await updateActionLog(plan.logId, action, "FAILED", plan.reason, false, null);
    logger.error(`[EXECUTOR] ${plan.reason}`);
    plan.executionMs = Date.now() - startedAt;
    return plan;
  }

  plan.verification = verification;

  if (verification.passed) {
    plan.status = "SUCCESS";
    plan.success = true;
    plan.reason = `${outcome.message}; verification passed (${verification.detail}).`;
    await circuitBreaker.recordSuccess();
    await updateActionLog(plan.logId, action, "SUCCESS", plan.reason, true, verification);
    logger.info(`[EXECUTOR] ${plan.reason}`);
    plan.executionMs = Date.now() - startedAt;
    return plan;
  }

  if (verification.outcome === "conflict") {
    plan.status = "VERIFIED_MISMATCH";
    plan.reason = `${outcome.message}; verification detected a concurrent modification (${verification.detail}). Not rolled back, so the other actor's change is preserved; alerting for manual review.`;
    await circuitBreaker.recordFailure("verification_mismatch", { reason: verification.detail });
    await updateActionLog(plan.logId, action, "VERIFIED_MISMATCH", plan.reason, false, verification);
    logger.error(`[EXECUTOR] ${plan.reason}`);
    plan.executionMs = Date.now() - startedAt;
    return plan;
  }

  if (outcome.rollbackAvailable && outcome.rollback) {
    try {
      await outcome.rollback();
      plan.rolledBack = true;
      plan.status = "VERIFY_FAILED";
      plan.reason = `${outcome.message}; verification failed (${verification.detail}). State rolled back successfully.`;
      await updateActionLog(plan.logId, action, "VERIFY_FAILED", plan.reason, false, verification);
      logger.error(`[EXECUTOR] ${plan.reason}`);
    } catch (rollbackError) {
      plan.status = "VERIFY_FAILED";
      plan.reason = `${outcome.message}; verification failed (${verification.detail}). ROLLBACK ATTEMPT FAILED: ${formatError(rollbackError)}. State may be inconsistent — manual review required.`;
      await updateActionLog(plan.logId, action, "VERIFY_FAILED", plan.reason, false, verification);
      logger.error(`[EXECUTOR] ${plan.reason}`);
    }
  } else {
    plan.status = "VERIFY_FAILED";
    plan.reason = `${outcome.message}; verification failed (${verification.detail}). No rollback was available.`;
    await updateActionLog(plan.logId, action, "VERIFY_FAILED", plan.reason, false, verification);
    logger.error(`[EXECUTOR] ${plan.reason}`);
  }

    plan.executionMs = Date.now() - startedAt;
    return plan;
  } finally {
    span.end(plan.success ? "OK" : "ERROR", plan.reason ? plan.reason : null, {
      status: plan.status,
      success: plan.success,
    });
  }
}