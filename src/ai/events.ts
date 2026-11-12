import { Client } from "pg";
import db from "@/lib/client";
import { logger } from "@/ai/logger";
import { circuitBreaker, type BreakerActionType } from "@/ai/circuit-breaker";
import { buildProjectContext } from "@/ai/context";
import { evaluate } from "@/ai/reasoner";
import { execute } from "@/ai/executor";
import { getStealthPolicy, getStealthPolicyAsync, setStealthPolicy } from "@/ai/policy";
import { processEmployeeUpdate } from "@/ai/nlu";
import { loadConfig } from "@/ai/config";
import { logAction, recordDecision, storeMemory, maskPii } from "@/ai/audit";
import { workflowLock } from "@/ai/scheduler";
import { emitEvent } from "@/lib/socket";
import { startSpan } from "@/ai/telemetry";
import { buildTrainingSample } from "@/ai/dataset-formatter";

export type AgentEventType = "TASK_COMPLETED" | "BLOCKER_CREATED" | "COMMENT_ADDED" | "PROJECT_INGESTED";

export const EVENT_CHANNELS: Record<AgentEventType, string> = {
  TASK_COMPLETED: "agent_task_completed",
  BLOCKER_CREATED: "agent_blocker_created",
  COMMENT_ADDED: "agent_comment_added",
  PROJECT_INGESTED: "agent_project_ingested",
};

export interface AgentEvent {
  id: string;
  type: AgentEventType;
  at: string;
  payload: Record<string, unknown>;
}

export type AgentEventHandler = (event: AgentEvent) => Promise<void>;

export function formatError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

export class EventListener {
  private handlers = new Map<AgentEventType, AgentEventHandler[]>();
  private client: Client | null = null;
  private stopped = false;
  private readonly agentId: string;

  constructor(agentId = loadConfig().agentId) {
    this.agentId = agentId;
  }

  subscribe(eventType: AgentEventType, handler: AgentEventHandler): void {
    const list = this.handlers.get(eventType) ?? [];
    list.push(handler);
    this.handlers.set(eventType, list);
    logger.info(`[EVENTS] Subscribed handler to "${eventType}" (total ${list.length}).`);
  }

  getSubscribedChannels(): string[] {
    return Object.values(EVENT_CHANNELS);
  }

  async handleEvent(event: AgentEvent): Promise<void> {
    const channel = EVENT_CHANNELS[event.type];
    const lockName = `event:${typeShort(event.type)}:${event.id}`;

    if (!channel) {
      logger.warn(`[EVENTS] Unknown event type "${event.type}"; ignoring.`);
      return;
    }

    const handlers = this.handlers.get(event.type) ?? [];
    if (handlers.length === 0) {
      logger.warn(`[EVENTS] No handlers subscribed for "${event.type}"; ignoring event ${event.id}.`);
      return;
    }

    const rootSpan = startSpan(`event_observer:${event.type}`, "events", {
      projectId: (event.payload?.projectId as string) ?? null,
      taskId: (event.payload?.taskId as string) ?? null,
      attributes: { eventType: event.type, eventId: event.id },
    });

    const processed = await workflowLock.run(lockName, async () => {
      let allowed = true;
      for (const handler of handlers) {
        try {
          const snapshot = await circuitBreaker.state;
          const actionType: BreakerActionType =
            event.type === "COMMENT_ADDED" ? "read" : "read";
          const circuitOk = await circuitBreaker.shouldAllow(actionType);
          if (!circuitOk) {
            logger.warn(
              `[EVENTS] Event "${event.type}" (${event.id}) skipped: circuit is ${snapshot.state.toUpperCase()}.`
            );
            allowed = false;
            break;
          }
          await handler(event);
        } catch (error) {
          logger.error(
            `[EVENTS] Handler for "${event.type}" (${event.id}) failed (${formatError(error)}). The failure is logged; the agent continues running.`
          );
          allowed = false;
          await logAction(
            this.agentId,
            "authorization_check",
            null,
            { eventType: event.type, eventId: event.id },
            false,
            { status: "FAILED", error: formatError(error) },
            `Event "${event.type}" handler failed: ${formatError(error)}.`,
            null
          );
        }
      }
      return allowed;
    });

    rootSpan.end(processed.status === "executed" ? "OK" : "CANCELLED", null, {
      lockStatus: processed.status,
    });

    if (processed.status === "skipped") {
      logger.warn(`[EVENTS] Event "${event.type}" (${event.id}) dropped: the channel is busy processing another event.`);
    }
  }

  private reconnectTimer: NodeJS.Timeout | null = null;
  private isReconnecting = false;

  async start(attempt = 1): Promise<void> {
    this.stopped = false;
    const dbUri = process.env.DB_URI;
    if (!dbUri) {
      throw new Error("[EVENTS] DB_URI is required to start the Postgres LISTEN connection.");
    }

    if (this.client) {
      try {
        await this.client.end();
      } catch (error) {
        // Ignored
      }
    }
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    this.client = new Client({ connectionString: dbUri });

    this.client.on("notification", (message) => {
      void this.onNotification(message.channel, message.payload);
    });

    this.client.on("error", (error) => {
      logger.error(`[EVENTS] Postgres LISTEN connection error: ${formatError(error)}`);
      void this.handleDisconnect();
    });

    this.client.on("end", () => {
      logger.warn("[EVENTS] Postgres LISTEN connection ended.");
      void this.handleDisconnect();
    });

    try {
      await this.client.connect();
      const channels = this.getSubscribedChannels();
      for (const channel of channels) {
        await this.client.query(`LISTEN ${channel}`);
      }
      logger.info(`[EVENTS] Reconnected and listening for NOTIFY on: ${channels.join(", ")}.`);
      this.isReconnecting = false;
    } catch (error) {
      this.isReconnecting = false;
      logger.error(`[EVENTS] Connection failed (attempt ${attempt}): ${formatError(error)}`);
      
      const delay = Math.min(1000 * Math.pow(2, attempt - 1), 30000);
      logger.info(`[EVENTS] Retrying connection in ${delay}ms...`);
      this.reconnectTimer = setTimeout(() => {
        void this.start(attempt + 1);
      }, delay);
    }
  }

  async stop(): Promise<void> {
    this.stopped = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.client) {
      try {
        await this.client.end();
      } catch (error) {
        logger.warn(`[EVENTS] Error closing LISTEN connection: ${formatError(error)}`);
      }
      this.client = null;
    }
    logger.info("[EVENTS] Stopped listening for database events.");
  }

  private async handleDisconnect(): Promise<void> {
    if (this.stopped || this.isReconnecting) return;
    this.isReconnecting = true;
    logger.warn("[EVENTS] Connection lost. Triggering reconnection loop...");
    await new Promise((resolve) => setTimeout(resolve, 1000));
    await this.start();
  }

  private async onNotification(channel: string, payload: string | undefined): Promise<void> {
    if (this.stopped) return;
    const eventType = (Object.keys(EVENT_CHANNELS) as AgentEventType[]).find(
      (candidate) => EVENT_CHANNELS[candidate] === channel
    );
    if (!eventType) {
      logger.debug(`[EVENTS] Ignoring notification on unknown channel "${channel}".`);
      return;
    }

    let body: Record<string, unknown>;
    try {
      body = JSON.parse(payload ?? "{}") as Record<string, unknown>;
    } catch {
      logger.warn(`[EVENTS] Channel "${channel}" delivered malformed JSON payload; ignoring.`);
      return;
    }

    const event: AgentEvent = {
      id: asString(body.eventId) || `evt-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      type: eventType,
      at: new Date().toISOString(),
      payload: body,
    };
    logger.info(`[EVENTS] Received "${eventType}" (${event.id}).`);
    await this.handleEvent(event);
  }
}

function typeShort(type: AgentEventType): string {
  return type.slice(0, 4).toLowerCase();
}

async function projectLeaders(projectId: string): Promise<{ userId: string; name: string }[]> {
  const rows = await db.projectMember.findMany({
    where: { projectId, isLeader: true },
    select: { userId: true, user: { select: { name: true } } },
  });
  return rows.map((row) => ({ userId: row.userId, name: row.user.name }));
}

async function notifyUser(
  userId: string,
  projectId: string | null,
  title: string,
  message: string,
  reason: string
): Promise<void> {
  const agentId = loadConfig().agentId;
  const result = await execute("send_notification", {
    agentId,
    projectId,
    targetUserId: userId,
    payload: { title, message },
  });
  await logAction(
    agentId,
    "authorization_check",
    null,
    { userId, title, message: maskPii(message) },
    result.success,
    { status: result.status, reason: result.reason },
    reason,
    projectId
  );
}

export async function handleTaskCompleted(event: AgentEvent): Promise<void> {
  const taskId = asString(event.payload.taskId);
  const projectId = asString(event.payload.projectId);
  if (!taskId || !projectId) {
    logger.warn(`[EVENTS] TASK_COMPLETED without taskId/projectId; ignoring.`);
    return;
  }

  const task = await db.task.findUnique({
    where: { id: taskId },
    select: { id: true, title: true, status: true },
  });
  if (!task) {
    logger.warn(`[EVENTS] TASK_COMPLETED references unknown task "${taskId}".`);
    return;
  }

  const dependents = await db.taskDependency.findMany({
    where: { dependsOnTaskId: taskId, projectId },
    select: { taskId: true },
  });

  for (const dependent of dependents) {
    try {
      const context = await buildProjectContext(projectId);
      const round = await evaluate(context, {
        taskId: dependent.taskId,
        runLlm: false,
      });
      const readiness = round.task;
      const recommendation = round.assignment?.ranked[0] ?? null;

      if (readiness && readiness.ready) {
        await logAction(
          this_agentId(),
          "task_updated",
          dependent.taskId,
          { projectId, dependencyCompleted: taskId, resolvedDependencies: readiness.reason },
          true,
          { status: "actionable", recommendedAssignee: recommendation?.name ?? null },
          `Task "${dependent.taskId}" became actionable after "${task.title}" completed.`,
          projectId
        );
        const assigneeId = await taskAssigneeId(dependent.taskId);
        if (assigneeId) {
          await notifyUser(
            assigneeId,
            projectId,
            "Task unlocked",
            `A dependency you are waiting on (task "${task.title}") is complete. Task "${dependent.taskId}" is now actionable.`,
            "Dependency unblocked after task completion."
          );
        }
        if (recommendation && !assigneeId) {
          await recordDecision(
            this_agentId(),
            `Recommend assigning "${dependent.taskId}" to ${recommendation.name} (score ${recommendation.score}).`,
            {
              action: "assign_task",
              riskLevel: "MEDIUM",
              reason: readiness.reason,
              projectId,
              taskId: dependent.taskId,
              confidence: recommendation.score >= 0.8 ? 0.9 : 0.5,
              scoring: { score: recommendation.score },
            },
            [readiness],
            projectId
          );
        }
      } else {
        await logAction(
          this_agentId(),
          "task_updated",
          dependent.taskId,
          { projectId, dependencyCompleted: taskId, stillBlocked: readiness?.unresolvedDependencies ?? [] },
          false,
          { status: "still_blocked" },
          `Task "${dependent.taskId}" still blocked after "${task.title}" completed: ${readiness?.reason ?? "unknown"}.`,
          projectId
        );
      }
    } catch (error) {
      logger.error(`[EVENTS] Task-completion cascade failed for dependent "${dependent.taskId}": ${formatError(error)}`);
    }
  }

  await logAction(
    this_agentId(),
    "task_status_changed",
    taskId,
    { projectId, dependents: dependents.length },
    true,
    { status: "completed" },
    `Task "${task.title}" completed; re-evaluated ${dependents.length} dependent(s).`,
    projectId
  );

  // Trigger full evaluation to catch group shifts and other unblocked tasks
  await triggerActiveAllocations(projectId).catch(err => {
    logger.error(`[EVENTS] Failed to trigger active allocations after task completed: ${formatError(err)}`);
  });
}

export async function handleBlockerCreated(event: AgentEvent): Promise<void> {
  const blockerId = asString(event.payload.blockerId);
  const projectId = asString(event.payload.projectId);
  const taskId = asString(event.payload.taskId);
  if (!blockerId || !projectId) {
    logger.warn(`[EVENTS] BLOCKER_CREATED without blockerId/projectId; ignoring.`);
    return;
  }

  const blocker = await db.riskBlockage.findUnique({
    where: { id: blockerId },
    select: { id: true, title: true, severity: true, description: true },
  });
  if (!blocker) {
    logger.warn(`[EVENTS] BLOCKER_CREATED references unknown blocker "${blockerId}".`);
    return;
  }

  let onCriticalPath = false;
  let blockedTaskTitle: string | null = null;
  if (taskId) {
    const task = await db.task.findUnique({
      where: { id: taskId },
      select: { id: true, title: true },
    });
    blockedTaskTitle = task?.title ?? null;
    const dependents = await db.taskDependency.count({ where: { dependsOnTaskId: taskId, projectId } });
    onCriticalPath = dependents > 0;
  }

  await storeMemory(
    projectId,
    "blocker_context",
    `Blocker "${blocker.title}" (severity ${blocker.severity}) reported on project "${projectId}". ` +
      `${blockedTaskTitle ? `Blocked task: ${blockedTaskTitle}. ` : ""}` +
      `${onCriticalPath ? "The blocked task is on the critical path (other tasks depend on it)." : "Not on the critical path (no dependents)."} ` +
      `${blocker.description || "No description provided."}`,
    { blockerId, taskId: taskId ?? null, severity: blocker.severity, onCriticalPath }
  );
  await logAction(
    this_agentId(),
    "blocker_recorded",
    taskId ?? null,
    { projectId, blockerId, onCriticalPath, severity: blocker.severity },
    true,
    { status: "assessed", onCriticalPath },
    `Blocker "${blocker.title}" assessed; criticalPath=${onCriticalPath}.`,
    projectId
  );

  if (onCriticalPath) {
    const leaders = await projectLeaders(projectId);
    for (const leader of leaders) {
      await notifyUser(
        leader.userId,
        projectId,
        `Critical blocker: ${blocker.title}`,
        `Blocker "${blocker.title}" (severity ${blocker.severity}) is on the critical path` +
          `${blockedTaskTitle ? ` — task "${blockedTaskTitle}"` : ""}. Immediate attention is recommended.`,
        "Critical-path blocker created; notifying the project leader."
      );
    }
    logger.warn(`[EVENTS] BLOCKER_CREATED on critical path for project "${projectId}"; notified ${leaders.length} leader(s).`);
  }
  
  // Auto-allocate other ready tasks since developer capacity might be freed by this blocker
  triggerActiveAllocations(projectId).catch(err => {
    logger.error(`[EVENTS] Blocker-triggered active allocations failed: ${formatError(err)}`);
  });
}

const UPDATE_PATTERN = /(progress|status|%|done|blocked|stuck|complete|finished|updated|on hold|waiting)/i;

export async function handleCommentAdded(event: AgentEvent): Promise<void> {
  const commentId = asString(event.payload.commentId);
  const taskId = asString(event.payload.taskId);
  const commenterId = asString(event.payload.userId);
  const content = asString(event.payload.content);
  if (!commentId || !taskId || !commenterId || !content.trim()) {
    logger.warn(`[EVENTS] COMMENT_ADDED missing commentId/taskId/userId/content; ignoring.`);
    return;
  }

  const comment = await db.comment.findUnique({
    where: { id: commentId },
    select: { id: true, content: true, taskId: true, userId: true },
  });
  if (!comment) {
    logger.warn(`[EVENTS] COMMENT_ADDED references unknown comment "${commentId}".`);
    return;
  }

  const task = await db.task.findUnique({
    where: { id: taskId },
    select: { id: true, projectId: true, title: true },
  });
  const projectId = task?.projectId ?? null;
  if (!projectId) {
    logger.warn(`[EVENTS] COMMENT_ADDED: task "${taskId}" has no project; skipping NLU processing.`);
    return;
  }

  if (commenterId === this_agentId()) {
    logger.debug("[EVENTS] COMMENT_ADDED from the agent itself; ignoring to avoid loops.");
    return;
  }

  const isManager = await db.projectMember.findFirst({
    where: { projectId, userId: commenterId, isLeader: true },
    select: { id: true },
  });
  if (isManager) {
    logger.debug("[EVENTS] COMMENT_ADDED from a project manager; not treated as an employee update.");
    return;
  }

  if (!UPDATE_PATTERN.test(content)) {
    logger.debug("[EVENTS] COMMENT_ADDED does not look like a status update; ignoring.");
    return;
  }

  const isBlocker = /(blocked|stuck|missing|timeout|permission|cannot|finalized)/i.test(content);
  if (isBlocker) {
    await notifyManagerOfBlocker(projectId, task.title, content);
    await db.task.update({
      where: { id: taskId },
      data: { status: "on-hold" }
    });
    await logAction(
      this_agentId(),
      "blocker_recorded",
      taskId,
      { projectId, commentId },
      true,
      {},
      `Blocker detected in comment "${commentId}" and task updated to On Hold.`,
      projectId
    );

    // Auto-allocate other ready tasks since developer capacity is now freed
    triggerActiveAllocations(projectId).catch(err => {
      logger.error(`[EVENTS] Comment-blocker active allocations failed: ${formatError(err)}`);
    });
  }

  const members = await db.projectMember.findMany({
    where: { projectId },
    select: { userId: true, user: { select: { name: true } } },
  });
  const knownUsers = members.map((member) => ({ id: member.userId, name: member.user.name }));

  try {
    const result = await processEmployeeUpdate(content, taskId, {
      taskId,
      taskTitle: task.title,
      projectId,
      knownUsers,
    });
    await logAction(
      this_agentId(),
      "employee_reply_parsed",
      taskId,
      { projectId, commentId, skill: result.status },
      result.status === "parsed",
      { status: result.status, step: result.step },
      `Employee update from comment "${commentId}" parsed with status "${result.status}".`,
      projectId
    );

    if (result.status === "parsed") {
      const parsed = result.parsed;
      if (parsed.confidence >= 0.8) {
        await applyParsedUpdate(taskId, projectId, task.title, parsed, commentId);
        if (parsed.blocker === true) {
          await notifyManagerOfBlocker(projectId, task.title, content);
        }
      }
    } else if (result.status === "deferred") {
      await notifyManagerOfDeferredUpdate(projectId, task.title, content);
    } else if (result.status === "clarify") {
      await storeMemory(
        projectId,
        "conversation",
        `Clarification needed for task "${taskId}" (comment "${commentId}"): ${result.question}`,
        { taskId, commentId, reason: "clarify", confidence: result.parsed.confidence ?? null }
      );
    }
  } catch (error) {
    logger.error(`[EVENTS] COMMENT_ADDED NLU processing failed (${formatError(error)}); storing raw context.`);
    await storeMemory(
      projectId,
      "conversation",
      `Employee update could not be parsed (comment "${commentId}", task "${taskId}"): ${content}`,
      { taskId, commentId, error: formatError(error) }
    );
  }
}

type ParsedEmployeeUpdate = {
  status?: string | null;
  progressPercent?: number | null;
  blocker?: boolean | null;
  blockerDescription?: string | null;
  nextSteps?: string | null;
};

async function applyParsedUpdate(
  taskId: string,
  projectId: string,
  taskTitle: string,
  parsed: ParsedEmployeeUpdate,
  commentId: string
): Promise<void> {
  const agentId = this_agentId();

  if (parsed.blocker === true) {
    const blockerResult = await execute("record_blocker", {
      agentId,
      projectId,
      taskId,
      payload: {
        clientId: "",
        title: `Blocked: ${taskTitle}`,
        riskTitle: `Blocked: ${taskTitle}`,
        severity: "high",
        description: parsed.blockerDescription ?? `Employee-reported blocker via comment "${commentId}".`,
        category: "employee_reported",
      },
    });
    await logAction(
      agentId,
      "blocker_recorded",
      taskId,
      { projectId, commentId, status: blockerResult.status },
      blockerResult.success,
      { status: blockerResult.status },
      blockerResult.reason,
      projectId
    );
  }

  if (parsed.status && ["pending", "assigned", "in-progress", "completed"].includes(parsed.status)) {
    const statusResult = await execute("update_task_status", {
      agentId,
      projectId,
      taskId,
      payload: { status: parsed.status },
    });
    await logAction(
      agentId,
      "task_status_changed",
      taskId,
      { projectId, commentId, status: parsed.status, result: statusResult.status },
      statusResult.success,
      { status: statusResult.status },
      statusResult.reason,
      projectId
    );
  }

  if (parsed.progressPercent !== null) {
    const progressResult = await execute("update_task_progress", {
      agentId,
      projectId,
      taskId,
      payload: { progress: parsed.progressPercent },
    });
    await logAction(
      agentId,
      "task_updated",
      taskId,
      { projectId, commentId, progress: parsed.progressPercent, result: progressResult.status },
      progressResult.success,
      { status: progressResult.status },
      progressResult.reason,
      projectId
    );
  }

  if (parsed.nextSteps && parsed.nextSteps.trim()) {
    const project = await db.project.findUnique({
      where: { id: projectId },
      select: { name: true },
    });
    const comment = await db.comment.findUnique({
      where: { id: commentId },
      select: { userId: true },
    });
    const addResult = await execute("add_comment", {
      agentId,
      projectId,
      taskId,
      payload: {
        content: `Agent note (from employee update in comment "${commentId}"): ${parsed.nextSteps}`,
        userId: comment?.userId ?? agentId,
      },
    });
    await logAction(
      agentId,
      "blocker_recorded",
      taskId,
      { projectId, commentId, status: addResult.status },
      true,
      { status: addResult.status, projectName: project?.name ?? null },
      addResult.reason,
      projectId
    );
  }
}

async function taskAssigneeId(taskId: string): Promise<string | null> {
  const task = await db.task.findUnique({
    where: { id: taskId },
    select: { userId: true },
  });
  return task?.userId ?? null;
}

export async function triggerProjectInitialAllocation(projectId: string): Promise<void> {
  return triggerActiveAllocations(projectId);
}

export async function triggerActiveAllocations(projectId: string): Promise<void> {
  const lockName = `allocations:${projectId}`;
  const result = await workflowLock.run(lockName, async () => {
    logger.info(`[EVENTS] Triggering active task allocations for project "${projectId}"`);
    
    // Concurrency Throttle: Limit pending recommendations to maximum 3 per project (unless Stealth Mode auto-pilot is active)
    const stealthPolicy = await getStealthPolicyAsync(projectId);
    const isStealthActive = Boolean(stealthPolicy && stealthPolicy.isActive);

    const pendingCount = await db.agentDecision.count({
      where: { projectId }
    });
    if (!isStealthActive && pendingCount >= 3) {
      logger.info(`[EVENTS] Project "${projectId}" already has ${pendingCount} pending decisions; skipping new allocations to avoid bulk spam.`);
      return;
    }

    // Fetch all unassigned tasks in the project
    const tasks = await db.task.findMany({
      where: { projectId, userId: null },
      select: { id: true, title: true }
    });

    if (tasks.length === 0) {
      logger.info(`[EVENTS] No unassigned tasks for project "${projectId}" to allocate.`);
      return;
    }

    // If in SELECT mode, prioritize tasks according to the admin's exact selectedTaskIds queue order
    if (stealthPolicy && stealthPolicy.modeType === "SELECT" && Array.isArray(stealthPolicy.selectedTaskIds) && stealthPolicy.selectedTaskIds.length > 0) {
      const selectedMap = new Map(stealthPolicy.selectedTaskIds.map((id, index) => [id, index]));
      tasks.sort((a, b) => {
        const indexA = selectedMap.has(a.id) ? selectedMap.get(a.id)! : 999999;
        const indexB = selectedMap.has(b.id) ? selectedMap.get(b.id)! : 999999;
        return indexA - indexB;
      });
    }

    let context = await buildProjectContext(projectId);
    for (const task of tasks) {
      try {
        const round = await evaluate(context, {
          taskId: task.id,
          runLlm: false,
        });

        const readiness = round.task;
        const recommendation = round.assignment?.ranked[0] ?? null;

        if (readiness && readiness.ready && recommendation) {
          const rawScore = recommendation.score;
          const confidence = rawScore >= 0.8 ? 0.9 : (rawScore >= 0.6 ? 0.7 : 0.5);
          const stealth = await getStealthPolicyAsync(projectId);
          const parseEndDate = (dStr?: string | null): number | null => {
            if (!dStr) return null;
            const s = dStr.trim();
            if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return Date.parse(`${s}T23:59:59.999Z`);
            return Date.parse(s);
          };
          const endMs = parseEndDate(stealth?.endDate);
          const inRange = (!stealth?.startDate || Date.parse(stealth.startDate) <= Date.now()) &&
                          (endMs === null || (!Number.isNaN(endMs) && endMs >= Date.now()));

          if (stealth && stealth.isActive && inRange && confidence >= stealth.minConfidenceScore) {
            // Check SELECT mode constraint: If in SELECT mode, only auto-assign if task.id is in selectedTaskIds
            const modeType = stealth.modeType || "AUTO";
            const selectedIds = stealth.selectedTaskIds || [];
            if (modeType === "SELECT" && !selectedIds.includes(task.id)) {
              logger.info(`[EVENTS][STEALTH] Task "${task.id}" is ready but skipped because it is not in the admin's SELECT queue.`);
              continue;
            }

            logger.info(`[EVENTS][STEALTH] Auto-assigning task "${task.id}" to "${recommendation.name}" via Stealth Auto-Pilot [Mode: ${modeType}] (confidence ${confidence} >= ${stealth.minConfidenceScore}).`);
            const execResult = await execute("assign_task", {
              agentId: this_agentId(),
              projectId,
              taskId: task.id,
              targetUserId: recommendation.userId,
              payload: { assignee: recommendation.name, confidence: rawScore },
              executionGrant: { by: "StealthAutoPilot" },
            });
            if (execResult.success) {
              emitEvent("task:status:changed", { taskId: task.id, projectId, status: "assigned", assignee: recommendation.name });
              
              // If in SELECT mode, remove the assigned task from the selectedTaskIds queue and persist update
              if (modeType === "SELECT" && selectedIds.includes(task.id)) {
                const updatedSelectedIds = selectedIds.filter((id) => id !== task.id);
                const newPolicy = {
                  ...stealth,
                  selectedTaskIds: updatedSelectedIds,
                };
                setStealthPolicy(newPolicy);
                stealth.selectedTaskIds = updatedSelectedIds;
                logger.info(`[EVENTS][STEALTH] Removed assigned task "${task.id}" from SELECT queue. Remaining queued: ${updatedSelectedIds.length}`);
              }

              // Refresh context for next task in batch loop so candidate capacity updates
              context = await buildProjectContext(projectId);
              continue;
            } else {
              logger.warn(`[EVENTS][STEALTH] Auto-assignment failed for task "${task.id}": ${execResult.reason}. Falling back to recommendation.`);
            }
          }

          // Prevent duplicate recommendations for the same task in this project using exact task ID phrasing
          const existing = await db.agentDecision.findFirst({
            where: {
              projectId,
              decision: { contains: task.id }
            }
          });

          if (existing) {
            logger.info(`[EVENTS] Decision for task "${task.id}" already exists; skipping duplicate creation.`);
            continue;
          }

          logger.info(`[EVENTS] Allocation recommendation generated for ready task "${task.title}": ${recommendation.name}`);
          
          const trainingSample = buildTrainingSample(
            task.title,
            task.priority,
            task.tags ?? [],
            projectId,
            round.assignment?.ranked.map((r) => ({
              name: r.name,
              userId: r.userId,
              workloadPercent: r.components.workloadScore !== null ? Math.round((1 - r.components.workloadScore) * 100) : null,
              score: r.score,
            })) ?? [],
            recommendation.name,
            recommendation.basis ?? []
          );

          await recordDecision(
            this_agentId(),
            `Recommend assigning "${task.id}" to ${recommendation.name} (score ${recommendation.score}).`,
            {
              action: "assign_task",
              riskLevel: "MEDIUM",
              reason: readiness.reason,
              projectId,
              taskId: task.id,
              confidence,
              scoring: { score: recommendation.score },
              training_sample: trainingSample,
            },
            [readiness],
            projectId
          );
        }
      } catch (taskErr) {
        logger.error(`[EVENTS] Failed to allocate task "${task.id}": ${taskErr instanceof Error ? taskErr.message : String(taskErr)}`);
      }
    }
  });

  if (result.status === "skipped") {
    logger.info(`[EVENTS] Allocation for project "${projectId}" skipped: another allocation pass is already running.`);
  }
}

function this_agentId(): string {
  return loadConfig().agentId;
}

async function getProjectLeaderId(projectId: string): Promise<string | null> {
  const leader = await db.projectMember.findFirst({
    where: { projectId, isLeader: true },
    select: { userId: true }
  });
  if (leader) return leader.userId;

  // Fallback to first superadmin
  const fallback = await db.user.findFirst({
    where: { subRole: { name: "SUPERADMIN" } },
    select: { id: true }
  });
  return fallback?.id ?? null;
}

export async function sendAgentDm(toUserId: string, content: string): Promise<void> {
  try {
    const dm = await db.directMessage.create({
      data: {
        content,
        fromUserId: "agent-service-001",
        toUserId,
      },
      include: {
        from: { select: { id: true, name: true, image: true } },
        to: { select: { id: true, name: true, image: true } },
        attachments: true,
      }
    });

    // Notify user over Socket
    await emitEvent(toUserId, "chat:dm:new", dm);
    logger.info(`[EVENTS] Sent proactive AI DM to user "${toUserId}": "${content.slice(0, 50)}..."`);
  } catch (err) {
    logger.error(`[EVENTS] Failed to send proactive AI DM: ${err instanceof Error ? err.message : String(err)}`);
  }
}

export async function notifyManagerOfBlocker(projectId: string, taskTitle: string, commentContent: string): Promise<void> {
  const managerId = await getProjectLeaderId(projectId);
  if (!managerId) return;
  await sendAgentDm(
    managerId,
    `🚨 **Blocker Detected:** A team member reported a blocker on task **"${taskTitle}"**: "${commentContent.slice(0, 100)}..."\n\nTask status has been updated to **Blocked**.`
  );
}

export async function notifyManagerOfDeferredUpdate(projectId: string, taskTitle: string, commentContent: string): Promise<void> {
  const managerId = await getProjectLeaderId(projectId);
  if (!managerId) return;
  await sendAgentDm(
    managerId,
    `⚠️ **Low-Confidence Update:** A task update on task **"${taskTitle}"** could not be verified with high confidence and has been routed to **Agent Memory** for your manual review: "${commentContent.slice(0, 100)}..."`
  );
}

export async function handleProjectIngested(event: AgentEvent): Promise<void> {
  const projectId = event.payload.projectId as string;
  if (!projectId) return;
  logger.info(`[EVENTS] Received PROJECT_INGESTED event for project "${projectId}". Running allocations...`);
  await new Promise((resolve) => setTimeout(resolve, 150));
  await triggerProjectInitialAllocation(projectId);
}

export function registerDefaultEventHandlers(listener: EventListener): void {
  listener.subscribe("TASK_COMPLETED", handleTaskCompleted);
  listener.subscribe("BLOCKER_CREATED", handleBlockerCreated);
  listener.subscribe("COMMENT_ADDED", handleCommentAdded);
  listener.subscribe("PROJECT_INGESTED", handleProjectIngested);
}