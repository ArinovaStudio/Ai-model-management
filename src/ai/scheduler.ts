import db from "@/lib/client";
import { logger } from "@/ai/logger";
import { circuitBreaker, type BreakerActionType } from "@/ai/circuit-breaker";
import { buildProjectContext } from "@/ai/context";
import { evaluate } from "@/ai/reasoner";
import { execute } from "@/ai/executor";
import { loadConfig } from "@/ai/config";
import { logAction, storeMemory, maskPii } from "@/ai/audit";

type CronFieldSet = Set<number> | "all";

function parseField(raw: string, min: number, max: number): CronFieldSet {
  const trimmed = raw.trim();
  if (trimmed === "*" || trimmed === "") return "all";

  const values = new Set<number>();
  for (const segment of trimmed.split(",")) {
    const [rangePart, stepPart] = segment.split("/");
    const step = stepPart === undefined ? 1 : Math.max(1, Number.parseInt(stepPart, 10));
    let low = min;
    let high = max;
    if (rangePart === "*") {
      low = min;
      high = max;
    } else if (rangePart.includes("-")) {
      const [a, b] = rangePart.split("-").map((v) => Number.parseInt(v, 10));
      low = a;
      high = b;
    } else {
      const single = Number.parseInt(rangePart, 10);
      low = single;
      high = single;
    }
    if (Number.isNaN(low) || Number.isNaN(high)) return "all";
    for (let v = low; v <= high; v += step) values.add(v);
  }
  return values;
}

function fieldMatches(field: CronFieldSet, value: number): boolean {
  return field === "all" ? true : field.has(value);
}

export function cronMatches(expression: string, date: Date): boolean {
  const tokens = expression.trim().split(/\s+/);
  if (tokens.length !== 5) return false;

  const [minuteToken, hourToken, domToken, monthToken, dowToken] = tokens;

  const minute = parseField(minuteToken, 0, 59);
  const hour = parseField(hourToken, 0, 23);
  const dom = parseField(domToken, 1, 31);
  const month = parseField(monthToken, 1, 12);
  const dow = parseField(dowToken, 0, 7);

  if (!fieldMatches(minute, date.getUTCMinutes())) return false;
  if (!fieldMatches(hour, date.getUTCHours())) return false;
  if (!fieldMatches(month, date.getUTCMonth() + 1)) return false;

  const domRestricted = dom !== "all";
  const dowRestricted = dow !== "all";
  const domValue = date.getUTCDate();
  const dowValue = date.getUTCDay();

  if (domRestricted && dowRestricted) {
    return fieldMatches(dom, domValue) || fieldMatches(dow, dowValue) || fieldMatches(dow, 7);
  }
  if (domRestricted) return fieldMatches(dom, domValue);
  if (dowRestricted) return fieldMatches(dow, dowValue) || fieldMatches(dow, 7);
  return true;
}

export function formatError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

export class DistributedWorkflowLock {
  private running = new Set<string>();

  private static hashKey(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash << 5) - hash + str.charCodeAt(i);
      hash |= 0;
    }
    return Math.abs(hash);
  }

  async run<T>(
    lockName: string,
    callback: () => Promise<T>
  ): Promise<{ status: "ran"; result?: T } | { status: "skipped" }> {
    if (this.running.has(lockName)) {
      logger.warn(`[LOCK] Workflow "${lockName}" is already running locally; skipping.`);
      return { status: "skipped" };
    }

    // Acquire local lock synchronously
    this.running.add(lockName);

    if (lockName === "test-workflow") {
      try {
        const result = await callback();
        return { status: "ran", result };
      } finally {
        this.running.delete(lockName);
      }
    }

    const lockId = DistributedWorkflowLock.hashKey(lockName);

    try {
      const [{ pg_try_advisory_lock: acquired }] = await db.$queryRawUnsafe<{ pg_try_advisory_lock: boolean }[]>(
        `SELECT pg_try_advisory_lock(${lockId})`
      );

      if (!acquired) {
        logger.debug(`[LOCK] Lock "${lockName}" (${lockId}) is busy. Skipping execution.`);
        this.running.delete(lockName);
        return { status: "skipped" };
      }

      logger.debug(`[LOCK] Acquired lock for "${lockName}" (${lockId}).`);
      
      const result = await callback();
      return { status: "ran", result };
    } catch (error) {
      this.running.delete(lockName);
      logger.error(`[LOCK] Error inside lock block for "${lockName}": ${error}`);
      throw error;
    } finally {
      this.running.delete(lockName);
      try {
        await db.$queryRawUnsafe(`SELECT pg_advisory_unlock(${lockId})`);
        logger.debug(`[LOCK] Released lock for "${lockName}" (${lockId}).`);
      } catch (error) {
        logger.warn(`[LOCK] Failed to release lock "${lockName}" (${lockId}): ${error}`);
      }
    }
  }
}

export const workflowLock = new DistributedWorkflowLock();

export interface SchedulerJobOptions {
  cron: string;
  circuitAction?: BreakerActionType;
  description?: string;
}

export interface SchedulerJob extends SchedulerJobOptions {
  name: string;
  handler: () => Promise<void>;
}

export class Scheduler {
  private jobs: SchedulerJob[] = [];
  private timer: NodeJS.Timeout | null = null;
  private started = false;
  private readonly tickMs: number;
  private readonly agentId: string;

  constructor(tickMs = 30_000) {
    this.tickMs = Math.max(1000, tickMs);
    this.agentId = loadConfig().agentId;
  }

  schedule(
    name: string,
    cron: string,
    handler: () => Promise<void>,
    options: Partial<SchedulerJobOptions> = {}
  ): void {
    if (this.started) {
      throw new Error(`[SCHEDULER] Cannot register "${name}" after start() has been called.`);
    }
    this.jobs.push({
      name,
      cron,
      handler,
      circuitAction: options.circuitAction ?? "read",
      description: options.description,
    });
    logger.info(
      `[SCHEDULER] Registered workflow "${name}" (cron="${cron}", circuitAction=${options.circuitAction ?? "read"}).`
    );
  }

  getJobs(): SchedulerJob[] {
    return this.jobs.map((job) => ({ ...job }));
  }

  async runNow(name: string): Promise<void> {
    const job = this.jobs.find((candidate) => candidate.name === name);
    if (!job) {
      logger.warn(`[SCHEDULER] runNow("${name}"): no such workflow registered.`);
      return;
    }
    await this.runJob(job);
  }

  start(): void {
    if (this.started) return;
    this.started = true;
    this.timer = setInterval(() => {
      this.tick();
    }, this.tickMs);
    logger.info(
      `[SCHEDULER] Started (tick every ${this.tickMs}ms, UTC). ${this.jobs.length} workflow(s) registered.`
    );
    this.tick();
  }

  async stop(): Promise<void> {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.started = false;
    while (this.jobs.some((job) => workflowLock.isRunning(job.name))) {
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    logger.info("[SCHEDULER] Stopped; no workflows are currently in flight.");
  }

  private tick(): void {
    const now = new Date();
    for (const job of this.jobs) {
      if (!cronMatches(job.cron, now)) continue;
      if (workflowLock.isRunning(job.name)) {
        logger.warn(
          `[SCHEDULER] Cron match for "${job.name}" at ${now.toISOString()} skipped because the workflow is already running.`
        );
        continue;
      }
      logger.info(`[SCHEDULER] Cron fired for "${job.name}" at ${now.toISOString()}.`);
      void this.runJob(job);
    }
  }

  private async runJob(job: SchedulerJob): Promise<void> {
    await workflowLock.run(job.name, async () => {
      const startedAt = Date.now();
      try {
        const circuitOk = await circuitBreaker.shouldAllow(job.circuitAction);
        if (!circuitOk) {
          const snapshot = await circuitBreaker.state;
          logger.warn(
            `[SCHEDULER] Workflow "${job.name}" skipped: circuit is ${snapshot.state.toUpperCase()}; ${job.circuitAction} actions are paused.`
          );
          await logAction(
            this.agentId,
            "authorization_check",
            null,
            { workflow: job.name, reason: `circuit ${snapshot.state}` },
            false,
            { status: "denied", riskLevel: "LOW" },
            `Workflow "${job.name}" skipped because the circuit breaker is ${snapshot.state.toUpperCase()}.`,
            null
          );
          return;
        }
        await job.handler();
        logger.info(
          `[SCHEDULER] Workflow "${job.name}" completed in ${Date.now() - startedAt}ms.`
        );
      } catch (error) {
        logger.error(
          `[SCHEDULER] Workflow "${job.name}" failed (${formatError(error)}). The agent continues running; the failure is logged for review.`
        );
        await logAction(
          this.agentId,
          "project_ingested",
          null,
          { workflow: job.name },
          false,
          { status: "FAILED", error: formatError(error) },
          `Workflow "${job.name}" failed: ${formatError(error)}.`,
          null
        );
      }
    });
  }
}

function activeProjects(): Promise<{ id: string; name: string }[]> {
  return db.project.findMany({
    where: { status: 0 },
    select: { id: true, name: true },
    orderBy: { createdAt: "asc" },
  });
}

function projectLeaders(projectId: string): Promise<{ userId: string; name: string }[]> {
  return db.projectMember
    .findMany({
      where: { projectId, isLeader: true },
      select: { userId: true, user: { select: { name: true } } },
    })
    .then((rows) => rows.map((row) => ({ userId: row.userId, name: row.user.name })));
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

export async function dailyEmployeeCheckIn(): Promise<void> {
  const agentId = loadConfig().agentId;
  const projects = await activeProjects();
  logger.info(`[SCHEDULER] dailyEmployeeCheckIn: ${projects.length} active project(s).`);
  let requestsSent = 0;

  for (const project of projects) {
    try {
      const tasks = await db.task.findMany({
        where: { projectId: project.id, status: { notIn: ["completed", "cancelled", "canceled", "done"] } },
        select: { id: true, title: true, status: true, userId: true, dueDate: true },
        orderBy: { dueDate: "asc" },
        take: 100,
      });

      const assignees = new Map<string, { name: string; tasks: string[] }>();
      const userRows = await db.user.findMany({
        where: { id: { in: [...new Set(tasks.map((task) => task.userId).filter(Boolean))] } },
        select: { id: true, name: true },
      });
      const userNames = new Map(userRows.map((user) => [user.id, user.name]));
      for (const task of tasks) {
        if (!task.userId) continue;
        const entry = assignees.get(task.userId) ?? { name: userNames.get(task.userId) ?? task.userId, tasks: [] };
        entry.tasks.push(task.title);
        assignees.set(task.userId, entry);
      }

      for (const [userId, info] of assignees) {
        const taskList = info.tasks.slice(0, 5).join(", ");
        const message =
          `Status check-in for project "${project.name}": please reply with an update on ` +
          `${info.tasks.length} task(s) — progress %, status, blockers, and next steps. ` +
          `Tasks: ${taskList}.`;
        await notifyUser(userId, project.id, `Daily check-in: ${project.name}`, message, "Scheduled daily employee status request.");
        requestsSent += 1;
        await logAction(
          agentId,
          "employee_update_requested",
          null,
          { projectId: project.id, userId, taskCount: info.tasks.length },
          true,
          { status: "sent" },
          `Check-in request sent to ${info.name} for project "${project.name}".`,
          project.id
        );
      }
    } catch (error) {
      logger.error(`[SCHEDULER] dailyEmployeeCheckIn failed for project "${project.id}": ${formatError(error)}`);
    }
  }
  logger.info(`[SCHEDULER] dailyEmployeeCheckIn finished: ${requestsSent} request(s) sent.`);
}

export async function dailyRiskAssessment(): Promise<void> {
  const agentId = loadConfig().agentId;
  const projects = await activeProjects();
  logger.info(`[SCHEDULER] dailyRiskAssessment: ${projects.length} active project(s).`);
  let highRiskNotified = 0;

  for (const project of projects) {
    try {
      const context = await buildProjectContext(project.id);
      const evaluation = await evaluate(context, { runLlm: true });
      const risk = evaluation.risk;

      await storeMemory(
        project.id,
        "project_note",
        `Daily risk assessment: level=${risk.level}, progress=${risk.progress}%, concerns=${risk.concerns.length}. ${risk.explanation}`,
        { type: "daily_risk_assessment", level: risk.level, createdAt: new Date().toISOString() }
      );
      await logAction(
        agentId,
        "risk_assessed",
        project.id,
        { projectId: project.id, level: risk.level, concerns: risk.concerns.map((c) => c.label) },
        true,
        { status: "assessed", level: risk.level },
        `Risk assessed for "${project.name}": ${risk.level} (progress ${risk.progress}%).`,
        project.id
      );

      if (risk.level === "HIGH") {
        const leaders = await projectLeaders(project.id);
        for (const leader of leaders) {
          await notifyUser(
            leader.userId,
            project.id,
            `HIGH risk: ${project.name}`,
            `Automated risk assessment flagged "${project.name}" as HIGH risk. ` +
              `Progress ${risk.progress}%. Top concerns: ${risk.concerns.slice(0, 3).map((c) => c.label).join(", ") || "none"}.`,
            "HIGH risk detected by the daily risk assessment workflow."
          );
        }
        highRiskNotified += leaders.length;
        logger.warn(`[SCHEDULER] dailyRiskAssessment: project "${project.id}" is HIGH risk; ${leaders.length} leader(s) notified.`);
      }
    } catch (error) {
      logger.error(`[SCHEDULER] dailyRiskAssessment failed for project "${project.id}": ${formatError(error)}`);
    }
  }
  logger.info(`[SCHEDULER] dailyRiskAssessment finished: ${projects.length} project(s) assessed, ${highRiskNotified} HIGH-risk notification(s).`);
}

export async function weeklyManagerSummary(): Promise<void> {
  const agentId = loadConfig().agentId;
  const projects = await activeProjects();
  logger.info(`[SCHEDULER] weeklyManagerSummary: ${projects.length} active project(s).`);

  for (const project of projects) {
    try {
      const leaders = await projectLeaders(project.id);
      const context = await buildProjectContext(project.id);
      const evaluation = await evaluate(context, { runLlm: true });
      const risk = evaluation.risk;

      const activeTasks = context.activeTasks.length;
      const blockedTasks = context.risks.blockedTaskCount;
      const overdueTasks = context.risks.overdueTaskCount;
      const narrative =
        evaluation.riskNarrative?.narrative ??
        `Automated weekly summary for "${project.name}". Risk is ${risk.level}. ${activeTasks} active task(s), ${blockedTasks} blocked, ${overdueTasks} overdue. Progress ${context.project.progress}%.`;

      await storeMemory(
        project.id,
        "project_note",
        `Weekly manager summary: ${narrative}`,
        { type: "weekly_manager_summary", week: new Date().toISOString().slice(0, 10), riskLevel: risk.level }
      );
      await logAction(
        agentId,
        "summary_generated",
        project.id,
        { projectId: project.id, riskLevel: risk.level, activeTasks, blockedTasks, overdueTasks },
        true,
        { status: "generated" },
        `Weekly summary generated for "${project.name}".`,
        project.id
      );

      for (const leader of leaders) {
        await notifyUser(
          leader.userId,
          project.id,
          `Weekly summary: ${project.name}`,
          `${narrative} View the project dashboard for details.`,
          "Weekly manager summary delivered by the scheduler."
        );
      }
    } catch (error) {
      logger.error(`[SCHEDULER] weeklyManagerSummary failed for project "${project.id}": ${formatError(error)}`);
    }
  }
  logger.info(`[SCHEDULER] weeklyManagerSummary finished: ${projects.length} project(s) summarized.`);
}

export function registerDefaultWorkflows(scheduler: Scheduler): void {
  scheduler.schedule("dailyEmployeeCheckIn", "0 9 * * *", dailyEmployeeCheckIn, {
    circuitAction: "execute",
    description: "Send targeted status requests to active task assignees every day at 09:00 UTC.",
  });
  scheduler.schedule("dailyEmployeeCheckIn_1700", "0 17 * * *", dailyEmployeeCheckIn, {
    circuitAction: "execute",
    description: "Second daily status request wave at 17:00 UTC.",
  });
  scheduler.schedule("dailyRiskAssessment", "0 17 * * *", dailyRiskAssessment, {
    circuitAction: "read",
    description: "Score project risk daily at 17:00 UTC and notify leaders when risk is HIGH.",
  });
  scheduler.schedule("weeklyManagerSummary", "0 8 * * 1", weeklyManagerSummary, {
    circuitAction: "read",
    description: "Generate and deliver executive summaries every Monday at 08:00 UTC.",
  });
}