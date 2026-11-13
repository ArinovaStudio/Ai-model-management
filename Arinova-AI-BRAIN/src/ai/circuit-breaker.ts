import type { Prisma } from "@prisma/client";
import db from "@/lib/client";
import { logger } from "@/ai/logger";
import { startSpan } from "@/ai/telemetry";

type CircuitBreakerStateRecord = Prisma.CircuitBreakerStateGetPayload<Record<string, never>>;

export type BreakerStateValue = "closed" | "open" | "half_open";
export type BreakerActionType = "read" | "write" | "execute";

export type FailureType =
  | "llm_timeout"
  | "nlu_parse"
  | "execution"
  | "authorization"
  | "verification_mismatch"
  | "fatal";

export const BREAKER_FAILURE_TYPES: FailureType[] = [
  "llm_timeout",
  "nlu_parse",
  "execution",
  "authorization",
  "verification_mismatch",
  "fatal",
];

export const TRIP_THRESHOLDS: Record<FailureType, number> = {
  llm_timeout: 3,
  nlu_parse: 5,
  execution: 5,
  authorization: 3,
  verification_mismatch: 3,
  fatal: 1,
};

function envOr(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw || raw.trim() === "") return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isNaN(parsed) || parsed <= 0 ? fallback : parsed;
}

const SINGLETON_ID = "agent-circuit-breaker";
const DEFAULT_DWELL_MS = 60 * 60 * 1000;
const DEFAULT_PROBE_STALE_MS = 5 * 60 * 1000;
const DWELL_MS = envOr("CIRCUIT_BREAKER_DWELL_MS", DEFAULT_DWELL_MS);
const PROBE_STALE_MS = envOr("CIRCUIT_BREAKER_PROBE_STALE_MS", DEFAULT_PROBE_STALE_MS);

export interface BreakerSnapshot {
  state: BreakerStateValue;
  failureCount: number;
  failureCounts: Partial<Record<FailureType, number>>;
  lastFailureAt: Date | null;
  lastTrippedAt: Date | null;
  updatedAt: Date;
}

function formatDelta(ms: number): string {
  const minutes = Math.round(ms / 60000);
  if (minutes < 1) return `${Math.round(ms / 1000)}s`;
  if (minutes < 60) return `${minutes}m`;
  return `${Math.round((minutes / 60) * 10) / 10}h`;
}

function formatError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

export class CircuitBreakerManager {
  private probeStartedAt: number | null = null;

  get state(): Promise<BreakerSnapshot> {
    return this.getSnapshot();
  }

  private async getSnapshot(): Promise<BreakerSnapshot> {
    const row = await this.ensureRow();
    return {
      state: (row.state as BreakerStateValue) ?? "closed",
      failureCount: row.failureCount ?? 0,
      failureCounts: (row.failureCounts as Partial<Record<FailureType, number>> | null) ?? {},
      lastFailureAt: row.lastFailureAt,
      lastTrippedAt: row.lastTrippedAt,
      updatedAt: row.updatedAt,
    };
  }

  private async ensureRow(): Promise<CircuitBreakerStateRecord> {
    const existing = await db.circuitBreakerState.findUnique({
      where: { id: SINGLETON_ID },
    });
    if (existing) return existing;
    return db.circuitBreakerState.create({
      data: { id: SINGLETON_ID },
    });
  }

  private async persistPartial(patch: {
    state?: BreakerStateValue;
    failureCount?: number;
    failureCounts?: Partial<Record<FailureType, number>>;
    lastFailureAt?: Date | null;
    lastTrippedAt?: Date | null;
  }): Promise<CircuitBreakerStateRecord> {
    return db.circuitBreakerState.update({
      where: { id: SINGLETON_ID },
      data: patch as object,
    });
  }

  private async setState(
    next: BreakerStateValue,
    from: BreakerStateValue,
    reason: string
  ): Promise<void> {
    await this.persistPartial({
      state: next,
      lastTrippedAt: next === "open" ? new Date() : undefined,
    });
    logger.error(
      `[CIRCUIT_BREAKER] STATE ${from} -> ${next}: ${reason}`
    );
  }

  private dwellElapsed(row: CircuitBreakerStateRecord): boolean {
    if (!row.lastTrippedAt) return true;
    return Date.now() - row.lastTrippedAt.getTime() >= DWELL_MS;
  }

  async isOpen(): Promise<boolean> {
    return (await this.getSnapshot()).state === "open";
  }

  async isClosed(): Promise<boolean> {
    return (await this.getSnapshot()).state === "closed";
  }

  async isHalfOpen(): Promise<boolean> {
    return (await this.getSnapshot()).state === "half_open";
  }

  async recordSuccess(): Promise<string> {
    const snapshot = await this.getSnapshot();
    if (snapshot.state === "half_open") {
      this.probeStartedAt = null;
      await this.persistPartial({
        state: "closed",
        failureCount: 0,
        failureCounts: {},
        lastFailureAt: null,
        lastTrippedAt: null,
      });
      logger.info(
        `[CIRCUIT_BREAKER] STATE half_open -> closed: probe succeeded, circuit closed.`
      );
      return "closed";
    }
    await this.persistPartial({
      failureCount: 0,
      failureCounts: {},
      lastFailureAt: null,
    });
    logger.debug(
      `[CIRCUIT_BREAKER] Success recorded; circuit remains "${snapshot.state}" and failure counters were reset.`
    );
    return snapshot.state;
  }

  async recordFailure(
    failureType: string,
    opts?: { reason?: string }
  ): Promise<string> {
    const type = BREAKER_FAILURE_TYPES.includes(failureType as FailureType)
      ? (failureType as FailureType)
      : "fatal";
    const threshold = TRIP_THRESHOLDS[type];

    const row = await this.ensureRow();
    const current = (row.state as BreakerStateValue) ?? "closed";
    const counts: Partial<Record<FailureType, number>> = {
      ...((row.failureCounts as Partial<Record<FailureType, number>> | null) ?? {}),
      [type]: ((row.failureCounts as Partial<Record<FailureType, number>> | null)?.[type] ?? 0) + 1,
    };
    const total = Object.values(counts).reduce((sum, n) => sum + (n ?? 0), 0);

    const reason =
      opts?.reason ?? `${failureType} failure (${counts[type]} consecutive, threshold ${threshold})`;

    if (current === "half_open") {
      this.probeStartedAt = null;
      await this.persistPartial({
        state: "open",
        failureCount: total,
        failureCounts: counts,
        lastFailureAt: new Date(),
        lastTrippedAt: new Date(),
      });
      logger.error(
        `[CIRCUIT_BREAKER] STATE half_open -> open: probe failed (${reason}); dwell timer reset.`
      );
      return "open";
    }

    if (current === "open") {
      await this.persistPartial({
        failureCount: total,
        failureCounts: counts,
        lastFailureAt: new Date(),
      });
      logger.warn(
        `[CIRCUIT_BREAKER] Additional failure while open (${failureType}, now ${counts[type]}/${threshold}): ${reason}. Circuit stays open.`
      );
      return "open";
    }

    const crossed = (counts[type] ?? 0) >= threshold;
    const patch: {
      failureCount: number;
      failureCounts: Partial<Record<FailureType, number>>;
      lastFailureAt: Date;
      state?: BreakerStateValue;
      lastTrippedAt?: Date;
    } = {
      failureCount: total,
      failureCounts: counts,
      lastFailureAt: new Date(),
    };

    if (crossed) {
      patch.state = "open";
      patch.lastTrippedAt = new Date();
      await this.persistPartial(patch);
      const span = startSpan("circuit_breaker_tripped", "circuit_breaker", {
        attributes: { failureType, failureCount: counts[type], threshold, reason },
      });
      span.end("ERROR", reason);
      logger.error(
        `[CIRCUIT_BREAKER] STATE closed -> open: tripped on ${type} (${counts[type]}/${threshold} consecutive): ${reason}.`
      );
      return "open";
    }

    await this.persistPartial(patch);
    logger.debug(
      `[CIRCUIT_BREAKER] Failure recorded (${failureType}, now ${counts[type]}/${threshold}); circuit stays closed.`
    );
    return "closed";
  }

  async shouldAllow(actionType: BreakerActionType): Promise<boolean> {
    const row = await this.ensureRow();
    const current = (row.state as BreakerStateValue) ?? "closed";

    if (current === "closed") return true;

    if (current === "open") {
      if (actionType === "read") return true;
      logger.warn(
        `[CIRCUIT_BREAKER] Blocked a ${actionType} action while circuit is OPEN (only reads pass).`
      );
      return false;
    }

    if (current === "half_open") {
      if (actionType !== "read") {
        logger.warn(
          `[CIRCUIT_BREAKER] Blocked a ${actionType} action while circuit is HALF_OPEN (only one probe read may run).`
        );
        return false;
      }
      if (this.probeStartedAt !== null) {
        if (Date.now() - this.probeStartedAt > PROBE_STALE_MS) {
          logger.warn(
            `[CIRCUIT_BREAKER] Discarding a stale probe (started ${formatDelta(Date.now() - this.probeStartedAt)} ago); allowing a fresh probe read.`
          );
          this.probeStartedAt = null;
        } else {
          logger.warn(
            `[CIRCUIT_BREAKER] Blocked a second read while HALF_OPEN: a probe is already running (started ${formatDelta(Date.now() - this.probeStartedAt)} ago).`
          );
          return false;
        }
      }
      this.probeStartedAt = Date.now();
      logger.info(
        `[CIRCUIT_BREAKER] HALF_OPEN: permitting one probe read. If it succeeds the circuit closes; if it fails, we return to OPEN and restart the dwell timer.`
      );
      return true;
    }

    return false;
  }

  async probe(): Promise<boolean> {
    const row = await this.ensureRow();
    const current = (row.state as BreakerStateValue) ?? "closed";

    if (current === "closed") return true;
    if (current === "open") {
      if (!this.dwellElapsed(row)) {
        logger.info(
          `[CIRCUIT_BREAKER] Not probing while OPEN: dwell timer has ${formatDelta(
            DWELL_MS - (Date.now() - (row.lastTrippedAt?.getTime() ?? Date.now()))
          )} remaining.`
        );
        return false;
      }
      await this.setState("half_open", current, "dwell time elapsed; starting probe.");
    }

    if (this.probeStartedAt !== null) {
      if (Date.now() - this.probeStartedAt <= PROBE_STALE_MS) {
        logger.info(
          `[CIRCUIT_BREAKER] A probe is already running; not starting a second one.`
        );
        return false;
      }
      this.probeStartedAt = null;
    }
    this.probeStartedAt = Date.now();

    try {
      const projectId = await this.pickProbeTarget();
      if (projectId === null) {
        logger.info(
          `[CIRCUIT_BREAKER] No project found to probe; treating the safe read as successful.`
        );
        await this.recordSuccess();
        return true;
      }
      const { buildProjectContext } = await import("@/ai/context");
      await buildProjectContext(projectId);
      await this.recordSuccess();
      return true;
    } catch (error) {
      logger.error(
        `[CIRCUIT_BREAKER] Probe read failed: ${formatError(error)}. Returning circuit to OPEN and resetting the dwell timer.`
      );
      await this.recordFailure("execution", { reason: "probe read failed" });
      return false;
    }
  }

  private async pickProbeTarget(): Promise<string | null> {
    const project = await db.project.findFirst({
      orderBy: { createdAt: "desc" },
      select: { id: true },
    });
    return project?.id ?? null;
  }

  async resetManually(reason: string): Promise<string> {
    this.probeStartedAt = null;
    await this.persistPartial({
      state: "closed",
      failureCount: 0,
      failureCounts: {},
      lastFailureAt: null,
      lastTrippedAt: null,
    });
    logger.error(
      `[CIRCUIT_BREAKER] STATE * -> closed: manual reset (${reason}). All failure counters cleared.`
    );
    return "closed";
  }
}

export const circuitBreaker = new CircuitBreakerManager();

export class CircuitBreakerOpenError extends Error {
  constructor(actionType: BreakerActionType, current: BreakerStateValue) {
    super(
      `[CIRCUIT_BREAKER] Blocked: a ${actionType} action was attempted while the circuit is ${current.toUpperCase()}.`
    );
    this.name = "CircuitBreakerOpenError";
  }
}