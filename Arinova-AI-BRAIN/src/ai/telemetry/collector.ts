import crypto from "crypto";
import db from "@/lib/client";
import { logger } from "@/ai/logger";
import { maskPii } from "@/ai/audit";
import type { FlushResult, TelemetrySpan } from "./types";

const MAX_BUFFER_SIZE = 5000;
const BATCH_FLUSH_INTERVAL_MS = 5000;
const MAX_BATCH_SIZE = 100;
const BURST_FLUSH_WATERMARK = 50;

function safeSanitizeAttributes(value: unknown, seen = new WeakSet()): unknown {
  if (value === null || value === undefined) return null;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (typeof value === "object") {
    if (seen.has(value as object)) return "[CIRCULAR_REF]";
    seen.add(value as object);
    if (Array.isArray(value)) {
      return value.map((item) => safeSanitizeAttributes(item, seen));
    }
    const result: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) {
      result[key] = safeSanitizeAttributes(item, seen);
    }
    return result;
  }
  return String(value);
}

class TelemetryCollector {
  private ringBuffer: TelemetrySpan[] = [];
  private flushTimer: NodeJS.Timeout | null = null;
  private isFlushing = false;

  constructor() {
    this.startBatchTimer();
  }

  public recordSpan(span: TelemetrySpan): void {
    const durationMs = span.durationMs ?? (span.endTimeMs ? span.endTimeMs - span.startTimeMs : Date.now() - span.startTimeMs);
    const sanitizedAttributes = safeSanitizeAttributes(span.attributes ?? {}) as Record<string, unknown>;
    const completedSpan: TelemetrySpan = {
      ...span,
      endTimeMs: span.endTimeMs ?? Date.now(),
      durationMs: Math.max(0, durationMs),
      attributes: sanitizedAttributes,
    };

    if (this.ringBuffer.length >= MAX_BUFFER_SIZE) {
      // Evict oldest span if buffer is full to guarantee 0ms non-blocking backpressure
      this.ringBuffer.shift();
    }
    this.ringBuffer.push(completedSpan);

    if (this.ringBuffer.length >= BURST_FLUSH_WATERMARK) {
      void this.flush();
    }
  }

  public startSpan(
    name: string,
    subsystem: TelemetrySpan["subsystem"],
    options: {
      traceId?: string;
      parentSpanId?: string | null;
      projectId?: string | null;
      taskId?: string | null;
      userId?: string | null;
      attributes?: Record<string, unknown>;
    } = {}
  ): {
    spanId: string;
    traceId: string;
    end: (status?: TelemetrySpan["status"], errorDetails?: string | null, extraAttr?: Record<string, unknown>) => void;
  } {
    const traceId = options.traceId ?? `trc-${Date.now()}-${crypto.randomBytes(4).toString("hex")}`;
    const spanId = `spn-${Date.now()}-${crypto.randomBytes(4).toString("hex")}`;
    const startTimeMs = Date.now();

    const end = (
      status: TelemetrySpan["status"] = "OK",
      errorDetails: string | null = null,
      extraAttr: Record<string, unknown> = {}
    ): void => {
      const endTimeMs = Date.now();
      this.recordSpan({
        traceId,
        spanId,
        parentSpanId: options.parentSpanId ?? null,
        name,
        subsystem,
        projectId: options.projectId ?? null,
        taskId: options.taskId ?? null,
        userId: options.userId ?? null,
        startTimeMs,
        endTimeMs,
        durationMs: endTimeMs - startTimeMs,
        status,
        errorDetails: errorDetails ? maskPii(errorDetails) : null,
        attributes: {
          ...(options.attributes ?? {}),
          ...extraAttr,
        },
      });
    };

    return { spanId, traceId, end };
  }

  public async flush(): Promise<FlushResult> {
    if (this.isFlushing || this.ringBuffer.length === 0) {
      return { flushedCount: 0, success: true };
    }

    this.isFlushing = true;
    const batch = this.ringBuffer.splice(0, MAX_BATCH_SIZE);

    try {
      const rows = batch.map((span) => ({
        agentId: "agent-service-001",
        action: "telemetry_span",
        target: span.taskId ?? span.spanId,
        projectId: span.projectId ?? null,
        payload: {
          traceId: span.traceId,
          spanId: span.spanId,
          parentSpanId: span.parentSpanId ?? null,
          name: span.name,
          subsystem: span.subsystem,
          startTimeMs: span.startTimeMs,
          endTimeMs: span.endTimeMs,
          durationMs: span.durationMs,
          userId: span.userId ?? null,
          attributes: span.attributes ?? {},
        },
        result: {
          status: span.status,
          errorDetails: span.errorDetails ?? null,
        },
        success: span.status === "OK",
        reason: span.name,
      }));

      await db.agentActionLog.createMany({
        data: rows,
        skipDuplicates: true,
      });

      logger.debug(`[TELEMETRY] Flushed batch of ${rows.length} span(s) to AgentActionLog.`);
      return { flushedCount: batch.length, success: true };
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      logger.warn(`[TELEMETRY] Async batch flush warning: ${errMsg}`);
      // Re-queue failed batch up to buffer capacity
      this.ringBuffer.unshift(...batch.slice(0, MAX_BUFFER_SIZE - this.ringBuffer.length));
      return { flushedCount: 0, success: false, error: errMsg };
    } finally {
      this.isFlushing = false;
    }
  }

  private startBatchTimer(): void {
    this.flushTimer = setInterval(() => {
      void this.flush();
    }, BATCH_FLUSH_INTERVAL_MS);
  }

  public stop(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
  }
}

export const collector = new TelemetryCollector();
