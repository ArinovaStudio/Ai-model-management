import crypto from "crypto";
import { collector } from "./collector";
import type { TelemetrySpan } from "./types";

export function generateTraceId(): string {
  return `trc-${Date.now()}-${crypto.randomBytes(4).toString("hex")}`;
}

export function startSpan(
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
) {
  return collector.startSpan(name, subsystem, options);
}

export function recordSpan(span: TelemetrySpan): void {
  collector.recordSpan(span);
}

export { collector };
export type { TelemetrySpan } from "./types";
