export interface TelemetrySpan {
  traceId: string;
  spanId: string;
  parentSpanId?: string | null;
  name: string;
  subsystem: "events" | "context" | "reasoner" | "policy" | "executor" | "nlu" | "circuit_breaker" | "scheduler";
  projectId?: string | null;
  taskId?: string | null;
  userId?: string | null;
  startTimeMs: number;
  endTimeMs?: number | null;
  durationMs?: number | null;
  status: "OK" | "ERROR" | "CANCELLED";
  errorDetails?: string | null;
  attributes?: Record<string, unknown>;
}

export interface FlushResult {
  flushedCount: number;
  success: boolean;
  error?: string;
}
