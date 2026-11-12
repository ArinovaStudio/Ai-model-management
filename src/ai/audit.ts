import type { Prisma } from "@prisma/client";
import db from "@/lib/client";
import { logger } from "@/ai/logger";

export const AUDIT_ACTIONS = [
  "task_created",
  "task_updated",
  "task_assigned",
  "task_status_changed",
  "blocker_recorded",
  "blocker_resolved",
  "summary_generated",
  "risk_assessed",
  "employee_update_requested",
  "employee_reply_parsed",
  "project_ingested",
  "authorization_check",
] as const;
export type AuditAction = (typeof AUDIT_ACTIONS)[number];

export const MEMORY_TYPES = [
  "conversation",
  "commitment",
  "blocker_context",
  "project_note",
] as const;
export type MemoryType = (typeof MEMORY_TYPES)[number];

type AgentActionLogRecord = Prisma.AgentActionLogGetPayload<Record<string, never>>;
type AgentMemoryRecord = Prisma.AgentMemoryGetPayload<Record<string, never>>;
type AgentDecisionRecord = Prisma.AgentDecisionGetPayload<Record<string, never>>;

const SENSITIVE_KEY_RE =
  /(passwd|password|pwd|secret|token|apikey|api[_ -]?key|authorization|cookie|credential|access[_ -]?secret|ssn|cardnumber|cvv|cvc|privatekey|jwt)/i;

const EMAIL_RE = /([a-zA-Z0-9._%+-]{2,}|[a-zA-Z0-9._%+-])@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})/g;
const JWT_RE = /eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g;
const PHONE_CANDIDATE_RE =
  /(?<![A-Za-z0-9])(\+?[\d][\d\s().-]{6,}[\d])(?![A-Za-z0-9])/g;

function maskLocalPart(local: string): string {
  if (local.length <= 2) {
    return `${local[0]}${"*".repeat(Math.max(1, local.length - 1))}`;
  }
  return `${local[0]}${"*".repeat(local.length - 2)}${local[local.length - 1]}`;
}

function maskPhoneCandidate(raw: string): string {
  const digits = raw.replace(/\D/g, "");
  if (digits.length < 7 || digits.length > 15) return raw;
  if (/\d{1,2}:\d{2}/.test(raw)) return raw;
  if (/^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$/.test(raw)) return raw;
  return `+****${digits.slice(-4)}`;
}

export function maskPii(value: string): string {
  if (!value) return value;
  let out = value.replace(JWT_RE, "eyJ****.****.****");
  out = out.replace(EMAIL_RE, (_match, local: string, domain: string) => {
    return `${maskLocalPart(local)}@${domain}`;
  });
  out = out.replace(PHONE_CANDIDATE_RE, (raw) => maskPhoneCandidate(raw));
  return out;
}

function sanitizeForAudit(value: unknown, currentKey: string | null = null): unknown {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") {
    if (currentKey !== null && SENSITIVE_KEY_RE.test(currentKey)) return "[REDACTED]";
    return maskPii(value);
  }
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (Array.isArray(value)) return value.map((item) => sanitizeForAudit(item));
  if (typeof value === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) {
      result[key] = sanitizeForAudit(item, key);
    }
    return result;
  }
  return maskPii(String(value));
}

export function formatError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

export interface ActionLogPayload {
  agentId: string;
  action: string;
  target?: string | null;
  projectId?: string | null;
  payload?: unknown;
  success?: boolean;
  result?: unknown;
  reason?: string | null;
}

export async function logAction(
  agentId: string,
  action: string,
  target: string | null,
  payload: unknown,
  success: boolean,
  result: unknown,
  reason: string | null,
  projectId: string | null = null
): Promise<AgentActionLogRecord | null> {
  try {
    const row = await db.agentActionLog.create({
      data: {
        agentId,
        action,
        target,
        projectId,
        payload: sanitizeForAudit(payload) as object,
        result: sanitizeForAudit(result) as object,
        success,
        reason: reason ? maskPii(reason) : null,
      },
    });
    logger.info(
      `[AUDIT][ACTION][${action}] agentId="${agentId}" target="${target ?? "-"}" success=${success} logId="${row.id}".`
    );
    return row;
  } catch (error) {
    logger.warn(
      `[AUDIT][ACTION][${action}] Failed to write AgentActionLog: ${formatError(error)}`
    );
    return null;
  }
}

export interface DecisionBasis {
  action: string;
  riskLevel?: string | null;
  reason?: string | null;
  confidence?: number | null;
  projectId?: string | null;
  taskId?: string | null;
  scoring?: Record<string, unknown> | null;
}

export async function recordDecision(
  agentId: string,
  decision: string,
  basis: DecisionBasis,
  evidence: unknown[] | null = null,
  projectId: string | null = null
): Promise<AgentDecisionRecord | null> {
  try {
    const row = await db.agentDecision.create({
      data: {
        agentId,
        projectId: basis.projectId ?? projectId,
        decision: maskPii(decision),
        basis: sanitizeForAudit({
          ...basis,
          evidence: evidence ?? [],
        }) as object,
      },
    });
    logger.info(
      `[AUDIT][DECISION] agentId="${agentId}" action="${basis.action}" projectId="${row.projectId ?? "-"}" decisionId="${row.id}".`
    );
    return row;
  } catch (error) {
    logger.warn(
      `[AUDIT][DECISION][${basis.action}] Failed to write AgentDecision: ${formatError(error)}`
    );
    return null;
  }
}

export async function storeMemory(
  projectId: string | null,
  type: string,
  content: string,
  meta: Record<string, unknown> | null = null
): Promise<AgentMemoryRecord | null> {
  try {
    const row = await db.agentMemory.create({
      data: {
        projectId,
        type,
        content: maskPii(content),
        meta: sanitizeForAudit(meta ?? {}) as object,
      },
    });
    logger.info(
      `[AUDIT][MEMORY] projectId="${projectId ?? "-"}" type="${type}" memoryId="${row.id}".`
    );
    return row;
  } catch (error) {
    logger.warn(
      `[AUDIT][MEMORY] projectId="${projectId ?? "-"}" type="${type}" failed to persist: ${formatError(error)}`
    );
    return null;
  }
}

export interface AuditQueryFilters {
  agentId?: string | null;
  success?: boolean | null;
  start?: Date | null;
  end?: Date | null;
}

const MAX_QUERY_LIMIT = 500;

export async function getActionHistory(
  projectId: string | null,
  limit: number = 50,
  filters: AuditQueryFilters = {}
): Promise<AgentActionLogRecord[]> {
  const where: Prisma.AgentActionLogWhereInput = {
    ...(projectId !== null && projectId !== undefined ? { projectId } : {}),
    ...(filters.agentId ? { agentId: filters.agentId } : {}),
    ...(filters.success !== null && filters.success !== undefined
      ? { success: filters.success }
      : {}),
    ...(filters.start || filters.end
      ? {
          createdAt: {
            ...(filters.start ? { gte: filters.start } : {}),
            ...(filters.end ? { lte: filters.end } : {}),
          },
        }
      : {}),
  };
  const safeLimit = Math.min(Math.max(limit, 1), MAX_QUERY_LIMIT);
  try {
    return await db.agentActionLog.findMany({
      where,
      orderBy: { createdAt: "desc" },
      take: safeLimit,
    });
  } catch (error) {
    logger.warn(`[AUDIT][QUERY] getActionHistory failed: ${formatError(error)}`);
    return [];
  }
}

export async function getMemory(
  projectId: string | null,
  type: string | null = null,
  limit: number = 20
): Promise<AgentMemoryRecord[]> {
  const where: Prisma.AgentMemoryWhereInput = {
    ...(projectId !== null && projectId !== undefined ? { projectId } : {}),
    ...(type ? { type } : {}),
  };
  const safeLimit = Math.min(Math.max(limit, 1), MAX_QUERY_LIMIT);
  try {
    return await db.agentMemory.findMany({
      where,
      orderBy: { createdAt: "desc" },
      take: safeLimit,
    });
  } catch (error) {
    logger.warn(`[AUDIT][QUERY] getMemory failed: ${formatError(error)}`);
    return [];
  }
}

export async function getDecisionHistory(
  projectId: string | null,
  limit: number = 20
): Promise<AgentDecisionRecord[]> {
  const where: Prisma.AgentDecisionWhereInput = {
    ...(projectId !== null && projectId !== undefined ? { projectId } : {}),
  };
  const safeLimit = Math.min(Math.max(limit, 1), MAX_QUERY_LIMIT);
  try {
    return await db.agentDecision.findMany({
      where,
      orderBy: { createdAt: "desc" },
      take: safeLimit,
    });
  } catch (error) {
    logger.warn(`[AUDIT][QUERY] getDecisionHistory failed: ${formatError(error)}`);
    return [];
  }
}