import type { PolicyDecision } from "@/ai/policy/types";

export type ExecutorAction =
  | "add_comment"
  | "record_blocker"
  | "update_task_status"
  | "update_task_progress"
  | "assign_task"
  | "create_task"
  | "send_notification";

export type VerificationOutcome =
  | "pass"
  | "field_missing"
  | "conflict"
  | "mismatch";

export interface VerificationResult {
  passed: boolean;
  outcome: VerificationOutcome;
  detail: string;
  actual: unknown;
  expected: unknown;
}

export type ExecutionStatus =
  | "SUCCESS"
  | "FAILED"
  | "PRECONDITION_FAILED"
  | "VERIFY_FAILED"
  | "SKIPPED_RECOMMENDED"
  | "REFUSED_POLICY"
  | "CIRCUIT_OPEN"
  | "VERIFIED_MISMATCH";

export interface ExecutionResult {
  action: string;
  status: ExecutionStatus;
  success: boolean;
  policy: {
    allowed: boolean;
    requiresApproval: boolean;
    status: string;
    riskLevel: string;
  };
  verification: VerificationResult | null;
  logId: string | null;
  reason: string;
  rolledBack: boolean;
  executionMs: number;
  createdAt: string;
}

export interface ExecutionGrant {
  by: string;
  at?: string;
  note?: string;
}

export interface ExecutionContext {
  policy?: PolicyDecision;
  executionGrant?: ExecutionGrant;
  agentId: string;
  agentToken?: string;
  projectId?: string | null;
  taskId?: string | null;
  targetUserId?: string | null;
  payload: Record<string, unknown>;
}

export interface ActionLogRow {
  id: string;
  agentId: string;
  action: string;
  target: string | null;
  payload: unknown;
  result: unknown;
  success: boolean;
  reason: string;
  createdAt: string;
}