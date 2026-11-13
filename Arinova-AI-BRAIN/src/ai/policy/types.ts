export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

export type ActionType =
  | "read_project"
  | "generate_summary"
  | "request_update"
  | "add_comment"
  | "record_blocker"
  | "send_notification"
  | "create_task"
  | "assign_task"
  | "update_task_status"
  | "update_task_progress"
  | "change_deadline"
  | "delete_task"
  | "remove_team_member"
  | "change_project_scope";

export type PolicyStatus = "allowed" | "requires_approval" | "denied";

export interface PolicyCheck {
  name: string;
  passed: boolean;
  detail: string;
}

export interface ApprovalInfo {
  by: string;
  at?: string;
  note?: string;
}

export interface PolicyContext {
  projectId?: string | null;
  taskId?: string | null;
  targetId?: string | null;
  token?: string;
  approval?: ApprovalInfo;
  payload?: unknown;
}

export type StealthModeType = "AUTO" | "SELECT";

export interface StealthPolicy {
  projectId: string;
  startDate?: string | null;
  endDate?: string | null;
  minConfidenceScore: number;
  isActive: boolean;
  modeType?: StealthModeType;
  selectedTaskIds?: string[];
  isException?: boolean;
}

export interface UniversalStealthConfig {
  isUniversalActive: boolean;
  globalMinConfidence: number;
  exceptionProjectIds?: string[];
}


export interface PolicyDecision {
  status: PolicyStatus;
  allowed: boolean;
  requiresApproval: boolean;
  riskLevel: RiskLevel;
  action: string;
  projectId: string | null;
  reason: string;
  checks: PolicyCheck[];
}