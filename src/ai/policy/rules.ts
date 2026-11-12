import type { ActionType, RiskLevel } from "./types";

export interface ActionRule {
  type: ActionType;
  label: string;
  riskLevel: RiskLevel;
  requiresProject: boolean;
  requiresTask: boolean;
}

export const ACTION_RULES: Record<ActionType, ActionRule> = {
  read_project: {
    type: "read_project",
    label: "Read project state",
    riskLevel: "LOW",
    requiresProject: true,
    requiresTask: false,
  },
  generate_summary: {
    type: "generate_summary",
    label: "Generate project summary",
    riskLevel: "LOW",
    requiresProject: true,
    requiresTask: false,
  },
  request_update: {
    type: "request_update",
    label: "Request an employee update",
    riskLevel: "LOW",
    requiresProject: true,
    requiresTask: false,
  },
  add_comment: {
    type: "add_comment",
    label: "Add a comment",
    riskLevel: "LOW",
    requiresProject: true,
    requiresTask: false,
  },
  record_blocker: {
    type: "record_blocker",
    label: "Record a risk blocker",
    riskLevel: "LOW",
    requiresProject: true,
    requiresTask: false,
  },
  send_notification: {
    type: "send_notification",
    label: "Send a notification",
    riskLevel: "LOW",
    requiresProject: false,
    requiresTask: false,
  },
  create_task: {
    type: "create_task",
    label: "Create a task",
    riskLevel: "MEDIUM",
    requiresProject: true,
    requiresTask: false,
  },
  assign_task: {
    type: "assign_task",
    label: "Assign a task",
    riskLevel: "MEDIUM",
    requiresProject: true,
    requiresTask: true,
  },
  update_task_status: {
    type: "update_task_status",
    label: "Update task status",
    riskLevel: "MEDIUM",
    requiresProject: true,
    requiresTask: true,
  },
  update_task_progress: {
    type: "update_task_progress",
    label: "Update task progress",
    riskLevel: "MEDIUM",
    requiresProject: true,
    requiresTask: true,
  },
  change_deadline: {
    type: "change_deadline",
    label: "Change project deadline",
    riskLevel: "MEDIUM",
    requiresProject: true,
    requiresTask: false,
  },
  delete_task: {
    type: "delete_task",
    label: "Delete a task",
    riskLevel: "HIGH",
    requiresProject: true,
    requiresTask: true,
  },
  remove_team_member: {
    type: "remove_team_member",
    label: "Remove a team member",
    riskLevel: "HIGH",
    requiresProject: true,
    requiresTask: false,
  },
  change_project_scope: {
    type: "change_project_scope",
    label: "Change project scope",
    riskLevel: "HIGH",
    requiresProject: true,
    requiresTask: false,
  },
};

export const DEFAULT_UNKNOWN_RISK: RiskLevel = "HIGH";

export function getActionRule(action: string): ActionRule | null {
  return ACTION_RULES[action as ActionType] ?? null;
}

export const ACTION_TYPES = Object.keys(ACTION_RULES) as ActionType[];