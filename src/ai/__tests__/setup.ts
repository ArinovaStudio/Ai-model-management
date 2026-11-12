import db from "@/lib/client";
import { circuitBreaker } from "@/ai/circuit-breaker";
import type { PolicyDecision } from "@/ai/policy/types";

/**
 * Shared helpers for the AI integration test suite. Tests run against the live
 * Postgres (Neon) instance; every row created by a test carries a unique
 * `test-...` prefix so tests are isolated, repeatable and clean up after
 * themselves (see `cleanupTestData`).
 */

export const TEST_AGENT_ID = "test-agent-integration-001";

export function makePrefix(label: string): string {
  return `test-${label}-${Math.random().toString(36).slice(2, 8)}`;
}

export function uid(prefix: string, label: string): string {
  return `${prefix}-${label}`;
}

export function allowedPolicy(
  action: string,
  projectId: string | null = null
): PolicyDecision {
  return {
    status: "allowed",
    allowed: true,
    requiresApproval: false,
    riskLevel: "LOW",
    action,
    projectId,
    reason: "policy injected by integration test",
    checks: [{ name: "injected", passed: true, detail: "test override" }],
  };
}

export function mediumApprovalPolicy(
  action: string,
  projectId: string | null = null
): PolicyDecision {
  return {
    status: "requires_approval",
    allowed: true,
    requiresApproval: true,
    riskLevel: "MEDIUM",
    action,
    projectId,
    reason: "MEDIUM-risk action, human approval required",
    checks: [{ name: "injected", passed: true, detail: "test override" }],
  };
}

export function deniedPolicy(action: string): PolicyDecision {
  return {
    status: "denied",
    allowed: false,
    requiresApproval: false,
    riskLevel: "HIGH",
    action,
    projectId: null,
    reason: "denied by injected test policy",
    checks: [{ name: "injected", passed: false, detail: "test override" }],
  };
}

/**
 * Creates (idempotently) the agent user used by the workflows. `canPerformAction`
 * verifies the agent via a SubRole named "SYSTEM_AGENT" plus the AI-agent marker,
 * so no Role row is required (the live Role table stays untouched).
 */
export async function ensureAgentUser(): Promise<void> {
  let subRole = await db.subRole.findFirst({
    where: { name: "SYSTEM_AGENT", id: { startsWith: TEST_AGENT_ID } },
  });
  if (!subRole) {
    subRole = await db.subRole.create({
      data: { id: `${TEST_AGENT_ID}-subrole`, name: "SYSTEM_AGENT" },
    });
  }
  const existing = await db.user.findUnique({ where: { id: TEST_AGENT_ID } });
  if (!existing) {
    await db.user.create({
      data: {
        id: TEST_AGENT_ID,
        email: `${TEST_AGENT_ID}@test.local`,
        name: "Test Agent",
        password: "x",
        workingAs: "AI Agent",
        isDev: true,
        subRoleId: subRole.id,
      },
    });
  } else {
    await db.user.update({
      where: { id: TEST_AGENT_ID },
      data: { workingAs: "AI Agent", isDev: true, subRoleId: subRole.id },
    });
  }
}

export async function createUser(
  prefix: string,
  label: string,
  overrides: { name?: string; workingAs?: string | null } = {}
): Promise<{ userId: string; email: string }> {
  const id = uid(prefix, `u-${label}`);
  const user = await db.user.create({
    data: {
      id,
      email: `${id}@test.local`,
      name: overrides.name ?? "Test Member",
      password: "x",
      workingAs: overrides.workingAs ?? null,
    },
  });
  return { userId: user.id, email: user.email };
}

export async function resetBreaker(): Promise<void> {
  await circuitBreaker.resetManually("test reset");
}

/**
 * Removes every row the test suite may have created (scoped by the unique
 * prefix / agent id) so no production-visible leftovers remain.
 */
export async function cleanupTestData(prefix: string): Promise<void> {
  const agentLogFilter = {
    OR: [
      { agentId: TEST_AGENT_ID },
      { projectId: { startsWith: prefix } },
    ],
  };

  await db.agentActionLog.deleteMany({ where: agentLogFilter });
  await db.agentDecision.deleteMany({ where: agentLogFilter });
  await db.agentMemory.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.notification.deleteMany({ where: { userId: { startsWith: prefix } } });
  await db.comment.deleteMany({ where: { taskId: { startsWith: prefix } } });
  await db.documentAccess.deleteMany({ where: { docId: { startsWith: prefix } } });
  await db.planDocs.deleteMany({ where: { id: { startsWith: prefix } } });
  await db.memo.deleteMany({ where: { meetingId: { startsWith: prefix } } });
  await db.meetingRequest.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.roadmapComment.deleteMany({ where: { roadmapId: { startsWith: prefix } } });
  await db.roadmapPermission.deleteMany({ where: { roadmapId: { startsWith: prefix } } });
  await db.roadmapAssignment.deleteMany({ where: { roadmapId: { startsWith: prefix } } });
  await db.roadmap.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.groupingTask.deleteMany({ where: { name: { contains: prefix } } });
  await db.workDone.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.ticket.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.usefulTip.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.latestUpdate.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.milestone.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.asset.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.featureRequest.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.riskBlockage.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.docs.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.taskDependency.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.task.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.designSystem.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.projectTechnology.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.projectInfo.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.projectMember.deleteMany({ where: { projectId: { startsWith: prefix } } });
  await db.project.deleteMany({ where: { id: { startsWith: prefix } } });
  await db.subRole.deleteMany({ where: { id: { startsWith: prefix } } });
  await db.subRole.deleteMany({ where: { id: { startsWith: TEST_AGENT_ID } } });
  await db.user.deleteMany({ where: { id: { startsWith: prefix } } });
  await db.user.deleteMany({ where: { id: TEST_AGENT_ID } });
  await resetBreaker();
}