/** @jest-environment node */
import db from "@/lib/client";
import { execute } from "@/ai/executor";
import { circuitBreaker } from "@/ai/circuit-breaker";
import {
  compareField,
  compareFields,
  mergeVerification,
  detectConflict,
} from "@/ai/executor/verifier";
import {
  makePrefix,
  cleanupTestData,
  createUser,
  allowedPolicy,
  mediumApprovalPolicy,
  deniedPolicy,
  resetBreaker,
  TEST_AGENT_ID,
} from "./setup";
import { seedProject, seedTask, daysFromNow } from "./fixtures";

describe("ai/executor verifier (pure)", () => {
  it("normalizes numbers and dates for comparison", () => {
    expect(compareField(1, "1", "n").passed).toBe(true);
    const iso = new Date("2026-01-01T00:00:00.000Z");
    expect(compareField(new Date(iso.toISOString()), iso.toISOString(), "d").passed).toBe(true);
    expect(compareField("x", "y", "f").passed).toBe(false);
  });

  it("merges multiple field comparisons", () => {
    const ok = mergeVerification(compareFields({ a: 1, b: "x" }, { a: "1", b: "x" }));
    expect(ok.passed).toBe(true);
    const bad = mergeVerification(
      compareFields({ a: 1, b: "nope" }, { a: "1", b: "x" })
    );
    expect(bad.passed).toBe(false);
    expect(bad.outcome).toBe("mismatch");
  });

  it("distinguishes pass, mismatch and conflict in detectConflict", () => {
    expect(detectConflict("completed", "completed", "pending", "status").passed).toBe(true);
    const unchanged = detectConflict("pending", "completed", "pending", "status");
    expect(unchanged.outcome).toBe("mismatch");
    const concurrent = detectConflict("blocked", "completed", "pending", "status");
    expect(concurrent.outcome).toBe("conflict");
  });
});

describe("ai/executor execute (live DB)", () => {
  const prefix = makePrefix("executor");
  let projectId: string;
  let taskId: string;
  let assigeeId: string;

  async function handlerContext(projectId: string): Promise<string> {
    const u = await createUser(prefix, "assigned");
    return u.userId;
  }

  beforeAll(async () => {
    await resetBreaker();
    const project = await seedProject(prefix, "Executor Project");
    projectId = project.projectId;
    assigeeId = await handlerContext(projectId);
    const task = await seedTask(prefix, projectId, "core", {
      title: "Core feature",
      status: "pending",
      assignee: "Assigned",
      userId: assigeeId,
      dueDate: daysFromNow(5),
    });
    taskId = task.taskId;
  });

  beforeEach(async () => {
    await resetBreaker();
  });

  afterAll(async () => {
    await cleanupTestData(prefix);
  });

  it("adds a comment and verifies it end to end", async () => {
    const result = await execute("add_comment", {
      agentId: TEST_AGENT_ID,
      projectId,
      taskId,
      payload: { content: "Status update recorded.", userId: assigeeId },
      policy: allowedPolicy("add_comment", projectId),
    });

    expect(result.status).toBe("SUCCESS");
    expect(result.success).toBe(true);
    expect(result.verification?.passed).toBe(true);
    expect(result.logId).not.toBeNull();

    const comment = await db.comment.findFirst({
      where: { taskId, content: "Status update recorded." },
    });
    expect(comment).not.toBeNull();
    expect(comment!.userId).toBe(assigeeId);

    const log = await db.agentActionLog.findUnique({ where: { id: result.logId! } });
    expect(log!.success).toBe(true);
    expect(log!.action).toBe("add_comment");
  });

  it("records a blocker", async () => {
    const result = await execute("record_blocker", {
      agentId: TEST_AGENT_ID,
      projectId,
      taskId,
      payload: { title: "External dependency missing", severity: "high" },
      policy: allowedPolicy("record_blocker", projectId),
    });

    expect(result.status).toBe("SUCCESS");
    const blocker = await db.riskBlockage.findFirst({
      where: { projectId, title: "External dependency missing" },
    });
    expect(blocker).not.toBeNull();
    expect(blocker!.severity).toBe("high");
  });

  it("updates task status with verification", async () => {
    const result = await execute("update_task_status", {
      agentId: TEST_AGENT_ID,
      projectId,
      taskId,
      payload: { status: "in-progress" },
      policy: allowedPolicy("update_task_status", projectId),
    });

    expect(result.status).toBe("SUCCESS");
    const updated = await db.task.findUnique({ where: { id: taskId } });
    expect(updated!.status).toBe("in-progress");
  });

  it("fails on an invalid status without mutating", async () => {
    const result = await execute("update_task_status", {
      agentId: TEST_AGENT_ID,
      projectId,
      taskId,
      payload: { status: "sideways" },
      policy: allowedPolicy("update_task_status", projectId),
    });

    expect(result.status).toBe("FAILED");
    const updated = await db.task.findUnique({ where: { id: taskId } });
    expect(updated!.status).toBe("in-progress");
  });

  it("enforces preconditions before mutating", async () => {
    const result = await execute("update_task_status", {
      agentId: TEST_AGENT_ID,
      projectId,
      payload: { status: "completed" },
      policy: allowedPolicy("update_task_status", projectId),
    });

    expect(result.status).toBe("PRECONDITION_FAILED");
  });

  it("refuses to guess a progress mapping when Task has no progress column", async () => {
    const result = await execute("update_task_progress", {
      agentId: TEST_AGENT_ID,
      projectId,
      taskId,
      payload: { progress: 50 },
      policy: allowedPolicy("update_task_progress", projectId),
    });

    expect(result.status).toBe("FAILED");
    expect(result.verification?.outcome).toBe("field_missing");
    expect(result.reason).toContain("field_missing");
  });

  it("creates a task", async () => {
    const result = await execute("create_task", {
      agentId: TEST_AGENT_ID,
      projectId,
      payload: {
        title: "Created by agent",
        dueDate: daysFromNow(9).toISOString(),
        priority: "high",
        tags: ["qa"],
      },
      policy: allowedPolicy("create_task", projectId),
    });

    expect(result.status).toBe("SUCCESS");
    const created = await db.task.findFirst({
      where: { projectId, title: "Created by agent" },
    });
    expect(created).not.toBeNull();
  });

  it("requires a due date when creating a task", async () => {
    const result = await execute("create_task", {
      agentId: TEST_AGENT_ID,
      projectId,
      payload: { title: "No date" },
      policy: allowedPolicy("create_task", projectId),
    });

    expect(result.status).toBe("PRECONDITION_FAILED");
    expect(result.reason).toContain("dueDate");
  });

  it("assigns a task to a user", async () => {
    const target = await createUser(prefix, "target");
    const result = await execute("assign_task", {
      agentId: TEST_AGENT_ID,
      projectId,
      taskId,
      targetUserId: target.userId,
      payload: {},
      policy: allowedPolicy("assign_task", projectId),
    });

    expect(result.status).toBe("SUCCESS");
    const updated = await db.task.findUnique({ where: { id: taskId } });
    expect(updated!.userId).toBe(target.userId);
    expect(updated!.assignee).toBe("Test Member");
  });

  it("skips execution and records an approval recommendation for MEDIUM actions", async () => {
    const result = await execute("update_task_status", {
      agentId: TEST_AGENT_ID,
      projectId,
      taskId,
      payload: { status: "completed" },
      policy: mediumApprovalPolicy("update_task_status", projectId),
    });

    expect(result.status).toBe("SKIPPED_RECOMMENDED");
    const recommendation = await db.agentDecision.findFirst({
      where: { agentId: TEST_AGENT_ID },
      orderBy: { createdAt: "desc" },
    });
    expect(recommendation).not.toBeNull();
    expect(recommendation!.basis).toMatchObject({ action: "update_task_status" });
    const updated = await db.task.findUnique({ where: { id: taskId } });
    expect(updated!.status).toBe("in-progress");
  });

  it("refuses when the injected policy denies the action", async () => {
    const result = await execute("update_task_status", {
      agentId: TEST_AGENT_ID,
      projectId,
      taskId,
      payload: { status: "completed" },
      policy: deniedPolicy("update_task_status"),
    });

    expect(result.status).toBe("REFUSED_POLICY");
    const updated = await db.task.findUnique({ where: { id: taskId } });
    expect(updated!.status).toBe("in-progress");
  });

  it("sends a notification to a user", async () => {
    const result = await execute("send_notification", {
      agentId: TEST_AGENT_ID,
      targetUserId: assigeeId,
      payload: { title: "Heads up", message: "You have a new task." },
      policy: allowedPolicy("send_notification"),
    });

    expect(result.status).toBe("SUCCESS");
    const notification = await db.notification.findFirst({
      where: { userId: assigeeId, title: "Heads up" },
    });
    expect(notification).not.toBeNull();
  });

  it("blocks writes while the circuit is OPEN", async () => {
    await circuitBreaker.recordFailure("fatal", { reason: "force open" });
    expect((await circuitBreaker.isOpen())).toBe(true);

    const result = await execute("add_comment", {
      agentId: TEST_AGENT_ID,
      projectId,
      taskId,
      payload: { content: "should not be written" },
      policy: allowedPolicy("add_comment", projectId),
    });

    expect(result.status).toBe("CIRCUIT_OPEN");
    const comment = await db.comment.findFirst({ where: { taskId, content: "should not be written" } });
    expect(comment).toBeNull();
  });
});