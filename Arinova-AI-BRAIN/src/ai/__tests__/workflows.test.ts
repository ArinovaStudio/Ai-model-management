/** @jest-environment node */
jest.mock("@/ai/nlu/ollama", () => {
  const actual = jest.requireActual("@/ai/nlu/ollama");
  return { ...actual, callOllama: jest.fn() };
});

import db from "@/lib/client";
import { cronMatches, workflowLock, Scheduler } from "@/ai/scheduler";
import {
  EventListener,
  handleTaskCompleted,
  handleBlockerCreated,
  handleCommentAdded,
  type AgentEvent,
} from "@/ai/events";
import { callOllama } from "@/ai/nlu/ollama";
import {
  makePrefix,
  uid,
  ensureAgentUser,
  createUser,
  cleanupTestData,
  resetBreaker,
  TEST_AGENT_ID,
} from "./setup";
import {
  seedProject,
  seedTask,
  seedMember,
  seedDependency,
  seedRiskBlockage,
  daysFromNow,
} from "./fixtures";

const mockCallOllama = jest.mocked(callOllama);

describe("ai/scheduler cron (pure)", () => {
  it("matches asterisk expressions", () => {
    expect(cronMatches("* * * * *", new Date("2026-08-14T10:15:00Z"))).toBe(true);
  });

  it("matches minute and hour constraints", () => {
    const at = new Date("2026-08-14T09:00:00Z");
    expect(cronMatches("0 9 * * *", at)).toBe(true);
    expect(cronMatches("30 9 * * *", at)).toBe(false);
    expect(cronMatches("0 8 * * *", at)).toBe(false);
  });

  it("matches ranges and lists", () => {
    const at = new Date("2026-08-14T17:30:00Z");
    expect(cronMatches("30 17 * * *", at)).toBe(true);
    expect(cronMatches("0,10,20,30 17 * * *", at)).toBe(true);
    expect(cronMatches("*/5 17 * * *", at)).toBe(true);
  });

  it("matches day-of-week without requiring a specific day-of-month", () => {
    // 2026-08-14 is a Friday; DOW=5.
    const friday = new Date("2026-08-14T08:00:00Z");
    expect(cronMatches("0 8 * * 5", friday)).toBe(true);
    expect(cronMatches("0 8 * * 1", friday)).toBe(false);
  });

  it("rejects malformed expressions", () => {
    expect(cronMatches("0 9 * *", new Date())).toBe(false);
    expect(cronMatches("", new Date())).toBe(false);
  });
});

describe("ai/scheduler workflow lock and scheduler (pure)", () => {
  it("runs a workflow once and dedupes an in-flight run", async () => {
    let calls = 0;
    let release: (() => void) | null = null;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });

    // Kick off the first run without awaiting it; it blocks on `gate`.
    const first = workflowLock.run("test-workflow", async () => {
      calls += 1;
      await gate;
    });

    const second = await workflowLock.run("test-workflow", async () => {
      calls += 1;
    });
    expect(second.status).toBe("skipped");
    expect(calls).toBe(1);

    release!();
    const result = await first;
    expect(result.status).toBe("ran");
    expect(calls).toBe(1);
  });

  it("registers and runs a job with runNow", async () => {
    const scheduler = new Scheduler(10_000);
    let ran = 0;
    scheduler.schedule("probe", "* * * * *", async () => {
      ran += 1;
    });

    await scheduler.runNow("probe");
    await scheduler.runNow("probe");
    expect(ran).toBe(2);

    const unknown = scheduler.getJobs().filter((job) => job.name === "probe");
    expect(unknown.length).toBe(1);
    await scheduler.runNow("nope");
  });
});

describe("ai/events EventListener (live DB)", () => {
  let listener: EventListener;
  const handled: string[] = [];

  beforeAll(() => {
    listener = new EventListener(TEST_AGENT_ID);
    listener.subscribe("TASK_COMPLETED", async (event) => {
      handled.push(event.id);
    });
  });

  beforeEach(async () => {
    handled.length = 0;
  });

  it("dispatches subscribed handlers for a known event type", async () => {
    const event: AgentEvent = {
      id: "evt-1",
      type: "TASK_COMPLETED",
      at: new Date().toISOString(),
      payload: { taskId: "abc", projectId: "proj" },
    };
    await listener.handleEvent(event);
    expect(handled).toEqual(["evt-1"]);
  });
});

describe("ai/events workflow handlers (live DB)", () => {
  const prefix = makePrefix("workflows");
  let projectId: string;
  let memberId: string;
  let leaderId: string;

  beforeAll(async () => {
    await resetBreaker();
    await ensureAgentUser();

    const project = await seedProject(prefix, "Workflow Project");
    projectId = project.projectId;
    memberId = (await createUser(prefix, "dev")).userId;
    leaderId = (await createUser(prefix, "lead")).userId;
    await seedMember(prefix, projectId, leaderId, { role: "manager", isLeader: true });
    await seedMember(prefix, projectId, memberId, { role: "member" });
  });

  beforeEach(async () => {
    mockCallOllama.mockReset();
    await resetBreaker();
  });

  afterAll(async () => {
    await cleanupTestData(prefix);
  });

  it("re-evaluates dependents once a task completes", async () => {
    const parent = await seedTask(prefix, projectId, "parent", {
      title: "Landing ready",
      status: "completed",
    });
    const child = await seedTask(prefix, projectId, "child", {
      title: "Build on landing",
      status: "pending",
      userId: memberId,
      assignee: "Dev",
    });
    await seedDependency(prefix, projectId, child.taskId, parent.taskId, "hard");

    await handleTaskCompleted({
      id: "evt-complete",
      type: "TASK_COMPLETED",
      at: new Date().toISOString(),
      payload: { taskId: parent.taskId, projectId },
    });

    const log = await db.agentActionLog.findFirst({
      where: { action: "task_updated", target: child.taskId },
      orderBy: { createdAt: "desc" },
    });
    expect(log).not.toBeNull();
    expect(log!.success).toBe(true);
    expect(JSON.stringify(log!.result)).toContain("actionable");

    const completionLog = await db.agentActionLog.findFirst({
      where: { action: "task_status_changed", target: parent.taskId },
      orderBy: { createdAt: "desc" },
    });
    expect(completionLog).not.toBeNull();
    expect(completionLog!.success).toBe(true);
  });

  it("assesses a critical-path blocker and notifies the project leader", async () => {
    const blocked = await seedTask(prefix, projectId, "blocked", {
      title: "Checkout flow",
      status: "in-progress",
    });
    const dependent = await seedTask(prefix, projectId, "dep", {
      title: "Mobile billing",
      status: "pending",
    });
    await seedDependency(prefix, projectId, dependent.taskId, blocked.taskId, "hard");
    const blockerId = await seedRiskBlockage(prefix, projectId, {
      title: "Stripe outage",
      severity: "critical",
    });

    await handleBlockerCreated({
      id: "evt-blocker",
      type: "BLOCKER_CREATED",
      at: new Date().toISOString(),
      payload: { blockerId, projectId, taskId: blocked.taskId },
    });

    const memory = await db.agentMemory.findFirst({
      where: { projectId, type: "blocker_context" },
      orderBy: { createdAt: "desc" },
    });
    expect(memory!.content).toContain("Stripe outage");
    expect(JSON.stringify(memory!.meta)).toContain('"onCriticalPath":true');

    const log = await db.agentActionLog.findFirst({
      where: { action: "blocker_recorded", projectId },
      orderBy: { createdAt: "desc" },
    });
    expect(log!.success).toBe(true);

    const notification = await db.notification.findFirst({
      where: { userId: leaderId, title: { contains: "Critical blocker" } },
      orderBy: { createdAt: "desc" },
    });
    expect(notification).not.toBeNull();
    expect(notification!.message).toContain("Stripe outage");
  });

  it("does not notify leaders for a non-critical blocker", async () => {
    const blockerId = await seedRiskBlockage(prefix, projectId, {
      title: "Design sign-off pending",
      severity: "medium",
    });

    await handleBlockerCreated({
      id: "evt-blocker2",
      type: "BLOCKER_CREATED",
      at: new Date().toISOString(),
      payload: { blockerId, projectId },
    });

    const criticalNotification = await db.notification.findFirst({
      where: { userId: leaderId, message: { contains: "Design sign-off pending" } },
    });
    expect(criticalNotification).toBeNull();
  });

  it("parses an employee update from a comment and acts on it", async () => {
    mockCallOllama.mockResolvedValueOnce({
      text: JSON.stringify({
        taskId: null,
        progressPercent: 50,
        status: "in-progress",
        blocker: false,
        blockerDescription: null,
        blockedByUserId: null,
        nextSteps: "Continue on the tasks list page",
        confidence: 0.9,
      }),
      model: "mock-llm",
    });

    const task = await seedTask(prefix, projectId, "comment-task", {
      title: "Tasks list page",
      status: "pending",
    });
    const comment = await db.comment.create({
      data: {
        content: "Working on the UI now; the layout is 50% done.",
        taskId: task.taskId,
        userId: memberId,
      },
      select: { id: true },
    });

    await handleCommentAdded({
      id: "evt-comment",
      type: "COMMENT_ADDED",
      at: new Date().toISOString(),
      payload: {
        commentId: comment.id,
        taskId: task.taskId,
        userId: memberId,
        content: "Working on the UI now; the layout is 50% done.",
      },
    });

    const parsedLog = await db.agentActionLog.findFirst({
      where: { action: "employee_reply_parsed", projectId },
      orderBy: { createdAt: "desc" },
    });
    expect(parsedLog).not.toBeNull();
    expect(parsedLog!.success).toBe(true);
    expect(JSON.stringify(parsedLog!.result)).toContain('"status":"parsed"');
  });

  it("ignores comments from the agent to avoid feedback loops", async () => {
    const loopTask = await seedTask(prefix, projectId, "loop-task", {
      title: "Loop task",
    });
    const agentComment = await db.comment.create({
      data: {
        content: "I am 100% done with this task.",
        taskId: loopTask.taskId,
        userId: TEST_AGENT_ID,
      },
      select: { id: true },
    });

    await handleCommentAdded({
      id: "evt-self",
      type: "COMMENT_ADDED",
      at: new Date().toISOString(),
      payload: {
        commentId: agentComment.id,
        taskId: loopTask.taskId,
        userId: TEST_AGENT_ID,
        content: "I am 100% done with this task.",
      },
    });

    const logs = await db.agentActionLog.findMany({
      where: { action: "employee_reply_parsed", target: loopTask.taskId },
    });
    expect(logs.length).toBe(0);
  });
});