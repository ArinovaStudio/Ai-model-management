/** @jest-environment node */

jest.mock("@/ai/nlu/ollama", () => {
  const actual = jest.requireActual("@/ai/nlu/ollama");
  return { ...actual, callOllama: jest.fn() };
});

import db from "@/lib/client";
import { buildProjectContext } from "@/ai/context";
import { taskReadiness, riskScore, assignmentScore } from "@/ai/reasoner/scoring";
import { evaluate } from "@/ai/reasoner";
import { callOllama } from "@/ai/nlu/ollama";
import type { ProjectContext } from "@/ai/context/types";
import type { AssignmentCandidate } from "@/ai/reasoner/types";
import { makePrefix, cleanupTestData, createUser, resetBreaker, TEST_AGENT_ID } from "./setup";
import {
  seedProject,
  seedProjectInfo,
  seedTask,
  seedMember,
  seedDependency,
  seedRiskBlockage,
  daysFromNow,
} from "./fixtures";

const mockCallOllama = jest.mocked(callOllama);

function baseContext(): ProjectContext {
  return {
    project: {
      id: "p1",
      name: "Base",
      summary: null,
      priority: "high",
      status: 0,
      progress: 50,
      currentPhase: "Build",
      deadline: new Date(Date.now() + 86400000).toISOString(),
      deadlineOverdue: false,
      memberCount: 1,
    },
    activeTasks: [],
    dependencyGraph: [],
    teamCapacity: [],
    risks: {
      activeTaskCount: 0,
      blockedTaskCount: 0,
      overdueTaskCount: 0,
      topRiskBlockers: [],
    },
    createdAt: new Date().toISOString(),
  };
}

describe("ai/reasoner scoring (pure)", () => {
  it("marks a task ready when all hard dependencies are terminal", () => {
    const edges = [
      { taskId: "a", dependsOnTaskId: "b", relationType: "hard" },
      { taskId: "a", dependsOnTaskId: "c", relationType: "soft" },
    ];
    const statusById = new Map<string, string>([
      ["a", "pending"],
      ["b", "completed"],
      ["c", "pending"],
    ]);
    const result = taskReadiness("a", edges, statusById);
    expect(result.ready).toBe(true);
    expect(result.hasCycle).toBe(false);
    expect(result.unresolvedDependencies).toHaveLength(0);
  });

  it("blocks a task on an unsatisfied hard dependency", () => {
    const edges = [{ taskId: "a", dependsOnTaskId: "b", relationType: "hard" }];
    const statusById = new Map<string, string>([
      ["a", "pending"],
      ["b", "in-progress"],
    ]);
    const result = taskReadiness("a", edges, statusById);
    expect(result.ready).toBe(false);
    expect(result.unresolvedDependencies).toEqual(["b"]);
  });

  it("flags circular dependencies as unactionable", () => {
    const edges = [
      { taskId: "a", dependsOnTaskId: "b", relationType: "hard" },
      { taskId: "b", dependsOnTaskId: "a", relationType: "hard" },
    ];
    const statusById = new Map<string, string>([
      ["a", "pending"],
      ["b", "pending"],
    ]);
    const result = taskReadiness("a", edges, statusById);
    expect(result.ready).toBe(false);
    expect(result.hasCycle).toBe(true);
  });

  it("scores a project as LOW when on track", () => {
    const risk = riskScore(baseContext());
    expect(risk.level).toBe("LOW");
    expect(risk.concerns).toHaveLength(0);
    expect(risk.progress).toBe(50);
    expect(risk.progressClamped).toBe(false);
  });

  it("scores a single overdue task as MEDIUM", () => {
    const ctx = baseContext();
    ctx.risks = {
      activeTaskCount: 1,
      blockedTaskCount: 0,
      overdueTaskCount: 1,
      topRiskBlockers: [],
    };
    ctx.activeTasks = [
      {
        id: "t1",
        title: "Late",
        status: "pending",
        priority: "high",
        assignee: "Dev",
        dueDate: new Date(Date.now() - 86400000).toISOString(),
        createdAt: new Date().toISOString(),
        isBlocked: false,
      },
    ];
    const risk = riskScore(ctx);
    expect(risk.level).toBe("MEDIUM");
    expect(risk.concerns.some((c) => c.id === "overdue_tasks")).toBe(true);
  });

  it("scores HIGH when the deadline is overdue with work remaining", () => {
    const ctx = baseContext();
    ctx.project.deadlineOverdue = true;
    ctx.project.progress = 40;
    const risk = riskScore(ctx);
    expect(risk.level).toBe("HIGH");
    expect(risk.explanation).toContain("deadline is in the past");
    expect(risk.confidence.score).toBeGreaterThanOrEqual(0.9);
  });

  it("scores HIGH when a critical path is blocked by a critical blocker", () => {
    const ctx = baseContext();
    ctx.risks = {
      activeTaskCount: 1,
      blockedTaskCount: 1,
      overdueTaskCount: 0,
      topRiskBlockers: [
        {
          id: "rb1",
          title: "Deploy outage",
          severity: "critical",
          status: "open",
          description: "",
          createdAt: new Date().toISOString(),
        },
      ],
    };
    const risk = riskScore(ctx);
    expect(risk.level).toBe("HIGH");
    expect(risk.concerns.some((c) => c.id === "critical_blockers")).toBe(true);
  });

  it("clamps progress outside [0,100] and reports it", () => {
    const ctx = baseContext();
    ctx.project.progress = 140;
    const risk = riskScore(ctx);
    expect(risk.progress).toBe(100);
    expect(risk.progressClamped).toBe(true);
  });

  it("scores a non-member as zero", () => {
    const candidate: AssignmentCandidate = {
      userId: "u1",
      name: "Outsider",
      membership: false,
      isLeader: false,
      role: null,
      workloadPercent: 10,
      workingAs: "React Developer",
      department: null,
      subRole: null,
      bio: null,
    };
    const score = assignmentScore(["react"], "React migration", "medium", candidate);
    expect(score.score).toBe(0);
    expect(score.components.membershipScore).toBe(0);
    expect(score.basis[0]).toContain("not a project member");
  });

  it("computes a weighted member score", () => {
    const candidate: AssignmentCandidate = {
      userId: "u1",
      name: "Dev",
      membership: true,
      isLeader: false,
      role: "developer",
      workloadPercent: 20,
      workingAs: "React Developer",
      department: null,
      subRole: null,
      bio: "focused on react",
    };
    const score = assignmentScore(["react"], "React migration", "medium", candidate);
    expect(score.components.workloadScore).toBe(0.8);
    expect(score.components.skillScore).toBe(0.5);
    expect(score.components.membershipScore).toBe(1);
    expect(score.score).toBe(0.715);
  });
});

describe("ai/reasoner evaluate (live DB + mocked LLM)", () => {
  const prefix = makePrefix("reasoner");
  let projectId: string;
  let memberId: string;
  let parentTaskId: string;
  let childTaskId: string;

  beforeAll(async () => {
    await resetBreaker();
    const project = await seedProject(prefix, "Reasoner Project");
    projectId = project.projectId;
    await seedProjectInfo(prefix, projectId, 30);

    const member = await createUser(prefix, "dev", { workingAs: "React Developer" });
    memberId = member.userId;
    await seedMember(prefix, projectId, memberId, { role: "developer" });

    const parent = await seedTask(prefix, projectId, "parent", {
      title: "Deploy pipeline",
      status: "in-progress",
      assignee: "Dev",
      userId: memberId,
      tags: ["react"],
    });
    parentTaskId = parent.taskId;
    const child = await seedTask(prefix, projectId, "child", {
      title: "React migration",
      status: "pending",
      assignee: "Dev",
      userId: memberId,
      tags: ["react"],
      dueDate: daysFromNow(2),
    });
    childTaskId = child.taskId;
    await seedDependency(prefix, projectId, childTaskId, parentTaskId, "hard");
  });

  beforeEach(async () => {
    mockCallOllama.mockReset();
    await resetBreaker();
  });

  afterAll(async () => {
    await cleanupTestData(prefix);
  });

  it("reports a dependent task as not ready while its parent is open", async () => {
    const context = await buildProjectContext(projectId);
    const evaluation = await evaluate(context, {
      taskId: childTaskId,
      candidates: [memberId],
      runLlm: false,
    });

    expect(evaluation.task).not.toBeNull();
    expect(evaluation.task!.ready).toBe(false);
    expect(evaluation.task!.unresolvedDependencies).toContain(parentTaskId);
    expect(evaluation.llmUsed).toBe(false);
  });

  it("lists project members as scored assignment candidates", async () => {
    const context = await buildProjectContext(projectId);
    const evaluation = await evaluate(context, {
      taskId: childTaskId,
      candidates: [memberId, TEST_AGENT_ID],
      runLlm: false,
    });

    expect(evaluation.assignment).not.toBeNull();
    const names = evaluation.assignment!.ranked.map((c) => c.userId);
    expect(names).toContain(memberId);
    expect(names.length).toBeLessThanOrEqual(3);
  });

  it("marks the task ready once its hard dependency completes", async () => {
    await db.task.update({ where: { id: parentTaskId }, data: { status: "completed" } });

    const context = await buildProjectContext(projectId);
    const evaluation = await evaluate(context, { taskId: childTaskId, runLlm: false });

    expect(evaluation.task!.ready).toBe(true);
  });

  it("generates a grounded risk narrative via the mocked LLM", async () => {
    await seedRiskBlockage(prefix, projectId, {
      title: "Deploy outage",
      severity: "critical",
    });
    mockCallOllama.mockResolvedValue({
      text: JSON.stringify({
        narrative: "Deploy pipeline has a single critical blocker holding back the child task.",
        recommendations: ["Resolve the critical blocker to unblock the plan"],
      }),
      model: "mock-llm",
    });

    const context = await buildProjectContext(projectId);
    const evaluation = await evaluate(context, { runLlm: true });

    expect(evaluation.llmUsed).toBe(true);
    expect(evaluation.riskNarrative).not.toBeNull();
    expect(evaluation.riskNarrative!.recommendations.length).toBeGreaterThanOrEqual(1);
  });
});