/** @jest-environment node */
import db from "@/lib/client";
import { buildProjectContext } from "@/ai/context";
import { makePrefix, cleanupTestData, createUser, resetBreaker } from "./setup";
import {
  seedProject,
  seedProjectInfo,
  seedTask,
  seedMember,
  seedDependency,
  seedWorkHours,
  seedRiskBlockage,
  daysFromNow,
} from "./fixtures";

describe("ai/context buildProjectContext", () => {
  const prefix = makePrefix("context");
  let projectId: string;
  let memberId: string;
  let leaderId: string;

  beforeAll(async () => {
    await resetBreaker();
    const project = await seedProject(prefix, "Context Project");
    projectId = project.projectId;

    const member = await createUser(prefix, "dev", { workingAs: "Software Developer" });
    memberId = member.userId;
    const leader = await createUser(prefix, "lead", { name: "Leader One" });
    leaderId = leader.userId;

    await seedProjectInfo(prefix, projectId, 30);
    await seedMember(prefix, projectId, memberId, { role: "developer" });
    await seedMember(prefix, projectId, leaderId, { role: "manager", isLeader: true });
    await seedWorkHours(memberId, 20, daysFromNow(-1));
    await seedWorkHours(leaderId, 40, daysFromNow(-1));

    const { taskId: depTask } = await seedTask(prefix, projectId, "dep", {
      title: "API written",
      status: "in-progress",
      assignee: "Leader One",
      userId: leaderId,
    });
    const { taskId: blkTask } = await seedTask(prefix, projectId, "blocked", {
      title: "UI integration",
      status: "pending",
      assignee: "Dev One",
      userId: memberId,
    });
    const { taskId: readyTask } = await seedTask(prefix, projectId, "ready", {
      title: "Tests",
      status: "in-progress",
      assignee: "Dev One",
      userId: memberId,
    });
    await seedTask(prefix, projectId, "ship", {
      title: "Shipped",
      status: "completed",
      assignee: "Leader One",
      userId: leaderId,
    });
    await seedDependency(prefix, projectId, blkTask, depTask, "hard");
    await seedDependency(prefix, projectId, readyTask, depTask, "soft");

    await seedRiskBlockage(prefix, projectId, {
      title: "Low risk",
      severity: "low",
    });
    await seedRiskBlockage(prefix, projectId, {
      title: "Critical blocker",
      severity: "critical",
    });
  });

  afterAll(async () => {
    await cleanupTestData(prefix);
  });

  it("builds a context snapshot of the live project graph", async () => {
    const ctx = await buildProjectContext(projectId);

    expect(ctx.project.id).toBe(projectId);
    expect(ctx.project.name).toBe("Context Project");
    expect(ctx.project.status).toBe(0);
    expect(ctx.project.progress).toBe(0);
    expect(ctx.project.memberCount).toBe(2);
    expect(ctx.project.deadline).not.toBeNull();
    expect(ctx.project.deadlineOverdue).toBe(false);

    expect(ctx.activeTasks).toHaveLength(3);
    expect(ctx.activeTasks.some((t) => t.title === "UI integration")).toBe(true);
    expect(ctx.activeTasks.some((t) => t.title === "Tests")).toBe(true);
    expect(ctx.activeTasks.some((t) => t.title === "API written")).toBe(true);
    expect(ctx.activeTasks.some((t) => t.title === "Shipped")).toBe(false);

    const blocked = ctx.activeTasks.find((t) => t.title === "UI integration");
    expect(blocked!.isBlocked).toBe(true);
    const soft = ctx.activeTasks.find((t) => t.title === "Tests");
    expect(soft!.isBlocked).toBe(false);

    expect(ctx.dependencyGraph).toHaveLength(2);
    const hardEdge = ctx.dependencyGraph.find((e) => e.taskId === blocked!.id);
    expect(hardEdge!.relationType).toBe("hard");

    const devCapacity = ctx.teamCapacity.find((c) => c.userId === memberId);
    expect(devCapacity!.totalHours7d).toBe(20);
    expect(devCapacity!.workloadPercent).toBe(50);
    const leadCapacity = ctx.teamCapacity.find((c) => c.userId === leaderId);
    expect(leadCapacity!.workloadPercent).toBe(100);
    expect(ctx.teamCapacity.find((c) => c.userId === leaderId)!.isLeader).toBe(true);

    const top = ctx.risks.topRiskBlockers;
    expect(top[0].title).toBe("Critical blocker");
    expect(top[0].severity).toBe("critical");
    expect(ctx.risks.blockedTaskCount).toBe(1);
  });

  it("flags an overdue project deadline", async () => {
    const proj = await seedProject(prefix, "Due Project", "proj2");
    const { deadline } = await db.projectInfo.create({
      data: {
        projectId: proj.projectId,
        budget: 10,
        clientName: "x",
        projectType: "internal",
        startDate: daysFromNow(-40),
        deadline: daysFromNow(-2),
      },
    });
    expect(deadline).toBeTruthy();

    const ctx = await buildProjectContext(proj.projectId);
    expect(ctx.project.deadlineOverdue).toBe(true);
  });

  it("rejects a missing project", async () => {
    await expect(buildProjectContext("missing-project-definitely-not-there")).rejects.toThrow(
      /Project not found/
    );
  });

  it("rejects an empty projectId", async () => {
    await expect(buildProjectContext("")).rejects.toThrow(/requires a non-empty projectId/);
  });
});