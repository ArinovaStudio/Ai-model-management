/** @jest-environment node */
jest.mock("uuid", () => {
  let counter = 0;
  return {
    v4: () => `test-uuid-${Date.now()}-${counter++}-${Math.random().toString(36).slice(2, 8)}`,
  };
});

import db from "@/lib/client";
import {
  ingestProject,
  remapDependencyString,
  remapTaskReferencesInJson,
  detectDependencyCycles,
} from "@/ai/ingestion";
import {
  makePrefix,
  uid,
  createUser,
  cleanupTestData,
} from "./setup";
import {
  seedProject,
  seedProjectInfo,
  seedTask,
  seedMember,
  seedDependency,
  seedMilestone,
  seedAsset,
  seedLatestUpdate,
  seedFeatureRequest,
  seedRiskBlockage,
  seedPlanDoc,
  seedDocumentAccess,
  seedDocs,
  seedRoadmapRaw,
  seedDesignSystem,
  seedProjectTechnology,
  seedGroupingTask,
  seedWorkDone,
  seedTicket,
  seedUsefulTip,
  seedMeetingRequest,
  seedMemo,
} from "./fixtures";

describe("ai/ingestion helpers (pure)", () => {
  it("remaps known task ids in a dependency string and keeps foreign ones", () => {
    const map = new Map([["src-a", "new-a"]]);
    expect(remapDependencyString("src-a,foreign", map)).toBe("new-a,foreign");
    expect(remapDependencyString(null, map)).toBe("");
    expect(remapDependencyString("", map)).toBe("");
  });

  it("deep-rewrites task references inside JSON structures", () => {
    const map = new Map([["src-a", "new-a"]]);
    const rewritten = remapTaskReferencesInJson(
      { taskIds: ["src-a", "other"], meta: { depends: ["src-a"] }, n: 3, ok: true, none: null },
      map
    ) as Record<string, unknown>;
    expect(rewritten.taskIds).toEqual(["new-a", "other"]);
    expect((rewritten.meta as { depends: string[] }).depends).toEqual(["new-a"]);
    expect(rewritten.n).toBe(3);
  });

  it("detects cycles and leaves acyclic graphs alone", () => {
    expect(
      detectDependencyCycles([
        { taskId: "a", dependsOnTaskId: "b" },
        { taskId: "b", dependsOnTaskId: "c" },
      ])
    ).toEqual([]);
    const cycles = detectDependencyCycles([
      { taskId: "a", dependsOnTaskId: "b" },
      { taskId: "b", dependsOnTaskId: "a" },
    ]);
    expect(cycles.length).toBeGreaterThan(0);
    expect(cycles[0]).toContain("a");
    expect(cycles[0]).toContain("b");
  });
});

describe("ai/ingestion ingestProject (live DB)", () => {
  const prefix = makePrefix("ingest");
  let sourceProjectId: string;
  let sourceTaskIds: { a: string; b: string };
  let clientUserId: string;
  const cloneProjectIds: string[] = [];

  async function cleanupCloneProjects(): Promise<void> {
    for (const targetProjectId of cloneProjectIds) {
      await db.roadmapComment.deleteMany({ where: { roadmapId: { contains: "test-uuid" } } });
      await db.roadmapPermission.deleteMany({ where: { roadmapId: { contains: "test-uuid" } } });
      await db.roadmapAssignment.deleteMany({ where: { roadmapId: { contains: "test-uuid" } } });
      await db.roadmap.deleteMany({ where: { projectId: targetProjectId } });
      await db.documentAccess.deleteMany({ where: { docId: { contains: prefix } } });
      await db.memo.deleteMany({ where: { meetingId: { contains: prefix } } });
      await db.meetingRequest.deleteMany({ where: { projectId: targetProjectId } });
      await db.workDone.deleteMany({ where: { projectId: targetProjectId } });
      await db.ticket.deleteMany({ where: { projectId: targetProjectId } });
      await db.usefulTip.deleteMany({ where: { projectId: targetProjectId } });
      await db.latestUpdate.deleteMany({ where: { projectId: targetProjectId } });
      await db.milestone.deleteMany({ where: { projectId: targetProjectId } });
      await db.asset.deleteMany({ where: { projectId: targetProjectId } });
      await db.featureRequest.deleteMany({ where: { projectId: targetProjectId } });
      await db.riskBlockage.deleteMany({ where: { projectId: targetProjectId } });
      await db.docs.deleteMany({ where: { projectId: targetProjectId } });
      await db.taskDependency.deleteMany({ where: { projectId: targetProjectId } });
      await db.task.deleteMany({ where: { projectId: targetProjectId } });
      await db.planDocs.deleteMany({ where: { projectId: targetProjectId } });
      await db.designSystem.deleteMany({ where: { projectId: targetProjectId } });
      await db.projectTechnology.deleteMany({ where: { projectId: targetProjectId } });
      await db.projectInfo.deleteMany({ where: { projectId: targetProjectId } });
      await db.projectMember.deleteMany({ where: { projectId: targetProjectId } });
      await db.project.deleteMany({ where: { id: targetProjectId } });
    }
  }

  beforeAll(async () => {
    const client = await createUser(prefix, "client");
    clientUserId = client.userId;

    const project = await seedProject(prefix, "Source Ingest Project");
    sourceProjectId = project.projectId;

    await seedProjectInfo(prefix, sourceProjectId, 30);
    await seedMember(prefix, sourceProjectId, clientUserId, { role: "manager" });
    await seedMilestone(prefix, sourceProjectId, clientUserId);
    await seedAsset(prefix, sourceProjectId);
    await seedLatestUpdate(prefix, sourceProjectId, clientUserId);
    await seedFeatureRequest(prefix, sourceProjectId, clientUserId);
    await seedRiskBlockage(prefix, sourceProjectId, { title: "Source risk" });
    const docId = await seedPlanDoc(prefix, sourceProjectId, "source");
    await seedDocumentAccess(prefix, docId, clientUserId);
    await seedDocs(prefix, sourceProjectId);
    await seedRoadmapRaw(sourceProjectId, { title: "Source Roadmap" });
    await seedDesignSystem(prefix, sourceProjectId);
    await seedProjectTechnology(prefix, sourceProjectId);

    const taskA = await seedTask(prefix, sourceProjectId, "a", { title: "Task A" });
    const taskB = await seedTask(prefix, sourceProjectId, "b", { title: "Task B" });
    sourceTaskIds = { a: taskA.taskId, b: taskB.taskId };
    await seedDependency(prefix, sourceProjectId, taskB.taskId, taskA.taskId, "hard");
    await seedGroupingTask(prefix, "source", [taskA.taskId, taskB.taskId]);
    await seedWorkDone(prefix, sourceProjectId, taskA.taskId, clientUserId);
    await seedTicket(prefix, sourceProjectId);
    await seedUsefulTip(prefix, sourceProjectId, clientUserId);

    const meetingId = await seedMeetingRequest(prefix, sourceProjectId, clientUserId, "kickoff");
    await seedMemo(prefix, meetingId, clientUserId);
  });

  afterAll(async () => {
    await cleanupCloneProjects();
    await cleanupTestData(prefix);
  });

  it("throws for an unknown source project", async () => {
    await expect(ingestProject(`${prefix}-nope`, "Clone")).rejects.toThrow(/does not exist/);
  });

  it("clones the whole project graph into a new project", async () => {
    const result = await ingestProject(sourceProjectId, `${prefix}-clone`);
    cloneProjectIds.push(result.targetProjectId);

    expect(result.success).toBe(true);
    expect(result.targetProjectName).toBe(`${prefix}-clone`);
    expect(result.targetProjectId).not.toBe(sourceProjectId);

    expect(result.records.project).toBe(1);
    expect(result.records.projectInfo).toBe(1);
    expect(result.records.projectMembers).toBe(1);
    expect(result.records.milestones).toBe(1);
    expect(result.records.assets).toBe(1);
    expect(result.records.latestUpdates).toBe(1);
    expect(result.records.featureRequests).toBe(1);
    expect(result.records.riskBlockages).toBe(1);
    expect(result.records.planDocs).toBe(1);
    expect(result.records.documentAccess).toBe(1);
    expect(result.records.docs).toBe(1);
    expect(result.records.roadmaps).toBe(1);
    expect(result.records.designSystems).toBe(1);
    expect(result.records.projectTechnologies).toBe(1);
    expect(result.records.tasks).toBe(2);
    expect(result.records.taskDependencies).toBe(1);
    expect(result.records.groupTasks).toBe(1);
    expect(result.records.workDone).toBe(1);
    expect(result.records.tickets).toBe(1);
    expect(result.records.usefulTips).toBe(1);
    expect(result.records.meetingRequests).toBe(1);
    expect(result.records.memos).toBe(1);

    expect(result.sections.wbsSchedule.tasks).toBe(2);
    expect(result.sections.wbsSchedule.taskDependencies).toBe(1);
    expect(result.sections.timeline.latestUpdates).toBe(1);
  });

  it("maps source task ids to cloned task ids", async () => {
    const result = await ingestProject(sourceProjectId, `${prefix}-clone2`);
    cloneProjectIds.push(result.targetProjectId);

    const newA = result.taskIdMap[sourceTaskIds.a];
    const newB = result.taskIdMap[sourceTaskIds.b];
    expect(newA).toBeTruthy();
    expect(newB).toBeTruthy();
    expect(newA).not.toBe(newB);
    expect(newA).not.toBe(sourceTaskIds.a);

    const dep = await db.taskDependency.findFirst({
      where: { projectId: result.targetProjectId },
    });
    expect(dep).not.toBeNull();
    expect(dep!.taskId).toBe(newB);
    expect(dep!.dependsOnTaskId).toBe(newA);
  });

  it("clones grouping tasks with remapped task references", async () => {
    const result = await ingestProject(sourceProjectId, `${prefix}-clone3`);
    cloneProjectIds.push(result.targetProjectId);

    const groupings = await db.groupingTask.findMany({
      where: { name: { contains: prefix } },
    });
    const cloneTaskIds = new Set(Object.values(result.taskIdMap));
    const cloned = groupings.find((g) => {
      const refs = ((g.task as { taskIds?: string[] } | null)?.taskIds ?? []);
      return refs.length > 0 && refs.every((ref) => cloneTaskIds.has(ref));
    });
    expect(cloned).toBeDefined();
    const taskRefs = ((cloned!.task as { taskIds?: string[] })?.taskIds ?? []);
    for (const ref of taskRefs) {
      expect(cloneTaskIds.has(ref)).toBe(true);
    }
  });
});