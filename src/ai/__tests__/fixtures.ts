import { randomUUID } from "crypto";
import db from "@/lib/client";
import { uid, createUser } from "./setup";

export function daysFromNow(days: number, base: Date = new Date()): Date {
  return new Date(base.getTime() + days * 24 * 60 * 60 * 1000);
}

export function isoDaysFromNow(days: number): string {
  return daysFromNow(days).toISOString();
}

export interface SeedProject {
  projectId: string;
  name: string;
}

export async function seedProject(
  prefix: string,
  name = "Test Project",
  label = "proj"
): Promise<SeedProject> {
  const id = uid(prefix, label);
  const project = await db.project.create({
    data: {
      id,
      name,
      summary: "integration test project",
      status: 0,
      progress: 0,
      currentPhase: "Design",
    },
    select: { id: true, name: true },
  });
  return { projectId: project.id, name: project.name };
}

export async function seedProjectInfo(
  prefix: string,
  projectId: string,
  deadlineDays: number
): Promise<void> {
  await db.projectInfo.create({
    data: {
      projectId,
      budget: 1000,
      clientName: "Test Client",
      projectType: "internal",
      startDate: daysFromNow(-30),
      deadline: daysFromNow(deadlineDays),
    },
  });
}

export interface SeedTask {
  taskId: string;
  title: string;
}

export async function seedTask(
  prefix: string,
  projectId: string,
  label: string,
  overrides: {
    title?: string;
    status?: string;
    priority?: string;
    assignee?: string;
    userId?: string | null;
    dueDate?: Date;
    tags?: string[];
  } = {}
): Promise<SeedTask> {
  const id = uid(prefix, `t-${label}`);
  const task = await db.task.create({
    data: {
      id,
      title: overrides.title ?? `Task ${label}`,
      description: "",
      assignee: overrides.assignee ?? "unassigned",
      priority: overrides.priority ?? "medium",
      dueDate: overrides.dueDate ?? daysFromNow(7),
      status: overrides.status ?? "pending",
      userId: overrides.userId ?? null,
      tags: overrides.tags ?? [],
      projectId,
    },
    select: { id: true, title: true },
  });
  return { taskId: task.id, title: task.title };
}

export async function seedMember(
  prefix: string,
  projectId: string,
  userId: string,
  overrides: { role?: string | null; isLeader?: boolean } = {}
): Promise<void> {
  await db.projectMember.create({
    data: {
      id: uid(prefix, `pm-${userId.slice(-6)}`),
      projectId,
      userId,
      role: overrides.role ?? "member",
      isLeader: overrides.isLeader ?? false,
    },
  });
}

export async function seedDependency(
  prefix: string,
  projectId: string,
  taskId: string,
  dependsOnTaskId: string,
  relationType = "hard"
): Promise<void> {
  await db.taskDependency.create({
    data: {
      id: uid(prefix, `dep-${taskId.slice(-4)}-${dependsOnTaskId.slice(-4)}`),
      projectId,
      taskId,
      dependsOnTaskId,
      relationType,
      source: "human",
    },
  });
}

export async function seedWorkHours(
  userId: string,
  hours: number,
  at: Date = new Date()
): Promise<void> {
  await db.workHours.create({
    data: { userId, date: at, hours, clockIn: "09:00", clockOut: "17:00" },
  });
}

export async function seedRiskBlockage(
  prefix: string,
  projectId: string,
  overrides: {
    title?: string;
    severity?: string;
    status?: string;
    description?: string;
  } = {}
): Promise<string> {
  const blocker = await db.riskBlockage.create({
    data: {
      projectId,
      clientId: "",
      riskTitle: overrides.title ?? "Test blocker",
      title: overrides.title ?? "Test blocker",
      severity: overrides.severity ?? "medium",
      status: overrides.status ?? "open",
      description: overrides.description ?? "",
      category: "general",
      reportType: "risk",
      type: "risk",
    },
    select: { id: true },
  });
  return blocker.id;
}

/**
 * Inserts a roadmap using raw SQL on the LIVE columns only (the live table
 * lacks `canvasData`/`activeMode` present in schema.prisma).
 */
export async function seedRoadmapRaw(
  projectId: string,
  overrides: { title?: string; isPrivate?: boolean } = {}
): Promise<string> {
  const id = randomUUID();
  const now = new Date();
  await db.$queryRaw`
    INSERT INTO "Roadmap"
      ("id", "title", "description", "isPrivate", "createdById", "projectId", "lastEditBy", "lastEditAt", "tldrawData", "createdAt", "updatedAt")
    VALUES
      (${id}, ${overrides.title ?? "Test Roadmap"}, ${"test description"}, ${overrides.isPrivate ?? false}, ${""}, ${projectId}, ${null}, ${null}, ${null}::jsonb, ${now}, ${now})
  `;
  return id;
}

export async function seedMeetingRequest(
  prefix: string,
  projectId: string,
  clientUserId: string,
  label: string
): Promise<string> {
  const meeting = await db.meetingRequest.create({
    data: {
      id: uid(prefix, `m-${label}`),
      clientId: clientUserId,
      reason: "kickoff",
      meetDate: daysFromNow(3),
      meetTime: "10:00",
      status: "pending",
      duration: 30,
      projectId,
      createdBy: clientUserId,
    },
    select: { id: true },
  });
  return meeting.id;
}

export async function seedMemo(
  prefix: string,
  meetingId: string,
  clientUserId: string
): Promise<void> {
  await db.memo.create({
    data: {
      id: uid(prefix, "memo"),
      meetingId,
      message: "noted",
      clientId: clientUserId,
      totaltime: "30m",
    },
  });
}

export async function seedPlanDoc(
  prefix: string,
  projectId: string,
  label: string
): Promise<string> {
  const doc = await db.planDocs.create({
    data: {
      id: uid(prefix, `pd-${label}`),
      name: `Plan doc ${label}`,
      content: "body",
      createdBy: "",
      projectId,
    },
    select: { id: true },
  });
  return doc.id;
}

export async function seedDocumentAccess(
  prefix: string,
  docId: string,
  userId: string
): Promise<void> {
  await db.documentAccess.create({
    data: { id: uid(prefix, "da"), docId, userId, role: "editor" },
  });
}

export async function seedGroupingTask(
  prefix: string,
  label: string,
  taskRefs: string[]
): Promise<string> {
  const grouping = await db.groupingTask.create({
    data: {
      id: uid(prefix, `gt-${label}`),
      name: `${prefix}-gt-${label}`,
      task: { taskIds: taskRefs },
      users: { userIds: [] },
    },
    select: { id: true },
  });
  return grouping.id;
}

export async function seedDesignSystem(
  prefix: string,
  projectId: string
): Promise<void> {
  await db.designSystem.create({
    data: {
      id: uid(prefix, "ds"),
      projectId,
      brandName: "TestBrand",
      colors: ["#000"],
      fonts: { primary: "Arial" },
      designType: ["minimal"],
      layoutStyle: { layout: "grid" },
      contentTone: ["professional"],
      visualGuidelines: { guidelines: "" },
      theme: ["light"],
      brandFeel: "clean",
      keyPages: ["home"],
      uniqueness: { points: [] },
    },
  });
}

export async function seedProjectTechnology(
  prefix: string,
  projectId: string
): Promise<void> {
  await db.projectTechnology.create({
    data: { id: uid(prefix, "tech"), projectId, tech: [{ name: "React" }] },
  });
}

export async function seedWorkDone(
  prefix: string,
  projectId: string,
  taskId: string,
  userId: string
): Promise<void> {
  await db.workDone.create({
    data: {
      id: uid(prefix, "wd"),
      projectId,
      title: "Work item",
      description: "done",
      priority: "high",
      dueDate: daysFromNow(5),
      tags: [],
      userId,
      taskId,
    },
  });
}

export async function seedMilestone(
  prefix: string,
  projectId: string,
  createdBy: string
): Promise<void> {
  await db.milestone.create({
    data: {
      id: uid(prefix, "ms"),
      projectId,
      title: "Milestone 1",
      dueDate: daysFromNow(10),
      createdBy,
    },
  });
}

export async function seedLatestUpdate(
  prefix: string,
  projectId: string,
  createdBy: string
): Promise<void> {
  await db.latestUpdate.create({
    data: {
      id: uid(prefix, "lu"),
      title: "Weekly update",
      date: new Date().toISOString().slice(0, 10),
      projectId,
      createdBy,
    },
  });
}

export async function seedFeatureRequest(
  prefix: string,
  projectId: string,
  clientUserId: string
): Promise<void> {
  await db.featureRequest.create({
    data: {
      id: uid(prefix, "fr"),
      clientId: clientUserId,
      projectId,
      title: "Feature",
      description: "desc",
      status: "pending",
    },
  });
}

export async function seedTicket(
  prefix: string,
  projectId: string
): Promise<void> {
  await db.ticket.create({
    data: {
      id: uid(prefix, "tk"),
      projectId,
      reportedBy: "anonymous",
      reason: "issue",
      blockedTeammates: [],
    },
  });
}

export async function seedUsefulTip(
  prefix: string,
  projectId: string,
  userId: string
): Promise<void> {
  await db.usefulTip.create({
    data: {
      id: uid(prefix, "tip"),
      projectId,
      userId,
      user: "Member",
      tip: "tip",
    },
  });
}

export async function seedAsset(prefix: string, projectId: string): Promise<void> {
  await db.asset.create({
    data: {
      id: uid(prefix, "as"),
      projectId,
      type: "image",
      url: "https://example.com/a.png",
      title: "asset",
    },
  });
}

export async function seedDocs(prefix: string, projectId: string): Promise<void> {
  await db.docs.create({
    data: {
      id: uid(prefix, "doc"),
      title: "Doc",
      fileUrl: "https://example.com/d.pdf",
      projectId,
    },
  });
}