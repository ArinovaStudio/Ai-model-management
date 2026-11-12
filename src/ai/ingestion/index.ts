import db from "@/lib/client";
import { logger } from "@/ai/logger";
import { v4 as uuidv4 } from "uuid";
import { callOllama } from "@/ai/nlu/ollama";
import type {
  IngestionResult,
  IngestionRecordCounts,
  IngestionSectionCounts,
  IngestionStepLog,
} from "./types";

const STEP_TIMEOUT_MS = 60_000;

function now(): string {
  return new Date().toISOString();
}

function elapsed(startedAt: number): number {
  return Date.now() - startedAt;
}

function pushLog(logs: IngestionStepLog[], step: string, startedAt: number, detail: string): void {
  logs.push({ step, startedAt: now(), endedAt: now(), elapsedMs: elapsed(startedAt), detail });
}

function emptyCounts(): IngestionRecordCounts {
  return {
    project: 0,
    projectInfo: 0,
    projectMembers: 0,
    milestones: 0,
    assets: 0,
    latestUpdates: 0,
    featureRequests: 0,
    riskBlockages: 0,
    planDocs: 0,
    documentAccess: 0,
    docs: 0,
    roadmaps: 0,
    roadmapAssignments: 0,
    roadmapPermissions: 0,
    roadmapComments: 0,
    designSystems: 0,
    projectTechnologies: 0,
    groupTasks: 0,
    tasks: 0,
    taskDependencies: 0,
    workDone: 0,
    tickets: 0,
    usefulTips: 0,
    meetingRequests: 0,
    memos: 0,
  };
}

function sectionsFrom(records: IngestionRecordCounts): IngestionSectionCounts {
  return {
    wbsSchedule: {
      groupingTasks: records.groupTasks,
      tasks: records.tasks,
      taskDependencies: records.taskDependencies,
    },
    timeline: {
      tasks: records.tasks,
      latestUpdates: records.latestUpdates,
    },
    resources: {
      projectMembers: records.projectMembers,
      milestones: records.milestones,
      assets: records.assets,
      meetingRequests: records.meetingRequests,
    },
    gaps: {
      featureRequests: records.featureRequests,
      riskBlockages: records.riskBlockages,
    },
    productStrategy: {
      roadmaps: records.roadmaps,
      roadmapComments: records.roadmapComments,
    },
    technicalArchitecture: {
      designSystems: records.designSystems,
      projectTechnologies: records.projectTechnologies,
    },
    documentation: {
      planDocs: records.planDocs,
      docs: records.docs,
    },
  };
}

/**
 * Remaps a comma-separated dependency string. Task IDs known to the clone are
 * mapped to their new IDs; any other token (cross-project or foreign) is kept
 * as-is.
 */
export function remapDependencyString(oldDeps: string | null, taskMap: Map<string, string>): string {
  if (!oldDeps || oldDeps.trim() === "") return "";
  return oldDeps
    .split(",")
    .map((token) => token.trim())
    .filter(Boolean)
    .map((token) => taskMap.get(token) ?? token)
    .join(",");
}

/**
 * Deep-rewrites any string value inside a JSON structure that equals a known
 * source task ID, replacing it with the cloned task ID. Used for GroupingTask
 * JSON payloads that embed task references.
 */
export function remapTaskReferencesInJson(
  value: unknown,
  taskMap: Map<string, string>
): unknown {
  if (value === null || value === undefined) return value;
  if (typeof value === "string") return taskMap.get(value) ?? value;
  if (Array.isArray(value)) return value.map((item) => remapTaskReferencesInJson(item, taskMap));
  if (typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      out[key] = remapTaskReferencesInJson(item, taskMap);
    }
    return out;
  }
  return value;
}

/**
 * Detects cycles in the task dependency graph. Returns the list of dependency
 * loops involved in a cycle; does not throw.
 */
export function detectDependencyCycles(
  dependencies: { taskId: string; dependsOnTaskId: string }[]
): string[][] {
  const outgoing = new Map<string, string[]>();
  const all = new Set<string>();
  for (const dep of dependencies) {
    if (!outgoing.has(dep.taskId)) outgoing.set(dep.taskId, []);
    outgoing.get(dep.taskId)!.push(dep.dependsOnTaskId);
    all.add(dep.taskId);
    all.add(dep.dependsOnTaskId);
  }
  const state = new Map<string, "visiting" | "visited">();
  const cycles: string[][] = [];
  const stack: string[] = [];

  const dfs = (node: string): void => {
    const st = state.get(node);
    if (st === "visited") return;
    if (st === "visiting") {
      const start = stack.indexOf(node);
      if (start !== -1) cycles.push(stack.slice(start));
      return;
    }
    state.set(node, "visiting");
    stack.push(node);
    for (const next of outgoing.get(node) ?? []) dfs(next);
    stack.pop();
    state.set(node, "visited");
  };

  for (const node of all) dfs(node);
  return cycles;
}

/**
 * Clones an entire project graph (all project-bound models) into a new project.
 * Runs inside a single Prisma interactive transaction; if any step throws the
 * whole clone rolls back and no partial data is persisted.
 */
export async function ingestProject(sourceProjectId: string, newName: string): Promise<IngestionResult> {
  const startedAt = Date.now();
  const logs: IngestionStepLog[] = [];
  const warnings: string[] = [];

  const sourceProject = await db.project.findUnique({
    where: { id: sourceProjectId },
  });
  if (!sourceProject) {
    throw new Error(`Source project "${sourceProjectId}" does not exist.`);
  }
  logger.info(`[INGEST] Beginning clone of project "${sourceProject.name}" (${sourceProjectId}) as "${newName}".`);

  const result = await db.$transaction(
    async (tx) => {
      const records = emptyCounts();

      const step1 = Date.now();
      const [
        projectInfoRows,
        memberRows,
        milestoneRows,
        assetRows,
        latestUpdateRows,
        featureRequestRows,
        riskBlockageRows,
        planDocRows,
        docRows,
        roadmaps,
        designSystemRows,
        projectTechnologyRows,
        taskRows,
        dependencyRows,
        workDoneRows,
        ticketRows,
        usefulTipRows,
        meetingRequestRows,
      ] = await Promise.all([
        tx.projectInfo.findMany({ where: { projectId: sourceProjectId } }),
        tx.projectMember.findMany({ where: { projectId: sourceProjectId } }),
        tx.milestone.findMany({ where: { projectId: sourceProjectId } }),
        tx.asset.findMany({ where: { projectId: sourceProjectId } }),
        tx.latestUpdate.findMany({ where: { projectId: sourceProjectId } }),
        tx.featureRequest.findMany({ where: { projectId: sourceProjectId } }),
        tx.riskBlockage.findMany({ where: { projectId: sourceProjectId } }),
        tx.planDocs.findMany({ where: { projectId: sourceProjectId } }),
        tx.docs.findMany({ where: { projectId: sourceProjectId } }),
        tx.roadmap.findMany({
          where: { projectId: sourceProjectId },
          select: {
            id: true,
            title: true,
            description: true,
            isPrivate: true,
            createdById: true,
            lastEditBy: true,
            lastEditAt: true,
            tldrawData: true,
          },
        }),
        tx.designSystem.findMany({ where: { projectId: sourceProjectId } }),
        tx.projectTechnology.findMany({ where: { projectId: sourceProjectId } }),
        tx.task.findMany({ where: { projectId: sourceProjectId } }),
        tx.taskDependency.findMany({ where: { projectId: sourceProjectId } }),
        tx.workDone.findMany({ where: { projectId: sourceProjectId } }),
        tx.ticket.findMany({ where: { projectId: sourceProjectId } }),
        tx.usefulTip.findMany({ where: { projectId: sourceProjectId } }),
        tx.meetingRequest.findMany({ where: { projectId: sourceProjectId } }),
      ]);
      pushLog(logs, "read-source", step1, "Read source project graph (project-bound rows).");

      const step2 = Date.now();
      const project = await tx.project.create({
        data: {
          name: newName,
          summary: sourceProject.summary,
          priority: sourceProject.priority,
          basicDetails: sourceProject.basicDetails,
          status: 0,
          progress: 0,
          repository: sourceProject.repository,
          currentPhase: sourceProject.currentPhase ?? "Design",
        },
        select: { id: true },
      });
      records.project = 1;
      pushLog(logs, "create-project", step2, `Created target project "${project.id}".`);

      const step3 = Date.now();
      for (const row of projectInfoRows) {
        await tx.projectInfo.create({
          data: {
            budget: row.budget,
            clientName: row.clientName,
            projectType: row.projectType,
            startDate: row.startDate,
            deadline: row.deadline ?? new Date(),
            supervisor: row.supervisorAdmin ? { connect: { id: row.supervisorAdmin } } : undefined,
            paidAmount: row.paidAmount ?? 0,
            Project: { connect: { id: project.id } },
          },
        });
        records.projectInfo += 1;
      }
      pushLog(logs, "clone-project-info", step3, `Cloned ${records.projectInfo} ProjectInfo row(s).`);

      const step4 = Date.now();
      for (const row of memberRows) {
        await tx.projectMember.create({
          data: { projectId: project.id, userId: row.userId, role: row.role, isLeader: row.isLeader },
        });
        records.projectMembers += 1;
      }
      pushLog(logs, "clone-members", step4, `Cloned ${records.projectMembers} team member(s).`);

      const step5 = Date.now();
      for (const row of milestoneRows) {
        await tx.milestone.create({
          data: {
            projectId: project.id,
            title: row.title,
            description: row.description,
            dueDate: row.dueDate,
            status: row.status,
            createdBy: row.createdBy,
          },
        });
        records.milestones += 1;
      }
      for (const row of assetRows) {
        await tx.asset.create({
          data: {
            projectId: project.id,
            type: row.type,
            url: row.url,
            publicId: row.publicId,
            title: row.title,
            uploadedBy: row.uploadedBy,
            userImage: row.userImage,
            liveUrl: row.liveUrl,
          },
        });
        records.assets += 1;
      }
      for (const row of latestUpdateRows) {
        await tx.latestUpdate.create({
          data: { projectId: project.id, title: row.title, date: row.date, createdBy: row.createdBy },
        });
        records.latestUpdates += 1;
      }
      pushLog(logs, "clone-schedule-artifacts", step5, `Cloned ${records.milestones} milestone(s), ${records.assets} asset(s), ${records.latestUpdates} update(s).`);

      const step6 = Date.now();
      for (const row of featureRequestRows) {
        await tx.featureRequest.create({
          data: {
            clientId: row.clientId,
            projectId: project.id,
            title: row.title,
            description: row.description,
            status: row.status,
          },
        });
        records.featureRequests += 1;
      }
      for (const row of riskBlockageRows) {
        await tx.riskBlockage.create({
          data: {
            projectId: project.id,
            clientId: row.clientId,
            riskTitle: row.riskTitle,
            category: row.category,
            description: row.description,
            metadata: row.metadata,
            reportType: row.reportType,
            severity: row.severity,
            sourceDocId: row.sourceDocId,
            sourceFeatureId: row.sourceFeatureId,
            status: row.status,
            title: row.title,
            type: row.type,
            confidenceScore: row.confidenceScore,
            planningWarnings: row.planningWarnings,
          },
        });
        records.riskBlockages += 1;
      }
      pushLog(logs, "clone-gaps", step6, `Cloned ${records.featureRequests} feature request(s) and ${records.riskBlockages} risk(s).`);

      const step7 = Date.now();
      const docIdMap = new Map<string, string>();
      for (const row of planDocRows) {
        const created = await tx.planDocs.create({
          data: {
            name: row.name,
            content: row.content,
            createdBy: row.createdBy,
            projectId: project.id,
          },
          select: { id: true },
        });
        docIdMap.set(row.id, created.id);
        records.planDocs += 1;
      }
      const docAccessRows = await tx.documentAccess.findMany({
        where: { docId: { in: planDocRows.map((row) => row.id) } },
      });
      for (const row of docAccessRows) {
        const newDocId = docIdMap.get(row.docId);
        if (!newDocId) {
          warnings.push(`DocumentAccess row "${row.id}" referenced an unknown plan doc; skipped.`);
          continue;
        }
        await tx.documentAccess.create({ data: { docId: newDocId, userId: row.userId, role: row.role } });
        records.documentAccess += 1;
      }
      for (const row of docRows) {
        await tx.docs.create({
          data: { title: row.title, fileUrl: row.fileUrl, userId: row.userId, projectId: project.id },
        });
        records.docs += 1;
      }
      pushLog(logs, "clone-docs", step7, `Cloned ${records.planDocs} plan doc(s), ${records.documentAccess} access grant(s), ${records.docs} doc(s).`);

      const step8 = Date.now();
      const roadmapIdMap = new Map<string, string>();
      for (const row of roadmaps) {
        const newId = uuidv4();
        const nowTimestamp = new Date();
        await tx.$queryRaw`
          INSERT INTO "Roadmap"
            ("id", "title", "description", "isPrivate", "createdById", "projectId", "lastEditBy", "lastEditAt", "tldrawData", "createdAt", "updatedAt")
          VALUES
            (${newId}, ${row.title}, ${row.description}, ${row.isPrivate}, ${row.createdById}, ${project.id}, ${row.lastEditBy}, ${row.lastEditAt}, ${row.tldrawData == null ? null : JSON.stringify(row.tldrawData)}::jsonb, ${nowTimestamp}, ${nowTimestamp})
        `;
        roadmapIdMap.set(row.id, newId);
        records.roadmaps += 1;
      }
      const roadmapIds = roadmaps.map((row) => row.id);
      const [roadmapAssignmentRows, roadmapPermissionRows, roadmapCommentRows] = await Promise.all([
        tx.roadmapAssignment.findMany({ where: { roadmapId: { in: roadmapIds } } }),
        tx.roadmapPermission.findMany({ where: { roadmapId: { in: roadmapIds } } }),
        tx.roadmapComment.findMany({ where: { roadmapId: { in: roadmapIds } } }),
      ]);
      for (const row of roadmapAssignmentRows) {
        const newRoadmapId = roadmapIdMap.get(row.roadmapId);
        if (!newRoadmapId) {
          warnings.push(`RoadmapAssignment row "${row.id}" referenced an unknown roadmap; skipped.`);
          continue;
        }
        await tx.roadmapAssignment.create({
          data: { roadmapId: newRoadmapId, userId: row.userId, assignedBy: row.assignedBy, assignedAt: row.assignedAt },
        });
        records.roadmapAssignments += 1;
      }
      for (const row of roadmapPermissionRows) {
        const newRoadmapId = roadmapIdMap.get(row.roadmapId);
        if (!newRoadmapId) {
          warnings.push(`RoadmapPermission row "${row.id}" referenced an unknown roadmap; skipped.`);
          continue;
        }
        await tx.roadmapPermission.create({
          data: {
            roadmapId: newRoadmapId,
            userId: row.userId,
            permission: row.permission,
            grantedBy: row.grantedBy,
            grantedAt: row.grantedAt,
          },
        });
        records.roadmapPermissions += 1;
      }
      for (const row of roadmapCommentRows) {
        const newRoadmapId = roadmapIdMap.get(row.roadmapId);
        if (!newRoadmapId) {
          warnings.push(`RoadmapComment row "${row.id}" referenced an unknown roadmap; skipped.`);
          continue;
        }
        await tx.roadmapComment.create({
          data: {
            roadmapId: newRoadmapId,
            userId: row.userId,
            comment: row.comment,
            nodeId: row.nodeId,
            isMarked: row.isMarked,
          },
        });
        records.roadmapComments += 1;
      }
      pushLog(logs, "clone-product-strategy", step8, `Cloned ${records.roadmaps} roadmap(s), ${records.roadmapAssignments} assignment(s), ${records.roadmapPermissions} permission(s), ${records.roadmapComments} comment(s).`);

      const step9 = Date.now();
      for (const row of designSystemRows) {
        await tx.designSystem.create({
          data: {
            projectId: project.id,
            brandName: row.brandName,
            colors: row.colors,
            fonts: row.fonts,
            designType: row.designType,
            layoutStyle: row.layoutStyle,
            contentTone: row.contentTone,
            visualGuidelines: row.visualGuidelines,
            theme: row.theme,
            brandFeel: row.brandFeel,
            keyPages: row.keyPages,
            uniqueness: row.uniqueness,
          },
        });
        records.designSystems += 1;
      }
      for (const row of projectTechnologyRows) {
        await tx.projectTechnology.create({
          data: { projectId: project.id, tech: row.tech },
        });
        records.projectTechnologies += 1;
      }
      pushLog(logs, "clone-architecture", step9, `Cloned ${records.designSystems} design system(s) and ${records.projectTechnologies} technology stack(s).`);

      const step10 = Date.now();
      const taskIdMap = new Map<string, string>();
      for (const row of taskRows) {
        const created = await tx.task.create({
          data: {
            title: row.title,
            description: row.description,
            assignee: row.assignee,
            assigneeAvatar: row.assigneeAvatar,
            priority: row.priority,
            dueDate: row.dueDate,
            employeeId: row.employeeId,
            tags: row.tags,
            attachments: row.attachments,
            status: row.status,
            projectId: project.id,
            userId: row.userId,
          },
          select: { id: true },
        });
        taskIdMap.set(row.id, created.id);
        records.tasks += 1;
      }
      pushLog(logs, "clone-tasks", step10, `Cloned ${records.tasks} task(s).`);

      const step11 = Date.now();
      for (const dep of dependencyRows) {
        const newTaskId = taskIdMap.get(dep.taskId);
        const newDependsOn = taskIdMap.get(dep.dependsOnTaskId) ?? dep.dependsOnTaskId;
        if (!newTaskId) {
          warnings.push(`TaskDependency row "${dep.id}" referenced task "${dep.taskId}" not owned by the source project; skipped.`);
          continue;
        }
        await tx.taskDependency.create({
          data: {
            projectId: project.id,
            taskId: newTaskId,
            dependsOnTaskId: newDependsOn,
            relationType: dep.relationType,
            source: dep.source,
            createdById: dep.createdById,
          },
        });
        records.taskDependencies += 1;
      }
      pushLog(logs, "remap-dependencies", step11, `Remapped and cloned ${records.taskDependencies} task dependency(ies).`);

      const step12 = Date.now();
      if (records.tasks > 0) {
        const sourceTaskIds = new Set(taskRows.map((row) => row.id));
        const allGroupingTasks = await tx.groupingTask.findMany();
        const ownedGroupingTasks = allGroupingTasks.filter((row) => {
          const combined = JSON.stringify(row.task) + JSON.stringify(row.users ?? null);
          return Array.from(sourceTaskIds).some((id) => combined.includes(id));
        });
        for (const row of ownedGroupingTasks) {
          await tx.groupingTask.create({
            data: {
              name: row.name,
              task: remapTaskReferencesInJson(row.task, taskIdMap) as object | null,
              users: remapTaskReferencesInJson(row.users, taskIdMap) as object | null,
            },
          });
          records.groupTasks += 1;
        }
      }
      for (const row of workDoneRows) {
        const newTaskId = taskIdMap.get(row.taskId);
        await tx.workDone.create({
          data: {
            projectId: project.id,
            title: row.title,
            description: row.description,
            priority: row.priority,
            dueDate: row.dueDate,
            tags: row.tags,
            userId: row.userId,
            taskId: newTaskId ?? row.taskId,
          },
        });
        records.workDone += 1;
      }
      for (const row of ticketRows) {
        await tx.ticket.create({
          data: { projectId: project.id, reportedBy: row.reportedBy, reason: row.reason, blockedTeammates: row.blockedTeammates },
        });
        records.tickets += 1;
      }
      for (const row of usefulTipRows) {
        await tx.usefulTip.create({
          data: { projectId: project.id, userId: row.userId, user: row.user, tip: row.tip },
        });
        records.usefulTips += 1;
      }
      pushLog(logs, "clone-aux-resources", step12, `Cloned ${records.workDone} work item(s), ${records.tickets} ticket(s), ${records.usefulTips} tip(s).`);

      const step13 = Date.now();
      const meetingIdMap = new Map<string, string>();
      for (const row of meetingRequestRows) {
        const created = await tx.meetingRequest.create({
          data: {
            clientId: row.clientId,
            reason: row.reason,
            meetDate: row.meetDate,
            meetTime: row.meetTime,
            status: row.status,
            duration: row.duration,
            projectId: project.id,
            createdBy: row.createdBy,
            meetingLink: row.meetingLink,
          },
          select: { id: true },
        });
        meetingIdMap.set(row.id, created.id);
        records.meetingRequests += 1;
      }
      const memoRows = await tx.memo.findMany({
        where: { meetingId: { in: meetingRequestRows.map((row) => row.id) } },
      });
      for (const row of memoRows) {
        const newMeetingId = meetingIdMap.get(row.meetingId);
        if (!newMeetingId) {
          warnings.push(`Memo row "${row.id}" referenced an unknown meeting request; skipped.`);
          continue;
        }
        await tx.memo.create({
          data: { meetingId: newMeetingId, message: row.message, clientId: row.clientId, totaltime: row.totaltime },
        });
        records.memos += 1;
      }
      pushLog(logs, "clone-meetings", step13, `Cloned ${records.meetingRequests} meeting request(s) and ${records.memos} memo(s).`);

      const cycles = detectDependencyCycles(dependencyRows);
      for (const cycle of cycles) {
        warnings.push(`Detected dependency cycle in source: ${cycle.join(" -> ")}. Cloned as-is; schedule evaluation may flag it.`);
      }

      return {
        projectId: project.id,
        records,
        taskIdMap: Object.fromEntries(taskIdMap),
        warnings,
        logs,
      };
    },
    { maxWait: STEP_TIMEOUT_MS, timeout: STEP_TIMEOUT_MS }
  );

  // Outside the transaction, check if we have zero task dependencies
  if (result.records.taskDependencies === 0 && result.records.tasks > 0) {
    logger.info(`[INGEST] Template has zero dependencies. Running AI task dependency generator...`);
    try {
      // Fetch all tasks for this project
      const tasks = await db.task.findMany({
        where: { projectId: result.projectId },
        select: { id: true, title: true, description: true }
      });

      const prompt = `You are a technical project manager. Below is a list of tasks for the project "${newName}".
Analyze the tasks and identify which tasks should logically depend on other tasks (e.g. database setup must happen before route setup, authentication must happen before profile management, design must happen before implementation, etc.).

Task List:
${JSON.stringify(tasks, null, 2)}

Identify logical dependency relationships. Return ONLY a valid JSON array of objects with schema:
[
  {
    "taskId": "string (the child task ID)",
    "dependsOnTaskId": "string (the parent task ID that must complete first)"
  }
]
Do not include any explanation or markdown formatting in your response. Just output raw JSON.`;

      const response = await callOllama({
        system: "You are a technical project manager.",
        prompt,
        formatJson: true,
        maxTokens: 1500
      });

      const rawJson = response.text.replace(/```json|```/g, "").trim();
      logger.info("[INGEST] AI returned raw JSON: " + rawJson);
      const parsedDeps = JSON.parse(rawJson);
      
      let depsArray = parsedDeps;
      if (parsedDeps && typeof parsedDeps === "object" && !Array.isArray(parsedDeps)) {
        depsArray = parsedDeps.tasks || parsedDeps.dependencies || Object.values(parsedDeps)[0];
      }

      if (Array.isArray(depsArray)) {
        let count = 0;
        for (const dep of depsArray) {
          if (dep.taskId && dep.dependsOnTaskId && dep.taskId !== dep.dependsOnTaskId) {
            // Verify tasks exist in project
            const childExists = tasks.some(t => t.id === dep.taskId);
            const parentExists = tasks.some(t => t.id === dep.dependsOnTaskId);
            if (childExists && parentExists) {
              await db.taskDependency.create({
                data: {
                  projectId: result.projectId,
                  taskId: dep.taskId,
                  dependsOnTaskId: dep.dependsOnTaskId,
                  relationType: "hard",
                  source: "ai"
                }
              });
              count++;
            }
          }
        }
        logger.info(`[INGEST] AI successfully generated and saved ${count} task dependencies.`);
        result.records.taskDependencies = count;
      }
    } catch (err) {
      logger.error(`[INGEST] AI task dependency generation failed: ${err instanceof Error ? err.message : String(err)}`);
      result.warnings.push(`AI task dependency generation failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  const durationMs = elapsed(startedAt);
  const summary =
    `[INGEST] Cloned project "${sourceProject.name}" -> "${newName}" (${result.projectId}) in ${durationMs}ms: ` +
    `${result.records.tasks} tasks, ${result.records.taskDependencies} dependencies, ${result.records.planDocs} plan docs, ` +
    `${result.records.roadmaps} roadmaps, ${result.records.riskBlockages} risks, ${result.records.featureRequests} features.`;
  logger.info(summary);

  return {
    success: true,
    sourceProjectId,
    sourceProjectName: sourceProject.name,
    targetProjectId: result.projectId,
    targetProjectName: newName,
    records: result.records,
    sections: sectionsFrom(result.records),
    taskIdMap: result.taskIdMap,
    warnings: result.warnings,
    logs: result.logs,
    durationMs,
  };
}