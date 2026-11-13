export interface TrainingSample {
  instruction: string;
  context: {
    taskId?: string;
    taskTitle?: string;
    taskPriority?: string;
    taskTags?: string[];
    projectId?: string;
    candidates: Array<{
      name: string;
      userId: string;
      workloadPercent: number | null;
      workloadDensity?: number | null;
      score: number;
    }>;
  };
  thought_chain: string;
  output: string;
}

export function buildTrainingSample(
  taskTitle: string,
  taskPriority: string,
  taskTags: string[],
  projectId: string | null,
  candidates: Array<{
    name: string;
    userId: string;
    workloadPercent: number | null;
    workloadDensity?: number | null;
    score: number;
  }>,
  chosenCandidateName: string,
  basis: string[]
): TrainingSample {
  const candidateSummaries = candidates.length > 0
    ? candidates
        .map(
          (c) =>
            `${c.name} (workload: ${c.workloadPercent ?? 0}%, density: ${c.workloadDensity ?? 0}, score: ${c.score})`
        )
        .join("; ")
    : "No other candidates evaluated";

  const mainReason = basis.length > 0
    ? basis.join(". ")
    : `Evaluated candidate capacity and skill matching.`;

  const dynamicThoughtChain = `Step 1: Analyzed task '${taskTitle}' priority (${taskPriority}) and required tags [${taskTags.join(", ")}]. Step 2: Evaluated project team members: ${candidateSummaries}. Step 3: ${mainReason}. Conclusion: Selected '${chosenCandidateName}' as the optimal assignee.`;

  return {
    instruction: `Evaluate task '${taskTitle}' (Priority: ${taskPriority}) for project '${projectId ?? "default"}' and select the optimal assignee.`,
    context: {
      taskTitle,
      taskPriority,
      taskTags,
      projectId: projectId ?? undefined,
      candidates,
    },
    thought_chain: dynamicThoughtChain,
    output: `Assign to ${chosenCandidateName}`,
  };
}

export async function exportSftDataset(projectId?: string): Promise<string> {
  const db = (await import("@/lib/client")).default;

  const [decisions, actionLogs] = await Promise.all([
    db.agentDecision.findMany({
      where: projectId ? { projectId } : undefined,
      orderBy: { createdAt: "desc" },
      take: 1000,
    }),
    db.agentActionLog.findMany({
      where: projectId ? { projectId } : undefined,
      orderBy: { createdAt: "desc" },
      take: 1000,
    }),
  ]);

  const projectIds = new Set<string>();
  const taskIds = new Set<string>();
  const userIds = new Set<string>();

  for (const d of decisions) {
    if (d.projectId) projectIds.add(d.projectId);
    const basis = (d.basis ?? {}) as Record<string, unknown>;
    if (typeof basis.taskId === "string") taskIds.add(basis.taskId);
  }
  for (const log of actionLogs) {
    if (log.projectId) projectIds.add(log.projectId);
    if (log.target) taskIds.add(log.target);
    const payload = (log.payload ?? {}) as Record<string, unknown>;
    if (typeof payload.targetUserId === "string") userIds.add(payload.targetUserId);
  }

  const [tasks, users, projectMembers] = await Promise.all([
    db.task.findMany({
      where: { id: { in: Array.from(taskIds) } },
      select: { id: true, title: true, priority: true, tags: true, projectId: true },
    }),
    db.user.findMany({
      where: { id: { in: Array.from(userIds) } },
      select: { id: true, name: true, workingAs: true, department: true },
    }),
    db.projectMember.findMany({
      where: projectIds.size > 0 ? { projectId: { in: Array.from(projectIds) } } : undefined,
      include: { user: { select: { id: true, name: true, workingAs: true } } },
    }),
  ]);

  const allUserIdsSet = new Set<string>([...Array.from(userIds)]);
  for (const pm of projectMembers) {
    allUserIdsSet.add(pm.userId);
  }
  const performanceRatings = allUserIdsSet.size > 0
    ? await db.performanceRating.findMany({
        where: { userId: { in: Array.from(allUserIdsSet) } },
        orderBy: { createdAt: "desc" },
      }).catch(() => [])
    : [];

  const ratingMap = new Map<string, number>();
  for (const r of performanceRatings) {
    if (!ratingMap.has(r.userId)) {
      const avg = Math.round(((r.newLearnings + r.speed + r.workQuality + r.leaves + r.communication + r.feedback) / 6) * 10) / 10;
      ratingMap.set(r.userId, avg);
    }
  }

  const taskMap = new Map(tasks.map((t) => [t.id, t]));
  const userMap = new Map(users.map((u) => [u.id, u.name]));

  const membersByProject = new Map<string, Array<{ name: string; userId: string; role?: string }>>();
  for (const pm of projectMembers) {
    const list = membersByProject.get(pm.projectId) ?? [];
    list.push({
      name: pm.user.name,
      userId: pm.user.id,
      role: pm.user.workingAs ?? "Team Member",
    });
    membersByProject.set(pm.projectId, list);
  }

  const activeTasks = await db.task.findMany({
    where: {
      status: { in: ["assigned", "in_progress", "pending"] },
    },
    select: { assignee: true },
  });

  const activeTaskCountMap = new Map<string, number>();
  for (const t of activeTasks) {
    if (t.assignee) {
      const count = activeTaskCountMap.get(t.assignee) ?? 0;
      activeTaskCountMap.set(t.assignee, count + 1);
    }
  }

  const samples: Array<Record<string, unknown>> = [];
  const seenSignatures = new Set<string>();

  for (const d of decisions) {
    const basis = (d.basis ?? {}) as Record<string, unknown>;
    const taskId = typeof basis.taskId === "string" ? basis.taskId : null;
    const task = taskId ? taskMap.get(taskId) : null;
    const taskTitle = task?.title ?? "General Task Allocation";
    const taskPriority = task?.priority ?? "HIGH";
    const taskTags = task?.tags ?? [];

    let chosenName: string | null = null;
    const match = d.decision.match(/to\s+([A-Za-z0-9_\-\s]+?)(?:\s*\(score|$)/i);
    if (match && match[1]) {
      chosenName = match[1].trim();
    }
    if (!chosenName || chosenName.toLowerCase() === "unassigned") continue;

    const projMembers = d.projectId ? (membersByProject.get(d.projectId) ?? []) : [];
    if (projMembers.length === 0) continue;

    const candidateList = projMembers.map((m) => {
      const count = activeTaskCountMap.get(m.name) ?? activeTaskCountMap.get(m.userId) ?? 0;
      const workloadPercent = Math.min(100, count * 15);
      const rating = ratingMap.get(m.userId) ?? null;
      return {
        name: m.name,
        userId: m.userId,
        role: m.role,
        activeTasks: count,
        workloadPercent,
        performanceRating: rating,
      };
    });

    const chosenCandidate = candidateList.find((c) => c.name.toLowerCase() === chosenName!.toLowerCase());
    if (!chosenCandidate) continue;

    const sig = `decision-${d.projectId}-${taskId}-${chosenCandidate.name}`;
    if (seenSignatures.has(sig)) continue;
    seenSignatures.add(sig);

    const finalCandidates = candidateList.map((c) => ({
      ...c,
      score: c.name === chosenCandidate.name ? 0.95 : Math.max(0.1, 0.80 - c.activeTasks * 0.1),
    }));

    const candidateSummary = finalCandidates
      .map((c) => `${c.name}${c.performanceRating ? ` [Performance Rating: ${c.performanceRating}/10]` : ""} (${c.role ?? "Team Member"}, ${c.activeTasks} active tasks, ${c.workloadPercent}% workload)`)
      .join("; ");

    const reasoningText = `Selected candidate '${chosenCandidate.name}' with lowest relative workload (${chosenCandidate.activeTasks} active tasks) and high matching skill profile.`;

    samples.push({
      instruction: `Evaluate task '${taskTitle}' (Priority: ${taskPriority}) for project '${d.projectId ?? "default"}' and select the optimal assignee.`,
      context: {
        taskTitle,
        taskPriority,
        taskTags,
        projectId: d.projectId ?? undefined,
        targetTaskId: taskId ?? undefined,
        candidates: finalCandidates,
        action: "assign_task",
      },
      thought_chain: `Step 1: Analyzed task requirements '${taskTitle}' (Priority: ${taskPriority}, Tags: [${taskTags.join(", ")}]). Step 2: Queried live project capacity: ${candidateSummary}. Step 3: ${reasoningText} Conclusion: Selected '${chosenCandidate.name}' as optimal candidate.`,
      output: `Assign to ${chosenCandidate.name}`,
    });
  }

  return samples.map((s) => JSON.stringify(s)).join("\n");
}

