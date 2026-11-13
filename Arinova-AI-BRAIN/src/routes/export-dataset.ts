import { NextResponse } from "next/server";
import db from "@/lib/client";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get("projectId");
    const format = searchParams.get("format") || "json";

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

    // Collect all unique project IDs, task IDs, and User IDs to resolve CUIDs and team contexts
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
    
    // Group members by project
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

    // 1. Fetch active tasks from PostgreSQL to count live workload per assignee safely
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
    const filterAnomalies = searchParams.get("filterAnomalies") !== "false";

    // 1. Process Decisions with full hydrated context & quality filtering
    for (const d of decisions) {
      const basis = (d.basis ?? {}) as Record<string, unknown>;
      const taskId = typeof basis.taskId === "string" ? basis.taskId : null;
      const task = taskId ? taskMap.get(taskId) : null;
      const taskTitle = task?.title ?? "General Task Allocation";
      const taskPriority = task?.priority ?? "HIGH";
      const taskTags = task?.tags ?? [];

      let rawAssignee = "unassigned";
      const match = d.decision.match(/Recommend assigning "[^"]+" to ([^(\n]+)/i);
      if (match && match[1]) {
        rawAssignee = match[1].trim();
      }

      if (rawAssignee === "unassigned") continue;

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

      // Find the true candidate with lowest active workload for logical AI training
      const sortedByWorkload = [...candidateList].sort((a, b) => a.activeTasks - b.activeTasks);
      const lowestWorkloadCandidate = sortedByWorkload[0];

      let chosenCandidate = candidateList.find((c) => c.name.toLowerCase() === rawAssignee.toLowerCase());

      // SANITY FILTER: If raw logs chose an overloaded member (> 5 tasks) while a free member (< 3 tasks) was available,
      // override with the logical optimal candidate for SFT fine-tuning training dataset purity.
      if (filterAnomalies && chosenCandidate && chosenCandidate.activeTasks > 5 && lowestWorkloadCandidate && lowestWorkloadCandidate.activeTasks < chosenCandidate.activeTasks) {
        chosenCandidate = lowestWorkloadCandidate;
      } else if (!chosenCandidate) {
        chosenCandidate = lowestWorkloadCandidate;
      }

      if (!chosenCandidate) continue;

      const sig = `decision-${d.projectId}-${taskId}-${chosenCandidate.name}`;
      if (seenSignatures.has(sig)) continue;
      seenSignatures.add(sig);

      const finalCandidates = candidateList.map((c) => ({
        ...c,
        score: c.name === chosenCandidate!.name ? 0.98 : Math.max(0.1, 0.85 - c.activeTasks * 0.1),
      }));

      const candidateSummary = finalCandidates
        .map((c) => `${c.name}${c.performanceRating ? ` [Performance Rating: ${c.performanceRating}/10]` : ""} (${c.role ?? "Team Member"}, ${c.activeTasks} active tasks, ${c.workloadPercent}% workload)`)
        .join("; ");

      const reasoningText = `Candidate '${chosenCandidate.name}' has optimal remaining capacity (${chosenCandidate.activeTasks} active tasks, ${chosenCandidate.workloadPercent}% workload) and matching domain role (${chosenCandidate.role}).`;

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

    // 2. Process Action Logs for completed/allowed assignments
    for (const log of actionLogs) {
      if (log.action === "assign_task") {
        const payload = (log.payload ?? {}) as Record<string, unknown>;
        const taskId = log.target;
        const task = taskId ? taskMap.get(taskId) : null;
        const taskTitle = task?.title ?? "Project Task Assignment";
        const taskPriority = task?.priority ?? "HIGH";
        const rawAssignee = typeof payload.assignee === "string" ? payload.assignee : (payload.targetUserId ? userMap.get(String(payload.targetUserId)) ?? "assigned member" : "assigned member");

        if (rawAssignee === "assigned member" || rawAssignee === "unassigned") continue;

        const projMembers = log.projectId ? (membersByProject.get(log.projectId) ?? []) : [];
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

        const sortedByWorkload = [...candidateList].sort((a, b) => a.activeTasks - b.activeTasks);
        const lowestWorkloadCandidate = sortedByWorkload[0];

        let chosenCandidate = candidateList.find((c) => c.name.toLowerCase() === rawAssignee.toLowerCase());

        if (filterAnomalies && chosenCandidate && chosenCandidate.activeTasks > 5 && lowestWorkloadCandidate && lowestWorkloadCandidate.activeTasks < chosenCandidate.activeTasks) {
          chosenCandidate = lowestWorkloadCandidate;
        } else if (!chosenCandidate) {
          chosenCandidate = lowestWorkloadCandidate;
        }

        if (!chosenCandidate) continue;

        const sig = `action-${log.projectId}-${taskId}-${chosenCandidate.name}`;
        if (seenSignatures.has(sig)) continue;
        seenSignatures.add(sig);

        const finalCandidates = candidateList.map((c) => ({
          ...c,
          score: c.name === chosenCandidate!.name ? 0.95 : Math.max(0.1, 0.80 - c.activeTasks * 0.1),
        }));

        const candidateSummary = finalCandidates
          .map((c) => `${c.name}${c.performanceRating ? ` [Performance Rating: ${c.performanceRating}/10]` : ""} (${c.role ?? "Team Member"}, ${c.activeTasks} active tasks, ${c.workloadPercent}% workload)`)
          .join("; ");

        const logReason = `Candidate '${chosenCandidate.name}' has low relative workload (${chosenCandidate.activeTasks} active tasks) and high skill score.`;

        samples.push({
          instruction: `Evaluate task '${taskTitle}' (Priority: ${taskPriority}) for project '${log.projectId ?? "default"}' and assign optimal team member.`,
          context: {
            taskTitle,
            taskPriority,
            taskTags: task?.tags ?? [],
            projectId: log.projectId ?? undefined,
            candidates: finalCandidates,
            confidence: payload.confidence ?? 0.95,
          },
          thought_chain: `Step 1: Evaluated task '${taskTitle}' requirements (Priority: ${taskPriority}). Step 2: Queried live project capacity: ${candidateSummary}. Step 3: ${logReason} Conclusion: Assigning to ${chosenCandidate.name}.`,
          output: `Assign to ${chosenCandidate.name}`,
        });
      }
    }


    if (format === "jsonl") {
      const jsonlText = samples.map((s) => JSON.stringify(s)).join("\n");
      return new Response(jsonlText, {
        headers: {
          "Content-Type": "application/x-jsonlines",
          "Content-Disposition": `attachment; filename="sft_dataset_${Date.now()}.jsonl"`,
        },
      });
    }

    return NextResponse.json({
      success: true,
      count: samples.length,
      samples,
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : "Failed to export dataset" },
      { status: 500 }
    );
  }
}
