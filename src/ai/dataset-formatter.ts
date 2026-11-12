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

