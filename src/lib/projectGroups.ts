/**
 * src/lib/projectGroups.ts
 *
 * Helper to dynamically calculate task groups (chunks) for a project
 * based on flat logical sorting and fixed-size visual slicing (slots).
 */

export interface TaskGroupInfo {
  name: string;
  tasks: any[];
  status: "COMPLETED" | "ACTIVE" | "UPCOMING";
  completedCount: number;
  totalCount: number;
}

export function getProjectGroups(tasks: any[], dependencies: any[]): TaskGroupInfo[] {
  // Helper to determine the SDLC phase of a task based on its title
  const getTaskPhase = (title: string): number => {
    const t = (title || "").toLowerCase();
    const isQA = /(deploy|go-live|qa|testing|verification|audit|monitoring)/i.test(t);
    if (isQA) return 3;
    const isBaseline = /(infrastructure|baseline|setup|initialize)/i.test(t);
    if (isBaseline) return 1;
    return 2; // Default to Feature
  };

  // Helper to check if a task is completed or blocked (on-hold)
  const isInactive = (status: string) => {
    const s = (status || "").toLowerCase();
    return s === "completed" || s === "done" || s === "blocked" || s === "on-hold" || s === "on_hold";
  };

  // 1. Build the clean task dependency graph (ignoring reversed dependencies)
  const adj = new Map<string, Set<string>>();
  tasks.forEach(t => adj.set(t.id, new Set<string>()));

  for (const dep of dependencies) {
    const task = tasks.find(t => t.id === dep.taskId);
    const dependsOn = tasks.find(t => t.id === dep.dependsOnTaskId);
    if (task && dependsOn) {
      const taskPhase = getTaskPhase(task.title);
      const dependsOnPhase = getTaskPhase(dependsOn.title);
      // Skip reversed dependency edges (e.g. Feature depending on QA)
      if (taskPhase < dependsOnPhase) {
        continue;
      }
      adj.get(dependsOn.id)?.add(task.id);
    }
  }

  // 2. Perform Topological Sort
  const visited = new Set<string>();
  const stack: string[] = [];

  function visit(taskId: string) {
    if (visited.has(taskId)) return;
    visited.add(taskId);
    const neighbors = adj.get(taskId) ?? new Set();
    for (const next of neighbors) {
      visit(next);
    }
    stack.push(taskId);
  }

  // Visit tasks in descending phase order (3 -> 2 -> 1) with stable ID fallback
  const sortedByPhaseDesc = [...tasks].sort((a, b) => {
    const phaseDiff = getTaskPhase(b.title) - getTaskPhase(a.title);
    if (phaseDiff !== 0) return phaseDiff;
    return a.id.localeCompare(b.id);
  });
  sortedByPhaseDesc.forEach(t => visit(t.id));
  const topoOrder = stack.reverse();

  // 3. Sort the flat tasks array with stable fallbacks
  const sortedTasks = [...tasks].sort((a, b) => {
    const phaseA = getTaskPhase(a.title);
    const phaseB = getTaskPhase(b.title);
    if (phaseA !== phaseB) {
      return phaseA - phaseB; // Baseline tasks always come first, then Features, then QA
    }
    const idxA = topoOrder.indexOf(a.id);
    const idxB = topoOrder.indexOf(b.id);
    if (idxA !== idxB) {
      return idxA - idxB;
    }
    return a.id.localeCompare(b.id);
  });

  // 4. Chunk/Slice the flat list into slots of size N (8 tasks per slot)
  const CHUNK_SIZE = 8;
  const groups: TaskGroupInfo[] = [];

  for (let i = 0; i < sortedTasks.length; i += CHUNK_SIZE) {
    const chunkTasks = sortedTasks.slice(i, i + CHUNK_SIZE);
    const slotIndex = Math.floor(i / CHUNK_SIZE) + 1;
    
    // Determine Slot Label dynamically based on the phases of its tasks
    const firstTaskPhase = getTaskPhase(chunkTasks[0]?.title);
    const label = firstTaskPhase === 1 ? `Slot ${slotIndex}: Setup & Baseline` :
                  firstTaskPhase === 3 ? `Slot ${slotIndex}: Testing & QA` :
                  `Slot ${slotIndex}: Feature Development`;

    const totalCount = chunkTasks.length;
    const completedCount = chunkTasks.filter(t => isInactive(t.status)).length;

    groups.push({
      name: label,
      tasks: chunkTasks,
      status: "UPCOMING", // Default status
      completedCount,
      totalCount
    });
  }

  // 5. Mark the first incomplete slot as ACTIVE
  let foundActive = false;
  for (const g of groups) {
    if (g.completedCount < g.totalCount) {
      if (!foundActive) {
        g.status = "ACTIVE";
        foundActive = true;
      } else {
        g.status = "UPCOMING";
      }
    } else {
      g.status = "COMPLETED";
    }
  }

  return groups;
}
