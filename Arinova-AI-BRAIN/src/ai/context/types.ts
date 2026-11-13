export interface ProjectSnapshot {
  id: string;
  name: string;
  summary: string | null;
  priority: string | null;
  status: number;
  progress: number;
  currentPhase: string | null;
  deadline: string | null;
  deadlineOverdue: boolean;
  memberCount: number;
}

export interface TaskSnapshot {
  id: string;
  title: string;
  status: string;
  priority: string;
  assignee: string;
  dueDate: string | null;
  createdAt: string;
  isBlocked: boolean;
  estimatedHours?: number | null;
  userId?: string | null;
}

export interface DependencyEdge {
  taskId: string;
  dependsOnTaskId: string;
  relationType: string;
  source: string;
}

export interface TeamCapacityEntry {
  userId: string;
  name: string;
  role: string | null;
  isLeader: boolean;
  totalHours7d: number;
  workloadPercent: number;
  pendingEffortHours?: number;
  workloadDensity?: number;
  performanceRating?: number | null;
  ratingBreakdown?: Record<string, number> | null;
}

export interface RiskBlocker {
  id: string;
  title: string;
  severity: string;
  status: string;
  description: string;
  createdAt: string | null;
}

export interface RiskIndicators {
  activeTaskCount: number;
  blockedTaskCount: number;
  overdueTaskCount: number;
  topRiskBlockers: RiskBlocker[];
}

export interface ProjectContext {
  project: ProjectSnapshot;
  activeTasks: TaskSnapshot[];
  dependencyGraph: DependencyEdge[];
  teamCapacity: TeamCapacityEntry[];
  risks: RiskIndicators;
  createdAt: string;
}