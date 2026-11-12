export interface DependencyInfo {
  taskId: string;
  dependsOnTaskId: string;
  relationType: string;
}

export interface TaskInfo {
  id: string;
  title: string;
  status: string;
  priority: string;
  tags: string[];
}

export interface TaskReadiness {
  taskId: string;
  ready: boolean;
  reason: string;
  unresolvedDependencies: string[];
  hasCycle: boolean;
}

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

export interface RiskConcern {
  id: string;
  label: string;
  severity: "low" | "medium" | "high";
}

export interface RiskConfidence {
  score: number;
  dataSourcesUsed: string[];
  dataSourcesTotal: number;
}

export interface RiskAssessment {
  level: RiskLevel;
  confidence: RiskConfidence;
  explanation: string;
  concerns: RiskConcern[];
  progress: number;
  progressClamped: boolean;
}

export interface AssignmentComponents {
  workloadScore: number | null;
  skillScore: number | null;
  complexityFit: number | null;
  membershipScore: number;
}

export interface AssignmentWeights {
  workload: number;
  skill: number;
  complexity: number;
}

export interface AssignmentScore {
  userId: string;
  name: string;
  score: number;
  components: AssignmentComponents;
  weights: AssignmentWeights;
  basis: string[];
}

export interface AssignmentCandidate {
  userId: string;
  name: string;
  membership: boolean;
  isLeader: boolean;
  role: string | null;
  workloadPercent: number | null;
  pendingEffortHours?: number | null;
  workloadDensity?: number | null;
  workingAs: string | null;
  department: string | null;
  subRole: string | null;
  bio: string | null;
}

export interface RiskNarrative {
  narrative: string;
  recommendations: string[];
  model: string | null;
}

export interface ReasonerEvaluation {
  task: TaskReadiness | null;
  risk: RiskAssessment;
  riskNarrative: RiskNarrative | null;
  assignment: {
    taskId: string;
    ranked: AssignmentScore[];
  } | null;
  createdAt: string;
  llmUsed: boolean;
}

export interface ReasonerOptions {
  taskId?: string;
  candidates?: string[];
  runLlm?: boolean;
}