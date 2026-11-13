export interface IngestionRecordCounts {
  project: number;
  projectInfo: number;
  projectMembers: number;
  milestones: number;
  assets: number;
  latestUpdates: number;
  featureRequests: number;
  riskBlockages: number;
  planDocs: number;
  documentAccess: number;
  docs: number;
  roadmaps: number;
  roadmapAssignments: number;
  roadmapPermissions: number;
  roadmapComments: number;
  designSystems: number;
  projectTechnologies: number;
  groupTasks: number;
  tasks: number;
  taskDependencies: number;
  workDone: number;
  tickets: number;
  usefulTips: number;
  meetingRequests: number;
  memos: number;
}

export interface IngestionSectionCounts {
  wbsSchedule: {
    groupingTasks: number;
    tasks: number;
    taskDependencies: number;
  };
  timeline: {
    tasks: number;
    latestUpdates: number;
  };
  resources: {
    projectMembers: number;
    milestones: number;
    assets: number;
    meetingRequests: number;
  };
  gaps: {
    featureRequests: number;
    riskBlockages: number;
  };
  productStrategy: {
    roadmaps: number;
    roadmapComments: number;
  };
  technicalArchitecture: {
    designSystems: number;
    projectTechnologies: number;
  };
  documentation: {
    planDocs: number;
    docs: number;
  };
}

export interface IngestionStepLog {
  step: string;
  startedAt: string;
  endedAt: string;
  elapsedMs: number;
  detail: string;
}

export interface IngestionResult {
  success: boolean;
  sourceProjectId: string;
  sourceProjectName: string;
  targetProjectId: string;
  targetProjectName: string;
  records: IngestionRecordCounts;
  sections: IngestionSectionCounts;
  taskIdMap: Record<string, string>;
  warnings: string[];
  logs: IngestionStepLog[];
  durationMs: number;
}

export class IngestionError extends Error {
  sourceProjectId: string;
  step: string;

  constructor(sourceProjectId: string, step: string, message: string) {
    super(message);
    this.name = "IngestionError";
    this.sourceProjectId = sourceProjectId;
    this.step = step;
  }
}