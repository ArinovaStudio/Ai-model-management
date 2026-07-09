from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    start_date: datetime
    deadline: Optional[datetime] = None
    metadata: Optional[dict[str, Any]] = None


class ProjectInfoUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[datetime] = None
    deadline: Optional[datetime] = None
    scope: Optional[str] = None
    constraints: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class PlanDocCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    metadata: Optional[dict[str, Any]] = None


class DocCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    doc_type: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class ClarificationInput(BaseModel):
    feature_id: str
    clarification: str = Field(..., min_length=1)


class AnalysisRequest(BaseModel):
    reindex: bool = False


class GapAnalysisRequest(BaseModel):
    reindex: bool = False


class ProjectInfoResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: Optional[str] = None
    start_date: datetime
    deadline: Optional[datetime] = None
    scope: Optional[str] = None
    constraints: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class PlanDocResponse(BaseModel):
    id: str
    project_id: str
    title: str
    content: str
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime


class DocResponse(BaseModel):
    id: str
    project_id: str
    title: str
    content: str
    doc_type: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime


class ProjectResponse(BaseModel):
    id: str
    name: str
    summary: Optional[dict[str, Any]] = None
    status: str
    project_info: Optional[ProjectInfoResponse] = None
    plan_docs_count: int = 0
    docs_count: int = 0
    feature_requests_count: int = 0
    technologies_count: int = 0
    risk_blockages_count: int = 0
    grouping_tasks_count: int = 0
    wbs_tasks_count: int = 0
    resource_allocations_count: int = 0
    total_effort_hours: Optional[float] = 0
    total_duration_days: Optional[float] = 0
    avg_complexity_score: Optional[float] = 0
    created_at: datetime
    updated_at: datetime


class ContextResponse(BaseModel):
    project_id: str
    project_name: str
    summary: dict[str, Any]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureRequestResponse(BaseModel):
    id: str
    project_id: str
    title: str
    description: str
    priority: str
    status: str
    category: Optional[str] = None
    source_doc_id: Optional[str] = None
    source_doc_type: Optional[str] = None
    is_ambiguous: bool = False
    clarification_prompt: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class ProjectTechnologyResponse(BaseModel):
    id: str
    project_id: str
    name: str
    category: str
    version: Optional[str] = None
    is_detected: bool = True
    dependencies: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class AnalysisResponse(BaseModel):
    project_id: str
    project_name: str
    feature_requests: list[FeatureRequestResponse]
    technologies: list[ProjectTechnologyResponse]
    ambiguous_requirements: list[FeatureRequestResponse]
    clarification_prompts: list[str]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class SemanticSearchResult(BaseModel):
    doc_id: str
    doc_type: str
    title: str
    content_snippet: str
    similarity_score: float


class SemanticSearchResponse(BaseModel):
    query: str
    results: list[SemanticSearchResult]
    answer: Optional[str] = None


class TechnicianClarificationInput(BaseModel):
    feature_id: str
    clarification: str = Field(..., min_length=1)


class RiskBlockageResponse(BaseModel):
    id: str
    project_id: str
    type: str
    category: str
    title: str
    description: str
    severity: str
    status: str
    source_feature_id: Optional[str] = None
    source_doc_id: Optional[str] = None
    report_type: str
    metadata: Optional[dict[str, Any]] = None
    confidence_score: Optional[float] = 0
    planning_warnings: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class GapAnalysisReport(BaseModel):
    assumptions: list[RiskBlockageResponse] = []
    clarifications: list[RiskBlockageResponse] = []
    risks: list[RiskBlockageResponse] = []


class GapAnalysisResponse(BaseModel):
    project_id: str
    project_name: str
    reports: GapAnalysisReport
    summary: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class WbsTaskResponse(BaseModel):
    id: str
    title: str
    description: str
    order: int = 0
    status: str
    priority: str
    dependencies: Optional[list[str]] = None
    estimated_hours: Optional[float] = None
    complexity_score: Optional[float] = 0
    effort_hours: Optional[float] = 0
    duration_days: Optional[float] = 0
    feature_request_id: Optional[str] = None
    feature_request_title: Optional[str] = None
    group_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class WbsGroupResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    order: int = 0
    feature_ids: Optional[list[str]] = None
    tasks: list[WbsTaskResponse] = []
    total_effort_hours: Optional[float] = 0
    total_duration_days: Optional[float] = 0
    avg_complexity_score: Optional[float] = 0
    created_at: datetime
    updated_at: datetime


class WbsResponse(BaseModel):
    project_id: str
    project_name: str
    groups: list[WbsGroupResponse]
    total_groups: int
    total_tasks: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class WbsRequest(BaseModel):
    reindex: bool = False
    sprint_duration_days: Optional[int] = 14
    productive_hours_per_day: Optional[float] = 6.0


class WbsTaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    assignee: Optional[str] = None
    estimated_hours: Optional[float] = None
    dependencies: Optional[list[str]] = None


class EstimationRequest(BaseModel):
    reindex: bool = False
    sprint_duration_days: Optional[int] = 14
    productive_hours_per_day: Optional[float] = 6.0


class TaskEstimateResponse(BaseModel):
    task_id: str
    task_title: str
    complexity_score: float = 0
    effort_hours: float = 0
    duration_days: float = 0
    estimated_hours: float = 0
    dependencies: Optional[list[str]] = None


class GroupEstimateResponse(BaseModel):
    group_id: str
    group_name: str
    total_effort_hours: float = 0
    total_duration_days: float = 0
    avg_complexity_score: float = 0
    task_count: int = 0
    tasks: list[TaskEstimateResponse] = []


class EstimationSummary(BaseModel):
    total_effort_hours: float = 0
    total_duration_days: float = 0
    avg_complexity_score: float = 0
    total_tasks: int = 0
    total_groups: int = 0


class EstimationResponse(BaseModel):
    project_id: str
    project_name: str
    summary: EstimationSummary
    groups: list[GroupEstimateResponse]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ResourceAllocationResponse(BaseModel):
    id: str
    project_id: str
    role: str
    required_count: int = 1
    allocated_hours: float = 0
    workload_percentage: Optional[float] = 0
    recommended_count: Optional[float] = 0
    created_at: datetime
    updated_at: datetime


class ResourcePlanRequest(BaseModel):
    reindex: bool = False
    sprint_duration_days: Optional[int] = 14
    productive_hours_per_day: Optional[float] = 6.0


class ResourcePlanResponse(BaseModel):
    project_id: str
    project_name: str
    allocations: list[ResourceAllocationResponse]
    total_team_size: int = 0
    total_effort_hours: float = 0
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# --- Phase 7: Schedule ---


class ScheduleRequest(BaseModel):
    reindex: bool = False
    sprint_duration_days: Optional[int] = 14
    productive_hours_per_day: Optional[float] = 6.0


class MilestoneResponse(BaseModel):
    id: str
    project_id: str
    title: str
    description: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    due_date: datetime
    status: str = "PENDING"
    created_at: datetime


class SprintInfo(BaseModel):
    sprint_number: int
    group_id: str
    group_name: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    total_effort_hours: float = 0
    task_count: int = 0


class ScheduledTaskInfo(BaseModel):
    task_id: str
    title: str
    group_id: Optional[str] = None
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    duration_days: float = 0
    dependencies: Optional[list[str]] = None


class ScheduleWarning(BaseModel):
    category: str
    message: str
    severity: str = "warning"


class ScheduleResponse(BaseModel):
    project_id: str
    project_name: str
    start_date: datetime
    deadline: datetime
    milestones: list[MilestoneResponse] = []
    sprints: list[SprintInfo] = []
    tasks: list[ScheduledTaskInfo] = []
    warnings: list[ScheduleWarning] = []
    feasible: bool = True
    required_headcount: Optional[int] = None
    expected_deadline: Optional[datetime] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# --- Phase 8: Validation & Confidence ---


class ValidationRequest(BaseModel):
    reindex: bool = False


class ConfidenceReport(BaseModel):
    score: float = 0
    requirement_completeness: float = 0
    document_coverage: float = 0
    dependency_clarity: float = 0
    timeline_feasibility: float = 0
    technology_certainty: float = 0
    max_score: float = 100


class PlanningWarningInfo(BaseModel):
    category: str
    message: str
    severity: str = "medium"


class RiskAssessmentInfo(BaseModel):
    risk_id: str
    title: str
    type: str
    severity: str
    status: str
    confidence_impact: float = 0
    warning: Optional[str] = None


class ValidationResponse(BaseModel):
    project_id: str
    project_name: str
    confidence: ConfidenceReport
    warnings: list[PlanningWarningInfo] = []
    risk_assessments: list[RiskAssessmentInfo] = []
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
