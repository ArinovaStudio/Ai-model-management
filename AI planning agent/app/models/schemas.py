from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    start_date: datetime
    deadline: datetime
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
    deadline: datetime
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


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
