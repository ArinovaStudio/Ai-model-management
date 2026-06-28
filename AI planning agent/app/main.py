import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.prisma import connect_db, disconnect_db
from app.routers import analysis, context, docs, estimation, projects, resource_planning, schedule, validation, wbs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    logger.info("Project Planning Engine started")
    yield
    await disconnect_db()
    logger.info("Project Planning Engine shut down")


app = FastAPI(
    title="AI Project Planning - Context & Analysis Engine",
    description="Phase 1+2: Ingest project inputs, produce unified context, extract requirements and technology stack",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router, prefix="/api/v1/projects", tags=["Projects"])
app.include_router(docs.router, prefix="/api/v1/projects", tags=["Documents"])
app.include_router(context.router, prefix="/api/v1/projects", tags=["Context"])
app.include_router(analysis.router, prefix="/api/v1/projects", tags=["Analysis"])
app.include_router(wbs.router, prefix="/api/v1/projects", tags=["WBS"])
app.include_router(estimation.router, prefix="/api/v1/projects", tags=["Estimation"])
app.include_router(resource_planning.router, prefix="/api/v1/projects", tags=["Resource Planning"])
app.include_router(schedule.router, prefix="/api/v1/projects", tags=["Schedule"])
app.include_router(validation.router, prefix="/api/v1/projects", tags=["Validation"])


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "project-planning-engine"}
