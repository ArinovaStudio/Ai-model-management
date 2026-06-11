from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from prisma import Prisma
from database import crud
from pydantic import BaseModel

# Initialize the global database client
db = Prisma()

# Manage the database connection lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    if db.is_connected():
        await db.disconnect()

# Update your FastAPI app initialization to use the lifespan
app = FastAPI(title="Employee Management Chatbot", lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "success", "message": "The local server is running properly."}

# --- Pydantic Models for Receiving Data ---
class TaskStatusUpdate(BaseModel):
    status: str

class NewTask(BaseModel):
    title: str
    project_id: int
    employee_id: int

class DailyReport(BaseModel):
    content: str
    employee_id: int

# --- API Endpoints ---

@app.get("/employees/{employee_id}")
async def get_employee(employee_id: int):
    employee = await crud.fetch_employee_details(db, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee

@app.get("/projects/{project_id}")
async def get_project(project_id: int):
    project = await crud.fetch_project_details(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@app.get("/employees/{employee_id}/tasks/pending")
async def get_pending_tasks(employee_id: int):
    return await crud.fetch_pending_tasks(db, employee_id)

@app.put("/tasks/{task_id}/status")
async def update_status(task_id: int, task_data: TaskStatusUpdate):
    return await crud.update_task_status(db, task_id, task_data.status)

@app.post("/tasks/assign")
async def assign_new_task(task_data: NewTask):
    return await crud.assign_task(db, task_data.title, task_data.project_id, task_data.employee_id)

@app.post("/reports/store")
async def store_report(report_data: DailyReport):
    return await crud.store_daily_report(db, report_data.content, report_data.employee_id)
