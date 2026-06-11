from prisma.models import Employee, Project, Task, DailyUpdate

# --- READ OPERATIONS ---

async def fetch_employee_details(db, employee_id: int):
    return await db.employee.find_unique(
        where={"id": employee_id},
        # FIX: Removed 'projects: True' to match your Prisma schema perfectly
        include={"tasks": True, "dailyUpdates": True} 
    )

async def fetch_project_details(db, project_id: int):
    return await db.project.find_unique(
        where={"id": project_id},
        include={"tasks": True}
    )

async def fetch_pending_tasks(db, employee_id: int):
    return await db.task.find_many(
        where={
            "employeeId": employee_id,
            "status": "PENDING"
        }
    )

# --- WRITE/UPDATE OPERATIONS ---

async def update_task_status(db, task_id: int, new_status: str):
    return await db.task.update(
        where={"id": task_id},
        data={"status": new_status}
    )

async def assign_task(db, title: str, project_id: int, employee_id: int):
    return await db.task.create(
        data={
            "title": title,
            "status": "PENDING",
            "projectId": project_id,
            "employeeId": employee_id
        }
    )

async def store_daily_report(db, content: str, employee_id: int):
    return await db.dailyupdate.create(
        data={
            "content": content,
            "employeeId": employee_id
        }
    )