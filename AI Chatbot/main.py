from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prisma import Prisma
from nlp_engine import NLPProcessor
import uvicorn
from datetime import datetime, timedelta

app = FastAPI(title="AI Management Chatbot")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = NLPProcessor()
db = Prisma()

conversation_context = {"last_entity": None}


class ChatRequest(BaseModel):
    query: str


@app.on_event("startup")
async def startup():
    await db.connect()


@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()


@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        analysis = nlp.process_query(request.query)
        intent = analysis["intent"]
        entities = analysis["extracted_entities"]

        # Memory Handling
        if entities.get("assignees"):
            conversation_context["last_entity"] = entities["assignees"][0]
        elif conversation_context["last_entity"]:
            entities["assignees"] = [conversation_context["last_entity"]]

        target = entities["assignees"][0] if entities.get("assignees") else None
        response = ""
        query_lower = request.query.lower()

        # ---------------------------------------------------------
        # GET ACTIVE EMPLOYEES / TODAY LOGIN
        # ---------------------------------------------------------
        if intent == "GET_ACTIVE_EMPLOYEES":

            # If asking about today login
            if "aaj" in query_lower or "today" in query_lower:
                today = datetime.today()
                start_of_day = datetime(today.year, today.month, today.day)
                end_of_day = start_of_day + timedelta(days=1)

                records = await db.workhours.find_many(
                    where={
                        "date": {
                            "gte": start_of_day,
                            "lt": end_of_day
                        }
                    },
                    include={"User": True}
                )

                if records:
                    names = list(set([r.User.name for r in records if r.User]))
                    response = f"Aaj login kiya: {', '.join(names)}"
                else:
                    response = "Aaj kisi ne login nahi kiya."

            # Otherwise show currently logged in users
            else:
                users = await db.user.find_many(where={"isLogin": True})
                if users:
                    names = ", ".join([u.name for u in users if u.name])
                    response = f"Active employees: {names}"
                else:
                    response = "No employees are currently logged in."

        # ---------------------------------------------------------
        # CHECK LOGIN STATUS (Specific User)
        # ---------------------------------------------------------
        elif intent == "CHECK_LOGIN_STATUS":
            if target:
                user = await db.user.find_first(
                    where={"name": {"contains": target, "mode": "insensitive"}}
                )
                if user:
                    status = "logged in" if user.isLogin else "logged out"
                    response = f"{user.name} is currently {status}."
                else:
                    response = "User not found."
            else:
                response = "Please specify employee name."

        # ---------------------------------------------------------
        # CHECK LOGOUT TIME
        # ---------------------------------------------------------
        elif intent == "CHECK_CLOCK_OUT_TIME":
            if target:
                user = await db.user.find_first(
                    where={"name": {"contains": target, "mode": "insensitive"}}
                )

                if user:
                    today = datetime.today()
                    start_of_day = datetime(today.year, today.month, today.day)
                    end_of_day = start_of_day + timedelta(days=1)

                    record = await db.workhours.find_first(
                        where={
                            "userId": user.id,
                            "date": {
                                "gte": start_of_day,
                                "lt": end_of_day
                            }
                        },
                        order={"date": "desc"}
                    )

                    if record and record.clockOut:
                        response = f"{user.name} logged out at {record.clockOut}."
                    else:
                        response = f"No logout record found for {user.name} today."

                else:
                    response = "User not found."
            else:
                response = "Please specify employee name."

        # ---------------------------------------------------------
        # CHECK COMPLETED WORK
        # ---------------------------------------------------------
        elif intent == "CHECK_COMPLETED_WORK":
            work = await db.workdone.find_many(
                order={"createdAt": "desc"},
                take=1
            )
            if work:
                response = f"Last completed task: {work[0].title}"
            else:
                response = "No completed tasks found."

        # ---------------------------------------------------------
        # GET GITHUB PROFILE
        # ---------------------------------------------------------
        elif intent == "GET_GITHUB_PROFILE":
            if target:
                user = await db.user.find_first(
                    where={"name": {"contains": target, "mode": "insensitive"}}
                )
                if user and user.githubProfile:
                    response = f"{user.name}'s GitHub: {user.githubProfile}"
                else:
                    response = "GitHub profile not found."
            else:
                response = "Please specify employee name."

        # ---------------------------------------------------------
        # LAST TASK ASSIGNED
        # ---------------------------------------------------------
        elif intent == "LAST_TASK_ASSIGNED":
            task = await db.task.find_many(
                order={"createdAt": "desc"},
                take=1
            )
            if task:
                response = f"Last assigned task: {task[0].title}"
            else:
                response = "No tasks found."

        # ---------------------------------------------------------
        # LAST PROJECT ASSIGNED
        # ---------------------------------------------------------
        elif intent == "LAST_PROJECT_ASSIGNED":
            task = await db.task.find_many(
                order={"createdAt": "desc"},
                take=1,
                include={"Project": True}
            )
            if task and task[0].Project:
                response = f"Last assigned project: {task[0].Project.name}"
            else:
                response = "No project found."

        # ---------------------------------------------------------
        # PROJECT STATUS
        # ---------------------------------------------------------
        elif intent == "PROJECT_STATUS":
            if target:
                project = await db.project.find_first(
                    where={"name": {"contains": target, "mode": "insensitive"}}
                )
                if project:
                    response = f"{project.name} is in {project.currentPhase} phase with {project.progress}% progress."
                else:
                    response = "Project not found."
            else:
                response = "Please specify project name."

        # ---------------------------------------------------------
        # GET PROJECT TEAM
        # ---------------------------------------------------------
        elif intent == "GET_PROJECT_TEAM":
            if target:
                project = await db.project.find_first(
                    where={"name": {"contains": target, "mode": "insensitive"}}
                )
                if project:
                    tasks = await db.task.find_many(where={"projectId": project.id})
                    members = list(set([t.assignee for t in tasks if t.assignee]))
                    if members:
                        response = f"Employees working on {project.name}: {', '.join(members)}"
                    else:
                        response = f"No employees assigned to {project.name}."
                else:
                    response = "Project not found."
            else:
                response = "Please specify project name."

        # ---------------------------------------------------------
        # GET ALL PROJECTS
        # ---------------------------------------------------------
        elif intent == "GET_ALL_PROJECTS":
            projects = await db.project.find_many()
            if projects:
                names = ", ".join([p.name for p in projects if p.name])
                response = f"Projects: {names}"
            else:
                response = "No projects found."

        # ---------------------------------------------------------
        # DEFAULT
        # ---------------------------------------------------------
        else:
            response = f"Intent detected: {intent}, but no logic implemented."

        return {"reply": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)