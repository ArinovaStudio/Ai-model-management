from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nlp_engine import NLPProcessor
from prisma import Prisma
import uvicorn

app = FastAPI(title="AI Management Chatbot API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading NLP Engine... (This takes a few seconds)")
nlp = NLPProcessor()
db = Prisma()

conversation_context = {
    "last_entity": None
}

class ChatRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup():
    await db.connect()

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        analysis = nlp.process_query(request.query)
        intent = analysis["intent"]
        entities = analysis["extracted_entities"]

        if entities["assignees"]:
            conversation_context["last_entity"] = entities["assignees"][0]
        else:
            if conversation_context["last_entity"] and (
                "that" in request.query.lower() or "this" in request.query.lower()
            ):
                entities["assignees"].append(conversation_context["last_entity"])

        target = entities["assignees"][0] if entities["assignees"] else None
        bot_response = ""

        # ---------------------------------------------------------
        # GET ALL PROJECTS
        # ---------------------------------------------------------
        if intent == "GET_ALL_PROJECTS":
            all_projects = await db.project.find_many()
            if all_projects:
                names = ", ".join([p.name for p in all_projects if p.name])
                bot_response = f"There are currently {len(all_projects)} projects in the system: {names}."
            else:
                bot_response = "There are no projects currently in the database."

        # ---------------------------------------------------------
        # GET ACTIVE EMPLOYEES
        # ---------------------------------------------------------
        elif intent == "GET_ACTIVE_EMPLOYEES":
            active_users = await db.user.find_many(where={"isLogin": True})
            if active_users:
                names = ", ".join([u.name for u in active_users if u.name])
                bot_response = f"There are currently {len(active_users)} active employees logged in: {names}."
            else:
                bot_response = "There are no employees currently logged in."

        # ---------------------------------------------------------
        # CHECK LOGIN STATUS
        # ---------------------------------------------------------
        elif intent == "CHECK_LOGIN_STATUS":
            if target:
                user = await db.user.find_first(
                    where={"name": {"contains": target, "mode": "insensitive"}}
                )
                if user:
                    status = "logged in" if user.isLogin else "logged out"
                    bot_response = f"{user.name} is currently {status}."
                else:
                    bot_response = f"I couldn't find an employee named '{target}'."
            else:
                bot_response = "Please specify the employee name."

        # ---------------------------------------------------------
        # CHECK CLOCK OUT TIME
        # ---------------------------------------------------------
        elif intent == "CHECK_CLOCK_OUT_TIME":
            if target:
                user = await db.user.find_first(
                    where={"name": {"contains": target, "mode": "insensitive"}}
                )
                if user:
                    record = await db.workhours.find_first(
                        where={"userId": user.id},
                        order={"date": "desc"}
                    )
                    if record and record.clockOut:
                        bot_response = f"{user.name} logged out at {record.clockOut} on {record.date.strftime('%Y-%m-%d')}."
                    else:
                        bot_response = f"No recent clock-out records found for {user.name}."
                else:
                    bot_response = f"I couldn't find an employee named '{target}'."
            else:
                bot_response = "Please specify the employee name."

        # ---------------------------------------------------------
        # CHECK COMPLETED WORK
        # ---------------------------------------------------------
        elif intent == "CHECK_COMPLETED_WORK":
            completed = await db.workdone.find_many(
                order={"createdAt": "desc"},
                take=1,
                include={"completedBy": True}
            )
            if completed:
                latest = completed[0]
                bot_response = f"The last completed task was '{latest.title}'."
            else:
                bot_response = "No completed work found."

        # ---------------------------------------------------------
        # PROJECT STATUS
        # ---------------------------------------------------------
        elif intent == "PROJECT_STATUS":
            if target:
                project = await db.project.find_first(
                    where={"name": {"contains": target, "mode": "insensitive"}}
                )
                if project:
                    bot_response = f"The '{project.name}' project is currently in the '{project.currentPhase}' phase with {project.progress}% progress."
                else:
                    bot_response = f"I couldn't find a project named '{target}'."
            else:
                bot_response = "Please specify the project name."

        # ---------------------------------------------------------
        # GET PROJECT TEAM (NEW FEATURE)
        # ---------------------------------------------------------
        elif intent == "GET_PROJECT_TEAM":
            if target:
                project = await db.project.find_first(
                    where={"name": {"contains": target, "mode": "insensitive"}}
                )

                if project:
                    tasks = await db.task.find_many(
                        where={"projectId": project.id}
                    )

                    if tasks:
                        assignees = list(set([t.assignee for t in tasks if t.assignee]))
                        if assignees:
                            names = ", ".join(assignees)
                            bot_response = f"The following employees are working on '{project.name}': {names}."
                        else:
                            bot_response = f"No employees are currently assigned to '{project.name}'."
                    else:
                        bot_response = f"No tasks found for project '{project.name}'."
                else:
                    bot_response = f"I couldn't find a project named '{target}'."
            else:
                bot_response = "Please specify the project name."

        # ---------------------------------------------------------
        # GET GITHUB PROFILE
        # ---------------------------------------------------------
        elif intent == "GET_GITHUB_PROFILE":
            if target:
                user = await db.user.find_first(
                    where={"name": {"contains": target, "mode": "insensitive"}}
                )
                if user and user.githubProfile:
                    bot_response = f"The GitHub profile for {user.name} is: {user.githubProfile}"
                else:
                    bot_response = f"No GitHub profile found for '{target}'."
            else:
                bot_response = "Please specify the employee name."

        # ---------------------------------------------------------
        # DEFAULT
        # ---------------------------------------------------------
        else:
            bot_response = f"I understood your intent as '{intent}', but I haven't been programmed with a database query for that yet."

        return {"reply": bot_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)