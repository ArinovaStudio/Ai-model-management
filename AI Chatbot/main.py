from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nlp_engine import NLPProcessor
from prisma import Prisma
import uvicorn

app = FastAPI(title="AI Management Chatbot API")

# ✅ CORS FIX
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

# --- SHORT TERM MEMORY ---
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
        
        # --- MEMORY INJECTION ---
        if entities["assignees"]:
            conversation_context["last_entity"] = entities["assignees"][0]
        else:
            if conversation_context["last_entity"] and ("that" in request.query.lower() or "this" in request.query.lower()):
                entities["assignees"].append(conversation_context["last_entity"])
        
        bot_response = ""
        target = entities["assignees"][0] if entities["assignees"] else None

        # --- DYNAMIC PROJECT NAME MATCHER ---
        if not target:
            all_projects = await db.project.find_many()
            normalized_query = request.query.lower().replace("-", "").replace(" ", "")
            
            for p in all_projects:
                if p.name:
                    normalized_p_name = p.name.lower().replace("-", "").replace(" ", "")
                    
                    if normalized_p_name and normalized_p_name in normalized_query:
                        target = p.name
                        conversation_context["last_entity"] = target 
                        break

        # ---------------------------------------------------------
        # 1. LAST TASK ASSIGNED
        # ---------------------------------------------------------
        if intent == "LAST_TASK_ASSIGNED":
            query_args = {"order": {"createdAt": "desc"}, "take": 1, "include": {"Project": True}}
            if target:
                query_args["where"] = {"assignee": {"contains": target, "mode": "insensitive"}}
                
            tasks = await db.task.find_many(**query_args)
            if tasks:
                t = tasks[0]
                project_name = t.Project.name if t.Project else "Unassigned"
                bot_response = f"Last task assigned to {t.assignee} and task is '{t.title}' and project name is '{project_name}'."
            else:
                if target:
                    old_work = await db.workdone.find_first(
                        where={"completedBy": {"is": {"name": {"contains": target, "mode": "insensitive"}}}},
                        order={"createdAt": "desc"}
                    )
                    if old_work:
                        bot_response = f"{target.title()} doesn't have an active task right now, but their last completed work from history was: '{old_work.title}'."
                    else:
                        bot_response = f"I checked both active tasks and old history, but couldn't find any records for '{target}'."
                else:
                    bot_response = "I couldn't find any recent tasks matching that criteria."

        # ---------------------------------------------------------
        # 1.5 LAST PROJECT ASSIGNED
        # ---------------------------------------------------------
        elif intent == "LAST_PROJECT_ASSIGNED":
            query_args = {"order": {"createdAt": "desc"}, "take": 1, "include": {"Project": True}}
            if target:
                query_args["where"] = {"assignee": {"contains": target, "mode": "insensitive"}}
                
            tasks = await db.task.find_many(**query_args)
            if tasks:
                t = tasks[0]
                if t.Project:
                    bot_response = f"The latest project assigned to {t.assignee} is '{t.Project.name}'."
                else:
                    bot_response = f"{t.assignee} was assigned a task, but it doesn't belong to a specific project."
            else:
                bot_response = f"I couldn't find any active projects assigned to '{target}'."

        # ---------------------------------------------------------
        # 2. PROJECT STATUS
        # ---------------------------------------------------------
        elif intent == "PROJECT_STATUS":
            if target:
                project = await db.project.find_first(where={"name": {"contains": target, "mode": "insensitive"}})
                if project:
                    bot_response = f"The '{project.name}' project is currently in the '{project.currentPhase}' phase with {project.progress}% progress."
                else:
                    bot_response = f"I couldn't find a project named '{target}'."
            else:
                bot_response = "Please specify which project's progress you want to check."

        # ---------------------------------------------------------
        # 3. GET FIGMA DESIGN
        # ---------------------------------------------------------
        elif intent == "GET_PROJECT_ASSET":
            if target:
                project = await db.project.find_first(
                    where={"name": {"contains": target, "mode": "insensitive"}},
                    include={"Asset": True}
                )
                if project and project.Asset:
                    figma_assets = [a for a in project.Asset if "figma" in a.url.lower() or "figma" in a.title.lower()]
                    if figma_assets:
                        bot_response = f"Here is the Figma design link for {project.name}: {figma_assets[0].url}"
                    else:
                        bot_response = f"I found assets for {project.name}, but no specific Figma links. The latest asset is: {project.Asset[0].url}"
                else:
                    bot_response = f"I couldn't find any design assets for '{target}'."
            else:
                bot_response = "Please specify the project name to get its design link."

        # ---------------------------------------------------------
        # 10. GET ACTIVE EMPLOYEES
        # ---------------------------------------------------------
        elif intent == "GET_ACTIVE_EMPLOYEES":
            active_users = await db.user.find_many(where={"isLogin": True})
            if active_users:
                names = ", ".join([u.name for u in active_users if u.name])
                bot_response = f"There are currently {len(active_users)} active employees logged in: {names}."
            else:
                bot_response = "There are no employees currently logged in or active."

        # ---------------------------------------------------------
        # 11. GET ALL PROJECTS
        # ---------------------------------------------------------
        elif intent == "GET_ALL_PROJECTS":
            all_projects = await db.project.find_many()
            if all_projects:
                names = ", ".join([p.name for p in all_projects if p.name])
                bot_response = f"There are currently {len(all_projects)} projects in the system: {names}."
            else:
                bot_response = "There are no projects currently in the database."

        # ---------------------------------------------------------
        # DEFAULT FALLBACK
        # ---------------------------------------------------------
        else:
            bot_response = f"I understood your intent as '{intent}', but I haven't been programmed with a database query for that yet."

        return {
            "reply": bot_response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)