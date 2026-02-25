from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nlp_engine import NLPProcessor
from prisma import Prisma
import uvicorn

app = FastAPI(title="AI Management Chatbot API")

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
        
        # --- MEMORY INJECTION (For Employees/Names) ---
        if entities["assignees"]:
            conversation_context["last_entity"] = entities["assignees"][0]
        else:
            if conversation_context["last_entity"] and ("that" in request.query.lower() or "this" in request.query.lower()):
                entities["assignees"].append(conversation_context["last_entity"])
        
        bot_response = ""
        db_data = None
        target = entities["assignees"][0] if entities["assignees"] else None

        # --- DYNAMIC PROJECT NAME MATCHER ---
        # If spaCy missed the custom project name, search the database directly!
        if not target:
            all_projects = await db.project.find_many()
            # Remove all spaces and hyphens from the user's sentence to make matching bulletproof
            normalized_query = request.query.lower().replace("-", "").replace(" ", "")
            
            for p in all_projects:
                if p.name:
                    # Remove spaces/hyphens from the database name too
                    normalized_p_name = p.name.lower().replace("-", "").replace(" ", "")
                    
                    if normalized_p_name and normalized_p_name in normalized_query:
                        target = p.name
                        # Save it to memory so asking about "that project" works next time!
                        conversation_context["last_entity"] = target 
                        break

       # ---------------------------------------------------------
        # 1. LAST TASK ASSIGNED (Upgraded with History Fallback)
        # ---------------------------------------------------------
        if intent == "LAST_TASK_ASSIGNED":
            # Step 1: Check the active Task table first
            query_args = {"order": {"createdAt": "desc"}, "take": 1, "include": {"Project": True}}
            if target:
                query_args["where"] = {"assignee": {"contains": target, "mode": "insensitive"}}
                
            tasks = await db.task.find_many(**query_args)
            if tasks:
                t = tasks[0]
                project_name = t.Project.name if t.Project else "Unassigned"
                bot_response = f"Last task assigned to {t.assignee} and task is '{t.title}' and project name is '{project_name}'."
            else:
                # Step 2: FALLBACK - Check the WorkDone (history) table if active is empty!
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
            # We look up their latest task, but we pull the Project name instead!
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
        # 2. PROJECT STATUS / PROGRESS
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
        # 3. GET FIGMA DESIGN / ASSET
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
        # 4. CHECK LOGIN STATUS
        # ---------------------------------------------------------
        elif intent == "CHECK_LOGIN_STATUS":
            if target:
                user = await db.user.find_first(where={"name": {"contains": target, "mode": "insensitive"}})
                if user:
                    status = "logged in" if user.isLogin else "logged out"
                    bot_response = f"Yes, {user.name} is currently {status}."
                else:
                    bot_response = f"I couldn't find an employee named '{target}'."
            else:
                bot_response = "Please specify which employee's login status you want to check."

        # ---------------------------------------------------------
        # 5. CHECK LOGOUT TIME (Upgraded to skip "-" hyphens)
        # ---------------------------------------------------------
        elif intent == "CHECK_CLOCK_OUT_TIME":
            if target:
                user = await db.user.find_first(where={"name": {"contains": target, "mode": "insensitive"}})
                if user:
                    # Fetch the last 5 records
                    records = await db.workhours.find_many(
                        where={"userId": user.id},
                        order={"date": "desc"},
                        take=5
                    )
                    
                    found_record = None
                    # Loop through records and explicitly ignore empty strings AND hyphens "-"
                    for r in records:
                        if r.clockOut and r.clockOut.strip() not in ["", "-"]:
                            found_record = r
                            break # Stop looking once we find a real time!
                            
                    if found_record:
                        bot_response = f"{user.name} logged out at {found_record.clockOut} on {found_record.date.strftime('%Y-%m-%d')}."
                    elif records:
                        bot_response = f"{user.name} has open shifts but no recent clock-out times recorded."
                    else:
                        bot_response = f"I couldn't find any work hours records for '{user.name}'."
                else:
                    bot_response = f"I couldn't find an employee named '{target}'."
            else:
                bot_response = "Please specify the employee."

        # ---------------------------------------------------------
        # 6. CHECK DAILY WORK SUMMARY (Upgraded to skip empty summaries)
        # ---------------------------------------------------------
        elif intent == "GET_EMPLOYEE_SUMMARY":
            if target:
                user = await db.user.find_first(where={"name": {"contains": target, "mode": "insensitive"}})
                if user:
                    # Fetch the last 5 records so we can look past today's empty shift
                    records = await db.workhours.find_many(
                        where={"userId": user.id},
                        order={"date": "desc"},
                        take=5
                    )
                    
                    found_summary = None
                    found_date = None
                    
                    # Loop through records and skip any where the summary is empty or just spaces
                    for r in records:
                        if r.summary and r.summary.strip():
                            found_summary = r.summary
                            found_date = r.date
                            break # Stop looking once we find a real summary!
                            
                    if found_summary:
                        bot_response = f"Latest work summary for {user.name} (from {found_date.strftime('%Y-%m-%d')}): '{found_summary}'"
                    else:
                        bot_response = f"I couldn't find any recent work summaries for {user.name}."
                else:
                    bot_response = f"I couldn't find an employee named '{target}'."
            else:
                bot_response = "Please specify the employee name."

        # ---------------------------------------------------------
        # 7. CHECK PROJECT SUMMARY UPDATE (Upgraded for People/Clients)
        # ---------------------------------------------------------
        elif intent == "CHECK_PROJECT_SUMMARY":
            if target:
                # Step 1: Try to find a project directly by its name
                project = await db.project.find_first(where={"name": {"contains": target, "mode": "insensitive"}})
                bot_prefix = ""
                
                # Step 2: If it's NOT a project name, see if it's a person/client!
                if not project:
                    latest_task = await db.task.find_first(
                        where={"assignee": {"contains": target, "mode": "insensitive"}},
                        order={"createdAt": "desc"},
                        include={"Project": True}
                    )
                    if latest_task and latest_task.Project:
                        project = latest_task.Project
                        bot_prefix = f"Checking {target}'s active project '{project.name}': "

                # Step 3: Check the latest update for whatever project we found
                if project:
                    update = await db.latestupdate.find_first(
                        where={"projectId": project.id},
                        order={"createdAt": "desc"}
                    )
                    if update:
                        bot_response = f"{bot_prefix}Yes, the summary for {project.name} is updated. Latest update: '{update.title}'."
                    else:
                        bot_response = f"{bot_prefix}No, the summary for {project.name} has not been updated recently."
                else:
                    bot_response = f"I couldn't find a project or active client named '{target}'."
            else:
                bot_response = "Please specify the project or client name."

        # ---------------------------------------------------------
        # 8. CHECK COMPLETED WORK
        # ---------------------------------------------------------
        elif intent == "CHECK_COMPLETED_WORK":
            query_args = {
                "order": {"createdAt": "desc"},
                "take": 5, 
                "include": {"completedBy": True}
            }
            if target:
                query_args["where"] = {"completedBy": {"is": {"name": {"contains": target, "mode": "insensitive"}}}}

            completed_work = await db.workdone.find_many(**query_args)
            if completed_work:
                latest = completed_work[0]
                bot_response = f"The last task completed was '{latest.title}'."
            else:
                bot_response = "I couldn't find any recent completed work."

        # ---------------------------------------------------------
        # 9. GET GITHUB PROFILE
        # ---------------------------------------------------------
        elif intent == "GET_GITHUB_PROFILE":
            if target:
                user = await db.user.find_first(where={"name": {"contains": target, "mode": "insensitive"}})
                if user:
                    if user.githubProfile and user.githubProfile.strip():
                        bot_response = f"The GitHub profile for {user.name} is: {user.githubProfile}"
                    else:
                        bot_response = f"{user.name} has not linked a GitHub profile to their account yet."
                else:
                    bot_response = f"I couldn't find an employee named '{target}'."
            else:
                bot_response = "Please specify the employee's name."

        # ---------------------------------------------------------
        # 10. GET ACTIVE EMPLOYEES
        # ---------------------------------------------------------
        elif intent == "GET_ACTIVE_EMPLOYEES":
            active_users = await db.user.find_many(where={"isLogin": True})
            
            if active_users:
                count = len(active_users)
                # Create a clean, comma-separated list of their names
                names = ", ".join([u.name for u in active_users if u.name])
                bot_response = f"There are currently {count} active employees logged in: {names}."
            else:
                bot_response = "There are no employees currently logged in or active."

        # ---------------------------------------------------------
        # 11. GET ALL PROJECTS
        # ---------------------------------------------------------
        elif intent == "GET_ALL_PROJECTS":
            # Search the Project table for all entries
            all_projects = await db.project.find_many()
            
            if all_projects:
                count = len(all_projects)
                # Create a clean, comma-separated list of the project names
                names = ", ".join([p.name for p in all_projects if p.name])
                bot_response = f"There are currently {count} projects in the system: {names}."
            else:
                bot_response = "There are no projects currently in the database."

        # THE CATCH-ALL MUST BE LAST!
        else:
            bot_response = f"I understood your intent as '{intent}', but I haven't been programmed with a database query for that yet."

        return {
            "reply": bot_response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
