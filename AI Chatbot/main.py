import json
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nlp_engine import NLPProcessor
from prisma import Prisma
import uvicorn
from thefuzz import process

app = FastAPI(title="AI Management Chatbot API (WebSocket)")

# --- CORS SETUP ---
# Crucial for allowing your future React app to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========================================================
# 🛠️ BUILT-IN WEBSOCKET TESTER (No React Needed!)
# =========================================================
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>WebSocket Tester</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f7f6; display: flex; justify-content: center; padding-top: 50px; }
            .chat-box { width: 600px; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
            #messages { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 5px; background: #f9fbfc; }
            input { width: 75%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 16px; }
            button { width: 20%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
            button:hover { background: #0056b3; }
            p { margin: 8px 0; line-height: 1.4; }
        </style>
    </head>
    <body>
        <div class="chat-box">
            <h2>🟢 AI WebSocket Tester</h2>
            <div id='messages'></div>
            <form action="" onsubmit="sendMessage(event)">
                <input type="text" id="messageText" placeholder="Ask about Khushi, projects, etc..." autocomplete="off"/>
                <button>Send</button>
            </form>
        </div>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws/chat");
            var messages = document.getElementById('messages');
            
            ws.onmessage = function(event) {
                var data = JSON.parse(event.data);
                messages.innerHTML += "<p>🤖 <b>Bot:</b> " + data.reply + "</p>";
                messages.scrollTop = messages.scrollHeight;
            };
            
            function sendMessage(event) {
                var input = document.getElementById("messageText");
                if (!input.value) return event.preventDefault();
                
                messages.innerHTML += "<p>👤 <b>You:</b> " + input.value + "</p>";
                ws.send(JSON.stringify({query: input.value}));
                
                input.value = '';
                messages.scrollTop = messages.scrollHeight;
                event.preventDefault();
            }
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)
# =========================================================

print("Loading NLP Engine... (This takes a few seconds)")
nlp = NLPProcessor()
db = Prisma()

@app.on_event("startup")
async def startup():
    await db.connect()

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()

# --- REAL-TIME WEBSOCKET ENDPOINT ---
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    # 1. Open the tunnel when React connects
    await websocket.accept()
    print("🟢 React Client Connected to WebSocket!")
    
    # --- SHORT TERM MEMORY (Scoped to this specific connection) ---
    conversation_context = {
        "last_entity": None
    }

    try:
        # 2. Keep the connection open forever
        while True:
            # Wait for the frontend to send a message
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)
            user_query = data.get("query", "")

            if not user_query:
                continue

            print(f"User Asked: {user_query}")

            analysis = nlp.process_query(user_query)
            intent = analysis["intent"]
            entities = analysis["extracted_entities"]
            
            # --- MEMORY INJECTION (For Employees/Names) ---
            if entities["assignees"]:
                conversation_context["last_entity"] = entities["assignees"][0]
            else:
                if conversation_context["last_entity"] and ("that" in user_query.lower() or "this" in user_query.lower() or "he" in user_query.lower() or "she" in user_query.lower() or "they" in user_query.lower()):
                    entities["assignees"].append(conversation_context["last_entity"])
            
            bot_response = ""
            db_data = None
            target = entities["assignees"][0] if entities["assignees"] else None

            # =========================================================
            # 🪄 THE FUZZY AUTOCORRECT INTERCEPTOR
            # =========================================================
            # If the NLP found a target name, let's make sure it's spelled perfectly!
            if target:
                # 1. Fetch all actual, correctly spelled names from your database
                all_users = await db.user.find_many()
                all_projects = await db.project.find_many()
                
                valid_names = [u.name for u in all_users if u.name] + [p.name for p in all_projects if p.name]
                
                # 2. Ask thefuzz to score the user's typo against the real names
                if valid_names:
                    # extractOne returns a tuple like: ("Asthapuram Chanakya", 92)
                    best_match, score = process.extractOne(target, valid_names)
                    
                    # 3. If the AI is at least 70% confident it found a match, fix the typo!
                    if score >= 70:
                        print(f"🪄 Autocorrected typo: '{target}' -> '{best_match}' (Confidence: {score}%)")
                        target = best_match
                        # Update short term memory with the correct spelling
                        conversation_context["last_entity"] = target 

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
            # 5. CHECK LOGOUT TIME
            # ---------------------------------------------------------
            elif intent == "CHECK_CLOCK_OUT_TIME":
                if target:
                    user = await db.user.find_first(where={"name": {"contains": target, "mode": "insensitive"}})
                    if user:
                        records = await db.workhours.find_many(
                            where={"userId": user.id},
                            order={"date": "desc"},
                            take=5
                        )
                        
                        found_record = None
                        for r in records:
                            if r.clockOut and r.clockOut.strip() not in ["", "-"]:
                                found_record = r
                                break 
                                
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
            # 6. CHECK DAILY WORK SUMMARY
            # ---------------------------------------------------------
            elif intent == "GET_EMPLOYEE_SUMMARY":
                if target:
                    user = await db.user.find_first(where={"name": {"contains": target, "mode": "insensitive"}})
                    if user:
                        records = await db.workhours.find_many(
                            where={"userId": user.id},
                            order={"date": "desc"},
                            take=5
                        )
                        
                        found_summary = None
                        found_date = None
                        
                        for r in records:
                            if r.summary and r.summary.strip():
                                found_summary = r.summary
                                found_date = r.date
                                break 
                                
                        if found_summary:
                            bot_response = f"Latest work summary for {user.name} (from {found_date.strftime('%Y-%m-%d')}): '{found_summary}'"
                        else:
                            bot_response = f"I couldn't find any recent work summaries for {user.name}."
                    else:
                        bot_response = f"I couldn't find an employee named '{target}'."
                else:
                    bot_response = "Please specify the employee name."

            # ---------------------------------------------------------
            # 7. CHECK PROJECT SUMMARY UPDATE
            # ---------------------------------------------------------
            elif intent == "CHECK_PROJECT_SUMMARY":
                if target:
                    project = await db.project.find_first(where={"name": {"contains": target, "mode": "insensitive"}})
                    bot_prefix = ""
                    
                    if not project:
                        latest_task = await db.task.find_first(
                            where={"assignee": {"contains": target, "mode": "insensitive"}},
                            order={"createdAt": "desc"},
                            include={"Project": True}
                        )
                        if latest_task and latest_task.Project:
                            project = latest_task.Project
                            bot_prefix = f"Checking {target}'s active project '{project.name}': "

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
                    names = ", ".join([u.name for u in active_users if u.name])
                    bot_response = f"There are currently {count} active employees logged in: {names}."
                else:
                    bot_response = "There are no employees currently logged in or active."

            # ---------------------------------------------------------
            # 11. GET ALL PROJECTS
            # ---------------------------------------------------------
            elif intent == "GET_ALL_PROJECTS":
                all_projects = await db.project.find_many()
                
                if all_projects:
                    count = len(all_projects)
                    names = ", ".join([p.name for p in all_projects if p.name])
                    bot_response = f"There are currently {count} projects in the system: {names}."
                else:
                    bot_response = "There are no projects currently in the database."

            # THE CATCH-ALL MUST BE LAST!
            else:
                bot_response = f"I understood your intent as '{intent}', but I haven't been programmed with a database query for that yet."

            # 3. Instantly stream the answer back through the open tunnel
            await websocket.send_json({"reply": bot_response})

    # Handle the user closing their browser tab
    except WebSocketDisconnect:
        print("🔴 React Client Disconnected.")
    # Catch internal errors so the server doesn't crash
    except Exception as e:
        print(f"WebSocket Error: {str(e)}")
        try:
            await websocket.send_json({"reply": f"Sorry, an internal error occurred: {str(e)}"})
        except:
            pass # Socket is already closed

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
