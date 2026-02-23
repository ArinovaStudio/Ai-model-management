from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nlp_engine import NLPProcessor
from prisma import Prisma
import uvicorn

app = FastAPI(title="AI Management Chatbot API")

print("Loading NLP Engine... (This takes a few seconds)")
nlp = NLPProcessor()
db = Prisma()

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
        # 1. Process Natural Language Query
        analysis = nlp.process_query(request.query)
        intent = analysis["intent"]
        entities = analysis["extracted_entities"]
        
        bot_response = ""
        db_data = None

        # 2. Map Intent to Database Query
        if intent == "LAST_TASK_ASSIGNED":
            query_args = {
                "order": {"createdAt": "desc"}, 
                "take": 1,
                "include": {"Project": True} # Fetch the actual Project details!
            }
            
            if entities["assignees"]:
                # Filter by developer name if extracted
                query_args["where"] = {
                    "assignee": {
                        "contains": entities["assignees"][0], 
                        "mode": "insensitive"
                    }
                }
                
            tasks = await db.task.find_many(**query_args)
            
            if tasks:
                t = tasks[0]
                
                # Check if a project is assigned, grab its 'name' column, otherwise use a fallback
                project_name = t.Project.name if t.Project else "Unassigned"
                
                # Updated to match your exact requested phrasing with the proper string variable
                bot_response = f"Last task assigned to {t.assignee} and task is '{t.title}' and project name is '{project_name}'."
                
                db_data = t.model_dump()
            else:
                bot_response = "I couldn't find any recent tasks matching that criteria."

        elif intent == "PROJECT_STATUS":
            total = await db.task.count()
            completed = await db.task.count(where={"status": "completed"})
            bot_response = f"Project Status: {completed} completed, {total - completed} pending."
            db_data = {"total": total, "completed": completed}

        elif intent == "CHECK_COMPLETED_WORK":
             query_args = {
                 "order": {"createdAt": "desc"},
                 "take": 5, 
                 "include": {"completedBy": True}
             }
             if entities["assignees"]:
                 query_args["where"] = {"completedBy": {"is": {"name": {"contains": entities["assignees"][0], "mode": "insensitive"}}}}

             completed_work = await db.workdone.find_many(**query_args)
             
             if completed_work:
                 latest = completed_work[0]
                 bot_response = f"The last task completed was '{latest.title}'."
                 db_data = [work.model_dump() for work in completed_work]
             else:
                 bot_response = "I couldn't find any recent completed work."
            
        else:
            bot_response = f"I understood your intent as '{intent}', but I haven't been programmed with a database query for that yet."

        # 3. Return final structured data
        return {
            "query": request.query,
            "intent": intent,
            "entities": entities,
            "response": bot_response,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)