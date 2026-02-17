from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_pipeline import generate_client_summary 
import random # Used just to generate a fake update_id for the example

app = FastAPI()

# --- CORS SETUP ---
# Crucial for allowing your Arinova Studio frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REQUEST SCHEMA ---
class DailyLogRequest(BaseModel):
    project_id: int
    employee_id: int
    technical_summary: str

# --- ENDPOINTS ---

@app.post("/submit-log/")
async def submit_daily_log(log: DailyLogRequest):
    """
    Receives the raw log from the frontend, runs it through the 1B model, 
    and returns the client-friendly translation.
    """
    if not log.technical_summary:
        raise HTTPException(status_code=400, detail="Log cannot be empty.")

    # 1. Send data to the LLM Pipeline
    client_friendly_text = generate_client_summary(log.technical_summary)
    
    # 2. Simulate saving to a database and getting an ID back
    new_update_id = random.randint(1000, 9999)

    # 3. Return the exact JSON structure your frontend expects
    return {
        "update_id": new_update_id,
        "original_log": log.technical_summary,
        "generated_client_summary": client_friendly_text,
        "message": "Draft created. Please confirm to publish."
    }

@app.post("/confirm-log/{update_id}")
async def confirm_log(update_id: int):
    """
    The frontend calls this when the user clicks 'Confirm & Clock Out'.
    """
    # In a real app, you would UPDATE your database here to mark visibility = 'Client-Facing'
    return {"status": "published", "message": f"Update {update_id} is now live for the client."}
    