# main.py
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prisma import Prisma
import uvicorn

# Imports from our new modular files
from nlp_engine import NLP
from functions import handle_query
from utils import get_html_ui

app = FastAPI(title="AI Office Text Chatbot")
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

print("Booting NLP Engine and Prisma...")
nlp = NLP()
db = Prisma()

@app.on_event("startup")
async def start_db(): 
    await db.connect()

@app.on_event("shutdown")
async def stop_db(): 
    await db.disconnect()

@app.get("/")
async def get_ui(): 
    return HTMLResponse(get_html_ui())

# ==========================================
# ENDPOINTS
# ==========================================
class ChatRequest(BaseModel): 
    query: str

http_ctx = {"last_tgt": None}

@app.post("/api/chat")
async def chat_api(req: ChatRequest):
    # Pass db and nlp into the function so it has access to them
    reply = await handle_query(req.query, http_ctx, db, nlp)
    return {"reply": reply}

@app.websocket("/ws/chat")
async def chat_socket(ws: WebSocket):
    await ws.accept()
    ws_ctx = {"last_tgt": None}
    try:
        while True:
            data = json.loads(await ws.receive_text())
            if "query" in data:
                # Pass db and nlp into the function
                reply = await handle_query(data["query"], ws_ctx, db, nlp)
                await ws.send_json({"reply": reply})
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"WebSocket Error: {e}")

if __name__ == "__main__":
    # Ensure this points to main:app
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)