from fastapi import FastAPI

app = FastAPI(title="Employee Management Chatbot")


@app.get("/health")
async def health_check():
    return {"status": "success", "message": "The local server is running properly."}
