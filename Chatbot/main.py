from fastapi import FastAPI

# Initialize the FastAPI application
app = FastAPI(title="Employee Management Chatbot")

# Create a simple health-check endpoint
@app.get("/health")
async def health_check():
    return {"status": "success", "message": "The local server is running properly."}
