import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
class UserInput(BaseModel):
    user_text: str 
@app.post("/generate-response/")
async def generate_response(input_data: UserInput):
    print(f"📥 Received Input: {input_data.user_text}")
    try:
        response = ollama.chat(model='llama3', messages=[
            {
                'role': 'system',
                'content': 'You are a helpful AI assistant. Rewrite the user input to be professional and concise.'
            },
            {
                'role': 'user',
                'content': input_data.user_text
            },
        ])
        generated_text = response['message']['content']
        
        print("✅ Response Generated")
        return {"response": generated_text}

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))