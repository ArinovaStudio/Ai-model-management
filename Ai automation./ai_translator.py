import ollama
def generate_client_summary(technical_log: str) -> str:
    system_prompt = """
    Role: You are an expert Client Relationship Manager.
    Task: Rewrite the following internal technical update into a non-technical, professional summary.
    Rules: 
    1. Simplify technical terms (e.g., "Refactored API" -> "Improved system speed").
    2. Keep tone professional and reassuring.
    3. Remove sensitive data like IPs, passwords, or specific variable names.
    4. Keep it concise (1-2 sentences).
    5. Do NOT include introductory filler like "Here is a summary". Just give the summary.
    """
    try:
        response = ollama.chat(model='llama3', messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': f"Technical Update: {technical_log}",
            },
        ])
        return response['message']['content'].strip()

    except Exception as e:
        return f"Error connecting to Local Llama: {str(e)}"
if __name__ == "__main__":
    print("Testing Local Llama connection...")
    test_log = "Fixed null pointer exception in auth flow and patched JWT token vulnerability."
    
    result = generate_client_summary(test_log)
    
    print("\n--- RESULTS ---")
    print(f"Input: {test_log}")
    print(f"Llama Output: {result}")