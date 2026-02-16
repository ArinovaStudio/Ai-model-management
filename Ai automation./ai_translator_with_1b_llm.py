import ollama

def generate_client_summary(technical_log: str) -> str:
    # UPDATED PROMPT: Bulletproof strict constraints for 1B models
    system_prompt = """
    You are a strict technical-to-client translator. Convert the developer log into a professional, jargon-free summary (2-3 sentences) explaining the business value.

    CRITICAL RULES:
    1. NO CODE. Never write, generate, or output code blocks. Plain text only.
    2. NO CHAT. Never use greetings, conversational filler, or intro text. 
    3. NO INVENTING. Stick exactly to the facts provided. Do not invent metrics or technologies.
    4. EXPLAIN VALUE. Translate technical jargon (like "refactoring" or "unit tests") into terms of stability, security, or performance.

    EXAMPLES:
    User: Cleaned up the spaghetti code in the authentication module and wrote unit tests for better coverage.
    Assistant: We restructured the underlying code in the login system to meet current best practices. We also added automated quality checks to ensure the login process remains highly stable and secure against future updates.

    User: Refactored the backend cron jobs to reduce latency by 40% on the AWS Lambda instances.
    Assistant: We optimized the background processes responsible for handling scheduled tasks. This architectural improvement reduces system latency by 40%, ensuring a noticeably faster and more responsive application.
    """
    
    try:
        response = ollama.chat(model='llama3.2:1b', messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': technical_log,
            },
        ])
        return response['message']['content'].strip()

    except Exception as e:
        return f"Error connecting to Local Llama: {str(e)}"

if __name__ == "__main__":
    print("--------------------------------------------------")
    print("   AI Client Summary Generator (No-Code Mode)     ")
    print("   Type 'exit' or 'quit' to stop the program.     ")
    print("--------------------------------------------------")

    while True:
        test_log = input("\n📝 Enter Technical Log: ")
        
        if test_log.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        if not test_log.strip():
            print("Please enter some text.")
            continue

        print("⚡ Translating log...")
        
        result = generate_client_summary(test_log)
        
        print(f"\n📢 Client Summary:\n{result}")
        print("-" * 50)