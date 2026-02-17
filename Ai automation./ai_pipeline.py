import ollama

def generate_client_summary(technical_log: str) -> str:
    """
    Takes raw technical text and uses Llama 3.2 1B to generate a strict, formal client summary.
    """
    system_prompt = """You are a strict technical-to-client translator. Your ONLY output must be the final 2-3 sentence client summary. Do not output anything else.

<rules>
1. NEVER output conversational text like "Here is the summary", "Alternatively", or "Here is the reformatted text".
2. NEVER provide multiple options. Provide exactly ONE paragraph.
3. Sentence 1: Translate the technical action into terms of system optimization, maintainability, and efficiency.
4. Sentence 2: Translate the testing/deployment into terms of stability and seamless integration.
</rules>

<example_1>
Input: Cleaned up the spaghetti code in the authentication module and wrote unit tests for better coverage.
Output: The authentication module underwent significant restructuring to enhance its overall efficiency, maintainability, and scalability. A thorough testing process was implemented to validate the updated logic, ensuring a highly reliable and seamless integration.
</example_1>

<example_2>
Input: Refactored the backend cron jobs to reduce latency by 40% on the AWS Lambda instances.
Output: The implementation of optimized background processes has yielded a significant improvement in system performance, resulting in a 40% decrease in response times. This architectural enhancement has streamlined system operations, leading to increased overall efficiency and reliability.
</example_2>
"""
    
    try:
        response = ollama.chat(model='llama3.2:1b', messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': f"Input: {technical_log}\nOutput:",
            },
        ])
        
        # Clean up the response just in case there's trailing whitespace
        return response['message']['content'].strip()

    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return "Error: Unable to generate summary. Please contact the administrator."