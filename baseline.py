import time
from openai import RateLimitError
from llm_client import client, MODEL_NAME, EXTRA_COMPLETION_KWARGS
from tools import get_schema, execute_sql

def run_baseline(question: str) -> dict:
    start_time = time.time()
    schema = get_schema()
    
    system_instruction = f"""
You are a data analyst answering business questions based on an SQLite database.
Here is the database schema:
{schema}

Write a SQL query to answer the question. Return ONLY the raw SQL query, nothing else. Do not use any markdown formatting, backticks, or explanatory text.
"""
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": question}
    ]
    
    steps = []
    input_tokens = 0
    output_tokens = 0
    final_answer = "Error: Did not complete"
    error_type = None

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            **EXTRA_COMPLETION_KWARGS
        )

        if response.usage:
            input_tokens += response.usage.prompt_tokens
            output_tokens += response.usage.completion_tokens
            
        message = response.choices[0].message
        query = message.content.strip()
        
        # Strip potential markdown code blocks if the model ignores the instruction
        if query.startswith("```sql"):
            query = query[6:]
        elif query.startswith("```"):
            query = query[3:]
        if query.endswith("```"):
            query = query[:-3]
        query = query.strip()
        
        steps.append({"type": "tool_call", "query": query})
        
        tool_result = execute_sql(query)
        steps.append({"type": "tool_result", "result": tool_result})
        
        messages.append({"role": "assistant", "content": query})
        messages.append({
            "role": "user", 
            "content": f"Here is the result of the query:\n{tool_result}\n\nNow provide your final natural language answer to the original question. Do not include the SQL query in your response, just the final answer."
        })
        
        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            **EXTRA_COMPLETION_KWARGS
        )

        if final_response.usage:
            input_tokens += final_response.usage.prompt_tokens
            output_tokens += final_response.usage.completion_tokens
            
        final_answer = final_response.choices[0].message.content
            
    except RateLimitError as e:
        final_answer = f"Error during baseline execution: {e}"
        error_type = "rate_limited"
    except Exception as e:
        final_answer = f"Error during baseline execution: {e}"
        error_type = "other"

    end_time = time.time()

    return {
        "question": question,
        "final_answer": final_answer,
        "steps": steps,
        "latency_seconds": round(end_time - start_time, 2),
        "tool_calls_count": 1 if any(s["type"] == "tool_call" for s in steps) else 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error_type": error_type
    }

if __name__ == "__main__":
    q = "How many total orders are there in the dataset?"
    res = run_baseline(q)
    print(res["final_answer"])
