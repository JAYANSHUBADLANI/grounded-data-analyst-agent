import time
import json
from openai import BadRequestError, RateLimitError
from llm_client import client, MODEL_NAME, EXTRA_COMPLETION_KWARGS
from tools import get_schema, execute_sql

MAX_STEPS = 10
MAX_MALFORMED_TOOL_CALL_RETRIES = 3

EXECUTE_SQL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": "Executes a read-only SQL query against the SQLite database.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The SQL SELECT query to execute"
                }
            },
            "required": ["query"]
        }
    }
}

GET_SCHEMA_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_schema",
        "description": "Gets the schema of the SQLite database.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}

def run_agent(question: str) -> dict:
    start_time = time.time()
    
    system_instruction = """
You are an expert data analyst with access to an SQLite database.
You have two tools: get_schema() to inspect the database schema, and execute_sql(query) to run SQL. Start by calling get_schema to understand what tables and columns exist.

INSTRUCTIONS:
1. When asked a question, first explain your plan (what tables you will query, how they join).
2. Write and execute a SQL query using the `execute_sql` tool.
3. Observe the output.
   - If there is a SQL error (e.g. column not found), revise your query and try again.
   - If the query returns 0 rows, check if your filters or joins are correct and try again.
4. Once you have the data needed, provide a clear, concise final natural language answer. Ensure every number in your final answer is exactly derived from the tool output. Do not make up numbers.

You may make multiple tool calls if needed, up to a limit.
"""
    
    messages = [
        {"role": "system", "content": system_instruction.strip()},
        {"role": "user", "content": question}
    ]
    
    steps = []
    input_tokens = 0
    output_tokens = 0
    final_answer = None
    error_type = None
    malformed_tool_call_retries = 0

    for step_count in range(MAX_STEPS):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=[GET_SCHEMA_TOOL_SCHEMA, EXECUTE_SQL_TOOL_SCHEMA],
                temperature=0.0,
                **EXTRA_COMPLETION_KWARGS
            )
            
            if response.usage:
                input_tokens += response.usage.prompt_tokens
                output_tokens += response.usage.completion_tokens
                
            message = response.choices[0].message
            messages.append(message)
            
            if message.content and message.content.strip():
                steps.append({"type": "thought", "content": message.content.strip()})
                
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                
                if tool_call.function.name == "get_schema":
                    steps.append({"type": "tool_call", "tool_name": "get_schema"})
                    tool_result = get_schema()
                    steps.append({"type": "tool_result", "result": tool_result})
                elif tool_call.function.name == "execute_sql":
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query", "")
                    steps.append({"type": "tool_call", "query": query})
                    tool_result = execute_sql(query)
                    steps.append({"type": "tool_result", "result": tool_result})
                else:
                    tool_result = f"Unknown tool: {tool_call.function.name}"
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": str(tool_result)
                })
            else:
                final_answer = message.content
                break
                
        except BadRequestError as e:
            body = getattr(e, "body", None) or {}
            code = (body.get("error") or {}).get("code") if isinstance(body, dict) else None

            if code == "tool_use_failed" and malformed_tool_call_retries < MAX_MALFORMED_TOOL_CALL_RETRIES:
                # The model wrote its tool call as text instead of using the real
                # function-calling protocol. This is exactly the kind of failure the
                # self-correction loop should recover from, not die on: nudge it and
                # let it try again on the next iteration, same as a bad SQL result.
                malformed_tool_call_retries += 1
                steps.append({"type": "tool_call_error", "error": str(e)})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your last response was not a valid tool call. You must invoke "
                        "get_schema or execute_sql through the function calling mechanism, "
                        "not by writing the call out as text. Please try again."
                    )
                })
                continue

            final_answer = f"Error during agent execution: {e}"
            steps.append({"type": "error", "error": str(e)})
            error_type = "malformed_tool_call" if code == "tool_use_failed" else "bad_request"
            break
        except RateLimitError as e:
            # Not retryable within a single run, Groq's own message says wait several
            # minutes. Stop immediately and tag it distinctly so the caller can exclude
            # it from scoring instead of grading a rate limit as a reasoning failure.
            final_answer = f"Error during agent execution: {e}"
            steps.append({"type": "error", "error": str(e)})
            error_type = "rate_limited"
            break
        except Exception as e:
            final_answer = f"Error during agent execution: {e}"
            steps.append({"type": "error", "error": str(e)})
            error_type = "other"
            break

    if final_answer is None:
        final_answer = "Error: Agent reached maximum steps without providing a final answer."
        
    end_time = time.time()

    tool_calls = sum(1 for s in steps if s["type"] == "tool_call")

    # A retry means: an execute_sql call happened right after a PRIOR execute_sql
    # call actually failed (a SQL error, or a query that came back with 0 rows).
    # Two prior, cruder attempts at this both overcounted: max(0, tool_calls - 1)
    # counted the single mandatory get_schema call as a retry on every question,
    # and max(0, execute_sql_calls - 1) still counted a second execute_sql call as
    # a "retry" even when the first one succeeded and the second was the model
    # deliberately chaining a follow-up query off a good result, not recovering
    # from anything. Only count it when the immediately preceding execute_sql
    # result was a genuine failure.
    sql_call_indices = [i for i, s in enumerate(steps) if s["type"] == "tool_call" and "query" in s]
    retries = 0
    for j in range(1, len(sql_call_indices)):
        prev_result_index = sql_call_indices[j - 1] + 1
        prev_result = steps[prev_result_index].get("result", "") if prev_result_index < len(steps) else ""
        if str(prev_result).startswith("Error") or "returned 0 rows" in str(prev_result):
            retries += 1
    retries += sum(1 for s in steps if s["type"] == "tool_call_error")

    return {
        "question": question,
        "final_answer": final_answer,
        "steps": steps,
        "latency_seconds": round(end_time - start_time, 2),
        "tool_calls_count": tool_calls,
        "retry_count": retries,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error_type": error_type
    }

if __name__ == "__main__":
    q = "What is the most popular product category by order count?"
    res = run_agent(q)
    print(res["final_answer"])
