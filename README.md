# Grounded Data-Analyst Agent

I built a self-correcting, grounded data-analyst agent that answers natural-language business questions by querying a real relational database. Unlike standard single-shot LLM wrappers or RAG pipelines, this agent actively explores the database schema, writes and executes real SQL, observes the returned rows, and mathematically verifies its final answer to prevent hallucinations.

This project uses the Kaggle **Brazilian E-Commerce Public Dataset by Olist** loaded into a fully relational SQLite database (`olist.db`), ensuring the agent is forced to navigate a realistic, multi-table schema with actual foreign keys rather than a pre-joined flat file.

## Why This Exists (Business Framing)

Many enterprise analytics teams deploy LLM chatbots that attempt to answer data questions from memory or by matching patterns. These "naive" wrappers fail when faced with complex, multi-table schemas or when an initial SQL query returns an error or empty result set. Worse, they confidently hallucinate numbers when they don't know the answer.

I built this agent for a specific persona: an operations or business analyst who needs trustworthy, grounded answers to ad-hoc questions without needing to learn SQL or wait for a data engineering ticket. 

To prove its value, I evaluated the agent against a naive single-shot baseline using a 25-question benchmark with independently computed ground-truth answers. The results demonstrate exactly why an explicit `plan -> act -> observe -> decide` loop is necessary for production data tasks.

## Evaluation Results

I ran both the single-shot Baseline and the full self-correcting Agent against all 25 benchmark questions using **Gemini 2.5 Flash-Lite on Vertex AI**, not the Llama-3.3-70B this project was originally designed and tested against on Groq. Groq's free tier has a 100,000-token daily cap that an earlier full run exhausted, and Llama 3.3 70B's partner-model access on Vertex AI wasn't available in time for this run either. `llm_client.py` picks the provider from what's set in `.env`, so switching back to Llama on either Groq or Vertex is a config change, not a code change, once either has real headroom.

| Metric | Naive Baseline | Grounded Agent |
| :--- | :--- | :--- |
| **Task Success Rate** | 72.0% | **76.0%** |
| **Average Latency** | 2.79s (median 2.76s) | 3.98s (median 3.80s) |
| **Average Tool Calls** | 1.00 | 2.08 |
| **Hallucination Guard Trigger Rate** | 36.0% | 36.0% |
| **Cost, all 25 questions** | $0.0036 | $0.0075 |

### The Tradeoff

The agent is 4 percentage points more reliable than the baseline and takes about 1.4x longer to answer (3.98s vs 2.79s average), the real, measured cost of planning a step, executing it, waiting on the result, and only then answering, not a number picked to make the story work.

**Self-correction actually fired once, and I can point to exactly which question.** Question 16, "What is the total freight value paid by customers who left a review with a comment message?", the agent's first query referenced `o.freight_value` on the wrong table alias, got back `Error executing query: no such column: o.freight_value`, and revised the query to join `order_items` and reference `oi.freight_value` correctly on the next attempt. That's the genuine article: an error observed, a query revised, demonstrated in the raw transcript in `evaluation_log.json`, not asserted.

Getting an honest number for this took a real bug fix. The first pass at this metric reported "100% of questions triggered a retry", which should have been an obvious red flag: nothing near that many of these questions are actually hard. The bug was counting the agent's mandatory first step, calling `get_schema` to look at the database before writing any SQL, as if it were a retry, so every clean, zero-error run got miscounted as one. A second, subtler issue survived that first fix too: question 24 makes two successful queries in a row (compute a price percentile, then use it to filter), which isn't a retry either, it's the agent deliberately chaining a follow-up query off a good result, not recovering from anything. Both are now counted correctly in `agent.py`: only a query that comes back **after a real failure**, a SQL error or an explicit 0-row result, counts as a retry. On the 25-question set actually run, that's 1 question out of 25, 4%.

The honest read: this benchmark's questions weren't hard enough to exercise the self-correction path more than once. The mechanism is real and demonstrated, not simulated, but a benchmark specifically designed to break naive SQL generation more often (more ambiguous phrasing, more columns that don't match business language) would be a better test of it than this one turned out to be.

## Weak Spots & Limitations

I want to be transparent about what this agent *doesn't* do:
1. **Semantic SQL Verification**: The hallucination guard mathematically verifies that every number in the final answer exists in the raw tool output. However, it *does not* verify that the SQL query itself was semantically correct. If the agent writes a flawless query that calculates the wrong metric (e.g., averaging instead of summing), the guard will still pass the wrong answer because the number is "grounded" in the returned rows. Question 12 ("total revenue generated by sellers located in sao paulo") is a real, verified example: the agent's query joined `order_items` to `order_payments` without accounting for orders with more than one item, which fans out and inflates the summed payment total, a classic SQL join bug, not a hallucination. The guard correctly marked the answer as grounded (`is_grounded: True`), because it genuinely was, that number really did come out of the query, the query itself was just wrong. Baseline made the same class of mistake independently on this question with a different flawed query.
2. **Strict Guarding**: The hallucination guard is extremely strict. If the agent calculates a percentage or rounds a decimal in its final text that wasn't exactly printed in the SQL output, the guard will flag it as a hallucination.
3. **This run used Gemini 2.5 Flash-Lite, not the Llama-3.3-70B this agent was designed around**, see Evaluation Results above for why. It also means the numbers here include some unavoidable overhead specific to this model: Vertex AI wouldn't let me set its `reasoning_effort` below `"minimal"`, so every call, baseline and agent, carries a real hidden reasoning-token cost that a non-reasoning model like Llama doesn't have. The agent and baseline code are provider-agnostic through `llm_client.py`, so none of this is a property of the architecture, a Llama run would very plausibly land on different numbers.

## Project Structure

*   `data_loader.py`: Ingests the raw Olist CSVs into the relational `olist.db` SQLite database.
*   `tools.py`: Contains the sandboxed `get_schema()` and `execute_sql()` functions. Read-only limits are enforced.
*   `llm_client.py`: Builds the one shared LLM client both `agent.py` and `baseline.py` call through, picking Groq or Vertex AI based on what's set in `.env`.
*   `agent.py`: The core `plan -> act -> observe` loop.
*   `baseline.py`: The naive single-shot comparator.
*   `verification.py`: The numeric hallucination guard that extracts and matches exact numbers.
*   `evaluate.py`: The benchmark harness that grades the agent against `benchmark_qa.json`.
*   `app.py`: A Streamlit UI that exposes the agent's step-by-step reasoning trace.

## Getting Started

1. Install dependencies: `pip install -r requirements.txt` (or install manually: `pandas`, `streamlit`, `python-dotenv`, `openai`, `kagglehub`).
2. Run `python3 data_loader.py` to fetch the data and build the database.
3. Pick a provider in `.env`:
   - **Groq (Llama-3.3-70B-Versatile)**: set `GROQ_API_KEY`. This is the model the agent was originally designed around.
   - **Vertex AI (Gemini 2.5 Flash-Lite)**: set `GCP_PROJECT_ID` and `GCP_REGION`, and have `gcloud auth login` plus `gcloud auth application-default login` already done. This is what the committed evaluation results in this repo were run against.
4. Run the UI trace: `streamlit run app.py`, or the full benchmark: `python3 evaluate.py`.

## License

MIT
