# Progress Tracker

## Done
- Set up `.gitignore` to exclude `.env`, `.venv`, `data/raw`, and `olist.db`.
- Built `data_loader.py`, `tools.py` (get_schema/execute_sql, sandboxed read-only), `verification.py`
  (hallucination guard), `baseline.py` (naive single-shot comparator), `agent.py`
  (plan/act/observe loop with self-correction), `evaluate.py` (benchmark harness), `app.py`
  (Streamlit reasoning-trace UI).
- Wrote and independently ground-truthed 25 benchmark questions in `benchmark_qa.json` via
  `populate_benchmark.py`, computed directly against `olist.db`, not by reusing the agent's own
  queries.
- Fixed a real self-correction gap: the agent had no handling for a malformed tool call
  (the model writing `<function=...>` as text instead of a real structured call), which used to
  die with zero retries. Now caught and retried up to 3 times with a correction nudge.
- Fixed the OpenAI SDK's default retry-with-backoff silently blocking for minutes on a 429, root
  cause of an earlier 465-second latency outlier. Both agent and baseline clients now set
  `max_retries=0`.
- Fixed `evaluate.py` to stop cleanly on a real rate limit instead of grinding through the rest
  of the benchmark producing zero-token junk rows, and to never score an error string's stray
  digits as if they were a real answer.
- Built `llm_client.py`, one shared client both `agent.py` and `baseline.py` import, picking Groq
  or Vertex AI from what's set in `.env` instead of duplicating provider logic in both files.
- Fixed a retry-counting bug in `agent.py`, twice. First pass: the mandatory one-time
  `get_schema` call was being counted as a retry on every question. Second pass, after that fix
  still didn't look right: a second successful query chained deliberately off a good result (no
  error at all) was also getting counted as a retry. Now only counts a query that comes back
  after a genuine failure, a SQL error or an explicit 0-row result.
- Fixed `evaluate.py` claiming "$0.00 (Groq Free Tier)" regardless of which provider actually
  ran. Cost is now computed from real per-model pricing in `llm_client.py`.
- Ran the full, clean 25-question benchmark on Gemini 2.5 Flash-Lite via Vertex AI (Groq's daily
  cap was exhausted from an earlier run, and Llama 3.3 70B's Vertex AI partner-model access
  wasn't live yet). All 25 questions attempted, no rate limiting. Baseline 72.0% success, agent
  76.0%, exactly 1 genuine self-correction (question 16), real cost $0.0111 total.
- README rewritten to match this real run: correct results table, the retry-metric bug and fix
  explained with the real numbers, the model substitution disclosed plainly, a verified example
  of the "grounded but semantically wrong" limitation (question 12's join fan-out bug), and the
  reasoning-effort caveat.
- Independently re-verified every number in the README against `evaluation_log.json` from
  scratch, including recomputing the hallucination guard trigger rate with the real
  `verify_grounding()` function since baseline's grounding result wasn't even persisted in the
  saved log. Also ran `app.py` for real (`streamlit run`, not just an import check), confirmed it
  serves.
- Fixed `requirements.txt`: `numpy==2.0.2` has no wheel for Python 3.13 and fails to build from
  source on this machine. Checked first that numpy is never directly imported anywhere in this
  project's own code, it's a transitive dependency of pandas and streamlit only, so relaxing to
  `numpy>=2.0.2` is safe. Verified on a completely fresh venv installing straight from the fixed
  file, not just patched and assumed.

## Pending
- Llama 3.3 70B access on Vertex AI was enabled in the console but is still returning 404 as of
  the last check, tried several model-name variations to rule out a naming issue, none worked.
  Might just need propagation time, or the enablement happened on the wrong project. If it comes
  through later, rerunning the benchmark on Llama is a five-minute job, `llm_client.py` already
  supports both providers, this is a config change, not a code change.
- Not pushed yet.
