"""
Builds the single shared LLM client both agent.py and baseline.py call through.

Two providers are supported, picked by what's set in .env:
- GCP_PROJECT_ID set: Vertex AI, serving Gemini 2.5 Flash-Lite through its
  OpenAI-compatible endpoint. Auth is a short-lived gcloud access token fetched
  once at import time, not a static API key, so this path needs `gcloud auth
  login` already done on the machine running it.
- Otherwise: Groq, serving Llama-3.3-70B-Versatile, the original setup.

Kept as one module so the provider only gets picked in one place instead of
duplicated across agent.py and baseline.py.
"""

import os
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")

if GCP_PROJECT_ID:
    PROVIDER = "vertex"
    MODEL_NAME = os.getenv("MODEL_NAME", "google/gemini-2.5-flash-lite")
    _access_token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"]
    ).decode().strip()
    client = OpenAI(
        api_key=_access_token,
        base_url=(
            f"https://{GCP_REGION}-aiplatform.googleapis.com/v1beta1/"
            f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}/endpoints/openapi"
        ),
        max_retries=0
    )
    # Vertex's OpenAI-compat layer for 2.5-series Gemini models does not accept
    # "none", "minimal" is the lowest it allows and still carries a real,
    # unavoidable reasoning-token cost, confirmed by testing before this run:
    # a trivial tool call still burned dozens of hidden reasoning tokens even
    # at this setting. Documented as a known limitation in the README rather
    # than presented as if reasoning were fully off.
    EXTRA_COMPLETION_KWARGS = {"reasoning_effort": "minimal"}
    # Per Google's published Vertex AI pricing, checked before this run.
    PRICE_PER_MILLION_INPUT_TOKENS = 0.10
    PRICE_PER_MILLION_OUTPUT_TOKENS = 0.40
else:
    PROVIDER = "groq"
    MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    # max_retries=0: the SDK's default backoff on a 429 can silently block for
    # several minutes before raising, which corrupts latency numbers and hides
    # what actually happened. Fail fast and let the caller decide what to do.
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        max_retries=0
    )
    EXTRA_COMPLETION_KWARGS = {}
    PRICE_PER_MILLION_INPUT_TOKENS = 0.0
    PRICE_PER_MILLION_OUTPUT_TOKENS = 0.0
