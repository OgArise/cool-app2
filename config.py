# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Configuration ---
# Prioritize OpenRouter Key if present
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Keep for potential fallback or direct use later

LLM_PROVIDER = "openrouter" if OPENROUTER_API_KEY else ("openai" if OPENAI_API_KEY else None)

# OpenRouter Settings
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# ===> CHANGE THIS LINE <===
OPENROUTER_MODEL_NAME = "google/gemini-2.5-pro-exp-03-25:free"
# ===> END OF CHANGE <===

# Recommended Headers for OpenRouter requests
# Replace with your actual site URL and App Name if desired
OPENROUTER_HEADERS = {
  "HTTP-Referer": os.getenv("YOUR_SITE_URL", "http://localhost"), # Or your deployed Streamlit URL
  "X-Title": os.getenv("YOUR_SITE_NAME", "AI Analyst Agent"), # Or your specific app name
}

# OpenAI Settings (if used directly)
OPENAI_MODEL_NAME = "gpt-4o" # Example fallback model

# --- Search/DB Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# --- Basic Validation ---
print(f"Using LLM Provider: {LLM_PROVIDER or 'None configured'}")
if LLM_PROVIDER == "openrouter":
     print(f"Using OpenRouter Model: {OPENROUTER_MODEL_NAME}")
elif LLM_PROVIDER == "openai":
     print(f"Using Direct OpenAI Model: {OPENAI_MODEL_NAME}")

if not LLM_PROVIDER:
    print("Warning: No LLM API Key found (OpenRouter or OpenAI). NLP functions will be skipped.")
if not SERPAPI_KEY:
    print("Warning: SERPAPI_KEY not found in .env (required for Baidu)")
if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
     print("Warning: Neo4j connection details missing. KG updates will be skipped.")
# Add more checks as needed