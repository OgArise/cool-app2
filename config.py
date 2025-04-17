# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Configuration ---
# Load ALL potential keys from environment
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Add other keys if needed (e.g., ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY"))

# Define standard base URLs / default models (can be overridden by user input later)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_GOOGLE_AI_MODEL = "models/gemini-1.5-flash-latest"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OPENROUTER_MODEL = "google/gemini-flash-1.5" # Example

# OpenRouter Headers (can still be configured via env vars if desired)
OPENROUTER_HEADERS = {
  "HTTP-Referer": os.getenv("YOUR_SITE_URL", "http://localhost"),
  "X-Title": os.getenv("YOUR_SITE_NAME", "AI Analyst Agent"),
}

# --- Search/DB Configuration ---
GOOGLE_API_KEY_SEARCH = os.getenv("GOOGLE_API_KEY") # For Google Search API
GOOGLE_CX = os.getenv("GOOGLE_CX")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# --- Google Sheets Configuration ---
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
GCP_SERVICE_ACCOUNT_JSON_STR = os.getenv("GCP_SERVICE_ACCOUNT_JSON_STR")

# --- Basic Validation & Info Logging ---
print(f"--- Configuration Settings Loaded ---")
print(f"Google AI Key Found: {'Yes' if GOOGLE_AI_API_KEY else 'No'}")
print(f"OpenAI Key Found: {'Yes' if OPENAI_API_KEY else 'No'}")
print(f"OpenRouter Key Found: {'Yes' if OPENROUTER_API_KEY else 'No'}")
# ... add logs for other keys ...

# Minimal validation - more checks could be added
if not any([GOOGLE_AI_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY]):
    print("Warning: No LLM API Keys found. NLP functions will be skipped.")
# ... (other validation messages for Search, Neo4j, Sheets) ...
if not SERPAPI_KEY: print("Warning: SERPAPI_KEY not found in .env (required for Baidu)")
if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD: print("Warning: Neo4j connection details missing.")
if not GOOGLE_SHEET_ID: print("Warning: GOOGLE_SHEET_ID not found. Results won't be saved to Google Sheets.")
if not GCP_SERVICE_ACCOUNT_JSON_STR: print("Warning: GCP_SERVICE_ACCOUNT_JSON_STR not found. Cannot save to Google Sheets.")
print(f"--- End Configuration ---")