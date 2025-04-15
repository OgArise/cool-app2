# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_PROVIDER = "openrouter" if OPENROUTER_API_KEY else ("openai" if OPENAI_API_KEY else None)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "google/gemini-2.5-pro-exp-03-25:free") # Allow override via env
OPENROUTER_HEADERS = {
  "HTTP-Referer": os.getenv("YOUR_SITE_URL", "http://localhost"),
  "X-Title": os.getenv("YOUR_SITE_NAME", "AI Analyst Agent"),
}
OPENAI_MODEL_NAME = "gpt-4o"

# --- Search/DB Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# --- Google Sheets Configuration ---
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID") # Get Sheet ID from .env
# Get Service Account JSON content from env var (for Render security)
GCP_SERVICE_ACCOUNT_JSON_STR = os.getenv("GCP_SERVICE_ACCOUNT_JSON_STR")

# --- Basic Validation ---
print(f"Using LLM Provider: {LLM_PROVIDER or 'None configured'}")
# ... (other validation messages) ...
if not GOOGLE_SHEET_ID:
    print("Warning: GOOGLE_SHEET_ID not found in .env. Results will not be saved to Google Sheets.")
if not GCP_SERVICE_ACCOUNT_JSON_STR:
     print("Warning: GCP_SERVICE_ACCOUNT_JSON_STR not found in environment variables. Cannot authenticate to Google Sheets.")