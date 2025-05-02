# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Configuration ---
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_GOOGLE_AI_MODEL = os.getenv("DEFAULT_GOOGLE_AI_MODEL", "models/gemini-1.5-flash-latest")
DEFAULT_OPENAI_MODEL = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_OPENROUTER_MODEL = os.getenv("DEFAULT_OPENROUTER_MODEL", "qwen/qwen3-235b-a22b:free")

OPENROUTER_HEADERS = {
  "HTTP-Referer": os.getenv("YOUR_SITE_URL", "http://localhost"),
  "X-Title": os.getenv("YOUR_SITE_NAME", "AI Analyst Agent"),
}

# --- Search/DB Configuration ---
GOOGLE_API_KEY_SEARCH = os.getenv("GOOGLE_API_KEY_SEARCH")
GOOGLE_CX = os.getenv("GOOGLE_CX")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY")
# LINKUP_BASE_URL = os.getenv("LINKUP_BASE_URL", "https://api.linkup.so/v1")

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# --- Google Sheets Configuration ---
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
GCP_SERVICE_ACCOUNT_JSON_STR = os.getenv("GCP_SERVICE_ACCOUNT_JSON_STR")

print(f"--- Configuration Settings Loaded ---")
print(f"Google AI Key Found: {'Yes' if GOOGLE_AI_API_KEY else 'No'}")
print(f"OpenAI Key Found: {'Yes' if OPENAI_API_KEY else 'No'}")
print(f"OpenRouter Key Found: {'Yes' if OPENROUTER_API_KEY else 'No'}")
# Corrected f-string syntax
print(f"-> Default OpenAI Model (Local Override): {DEFAULT_OPENAI_MODEL}")
print(f"-> Default Google AI Model: {DEFAULT_GOOGLE_AI_MODEL}")
# Corrected f-string syntax
print(f"-> Default OpenRouter Model: {DEFAULT_OPENROUTER_MODEL}")
if not any([GOOGLE_AI_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY]): print("Warning: No LLM API Keys found.")

print(f"Google Search API Key Found: {'Yes' if GOOGLE_API_KEY_SEARCH else 'No'}")
print(f"Google Search CX ID Found: {'Yes' if GOOGLE_CX else 'No'}")
if not GOOGLE_API_KEY_SEARCH or not GOOGLE_CX: print("ERROR: Google Search API Key or CX ID missing. Google Search WILL fail.")

print(f"SerpApi Key Found: {'Yes' if SERPAPI_KEY else 'No'} (Used as backup or for Baidu)")
print(f"Neo4j URI Found: {'Yes' if NEO4J_URI else 'No'}")
print(f"Linkup API Key Found: {'Yes' if LINKUP_API_KEY else 'No'}")
if not LINKUP_API_KEY: print("Warning: Linkup API Key missing. Linkup searches WILL fail.")
print(f"Google Sheet ID Found: {'Yes' if GOOGLE_SHEET_ID else 'No'}")
print(f"GCP Service Account JSON Found: {'Yes' if GCP_SERVICE_ACCOUNT_JSON_STR else 'No'}")
print(f"--- End Configuration ---")