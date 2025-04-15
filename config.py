import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Basic validation
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in .env")
if not SERPAPI_KEY:
    print("Warning: SERPAPI_KEY not found in .env (required for Baidu)")
# Add more checks as needed