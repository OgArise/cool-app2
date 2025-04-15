import os
import time
import requests
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from googleapiclient.discovery import build as build_google_service
from serpapi import GoogleSearch as SerpApiSearch # Renamed for clarity

# --- Environment & Config ---
load_dotenv()
app = FastAPI(
    title="AI Analyst Search API Service",
    description="Provides unified search capabilities for n8n.",
    version="1.0.0"
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY") # Optional fallback

# --- Request Models ---
class SearchRequest(BaseModel):
    query: str
    language_code: Optional[str] = 'en' # Google language code (e.g., 'en', 'zh-CN')
    country_code: Optional[str] = 'us' # SerpApi country code (e.g., 'us', 'cn')
    max_results: Optional[int] = 10
    search_provider: str = Field("google", description="Which provider to use: 'google', 'baidu', 'free'")
    # Add other params like date range if needed by underlying APIs

class WaybackRequest(BaseModel):
    url: str

# --- Helper Functions (Adapted from Streamlit Tester) ---

def format_results(items: List[Dict[str, Any]], source: str, language: str) -> List[Dict[str, Any]]:
    """Standardizes results from different sources."""
    results = []
    for item in items:
        # --- Try extracting dates (adapt as needed per source) ---
        published_date = item.get("published_date") # Assume pre-formatted if exists
        if not published_date and source == "google_api":
             try:
                metatags = item.get("pagemap", {}).get("metatags", [{}])[0]
                date_str = metatags.get("article:published_time") or metatags.get("og:published_time") or metatags.get("publication_date") or metatags.get("date")
                if date_str:
                    published_date = date_str.split('T')[0] # Basic split, might need better parsing
             except Exception: pass
        elif not published_date and source == "serpapi":
            # SerpApi sometimes has 'date' field directly
             published_date = item.get("date")

        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", item.get("url")), # Google uses 'link', SerpApi uses 'link' or 'url'
            "snippet": item.get("snippet", item.get("description")), # Google uses 'snippet', SerpApi often 'snippet'
            "source": source,
            "language": language,
            "published_date": published_date,
            "raw_data": item # Include raw data for potential downstream use
        })
    return results

def search_google_api(query: str, language: str, max_results: int, **kwargs) -> Dict[str, Any]:
    """Search using Google Custom Search JSON API."""
    start_time = time.time()
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        raise HTTPException(status_code=400, detail="Google API Key/CX not configured in API service.")

    try:
        service = build_google_service("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        # Map language_code to Google's 'lr' parameter format if needed
        # Google uses 'lr': 'lang_xx', e.g. 'lang_en', 'lang_zh-CN'
        lr_param = f"lang_{language}" if language else None

        api_results = service.cse().list(
            q=query,
            cx=GOOGLE_CX,
            num=min(max_results, 10), # Google API max 10 per request page
            lr=lr_param
            # Add other params like 'dateRestrict' if needed: dateRestrict='d[number]' (days), 'w[number]', 'm[number]'
        ).execute()

        items = api_results.get("items", [])
        formatted = format_results(items, "google_api", language)
        processing_time = time.time() - start_time
        return {
            "status": "success",
            "query": query,
            "results_count": len(formatted),
            "processing_time_seconds": round(processing_time, 2),
            "results": formatted
        }
    except Exception as e:
        # Log the error ideally
        print(f"Google API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Google API search failed: {str(e)}")

def search_baidu_via_serpapi(query: str, country_code: str, max_results: int, **kwargs) -> Dict[str, Any]:
    """Search Baidu using SerpApi."""
    start_time = time.time()
    if not SERPAPI_KEY:
        raise HTTPException(status_code=400, detail="SerpApi Key not configured in API service.")

    try:
        params = {
            "engine": "baidu",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": max_results, # SerpApi generally respects num
            "cc": country_code if country_code else "cn", # Default to China for Baidu
             # Add other SerpApi params like 'date' if needed
        }
        search = SerpApiSearch(params)
        api_results = search.get_dict()

        # Check for SerpApi errors
        if "error" in api_results:
             raise HTTPException(status_code=500, detail=f"SerpApi Baidu Error: {api_results['error']}")

        items = api_results.get("organic_results", [])
        formatted = format_results(items, "serpapi_baidu", country_code) # Use country as language proxy
        processing_time = time.time() - start_time
        return {
            "status": "success",
            "query": query,
            "results_count": len(formatted),
            "processing_time_seconds": round(processing_time, 2),
            "results": formatted
        }
    except Exception as e:
        print(f"SerpApi Baidu Error: {e}")
        raise HTTPException(status_code=500, detail=f"SerpApi Baidu search failed: {str(e)}")

def search_google_via_serpapi(query: str, country_code: str, language_code: str, max_results: int, **kwargs) -> Dict[str, Any]:
    """Search Google using SerpApi (as an alternative or for specific locations)."""
    start_time = time.time()
    if not SERPAPI_KEY:
        raise HTTPException(status_code=400, detail="SerpApi Key not configured in API service.")

    try:
        params = {
            "engine": "google",
            "q": query,
            "google_domain": f"google.{country_code}", # Adjust domain if needed
            "gl": country_code, # Country code
            "hl": language_code, # Language code
            "api_key": SERPAPI_KEY,
            "num": max_results,
             # Add other SerpApi params like 'tbs' for date ranges (e.g., tbs='qdr:m' for past month)
        }
        search = SerpApiSearch(params)
        api_results = search.get_dict()

        if "error" in api_results:
             raise HTTPException(status_code=500, detail=f"SerpApi Google Error: {api_results['error']}")

        items = api_results.get("organic_results", [])
        formatted = format_results(items, "serpapi_google", language_code)
        processing_time = time.time() - start_time
        return {
            "status": "success",
            "query": query,
            "results_count": len(formatted),
            "processing_time_seconds": round(processing_time, 2),
            "results": formatted
        }
    except Exception as e:
        print(f"SerpApi Google Error: {e}")
        raise HTTPException(status_code=500, detail=f"SerpApi Google search failed: {str(e)}")


def search_free_api(query: str, language: str, max_results: int, **kwargs) -> Dict[str, Any]:
    """Search using DuckDuckGo Zero Click Info API via RapidAPI (Fallback)."""
    start_time = time.time()
    if not RAPIDAPI_KEY:
        return search_fallback(query, language, max_results) # Go straight to mock if no key

    try:
        url = "https://duckduckgo-duckduckgo-zero-click-info.p.rapidapi.com/"
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "duckduckgo-duckduckgo-zero-click-info.p.rapidapi.com"
        }
        # Note: DDG API doesn't reliably support language/region filtering or max_results
        params = {"q": query, "format": "json", "no_html": 1}
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()
        api_results = []
        # Basic parsing - adapt based on DDG API structure if needed
        if data.get("AbstractText"):
            api_results.append({"title": data.get("Heading"), "link": data.get("AbstractURL"), "snippet": data.get("AbstractText")})
        for topic in data.get("RelatedTopics", []):
             if topic.get("Result") and len(api_results) < max_results:
                # Simplistic parsing, may need html cleaning (bs4) if no_html=0
                 link = topic.get("FirstURL")
                 text = topic.get("Text") # Title might be embedded here
                 snippet = topic.get("Result")[:200] + "..." # Very basic snippet
                 api_results.append({"title": text, "link": link, "snippet": snippet})

        formatted = format_results(api_results, "duckduckgo", language)
        processing_time = time.time() - start_time
        return {
            "status": "success",
            "query": query,
            "results_count": len(formatted),
            "processing_time_seconds": round(processing_time, 2),
            "results": formatted[:max_results] # Trim results here
        }

    except Exception as e:
        print(f"Free API (DDG) Error: {e}")
        return search_fallback(query, language, max_results) # Use mock on error


def search_fallback(query: str, language: str, max_results: int, **kwargs) -> Dict[str, Any]:
    """Generates mock results if other searches fail or aren't configured."""
    start_time = time.time()
    results = []
    for i in range(min(max_results, 5)): # Generate fewer mock results
        results.append({
            "title": f"Mock Result {i+1} for '{query}'",
            "url": f"https://example.com/mock/{i+1}",
            "snippet": f"This is mock result {i+1}. Source API unavailable or errored.",
            "source": "mock",
            "language": language,
            "published_date": None
        })
    processing_time = time.time() - start_time
    return {
        "status": "success_fallback",
        "query": query,
        "results_count": len(results),
        "processing_time_seconds": round(processing_time, 2),
        "results": results
    }

def search_wayback_machine(url_to_check: str) -> Dict[str, Any]:
    """Check Wayback Machine Availability API."""
    start_time = time.time()
    api_url = f"https://archive.org/wayback/available?url={requests.utils.quote(url_to_check)}"
    try:
        response = requests.get(api_url, timeout=15)
        response.raise_for_status()
        data = response.json()
        archived_snapshots = data.get("archived_snapshots", {})
        closest_snapshot = archived_snapshots.get("closest", {})
        processing_time = time.time() - start_time

        if closest_snapshot and closest_snapshot.get("available"):
             return {
                 "status": "found",
                 "original_url": url_to_check,
                 "wayback_url": closest_snapshot.get("url"),
                 "timestamp": closest_snapshot.get("timestamp"),
                 "processing_time_seconds": round(processing_time, 2),
             }
        else:
             return {
                 "status": "not_found",
                 "original_url": url_to_check,
                 "processing_time_seconds": round(processing_time, 2),
             }

    except Exception as e:
        print(f"Wayback Machine Error for {url_to_check}: {e}")
        # Don't raise HTTP Exception, just return an error status
        return {
             "status": "error",
             "original_url": url_to_check,
             "message": str(e)
         }

# --- API Endpoints ---

@app.post("/search")
async def run_search(request: SearchRequest = Body(...)):
    """
    Performs a search using the specified provider (google, baidu, free)
    or falls back to mock data.
    """
    provider = request.search_provider.lower()
    query_params = request.dict() # Pass all request params

    if provider == "google":
        if GOOGLE_API_KEY and GOOGLE_CX:
             # Prioritize official API if configured
             return search_google_api(**query_params)
        elif SERPAPI_KEY:
             # Fallback to SerpApi Google if official keys missing but SerpApi exists
             print("Warning: Google keys missing, falling back to SerpApi for Google search.")
             return search_google_via_serpapi(**query_params)
        else:
             print("Warning: No Google or SerpApi keys configured, using fallback.")
             return search_fallback(**query_params)
    elif provider == "baidu":
        if SERPAPI_KEY:
            return search_baidu_via_serpapi(**query_params)
        else:
            print("Warning: SerpApi key needed for Baidu search, using fallback.")
            return search_fallback(**query_params)
    elif provider == "free":
        return search_free_api(**query_params)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported search_provider: {request.search_provider}. Use 'google', 'baidu', or 'free'.")


@app.post("/wayback")
async def check_wayback(request: WaybackRequest = Body(...)):
    """Checks Wayback Machine for a given URL."""
    return search_wayback_machine(request.url)

# --- Run Instruction (for local testing) ---
# Use: uvicorn main:app --reload
# Example POST to /search using curl:
# curl -X POST "http://127.0.0.1:8000/search" -H "Content-Type: application/json" -d '{"query": "supply chain compliance issues 2023", "search_provider": "google", "language_code": "en", "country_code": "us", "max_results": 5}'
# Example POST to /wayback:
# curl -X POST "http://127.0.0.1:8000/wayback" -H "Content-Type: application/json" -d '{"url": "https://example.com"}'

if __name__ == "__main__":
    import uvicorn
    # Host on 0.0.0.0 to be accessible within a network/container if needed
    # For deployment, use a proper ASGI server like uvicorn managed by supervisor/systemd or a PaaS provider
    uvicorn.run(app, host="0.0.0.0", port=8000)