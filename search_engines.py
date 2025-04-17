# search_engines.py

import requests
import time
from typing import List, Dict, Any

# Import config to get API keys
import config

# Import Google API client library (ensure it's installed)
try:
    from googleapiclient.discovery import build as build_google_service
    google_api_client_available = True
except ImportError:
    print("Warning: google-api-python-client not installed. Official Google Search disabled.")
    google_api_client_available = False
    build_google_service = None # Define as None to avoid NameError later

# Import SerpApi library (ensure it's installed)
try:
    from serpapi import GoogleSearch as SerpApiSearch
    serpapi_available = True
except ImportError:
    print("Warning: google-search-results (SerpApi) library not installed. SerpApi searches disabled.")
    serpapi_available = False
    SerpApiSearch = None # Define as None to avoid NameError later


# --- Google Custom Search API ---
def search_google_official(query: str, lang: str = 'en', num: int = 10) -> list:
    """Search using Google Custom Search JSON API."""
    if not google_api_client_available:
        print("Skipping Google Official search: google-api-python-client not installed.")
        return []

    # Use the correct config variables for Search API Key and CX ID
    if not config.GOOGLE_API_KEY_SEARCH or not config.GOOGLE_CX:
        print("Warning: Google Search API Key/CX missing in config. Cannot use official search.")
        return []

    try:
        # Use the correct key variable here
        service = build_google_service("customsearch", "v1", developerKey=config.GOOGLE_API_KEY_SEARCH)

        lr_param = f"lang_{lang}" if lang else None
        print(f"Executing Google Official Search: q='{query}', lang='{lr_param}', num={min(num, 10)}")
        result = service.cse().list(
            q=query,
            cx=config.GOOGLE_CX,
            num=min(num, 10), # Google API max 10 per request page
            lr=lr_param
            # Add other params like 'dateRestrict' if needed
        ).execute()
        return result.get("items", []) # Return the list of result items
    except Exception as e:
        print(f"ERROR during Google Official search: {e}")
        return []

# --- SerpApi (for Google, Baidu, etc.) ---
def search_via_serpapi(query: str, engine: str, country_code: str = 'us', lang_code: str = 'en', num: int = 10) -> list:
    """Search using SerpApi for various engines."""
    if not serpapi_available:
        print(f"Skipping SerpApi search ({engine}): google-search-results library not installed.")
        return []
    if not config.SERPAPI_KEY:
        print(f"Warning: SERPAPI_KEY missing in config. Cannot use SerpApi for {engine}.")
        return []

    try:
        params = {
            "engine": engine,
            "q": query,
            "api_key": config.SERPAPI_KEY,
            "num": num,
        }
        # Engine-specific params
        if engine == "google":
            params["gl"] = country_code
            params["hl"] = lang_code
        elif engine == "baidu":
            params["cc"] = country_code # Baidu uses 'cc' for country

        print(f"Executing SerpApi Search: engine='{engine}', q='{query}', cc/gl='{country_code}', num={num}")
        search = SerpApiSearch(params) # Use the imported class name
        results = search.get_dict()

        if "error" in results:
            print(f"SerpApi Error ({engine}): {results['error']}")
            return []

        # SerpApi typically returns results under 'organic_results'
        return results.get("organic_results", []) # Return the list of result items
    except Exception as e:
        print(f"ERROR during SerpApi search ({engine}): {e}")
        return []

# --- Wayback Machine ---
def check_wayback_machine(url_to_check: str) -> dict:
    """Check Wayback Machine Availability API."""
    # Ensure URL is a non-empty string
    if not url_to_check or not isinstance(url_to_check, str):
        print("Warning: Invalid URL provided for Wayback check.")
        return {"status": "error", "original_url": url_to_check, "message": "Invalid URL"}

    api_url = f"https://archive.org/wayback/available?url={requests.utils.quote(url_to_check)}"
    try:
        response = requests.get(api_url, timeout=20)
        response.raise_for_status()
        data = response.json()

        archived_snapshots = data.get("archived_snapshots", {})
        closest_snapshot = archived_snapshots.get("closest", {})

        if closest_snapshot and closest_snapshot.get("available") and closest_snapshot.get("status", "").startswith("2"):
            return {
                "status": "found",
                "original_url": url_to_check,
                "wayback_url": closest_snapshot.get("url"),
                "timestamp": closest_snapshot.get("timestamp"),
            }
        else:
            return {"status": "not_found", "original_url": url_to_check}

    except requests.exceptions.Timeout:
         print(f"Timeout checking Wayback Machine for {url_to_check}")
         return {"status": "error", "original_url": url_to_check, "message": "Timeout"}
    except requests.exceptions.RequestException as req_e:
         print(f"Request Error checking Wayback Machine for {url_to_check}: {req_e}")
         return {"status": "error", "original_url": url_to_check, "message": f"Request Error: {req_e}"}
    except Exception as e:
        print(f"Unexpected Error checking Wayback Machine for {url_to_check}: {e}")
        return {"status": "error", "original_url": url_to_check, "message": str(e)}

# --- Helper to Standardize Results ---
def standardize_result(item: dict, source: str) -> dict | None:
    """
    Standardizes results from different sources into a common format.
    Returns None if essential fields (link, title/snippet) are missing.
    """
    if not item or not isinstance(item, dict):
        return None

    title = item.get("title", "")
    link = item.get("link", item.get("url", item.get("displayed_link")))
    snippet = item.get("snippet", item.get("description", ""))
    date_str = item.get("date")

    # Basic validation: Need at least a link and some text content
    if not link or not isinstance(link, str) or not (title or snippet):
         # print(f"Warning: Skipping result from {source} due to missing link or text content: {item.get('title', 'N/A')}")
         return None

    parsed_date = None
    if date_str and isinstance(date_str, str):
        # Basic date parsing attempt (can be improved with dateutil)
        try:
            # Attempt to grab YYYY-MM-DD like pattern
            if len(date_str) >= 10:
                 parsed_date = date_str[:10] # Grab first 10 chars, hoping for YYYY-MM-DD
                 # Add validation if needed: datetime.strptime(parsed_date, "%Y-%m-%d")
            else: # Fallback for simpler formats if needed
                 parsed_date = date_str.split(' ')[0]
        except Exception:
            parsed_date = date_str # Keep original string if basic parsing fails

    # Add more sophisticated date extraction from Google API pagemap if needed
    # if source == 'google_api' and not parsed_date:
    #    try:
    #        metatags = item.get("pagemap", {}).get("metatags", [{}])[0]
    #        date_string = metatags.get("article:published_time") or ...
    #        # parse date_string here
    #    except Exception: pass

    return {
        "title": title,
        "url": link,
        "snippet": snippet,
        "source": source,
        "published_date": parsed_date, # Store parsed or original date string
        # "raw": item # Generally avoid passing raw data unless necessary
    }