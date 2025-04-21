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
    build_google_service = None

# Import SerpApi library (ensure it's installed)
try:
    from serpapi import GoogleSearch as SerpApiSearch
    serpapi_available = True
except ImportError:
    print("Warning: google-search-results (SerpApi) library not installed. SerpApi searches disabled.")
    serpapi_available = False
    SerpApiSearch = None

# --- Google Custom Search API ---
def search_google_official(query: str, lang: str = 'en', num: int = 10) -> list:
    """Search using Google Custom Search JSON API."""
    if not google_api_client_available:
        print("Skipping Google Official search: library not installed.")
        return []
    if not config.GOOGLE_API_KEY_SEARCH or not config.GOOGLE_CX:
        print("Warning: Google Search API Key/CX missing. Cannot use official search.")
        return []
    try:
        service = build_google_service("customsearch", "v1", developerKey=config.GOOGLE_API_KEY_SEARCH)
        lr_param = f"lang_{lang}" if lang else None
        print(f"Executing Google Official Search: q='{query}', lang='{lr_param}', num={min(num, 10)}")
        result = service.cse().list( q=query, cx=config.GOOGLE_CX, num=min(num, 10), lr=lr_param ).execute()
        return result.get("items", [])
    except Exception as e:
        print(f"ERROR during Google Official search: {e}")
        return []

# --- SerpApi (for Google, Baidu, etc.) ---
def search_via_serpapi(query: str, engine: str, country_code: str = 'us', lang_code: str = 'en', num: int = 10) -> list:
    """Search using SerpApi for various engines."""
    if not serpapi_available:
        print(f"Skipping SerpApi search ({engine}): library not installed.")
        return []
    if not config.SERPAPI_KEY:
        print(f"Warning: SERPAPI_KEY missing. Cannot use SerpApi for {engine}.")
        return []
    try:
        params = { "engine": engine, "q": query, "api_key": config.SERPAPI_KEY, "num": num }
        if engine == "google": params["gl"] = country_code; params["hl"] = lang_code
        elif engine == "baidu": params["cc"] = country_code
        print(f"Executing SerpApi Search: engine='{engine}', q='{query}', cc/gl='{country_code}', num={num}")
        search = SerpApiSearch(params)
        results = search.get_dict()
        if "error" in results: print(f"SerpApi Error ({engine}): {results['error']}"); return []
        return results.get("organic_results", [])
    except Exception as e:
        print(f"ERROR during SerpApi search ({engine}): {e}")
        return []

# --- Wayback Machine ---
def check_wayback_machine(url_to_check: str) -> dict:
    """Check Wayback Machine Availability API."""
    if not url_to_check or not isinstance(url_to_check, str): return {"status": "error", "message": "Invalid URL"}
    api_url = f"https://archive.org/wayback/available?url={requests.utils.quote(url_to_check)}"
    try:
        response = requests.get(api_url, timeout=20); response.raise_for_status(); data = response.json()
        closest = data.get("archived_snapshots", {}).get("closest", {})
        if closest and closest.get("available") and closest.get("status", "").startswith("2"):
            return {"status": "found", "original_url": url_to_check, "wayback_url": closest.get("url"), "timestamp": closest.get("timestamp")}
        else: return {"status": "not_found", "original_url": url_to_check}
    except requests.exceptions.Timeout: print(f"Timeout checking Wayback: {url_to_check}"); return {"status": "error", "message": "Timeout"}
    except requests.exceptions.RequestException as req_e: print(f"Request Error checking Wayback: {req_e}"); return {"status": "error", "message": f"Request Error: {req_e}"}
    except Exception as e: print(f"Unexpected Error checking Wayback: {e}"); return {"status": "error", "message": str(e)}

# ===> NEW FUNCTION for Ownership Docs <===
def search_for_ownership_docs(entity_name: str, num_per_query: int = 2) -> List[Dict[str, Any]]:
    """
    Performs targeted searches to find potential ownership documents (reports, filings).
    Prioritizes Google Official API, then falls back to SerpApi Google.
    Args:
        entity_name: The name of the company (potential parent) to search for.
        num_per_query: Max results to request for each specific query variation.
    Returns:
        A list of standardized search result dictionaries.
    """
    print(f"\n--- Searching for ownership docs for: {entity_name} ---")
    # Construct targeted queries - add more specific terms if needed
    queries = [
        f'"{entity_name}" annual report ownership structure',
        f'"{entity_name}" subsidiaries list',
        f'"{entity_name}" 10-K filing exhibit 21', # Often lists subsidiaries in US filings
        f'"{entity_name}" holdings OR investment in OR stake in',
        f'"{entity_name}" group companies OR affiliates list',
    ]

    all_raw_results = {} # Use dict to store by URL for basic deduplication
    search_source = "unknown_ownership"

    # --- Attempt 1: Google Official API ---
    if config.GOOGLE_API_KEY_SEARCH and config.GOOGLE_CX and google_api_client_available:
        search_source = 'google_api_ownership'
        print("Using Google Official API for ownership search...")
        for q in queries:
            try:
                results = search_google_official(q, lang='en', num=num_per_query)
                for r in results:
                    url = r.get('link')
                    if url and url not in all_raw_results: all_raw_results[url] = r
                time.sleep(0.3) # Small delay
            except Exception as e: print(f"Error during Google API ownership query '{q}': {e}")

    # --- Attempt 2: SerpApi Google (Fallback or if more needed) ---
    # Only run if SerpApi is available AND (either Google official wasn't used OR didn't find many results)
    # Let's target getting roughly num_per_query * 2 total unique results if possible
    target_total_results = num_per_query * 2
    if serpapi_available and config.SERPAPI_KEY and \
       (not (config.GOOGLE_API_KEY_SEARCH and config.GOOGLE_CX and google_api_client_available) or len(all_raw_results) < target_total_results):
        search_source = 'serpapi_google_ownership'
        print("Using SerpApi Google for ownership search...")
        for q in queries:
            if len(all_raw_results) >= target_total_results: break # Stop if we have enough
            try:
                results = search_via_serpapi(q, engine='google', country_code='us', lang_code='en', num=num_per_query)
                for r in results:
                    url = r.get('link', r.get('url'))
                    if url and url not in all_raw_results: all_raw_results[url] = r
                time.sleep(0.5) # SerpApi might need slightly longer delay
            except Exception as e: print(f"Error during SerpApi ownership query '{q}': {e}")


    # Standardize the unique results found
    standardized = [standardize_result(r, search_source) for r in all_raw_results.values()]
    final_results = [res for res in standardized if res is not None] # Filter out None values from standardization

    print(f"Found {len(final_results)} unique potential ownership documents.")
    return final_results # Return all unique, standardized results found
# ===> END NEW FUNCTION <===


# --- Helper to Standardize Results ---
def standardize_result(item: dict, source: str) -> dict | None:
    """Standardizes results, returns None if invalid."""
    # ... (keep the existing standardize_result function as corrected before) ...
    if not item or not isinstance(item, dict): return None
    title = item.get("title", ""); link = item.get("link", item.get("url", item.get("displayed_link"))); snippet = item.get("snippet", item.get("description", "")); date_str = item.get("date")
    if not link or not isinstance(link, str) or not (title or snippet): return None
    parsed_date = None
    if date_str and isinstance(date_str, str):
        try:
            if len(date_str) >= 10: parsed_date = date_str[:10]
            else: parsed_date = date_str.split(' ')[0]
        except Exception: parsed_date = date_str
    return { "title": title, "url": link, "snippet": snippet, "source": source, "published_date": parsed_date }