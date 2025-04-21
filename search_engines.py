# search_engines.py

import requests
import time
from typing import List, Dict, Any

# Import config to get API keys
import config

# Import Google API client library
try:
    from googleapiclient.discovery import build as build_google_service
    google_api_client_available = True
except ImportError:
    print("Warning: google-api-python-client not installed. Official Google Search disabled.")
    google_api_client_available = False
    build_google_service = None

# Import SerpApi library
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
    # ... (keep function as before) ...
    if not google_api_client_available: return []
    if not config.GOOGLE_API_KEY_SEARCH or not config.GOOGLE_CX: return []
    try:
        service = build_google_service("customsearch", "v1", developerKey=config.GOOGLE_API_KEY_SEARCH)
        lr_param = f"lang_{lang}" if lang else None
        print(f"Executing Google Official Search: q='{query}', lang='{lr_param}', num={min(num, 10)}")
        result = service.cse().list( q=query, cx=config.GOOGLE_CX, num=min(num, 10), lr=lr_param ).execute()
        return result.get("items", [])
    except Exception as e: print(f"ERROR during Google Official search: {e}"); return []


# --- SerpApi (for Google, Baidu, etc.) ---
def search_via_serpapi(query: str, engine: str, country_code: str = 'us', lang_code: str = 'en', num: int = 10) -> list:
    """Search using SerpApi for various engines."""
    # ... (keep function as before) ...
    if not serpapi_available: return []
    if not config.SERPAPI_KEY: return []
    try:
        params = { "engine": engine, "q": query, "api_key": config.SERPAPI_KEY, "num": num }
        if engine == "google": params["gl"] = country_code; params["hl"] = lang_code
        elif engine == "baidu": params["cc"] = country_code
        print(f"Executing SerpApi Search: engine='{engine}', q='{query}', cc/gl='{country_code}', num={num}")
        search = SerpApiSearch(params)
        results = search.get_dict()
        if "error" in results: print(f"SerpApi Error ({engine}): {results['error']}"); return []
        return results.get("organic_results", [])
    except Exception as e: print(f"ERROR during SerpApi search ({engine}): {e}"); return []


# --- Wayback Machine ---
def check_wayback_machine(url_to_check: str) -> dict:
    """Check Wayback Machine Availability API."""
    # ... (keep function as before) ...
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


# ===> UPDATED FUNCTION for Ownership Docs <===
def search_for_ownership_docs(entity_name: str, num_per_query: int = 3) -> List[Dict[str, Any]]: # Request slightly more per query
    """
    Performs targeted searches to find potential ownership documents (reports, filings, news).
    Prioritizes Google Official API, then falls back to SerpApi Google. Includes broader terms.
    """
    print(f"\n--- Searching for ownership docs for: {entity_name} ---")
    # Construct broader targeted queries
    queries = [
        f'"{entity_name}" "subsidiaries" OR "affiliates" OR "related parties" list',
        f'"{entity_name}" "ownership structure" OR "shareholding" OR "investment in"',
        f'"{entity_name}" annual report OR 10-K "equity method investments"', # Look for specific accounting terms
        f'"{entity_name}" "controlling interest" OR "majority stake" OR "minority stake"',
        f'"{entity_name}" "joint venture" OR "partnership structure"',
        # Less direct, but might catch news about relationships
        f'"{entity_name}" parent company OR subsidiary news',
    ]

    all_raw_results = {} # Use dict to store by URL for basic deduplication
    search_source = "unknown_ownership"
    total_results_target = num_per_query * 3 # Aim for slightly more potential docs

    # --- Attempt 1: Google Official API ---
    if config.GOOGLE_API_KEY_SEARCH and config.GOOGLE_CX and google_api_client_available:
        search_source = 'google_api_ownership'
        print("Using Google Official API for ownership search...")
        for q in queries:
            if len(all_raw_results) >= total_results_target: break # Stop if we potentially have enough unique URLs
            try:
                results = search_google_official(q, lang='en', num=num_per_query)
                for r in results:
                    url = r.get('link')
                    if url and url not in all_raw_results: all_raw_results[url] = r
                time.sleep(0.3)
            except Exception as e: print(f"Error during Google API ownership query '{q}': {e}")

    # --- Attempt 2: SerpApi Google (Fallback or if more needed) ---
    if serpapi_available and config.SERPAPI_KEY and \
       (not (config.GOOGLE_API_KEY_SEARCH and config.GOOGLE_CX and google_api_client_available) or len(all_raw_results) < total_results_target):
        # Determine the primary source even if we augment with SerpApi
        search_source = search_source if search_source != "unknown_ownership" else 'serpapi_google_ownership'
        print("Using SerpApi Google for ownership search augmentation...")
        for q in queries:
            if len(all_raw_results) >= total_results_target * 1.5: break # Allow slight overshoot with SerpApi
            try:
                results = search_via_serpapi(q, engine='google', country_code='us', lang_code='en', num=num_per_query)
                for r in results:
                    url = r.get('link', r.get('url'))
                    # Use the primary source determined earlier if mixing results
                    source_to_assign = search_source
                    if url and url not in all_raw_results: all_raw_results[url] = r; all_raw_results[url]['_source_api'] = 'serpapi' # Tag SerpApi results if needed
                    elif url: all_raw_results[url]['_source_api'] = 'google_api' # Tag existing if needed
                time.sleep(0.5)
            except Exception as e: print(f"Error during SerpApi ownership query '{q}': {e}")


    # Standardize the unique results found
    # Pass the determined primary source ('google_api_ownership' or 'serpapi_google_ownership')
    standardized = [standardize_result(r, search_source) for r in all_raw_results.values()]
    final_results = [res for res in standardized if res is not None]

    print(f"Found {len(final_results)} unique potential ownership documents.")
    # Return maybe top 5-7 unique results overall for analysis
    return final_results[:max(5, num_per_query * 2)]
# ===> END UPDATED FUNCTION <===


# --- Helper to Standardize Results ---
def standardize_result(item: dict, source: str) -> dict | None:
    """Standardizes results, returns None if invalid."""
    # ... (keep as before) ...
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