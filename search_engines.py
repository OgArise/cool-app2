import requests
import time
from googleapiclient.discovery import build as build_google_service
from serpapi import GoogleSearch as SerpApiSearch
import config

# --- Google Custom Search API ---
def search_google_official(query: str, lang: str = 'en', num: int = 10) -> list:
    if not config.GOOGLE_API_KEY or not config.GOOGLE_CX:
        print("Error: Google API Key/CX missing for official search.")
        return []
    try:
        service = build_google_service("customsearch", "v1", developerKey=config.GOOGLE_API_KEY)
        lr_param = f"lang_{lang}" if lang else None
        result = service.cse().list(
            q=query, cx=config.GOOGLE_CX, num=min(num, 10), lr=lr_param
        ).execute()
        return result.get("items", [])
    except Exception as e:
        print(f"Error during Google Official search: {e}")
        return []

# --- SerpApi (for Google, Baidu, etc.) ---
def search_via_serpapi(query: str, engine: str, country_code: str = 'us', lang_code: str = 'en', num: int = 10) -> list:
    if not config.SERPAPI_KEY:
        print(f"Error: SERPAPI_KEY missing, cannot search {engine}.")
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

        search = SerpApiSearch(params)
        results = search.get_dict()

        if "error" in results:
            print(f"SerpApi Error ({engine}): {results['error']}")
            return []

        return results.get("organic_results", [])
    except Exception as e:
        print(f"Error during SerpApi search ({engine}): {e}")
        return []

# --- Wayback Machine ---
def check_wayback_machine(url_to_check: str) -> dict:
    api_url = f"https://archive.org/wayback/available?url={requests.utils.quote(url_to_check)}"
    try:
        response = requests.get(api_url, timeout=15)
        response.raise_for_status()
        data = response.json()
        archived = data.get("archived_snapshots", {}).get("closest", {})
        if archived and archived.get("available"):
            return {
                "status": "found",
                "original_url": url_to_check,
                "wayback_url": archived.get("url"),
                "timestamp": archived.get("timestamp"),
            }
        else:
            return {"status": "not_found", "original_url": url_to_check}
    except Exception as e:
        print(f"Wayback Machine check error for {url_to_check}: {e}")
        return {"status": "error", "original_url": url_to_check, "message": str(e)}

# --- Helper to Standardize Results ---
def standardize_result(item: dict, source: str) -> dict:
    # Basic standardization - adjust based on observed API differences
    title = item.get("title", "")
    link = item.get("link", item.get("url", "")) # Google vs SerpApi link field
    snippet = item.get("snippet", item.get("description", "")) # Google vs SerpApi snippet field
    date = item.get("date") # SerpApi often has this

    # Add more date parsing logic here if needed from Google pagemap etc.

    return {
        "title": title,
        "url": link,
        "snippet": snippet,
        "source": source,
        "published_date": date, # Needs refinement
        "raw": item # Keep raw data if needed
    }