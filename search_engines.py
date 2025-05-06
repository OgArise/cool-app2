# search_engines.py

import time
import asyncio # Import asyncio
import urllib.parse # Import urllib.parse for URL quoting
import httpx # Import httpx for asynchronous HTTP requests
import json
import re
import traceback

from typing import List, Dict, Any, Optional, Union # Import Union

import config

# We need nlp_processor for targeted search keyword translation in search_for_ownership_docs
# Import it conditionally or ensure it's available if search_for_ownership_docs is called.
# Given orchestrator imports nlp_processor and then search_engines, it should be available.
try:
    import nlp_processor
    nlp_processor_available_search = True # Use a different name to avoid conflict
except ImportError:
    nlp_processor = None
    nlp_processor_available_search = False


LinkupClient = None
linkup_library_available = False
LinkupSearchResults = None
LinkupSearchTextResult = None
# FIX: Removed import of LinkupSearchStructuredResult as it doesn't exist in the SDK


try:
    # FIX: Corrected the import statement for the Linkup library based on user confirmation
    from linkup import LinkupClient as _LinkupClient
    from linkup import LinkupSearchResults as _LinkupSearchResults
    from linkup import LinkupSearchTextResult as _LinkupSearchTextResult
    # FIX: Removed import of LinkupSearchStructuredResult

    # Check if the imported components are valid (optional but good practice)
    # FIX: Adjusted check as LinkupSearchStructuredResult is removed
    if _LinkupClient is not None and _LinkupSearchResults is not None and _LinkupSearchTextResult is not None:
         LinkupClient = _LinkupClient
         LinkupSearchResults = _LinkupSearchResults
         LinkupSearchTextResult = _LinkupSearchTextResult
         linkup_library_available = True
    else:
         # If imports didn't fail but components are None, something is still wrong
         print("Warning: Imported Linkup library (linkup) but required components are None. Linkup searches disabled.")
         linkup_library_available = False
except ImportError:
    # FIX: Updated error message to reflect the correct library name
    print("Warning: linkup library not installed. Linkup searches disabled.")
    linkup_library_available = False
except Exception as e:
     # Catch any other potential import errors
     # FIX: Updated error message to reflect the correct library name
     print(f"Error importing Linkup library (linkup): {e}. Linkup searches disabled.")
     traceback.print_exc()
     linkup_library_available = False


try:
    # google-api-python-client requires httplib2/oauthlib which are synchronous
    # For async, we might need to wrap synchronous calls in thread pools (less ideal for pure async)
    # OR rely on httpx directly for Google Search JSON API calls.
    # Given search is not the primary purpose of this module anymore (Linkup/SerpApi handle it)
    # and Google CSE has rate limits, we will stick to SerpApi or Linkup primarily.
    # However, if needed, httpx can call the Google CSE endpoint directly.
    # Let's keep the old import for now but note it might be synchronous
    from googleapiclient.discovery import build as build_google_service
    google_api_client_available = True
except ImportError:
    print("Warning: google-api-python-client not installed. Official Google Search disabled.")
    google_api_client_available = False
    build_google_service = None

try:
    from serpapi import GoogleSearch as SerpApiSearch # SerpApi client might also be synchronous
    serpapi_library_available = True
except ImportError:
    print("Warning: google-search-results (SerpApi) library not installed. SerpApi searches disabled.")
    serpapi_library_available = False
    SerpApiSearch = None

# Use a shared async client for multiple requests if needed (careful with lifecycle)
# For simplicity and safety with timeouts/errors, we'll use `async with httpx.AsyncClient()`
# within each async function that needs it, which creates and closes a client per call.
# This might be slightly less performant for very frequent tiny calls, but more robust.


async def check_linkup_balance() -> Optional[float]:
    """Checks the remaining Linkup API credits using an async httpx call."""
    if not config or not getattr(config, 'LINKUP_API_KEY', None):
        return None

    balance_url = "https://api.linkup.so/v1/credits/balance"
    headers = {"Authorization": f"Bearer {config.LINKUP_API_KEY}"}

    try:
        async with httpx.AsyncClient() as client:
             response = await client.get(balance_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        balance = data.get("balance")
        if isinstance(balance, (int, float)):
            return float(balance)
        else:
            print(f"Warning: Linkup balance response format unexpected. Got: {data}")
            return None
    except httpx.TimeoutException:
        print("Warning: Timeout checking Linkup credits.")
        return None
    except httpx.RequestError as req_e:
        print(f"Warning: Request error checking Linkup credits: {req_e}")
        return None
    except Exception as e:
        print(f"Warning: Unexpected error checking Linkup credits: {e}")
        traceback.print_exc()
        return None

linkup_client = None
linkup_search_enabled = False

# Initialize Linkup client only if library is available and key is configured
if linkup_library_available:
    if not config or not getattr(config, 'LINKUP_API_KEY', None):
        print("Warning: Linkup API Key missing in config/.env. Linkup searches disabled.")
    else:
        try:
            # Initialize the async client
            linkup_client = LinkupClient(api_key=config.LINKUP_API_KEY)
            # FIX: Removed the synchronous call to check_linkup_balance from here.
            # The balance will be checked asynchronously during application startup.
            # print("Linkup client initialized successfully.")
            linkup_search_enabled = True # Assume enabled if client initialized and key is present
            print("Linkup searches enabled (subject to balance check during startup).")
        except Exception as e:
            print(f"ERROR initializing Linkup client with API key: {type(e).__name__}: {e}")
            traceback.print_exc()
            linkup_client = None
            print("Warning: Linkup client initialization failed. Linkup searches disabled.")
else:
     print("Linkup library (linkup) not available, Linkup searches disabled.") # Re-iterate if library import failed


serpapi_available = False
if serpapi_library_available and config and getattr(config, 'SERPAPI_KEY', None):
     serpapi_available = True
     print("SerpApi searches enabled.")
elif serpapi_library_available:
     print("Warning: SerpApi library is available, but SerpApi Key missing in config/.env. SerpApi searches disabled.")
else:
     print("SerpApi library not available, SerpApi searches disabled.")

# Google CSE check moved here to be alongside other search engine checks
google_api_client_available_and_configured = google_api_client_available and config and getattr(config, 'GOOGLE_API_KEY_SEARCH', None) and getattr(config, 'GOOGLE_CX', None)

if google_api_client_available_and_configured:
    print("Google CSE searches enabled.")
else:
    print("Warning: Google API client or configuration missing. Google CSE searches disabled.")


async def search_google_official(query: str, lang: str = 'en', num: int = 10) -> list:
    """Search using Google Custom Search JSON API (using httpx)."""
    if not google_api_client_available_and_configured:
        # print("Skipping Google Official Search: Not configured.") # Suppress frequent message
        return []
    if not query or not isinstance(query, str) or not query.strip():
         print("Skipping Google Official Search: Invalid query provided.")
         return []

    # Google CSE API limits 'num' to a maximum of 10 per request.
    effective_num = min(num, 10)
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': config.GOOGLE_API_KEY_SEARCH,
        'cx': config.GOOGLE_CX,
        'q': query,
        'num': effective_num,
        'lr': f"lang_{lang}" if lang else None,
        'alt': 'json' # Explicitly request JSON
    }
    # Remove None values from params
    params = {k: v for k, v in params.items() if v is not None}


    try:
        print(f"Executing async Google Official Search: q='{query[:50]}...', lang='{lang}', num={effective_num}")
        async with httpx.AsyncClient() as client:
             response = await client.get(base_url, params=params, timeout=30) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes
        result = response.json()
        return result.get("items", [])
    except httpx.TimeoutException:
        print(f"Timeout during async Google Official search for '{query[:50]}...'.")
        return []
    except httpx.RequestError as e:
        print(f"ERROR during async Google Official search for '{query[:50]}...': {type(e).__name__}: {e}")
        # traceback.print_exc() # Too verbose
        return []
    except Exception as e:
        print(f"ERROR during async Google Official search for '{query[:50]}...': {type(e).__name__}: {e}")
        traceback.print_exc() # Log unexpected errors
        return []


async def search_via_serpapi(query: str, engine: str, country_code: str = 'cn', lang_code: str = 'en', num: int = 20) -> list:
    """Search using SerpApi for various engines (wrapping synchronous SerpApiSearch)."""
    # NOTE: The SerpApi Python client (google-search-results) is typically synchronous.
    # To make this function awaitable, we must run the synchronous SerpApiSearch call
    # within an executor (like asyncio's default ThreadPoolExecutor).
    # This allows the async event loop to continue while the synchronous call blocks a thread.
    # This is the standard pattern for integrating sync libraries into async code.

    if not serpapi_available:
        # print("Skipping SerpApi search: Not configured.") # Suppress frequent message
        return []
    if not query or not isinstance(query, str) or not query.strip():
         print(f"Skipping async Serpapi search ({engine}): Invalid query provided.")
         return []

    try:
        params = { "engine": engine, "q": query, "api_key": config.SERPAPI_KEY, "num": num }
        if engine == "google":
            # SerpApi's Google engine accepts gl (country) and hl (language)
            params["gl"] = country_code
            params["hl"] = lang_code
        elif engine == "baidu":
             # SerpApi's Baidu engine accepts cc (country code)
             params["cc"] = country_code
             # Note: SerpApi Baidu doesn't seem to have a direct 'lang' parameter like Google does.
             # Language is primarily controlled by the query string itself and the engine/country.


        print(f"Executing async SerpApi Search ({engine}): q='{query[:50]}...', params={{'engine': '{engine}', 'num': {num}, ...}}") # Log simplified params

        # Run the synchronous SerpApiSearch.get_dict() method in a thread pool
        loop = asyncio.get_running_loop()
        # Pass keyword arguments directly to get_dict
        # Assuming SerpApiSearch is thread-safe for separate instances
        results = await loop.run_in_executor(None, lambda: SerpApiSearch(params).get_dict())

        if "error" in results:
            print(f"SerpApi Error ({engine}) for '{query[:50]}...': {results['error']}")
            return []
        return results.get("organic_results", [])
    except Exception as e:
        print(f"ERROR during async SerpApi search ({engine}) for '{query[:50]}...': {type(e).__name__}: {e}")
        traceback.print_exc()
        return []

async def search_linkup_snippets(query: Union[str, List[str]], num: int = 20, country_code: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search using Linkup API client with output_type='searchResults' (async wrapper).
    Accepts a single query string or a list of query strings.
    Maps Linkup results to the standard snippet format expected by NLP.
    Returns all unique results found across executed queries.
    The 'num' parameter is passed to the Linkup API call for each query,
    but this function does NOT impose an external limit on the *total* results.
    """
    if not linkup_search_enabled or linkup_client is None:
        # print("Linkup searches are not enabled.") # Suppress frequent message
        return []

    all_linkup_results: List[Dict[str, Any]] = []
    processed_urls = set()

    queries_to_run = [query] if isinstance(query, str) else query
    queries_to_run = [q.strip() for q in queries_to_run if q.strip()] # Ensure queries are valid strings

    if not queries_to_run:
         print("No valid queries provided for async Linkup Snippet Search.")
         return []

    print(f"Executing async Linkup Snippet Search for {len(queries_to_run)} queries (num parameter passed per query: {num}, requested_country_code={country_code})...")

    # We can make multiple Linkup calls concurrently for different queries if the SDK supports it.
    # Since the underlying client is synchronous, we'll run each search in an executor thread.
    async def _perform_single_linkup_snippet_query(q):
         try:
              print(f"  Linkup Query: '{q[:50]}...'") # Print truncated query
              params = {
                  "query": q,
                  "depth": "standard", # or "deep" depending on desired depth
                  "output_type": "searchResults",
                  "include_images": False,
                  # FIX: Removed 'num' parameter as it caused TypeError.
                  # Rely on Linkup's internal limits or default SDK behavior.
                  # "num": num # Removed
              }
              # country_code is NOT a direct parameter according to previous errors.
              # Country context should ideally be in the query string itself.

              # FIX: Wrap the synchronous call in run_in_executor
              loop = asyncio.get_running_loop()
              response = await loop.run_in_executor(None, lambda: linkup_client.search(**params))


              if isinstance(response, LinkupSearchResults) and hasattr(response, 'results') and isinstance(response.results, list):
                   print(f"    Linkup API returned {len(response.results)} results for query '{q[:50]}...'.")
                   return response.results # Return the list of results
              elif isinstance(response, list) and all(isinstance(item, (LinkupSearchTextResult, dict)) for item in response): # Handle if Linkup returns a list directly
                   print(f"    Linkup API returned a list directly for query '{q[:50]}...'. Found {len(response)} items.")
                   return response
              else:
                   print(f"    Linkup Snippet Search for query '{q[:50]}...' returned no results or unexpected format (Type: {type(response)}).")
                   return [] # Return empty list on failure/no results

         except Exception as e:
              print(f"ERROR during async Linkup snippet search for query '{q[:50]}...': {type(e).__name__}: {e}")
              traceback.print_exc()
              return [] # Return empty list on error


    # Gather all the async query tasks (each internally wraps a sync call)
    query_tasks = [_perform_single_linkup_snippet_query(q) for q in queries_to_run]
    # Run them concurrently
    list_of_results_lists = await asyncio.gather(*query_tasks, return_exceptions=True) # Gather results, capture exceptions

    # Process and deduplicate results from all concurrent queries
    for results_list_or_exc in list_of_results_lists:
        if isinstance(results_list_or_exc, Exception):
             print(f"  Warning: One of the Linkup snippet search tasks raised an exception: {type(results_list_or_exc).__name__}: {results_list_or_exc}")
             traceback.print_exc() # Print traceback for search errors
             continue # Skip processing this result list

        for item in results_list_or_exc:
             # Ensure item is not an exception if return_exceptions was used above
             if isinstance(item, Exception):
                 print(f"  Warning: Item in Linkup results list was an exception: {type(item).__name__}: {item}")
                 continue

             # Standardize each item. standardize_result handles LinkupSDK objects.
             standardized = standardize_result(item, source='linkup_snippet_search')
             if standardized and isinstance(standardized, dict) and standardized.get('url') and isinstance(standardized.get('url'), str) and standardized['url'] not in processed_urls:
                  all_linkup_results.append(standardized)
                  processed_urls.add(standardized['url'])

    # No final truncation here, return all unique results found across sources up to their individual limits/thresholds.

    print(f"Linkup combined async snippet search returned {len(all_linkup_results)} unique results after internal deduplication.")
    return all_linkup_results


async def search_linkup_structured(query: str, structured_output_schema: str, depth: str = "deep", country_code: Optional[str] = None) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Search using Linkup API client with output_type='structured' (async wrapper).
    Returns the parsed JSON response (dict or list of dicts) or None on failure.
    """
    if not linkup_search_enabled or linkup_client is None:
        # print("Linkup searches are not enabled.") # Suppress frequent message
        return None
    if not structured_output_schema:
        print("Skipping async Linkup structured search: No schema provided.")
        return None
    if not query or not isinstance(query, str) or not query.strip():
         print("Skipping async Linkup structured search: No valid query provided.")
         return None


    try:
        print(f"Executing async Linkup Structured Search: q='{query[:50]}...', depth='{depth}', requested_country_code={country_code}") # Print truncated query
        # Construct parameters dictionary
        params = {
            "query": query,
            "depth": depth,
            "output_type": "structured",
            "structured_output_schema": structured_output_schema,
            "include_images": False,
            # Removed country_code and num here as per previous errors
        }

        # FIX: Wrap the synchronous call in run_in_executor
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: linkup_client.search(**params))

        if response is not None:
            # Linkup structured search can return various types.
            # Based on Linkup docs/examples, structured search *can* return:
            # 1. A raw dict matching the schema (often for 'extract' depth?)
            # 2. A LinkupSearchResults object containing LinkupSearchTextResult objects (where the structured data is in item.data)
            # 3. A list of dicts or LinkupSearchTextResult objects

            if isinstance(response, dict):
                 # Case 1: Raw dict matching the schema
                 return response
            elif isinstance(response, (LinkupSearchResults, list)): # Handle SearchResults object or a direct list
                 valid_structured_data_list = []
                 # Iterate through the list of results
                 results_list_to_process = response.results if isinstance(response, LinkupSearchResults) else response # Handle both SearchResults object and direct list
                 for item in results_list_to_process:
                      # Check if the item is a LinkupSearchTextResult with a data attribute (structured content)
                      if isinstance(item, LinkupSearchTextResult) and hasattr(item, 'data') and isinstance(getattr(item, 'data', None), dict):
                           valid_structured_data_list.append(item.data)
                      # Check if the item is a dict, assume it's structured data content
                      elif isinstance(item, dict):
                          valid_structured_data_list.append(item)
                      # else: # Optional: log items that don't fit the expected structured format
                      #      print(f"Warning: Linkup Structured list item lacks valid 'data' attribute or is not a dict: {item}")

                 if valid_structured_data_list:
                      print(f"Linkup async Structured Search returned list, extracted {len(valid_structured_data_list)} structured data items.")
                      # If there's only one structured data item in the list, return it directly as a dict for consistency
                      if len(valid_structured_data_list) == 1:
                           return valid_structured_data_list[0]
                      else:
                           # If there are multiple structured data items, return the list
                           return valid_structured_data_list
                 else:
                      print(f"Linkup async Structured Search returned a list, but no valid structured data items found within.")
                      return None

            elif isinstance(response, str):
                 # If the response is a string, attempt to parse it as JSON (fallback)
                 print(f"Linkup async Structured Search returned string. Attempting JSON parse.")
                 try:
                      parsed_response = json.loads(response)
                      if isinstance(parsed_response, dict):
                          return parsed_response
                      elif isinstance(parsed_response, list) and parsed_response and isinstance(parsed_response[0], dict):
                           # If it's a list of dictionaries, validate and return it
                           print(f"Linkup async Structured Search string parsed into a list of dictionaries ({len(parsed_response)} items).")
                           valid_list = [item for item in parsed_response if isinstance(item, dict)]
                           if len(valid_list) != len(parsed_response):
                                print(f"Warning: Parsed list from string contained non-dictionary items. Kept {len(valid_list)} out of {len(parsed_response)}.")
                           # If there's only one item in the list, return the dict directly
                           if len(valid_list) == 1:
                                return valid_list[0]
                           else:
                                return valid_list
                      else:
                          print(f"Warning: Parsed Linkup async Structured response string is not a dictionary or list of dicts. Type: {type(parsed_response)}. Response: {response[:200]}...")
                          return None
                 except json.JSONDecodeError:
                      print(f"ERROR decoding JSON from Linkup async Structured response string: {response[:200]}...")
                      return None
            else:
                 # Handle any other unexpected response types
                 print(f"Linkup async Structured Search returned unexpected type: {type(response)}. Response: {response}")
                 return None
        else:
            print("Linkup async Structured Search returned None response.")
            return None

    except Exception as e:
        print(f"ERROR during async Linkup structured search for '{query[:50]}...': {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

async def check_wayback_machine(url_to_check: str) -> dict:
    """Check Wayback Machine Availability API (async)."""
    if not url_to_check or not isinstance(url_to_check, str) or not url_to_check.strip():
        # print("Skipping Wayback check for invalid or empty URL.") # Suppress frequent message
        return {"status": "skipped_invalid_url", "original_url": url_to_check, "message": "Invalid URL provided"}
    # FIX: Use urllib.parse.quote instead of httpx.quoting.quote
    api_url = f"https://archive.org/wayback/available?url={urllib.parse.quote(url_to_check)}" # Use urllib.parse for quoting
    try:
        async with httpx.AsyncClient() as client:
             response = await client.get(api_url, timeout=30) # Added timeout
        response.raise_for_status()
        data = response.json()
        closest = data.get("archived_snapshots", {}).get("closest", {})
        if closest and closest.get("available") and closest.get("status", "").startswith("2"):
            return {"status": "found", "original_url": url_to_check, "wayback_url": closest.get("url"), "timestamp": closest.get("timestamp")}
        else: return {"status": "not_found", "original_url": url_to_check, "message": "No archive found"}
    except httpx.TimeoutException: print(f"Timeout checking Wayback: {url_to_check[:100]}..."); return {"status": "error", "original_url": url_to_check, "message": "Timeout checking Wayback Machine."}
    except httpx.RequestError as req_e: print(f"Request Error checking Wayback: {url_to_check[:100]}... Error: {req_e}"); return {"status": "error", "original_url": url_to_check, "message": f"Request Error checking Wayback: {req_e}"}
    except Exception as e: print(f"Unexpected Error checking Wayback: {url_to_check[:100]}... Error: {e}"); return {"status": "error", "original_url": url_to_check, "message": f"Unexpected error checking Wayback: {str(e)}"}

async def search_for_ownership_docs(entity_name: str,
                              num_per_query: int = 10, # This is still passed to individual search functions that support it
                              country_code: str = 'cn', # This influences query generation and search engine parameters
                              lang_code: str = 'en'
                              ) -> List[Dict[str, Any]]:
    """
    Performs targeted async searches to find potential ownership documents/mentions
    related to a specific entity. Prioritizes Linkup snippet search and Google CSE concurrently,
    then falls back to Serpapi if needed based on a threshold.
    Returns standardized snippet results.
    Collects all unique results found across enabled sources.
    """
    print(f"\n--- Searching for ownership docs (snippets) for: {entity_name} (Focus Country: {country_code}, Lang: {lang_code}) ---")
    if not entity_name or not isinstance(entity_name, str) or not entity_name.strip():
        print("Skipping ownership docs search: Invalid entity name.")
        return []

    # Initial queries - used for Google CSE and Serpapi fallback
    queries_base = [
        f'"{entity_name}" "subsidiaries" OR "affiliates" list',
        f'"{entity_name}" "ownership structure" OR "shareholding"',
        f'"{entity_name}" annual report OR filing "equity method investments"',
        f'"{entity_name}" "controlling interest" OR "majority stake"',
        f'"{entity_name}" "joint venture" OR "partnership structure"',
        f'"{entity_name}" parent company OR subsidiary news',
    ]
    # Add country context to queries for web search engines
    country_name = country_code.upper()
    if pycountry_available and pycountry is not None:
        try:
             country = pycountry.countries.get(alpha_2=country_code.upper())
             if country: country_name = country.name
        except Exception: # Catch any pycountry error
             pass

    web_search_queries = [f"{q} {country_name}" for q in queries_base]
    web_search_queries = list(set([q.strip() for q in web_search_queries if q.strip()]))


    all_raw_results_map = {} # Use a map to store results by URL for deduplication
    search_source_prefix = "ownership_multi" # Used for tracking source in standardized results

    # Check availability of async search functions
    google_cse_available_bool = bool(google_api_client_available_and_configured and hasattr(search_google_official, '__call__') and asyncio.iscoroutinefunction(search_google_official))
    serpapi_available_bool = bool(serpapi_available and hasattr(search_via_serpapi, '__call__') and asyncio.iscoroutinefunction(search_via_serpapi))
    linkup_search_enabled_bool = bool(linkup_library_available and linkup_search_enabled and hasattr(search_linkup_snippets, '__call__') and asyncio.iscoroutinefunction(search_linkup_snippets))
    # Note: linkup_structured_search_available is not directly used here, but included for completeness
    linkup_structured_search_available_bool = bool(linkup_library_available and linkup_search_enabled_orchestrator and hasattr(search_engines, 'search_linkup_structured') and callable(search_engines.search_linkup_structured) and asyncio.iscoroutinefunction(search_engines.search_linkup_structured))


    # Define a reasonable target for this specific type of search before using fallback
    fallback_threshold = max(num_per_query * 2, 20) # e.g., try fallback if we don't find at least 20 results

    print(f"Starting concurrent Linkup and Google CSE search for ownership docs...")

    # List to hold async tasks for concurrent execution
    concurrent_tasks = []

    # 1. Add Linkup Snippet Search Task (Async)
    if linkup_search_enabled_bool:
         print(f"  Adding Linkup snippet search task (broad query: '{entity_name} ownership...')")
         # search_linkup_snippets handles multiple queries internally and returns a list of results
         # It no longer accepts num as a parameter to the search call itself
         linkup_query_combined = f'"{entity_name}" ownership OR subsidiary OR affiliate OR stake OR filing OR "joint venture" OR "acquired" OR parent OR "equity method" {country_code}'
         # Ensure num_per_query is passed to search_linkup_snippets if it uses it internally for EACH query
         # Based on the implementation, search_linkup_snippets processes multiple queries itself
         # and might use the num parameter for each.
         concurrent_tasks.append(search_linkup_snippets(query=linkup_query_combined, num=num_per_query))
    else:
         print("  Linkup snippet search not enabled or not async, skipping task.")


    # 2. Add Google CSE Search Tasks (Async - one task per query variant)
    if google_cse_available_bool:
        print(f"  Adding Google CSE search tasks ({len(web_search_queries)} English queries)")
        # Create a task for each Google query
        google_tasks = [search_google_official(query=q, num=num_per_query) for q in web_search_queries]
        concurrent_tasks.extend(google_tasks)
    else:
        print("  Google CSE searches not enabled or not async, skipping tasks.")


    # Run initial concurrent searches
    initial_concurrent_results_lists = []
    if concurrent_tasks:
        print("  Running initial concurrent searches...")
        # asyncio.gather preserves order of results corresponding to tasks
        initial_concurrent_results_lists = await asyncio.gather(*concurrent_tasks, return_exceptions=True) # Gather results, capture exceptions

        # Process and deduplicate results from Linkup and Google CSE
        for result_list_or_exc in initial_concurrent_results_lists:
             if isinstance(result_list_or_exc, Exception):
                  print(f"  Warning: One of the initial concurrent search tasks raised an exception: {type(result_list_or_exc).__name__}: {result_list_or_exc}")
                  traceback.print_exc() # Print traceback for search errors
                  continue # Skip processing this result list

             for r in result_list_or_exc:
                  # standardize_result handles various formats including LinkupSDK objects
                  standardized = standardize_result(r, source=f'{search_source_prefix}_initial') # Generic source for initial concurrent batch
                  if standardized and isinstance(standardized, dict) and standardized.get('url') and isinstance(standardized.get('url'), str) and standardized['url'] not in all_raw_results_map:
                       # We are building a map here first, convert to list at the end
                       all_raw_results_map[standardized['url']] = standardized # Store standardized result directly

        print(f"  Finished initial concurrent searches (Linkup + Google CSE). Found {len(all_raw_results_map)} unique results.")
    else:
        print("  No initial concurrent search tasks were added.")


    # 3. Check Threshold and Run SerpApi Fallback if Needed (Async)
    # Only run SerpApi if the number of results is below the fallback threshold AND SerpApi is available
    if serpapi_available_bool and len(all_raw_results_map) < fallback_threshold:
         print(f"Result count ({len(all_raw_results_map)}) below fallback threshold ({fallback_threshold}). Attempting SerpApi fallback search...")
         serpapi_engine = 'baidu' if country_code.lower() == 'cn' else 'google'
         # Use all generated queries for Serpapi fallback (including Chinese ones if cn)
         # For simplicity, let's use a combined query approach for SerpApi fallback similar to Linkup,
         # using the main initial query and adding relevant terms/country.
         # A more sophisticated approach could use the translated keywords from Step 2 if available.
         serpapi_fallback_query_combined = f'"{entity_name}" ownership OR subsidiary OR affiliate OR stake OR filing OR "joint venture" OR "acquired" OR parent OR "equity method" {country_name}' # Use country name for Google SerpApi

         serpapi_fallback_tasks = []
         # Create a task for the SerpApi query. Pass the combined query.
         # search_via_serpapi accepts num and country_code, but uses engine parameter to handle them.
         # We can call search_via_serpapi directly with the combined query.
         serpapi_fallback_tasks.append(search_via_serpapi(query=serpapi_fallback_query_combined, engine=serpapi_engine, country_code=country_code, lang_code=lang_code, num=max(num_per_query, fallback_threshold - len(all_raw_results_map) + 5))) # Request enough results to potentially hit the threshold + buffer

         if serpapi_fallback_tasks:
              print(f"  Running {len(serpapi_fallback_tasks)} SerpApi fallback search task...")
              serpapi_fallback_results_lists = await asyncio.gather(*serpapi_fallback_tasks, return_exceptions=True) # Gather results, capture exceptions

              # Process and deduplicate results from Serpapi
              for results_list_or_exc in serpapi_fallback_results_lists:
                   if isinstance(results_list_or_exc, Exception):
                        print(f"  Warning: The SerpApi fallback search task raised an exception: {type(results_list_or_exc).__name__}: {results_list_or_exc}")
                        traceback.print_exc() # Print traceback for search errors
                        continue # Skip processing this result list

                   for r in results_list_or_exc:
                        standardized = standardize_result(r, source=f'serpapi_{serpapi_engine}_ownership_fallback')
                        # standardize_result attempts to set original_language. If not set, default based on engine
                        if standardized and standardized.get('original_language') is None:
                           if serpapi_engine == 'baidu': standardized['original_language'] = 'zh' # Assume Baidu results are Chinese

                        if standardized and isinstance(standardized, dict) and standardized.get('url') and isinstance(standardized.get('url'), str) and standardized['url'] not in all_raw_results_map:
                             # Add SerpApi results to the main map
                             all_raw_results_map[standardized['url']] = standardized

              print(f"  Finished Serpapi fallback search. Total unique results now: {len(all_raw_results_map)}")
         else:
              print("  No SerpApi fallback search tasks were added.")

    elif serpapi_available_bool:
        print(f"Skipping Serpapi fallback. Initial concurrent searches yielded {len(all_raw_results_map)} results (>= {fallback_threshold} threshold).")
    elif not serpapi_available_bool:
        print("SerpApi fallback search not available, skipping.")


    final_results = list(all_raw_results_map.values())

    # No final truncation here, return all unique results found across sources up to their individual limits/thresholds.

    print(f"Found {len(final_results)} unique potential ownership documents across all sources (Linkup, Google CSE, optional Serpapi fallback).")
    return final_results

def standardize_result(item: Any, source: str) -> dict | None:
    """
    Standardizes results from various search APIs (Google, SerpApi, Linkup).
    Returns None if item is invalid or lacks minimum required fields.
    Handles potentially different key names from various sources.
    Adds 'original_language' and 'translated_from' fields.
    """
    if not item: return None

    title = ""
    link = ""
    snippet = ""
    date_str = ""
    original_language = None # Default to None
    translated_from = None # Default to None
    source_type = None # e.g., 'snippet', 'structured'

    # Check for Linkup SDK object first (TextResult, SearchResults)
    if linkup_library_available and isinstance(item, (LinkupSearchTextResult, LinkupSearchResults)):
         # Handle LinkupSearchTextResult (from searchResults output)
         if isinstance(item, LinkupSearchTextResult):
              title = getattr(item, 'name', '')
              link = getattr(item, 'url', '')
              snippet = getattr(item, 'content', '')
              date_str = getattr(item, 'date', '') # Linkup TextResult might have a date attribute
              original_language = getattr(item, 'language', None)
              source_type = 'linkup_snippet_sdk'

         # Handle LinkupSearchResults (which contains a list of TextResults or StructuredResults) - Less likely to be passed directly after processing
         elif isinstance(item, LinkupSearchResults) and hasattr(item, 'results') and isinstance(item.results, list) and item.results:
             # Process the first item as a representative - ideally search functions return the inner list items
             first_item = item.results[0]
             return standardize_result(first_item, source) # Recursively standardize the first item

        # FIX: Removed handling for LinkupSearchStructuredResult as it doesn't exist

    elif isinstance(item, dict):
        # Handle dictionary-based results from Google CSE, SerpApi, or raw Linkup JSON/dictionary outputs
        title = item.get("title", item.get("Title", ""))
        link = item.get("link", item.get("url", item.get("displayed_link", item.get("Link", item.get("Url", item.get("sourceUrl", item.get("source_url", ""))))))) # Added source_url from Linkup structured
        snippet = item.get("snippet", item.get("description", item.get("Snippet", item.get("Description", item.get("Summary", item.get("textSnippet", item.get("excerpt", item.get("content", ""))))))))
        date_str = item.get("date", item.get("published_date", item.get("Date", item.get("PublishedDate", item.get("publish_date", item.get("timestamp", item.get("source_date", ""))))))) # Added source_date from Linkup structured
        # Attempt to get language from known keys if available
        original_language = item.get("language", item.get("lang", item.get("original_language", None))) # Added original_language

        # Check if this dict item looks like raw structured data content rather than a snippet
        # Check for schema/data keys (from our wrapper dict format) or keys typical of structured data content
        # Linkup structured search often returns a dict that IS the schema payload, not nested in "data".
        # It might also return a list of LinkupSearchTextResult where the schema payload is in item.data.
        # Let's check for keys typical of the schema payloads directly.
        if item.get("ownership_relationships") is not None or item.get("key_risks_identified") is not None or item.get("actions_found") is not None: # Added check for hypothetical types
             # This dict appears to be structured data content directly
             source_type = 'linkup_structured_content' # Mark as structured content
             # Use a summary of the data content as the snippet
             snippet = f"Structured Data Content: {json.dumps(item)[:200]}..."
             title = item.get("company_name", item.get("regulator_name", "Structured Data Content")) # Try company_name or regulator_name
             # Link/date might be extracted from items within these lists by process_linkup_structured_data
             link = item.get("source_url", link) # Maybe a top-level source_url exists?

        # Check for the wrapper format we might have created earlier in the pipeline if processing lists of structured results
        elif item.get("schema") and item.get("data") is not None:
             source_type = 'linkup_structured_raw' # From our wrapper dict format
             # Use a summary of the data content as the snippet
             data_content = item.get('data', {})
             snippet = f"Structured Data ({item['schema']}): {json.dumps(data_content)[:200]}..."
             # Title might be the entity name from the wrapper
             title = item.get("entity", "Structured Data")
             # Link might be a source_url within the data itself, extracted later in processing
             link = data_content.get("source_url", link) # Try to get source_url from data if not top-level


        else:
             # Default to snippet if it doesn't look like known structured data formats
             source_type = 'snippet' # Assume it's a snippet from a search engine


    else:
         print(f"Warning: Standardize_result received unexpected item type: {type(item)}. Item: {item}")
         return None

    # Basic validation for critical fields after determining type
    # Decide whether to keep items without URLs. For now, dropping them as they can't be sourced/verified snippets.
    # But structured data items might not have a direct URL at the top level. Keep those.
    if source_type not in ['linkup_structured_sdk', 'linkup_structured_raw', 'linkup_structured_content']:
        # Apply URL validation only to snippet-like results
        if not link or not isinstance(link, str) or not link.strip():
             # print(f"Skipping item from source '{source}' due to missing or invalid URL: {item}") # Optional logging
             return None
        pass # URL exists for snippet types


    # Allow items with just a link and either title OR snippet OR date
    # This logic might need adjustment if we strictly require snippets for NLP
    # But for standardization, let's allow it if it has a URL (for snippets) or is structured data
    if (source_type in ['snippet', 'linkup_snippet_sdk', 'snippet_list_item'] and (not title.strip() and not snippet.strip() and not date_str.strip())):
         # Keep if URL exists, even if other fields are empty for snippets
         pass
    elif (source_type in ['linkup_structured_sdk', 'linkup_structured_raw', 'linkup_structured_content'] and (not title.strip() and not snippet.strip())):
         # For structured data, require at least a title or the generated snippet summary
         pass # Keep if it's structured data


    # Ensure fields are strings if not None
    title = str(title).strip() if title is not None else ""
    snippet = str(snippet).strip() if snippet is not None else ""
    link = str(link).strip()
    date_str = str(date_str).strip() if date_str is not None else ""


    standardized_item = {
        "title": title,
        "url": link,
        "snippet": snippet,
        "source": source, # Keep the original source info passed in
        "published_date": date_str,
        "original_language": original_language,
        "translated_from": translated_from, # This will be set later by nlp_processor if translated
        "_source_type": source_type # Internal field to indicate item type
    }

    # Ensure language fields are strings if not None
    if standardized_item.get('original_language') is not None and not isinstance(standardized_item['original_language'], str):
         standardized_item['original_language'] = str(standardized_item['original_language'])
    if standardized_item.get('translated_from') is not None and not isinstance(standardized_item['translated_from'], str):
         standardized_item['translated_from'] = str(standardized_item['translated_from'])


    return standardized_item

if __name__ == "__main__":
    # Basic async test execution block
    async def async_test_suite():
        print("\n--- Running Local Async Search Engine Tests ---")
        print("NOTE: Local testing requires API keys for Google CSE, SerpApi, Linkup configured in .env")

        print("\nTesting Linkup Credit Check...")
        test_balance = await check_linkup_balance()
        if test_balance is not None:
            print(f"Linkup Available Credits during test: {test_balance:.4f}")
        else:
            print("Linkup credit check test failed or key is missing.")

        if google_api_client_available_and_configured:
            print("\nTesting async Google Official Search...")
            google_results = await search_google_official("example search query China", num=10)
            print(f"Found {len(google_results)} async Google Official results (max 10 per query).")
            # print(json.dumps(google_results[:2], indent=2)) # Print sample
        else:
            print("\nSkipping async Google Official Search test: Configuration missing or client not available.")

        if serpapi_available:
            print("\nTesting async SerpApi Google Search...")
            serpapi_google_results = await search_via_serpapi("example search query China", engine='google', country_code='cn', num=20)
            print(f"Found {len(serpapi_google_results)} async SerpApi Google results (requested num 20).")
            # print(json.dumps(serpapi_google_results[:2], indent=2)) # Print sample
        else:
            print("\nSkipping async SerpApi Google Search test: Configuration missing or client not available.")

        if serpapi_available:
             print("\nTesting async SerpApi Baidu Search...")
             serpapi_baidu_results = await search_via_serpapi("企业合规风险", engine='baidu', country_code='cn', num=20)
             print(f"Found {len(serpapi_baidu_results)} async SerpApi Baidu results (requested num 20).")
             # print(json.dumps(serpapi_baidu_results[:2], indent=2)) # Print sample
        else:
             print("\nSkipping async SerpApi Baidu Search test: Configuration missing or client not available.")

        if linkup_search_enabled:
             print("\nTesting async Linkup Snippet Search (Single Query)...")
             try:
                 # Pass a query that should yield more than 20 results if possible
                 # Removed num=20 from here to rely on Linkup's default per query
                 linkup_snippets_results = await search_linkup_snippets("Apple Inc OR Google OR Microsoft OR Amazon financial news")
                 print(f"Found {len(linkup_snippets_results)} async Linkup Snippet Search unique results (single query).")
                 # Print sample to verify no truncation occurred in the function itself
                 print(json.dumps(linkup_snippets_results[:5], indent=2))
             except Exception as e:
                 print(f"Error during async Linkup Snippet Search (Single) test: {type(e).__name__}: {e}")
                 traceback.print_exc()
        else:
            print("\nSkipping async Linkup Snippet Search test: Linkup search is not enabled.")

        if linkup_search_enabled:
             print("\nTesting async Linkup Snippet Search (Multiple Queries)...")
             try:
                 # Pass multiple queries that, when combined, might yield more than 20 unique results
                 multi_queries = ["Apple Inc China", "Apple compliance China", "Apple tax China", "Apple regulatory China"]
                 # Removed num=20 from here to rely on Linkup's default per query
                 linkup_snippets_multi_results = await search_linkup_snippets(multi_queries)
                 print(f"Found {len(linkup_snippets_multi_results)} Linkup Snippet Search unique results (multiple queries).")
                 # Print sample to verify no truncation occurred in the function itself
                 print(json.dumps(linkup_snippets_multi_results[:5], indent=2))
             except Exception as e:
                 print(f"Error during async Linkup Snippet Search (Multiple) test: {type(e).__name__}: {e}")
                 traceback.print_exc()
        else:
            print("\nSkipping async Linkup Snippet Search (Multiple) test: Linkup search is not enabled.")


        if linkup_search_enabled:
             print("\nTesting async Linkup Structured Search (with dummy schema)...")
             dummy_structured_schema = json.dumps({"type": "object", "properties": {"company_name": {"type": "string"}}})
             try:
                 # Pass country code to structured search as well, just in case SDK uses it. Removed country_code parameter.
                 linkup_structured_results = await search_linkup_structured("Apple Inc ownership structure China", structured_output_schema=dummy_structured_schema, depth="deep")
                 print(f"Linkup async Structured Search Result Type: {type(linkup_structured_results)}")
                 if isinstance(linkup_structured_results, dict):
                      print(f"Linkup async Structured Search found data with keys: {list(linkup_structured_results.keys())}")
                 elif isinstance(linkup_structured_results, list) and linkup_structured_results and isinstance(linkup_structured_results[0], dict):
                      print(f"Linkup async Structured Search found list of data items. Count: {len(linkup_structured_results)}. First item keys: {list(linkup_structured_results[0].keys())}")
                 else:
                      print("Linkup async Structured Search did not return expected type (dict or list of dict).")
                 # print(json.dumps(linkup_structured_results, indent=2)) # Print full result
             except Exception as e:
                 print(f"Error during async Linkup Structured Search test: {type(e).__name__}: {e}")
                 traceback.print_exc()
        else:
             print("\nSkipping async Linkup Structured Search test: Linkup search is not enabled.")


        print("\nTesting async Wayback Machine Check...")
        wayback_result = await check_wayback_machine("https://www.example.com")
        print(f"Wayback Check Result: {wayback_result}")
        wayback_result_404 = await check_wayback_machine("https://www.example.com/nonexistent-page-12345")
        print(f"Wayback Check Result (404): {wayback_result_404}")
        wayback_result_invalid = await check_wayback_machine("")
        print(f"Wayback Check Result (Invalid URL): {wayback_result_invalid}")


        print("\nTesting Targeted Ownership Docs Search (Combines async sources with fallback)...")
        # This function calls the individual async search functions concurrently.
        # SerpApi is fallback if primary (Linkup+Google) don't hit threshold.
        ownership_docs_results = await search_for_ownership_docs("Example Chinese Company", num_per_query=10, country_code='cn') # num_per_query affects internal search calls
        print(f"Found {len(ownership_docs_results)} unique potential ownership documents across all sources.") # Report total unique found
        # print(json.dumps(ownership_docs_results[:5], indent=2)) # Print sample


        print("\n--- Local Async Search Engine Tests Complete ---")

    # Run the async test suite
    if __name__ == "__main__":
        try:
            asyncio.run(async_test_suite())
        except KeyboardInterrupt:
            print("\nAsync test suite interrupted.")
        except Exception as e:
            print(f"\n--- Async Test Suite Exception ---")
            print(f"An exception occurred during the async test run: {type(e).__name__}: {e}")
            traceback.print_exc()