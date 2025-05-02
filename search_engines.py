# search_engines.py

import requests
import time
from typing import List, Dict, Any, Optional, Union
import traceback
import json
import re

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

try:
    # Attempt to import Linkup components
    from linkup import LinkupClient as _LinkupClient
    from linkup import LinkupSearchResults as _LinkupSearchResults
    from linkup import LinkupSearchTextResult as _LinkupSearchTextResult

    # Check if the imported components are valid (optional but good practice)
    if _LinkupClient is not None and _LinkupSearchResults is not None and _LinkupSearchTextResult is not None:
         LinkupClient = _LinkupClient
         LinkupSearchResults = _LinkupSearchResults
         LinkupSearchTextResult = _LinkupSearchTextResult
         linkup_library_available = True
    else:
         # If imports didn't fail but components are None, something is still wrong
         print("Warning: Imported Linkup library but required components are None. Linkup searches disabled.")
         linkup_library_available = False
except ImportError:
    print("Warning: linkup library not installed. Linkup searches disabled.")
    linkup_library_available = False
except Exception as e:
     # Catch any other potential import errors
     print(f"Error importing Linkup library: {e}. Linkup searches disabled.")
     linkup_library_available = False


try:
    from googleapiclient.discovery import build as build_google_service
    google_api_client_available = True
except ImportError:
    print("Warning: google-api-python-client not installed. Official Google Search disabled.")
    google_api_client_available = False
    build_google_service = None

try:
    from serpapi import GoogleSearch as SerpApiSearch
    serpapi_library_available = True
except ImportError:
    print("Warning: google-search-results (SerpApi) library not installed. SerpApi searches disabled.")
    serpapi_library_available = False
    SerpApiSearch = None

def check_linkup_balance() -> Optional[float]:
    """Checks the remaining Linkup API credits using a direct requests call."""
    if not config or not getattr(config, 'LINKUP_API_KEY', None):
        return None

    balance_url = "https://api.linkup.so/v1/credits/balance"
    headers = {"Authorization": f"Bearer {config.LINKUP_API_KEY}"}

    try:
        response = requests.get(balance_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        balance = data.get("balance")
        if isinstance(balance, (int, float)):
            return float(balance)
        else:
            print(f"Warning: Linkup balance response format unexpected. Got: {data}")
            return None
    except requests.exceptions.Timeout:
        print("Warning: Timeout checking Linkup credits.")
        return None
    except requests.exceptions.RequestException as req_e:
        print(f"Warning: Request error checking Linkup credits: {req_e}")
        return None
    except Exception as e:
        print(f"Warning: Unexpected error checking Linkup credits: {e}")
        return None

linkup_client = None
linkup_search_enabled = False

# Initialize Linkup client only if library is available and key is configured
if linkup_library_available:
    if not config or not getattr(config, 'LINKUP_API_KEY', None):
        print("Warning: Linkup API Key missing in config/.env. Linkup searches disabled.")
    else:
        try:
            linkup_client = LinkupClient(api_key=config.LINKUP_API_KEY)
            # Optional: Verify connectivity if SDK provides a method, or just proceed.
            # print("Linkup client initialized successfully.")
            balance = check_linkup_balance()
            if balance is not None:
                print(f"Linkup Available Credits: {balance:.4f}")
                linkup_search_enabled = True # Only enable if balance check (or init) is successful
                print("Linkup searches enabled.")
            else:
                print("Could not retrieve Linkup credit balance or initialization failed. Linkup searches disabled.")
                linkup_client = None # Set client to None if balance check fails
        except Exception as e:
            print(f"ERROR initializing Linkup client with API key: {type(e).__name__}: {e}")
            traceback.print_exc()
            linkup_client = None
            print("Warning: Linkup client initialization failed. Linkup searches disabled.")
else:
     print("Linkup library not available, Linkup searches disabled.") # Re-iterate if library import failed


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


def search_google_official(query: str, lang: str = 'en', num: int = 10) -> list:
    """Search using Google Custom Search JSON API."""
    if not google_api_client_available_and_configured:
        return []
    if not query or not isinstance(query, str) or not query.strip():
         print("Skipping Google Official Search: Invalid query provided.")
         return []
    try:
        service = build_google_service("customsearch", "v1", developerKey=config.GOOGLE_API_KEY_SEARCH)
        lr_param = f"lang_{lang}" if lang else None
        # Google CSE API limits 'num' to a maximum of 10 per request.
        # We still pass the requested 'num' but know the API caps it.
        effective_num = min(num, 10)
        print(f"Executing Google Official Search: q='{query}', lang='{lang}', num={effective_num}")
        result = service.cse().list( q=query, cx=config.GOOGLE_CX, num=effective_num, lr=lr_param ).execute()
        return result.get("items", [])
    except Exception as e:
        print(f"ERROR during Google Official search for '{query}': {type(e).__name__}: {e}")
        traceback.print_exc()
        return []

def search_via_serpapi(query: str, engine: str, country_code: str = 'cn', lang_code: str = 'en', num: int = 20) -> list:
    """Search using SerpApi for various engines."""
    if not serpapi_available:
        return []
    if not query or not isinstance(query, str) or not query.strip():
         print(f"Skipping SerpApi search ({engine}): Invalid query provided.")
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


        print(f"Executing SerpApi Search: engine='{engine}', q='{query[:50]}...', params={params}")
        search = SerpApiSearch(params)
        results = search.get_dict()
        if "error" in results:
            print(f"SerpApi Error ({engine}) for '{query[:50]}...': {results['error']}")
            return []
        return results.get("organic_results", [])
    except Exception as e:
        print(f"ERROR during SerpApi search ({engine}) for '{query[:50]}...': {type(e).__name__}: {e}")
        traceback.print_exc()
        return []

def search_linkup_snippets(query: Union[str, List[str]], num: int = 20, country_code: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search using Linkup API client with output_type='searchResults'.
    Accepts a single query string or a list of query strings.
    Maps Linkup results to the standard snippet format expected by NLP.
    Returns all unique results found across executed queries.
    The 'num' parameter is passed to the Linkup API call for each query,
    but this function does NOT impose an external limit on the *total* results.
    """
    if not linkup_search_enabled:
        print("Linkup searches are not enabled.")
        return []

    all_linkup_results: List[Dict[str, Any]] = []
    processed_urls = set()

    queries_to_run = [query] if isinstance(query, str) else query
    queries_to_run = [q.strip() for q in queries_to_run if q.strip()] # Ensure queries are valid strings

    if not queries_to_run:
         print("No valid queries provided for Linkup Snippet Search.")
         return []

    print(f"Executing Linkup Snippet Search for {len(queries_to_run)} queries (num parameter per query: {num}, requested_country_code={country_code})...")

    for q in queries_to_run:
        try:
            print(f"  Linkup Query: '{q[:50]}...'") # Print truncated query

            params = {
                "query": q,
                "depth": "standard", # or "deep" depending on desired depth
                "output_type": "searchResults",
                "include_images": False,
                # Removed 'num' parameter from here as it caused the TypeError.
                # Linkup API might have other ways to control result quantity, or a default limit.
                # We rely on Linkup's internal limits per query.
            }
            # Removed 'country_code' as a direct parameter based on previous errors.
            # Country context should ideally be in the query string itself for Linkup.

            response = linkup_client.search(**params) # Corrected params no longer include 'num' or 'country_code'

            if isinstance(response, LinkupSearchResults) and hasattr(response, 'results') and isinstance(response.results, list):
                 linkup_results_list_for_query = response.results
                 print(f"    Linkup API returned {len(linkup_results_list_for_query)} results for query '{q[:50]}...'.")

                 # Iterate through the results from this query and add unique ones to the combined list
                 for item in linkup_results_list_for_query:
                      # Standardize each item. standardize_result handles LinkupSDK objects.
                      standardized = standardize_result(item, source='linkup_snippet_search')
                      if standardized and standardized.get('url') and standardized['url'] not in processed_urls:
                           all_linkup_results.append(standardized)
                           processed_urls.add(standardized['url'])
            else:
                 print(f"    Linkup Snippet Search for query '{q[:50]}...' returned no results or unexpected format.")

        except Exception as e:
            print(f"ERROR during Linkup snippet search for query '{q[:50]}...': {type(e).__name__}: {e}")
            traceback.print_exc()

        if len(queries_to_run) > 1:
             time.sleep(0.5) # Small delay between multiple Linkup queries

    # No final truncation here, return all unique results found across sources up to their individual limits/thresholds.

    print(f"Linkup combined snippet search returned {len(all_linkup_results)} unique results after internal deduplication.")
    return all_linkup_results


def search_linkup_structured(query: str, structured_output_schema: str, depth: str = "deep", country_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Search using Linkup API client with output_type='structured'.
    Returns the parsed JSON response or None on failure.
    """
    if not linkup_search_enabled:
        print("Linkup searches are not enabled.")
        return None
    if not structured_output_schema:
        print("Skipping Linkup structured search: No schema provided.")
        return None
    if not query or not isinstance(query, str) or not query.strip():
         print("Skipping Linkup structured search: No valid query provided.")
         return None


    try:
        print(f"Executing Linkup Structured Search: q='{query[:50]}...', depth='{depth}', requested_country_code={country_code}") # Print truncated query
        params = {
            "query": query,
            "depth": depth,
            "output_type": "structured",
            "structured_output_schema": structured_output_schema,
            "include_images": False,
            # Removed country_code here as it caused TypeError. Rely on query string.
        }

        response = linkup_client.search(**params) # Corrected params no longer include 'country_code'

        if response is not None:
            # Linkup structured search can return a dict, a list of LinkupSDK objects, or other formats.
            # We expect a dict representing the structured data.
            if isinstance(response, dict):
                 # If the response is a dict, return it directly
                 return response
            elif hasattr(response, 'data') and isinstance(response.data, dict):
                 # If the response is an object with a 'data' attribute that is a dict, return that
                 return response.data
            # Add handling for LinkupSearchResults type if a structured search could potentially return it
            elif isinstance(response, LinkupSearchResults) and hasattr(response, 'results') and isinstance(response.results, list) and len(response.results) > 0:
                 # If it returns a LinkupSearchResults object containing a list of results,
                 # it might be a list of structured items, each with a 'data' attribute.
                 # Let's check if items in the list have a 'data' attribute that is a dict.
                 valid_structured_data_list = []
                 for item in response.results:
                      if hasattr(item, 'data') and isinstance(getattr(item, 'data', None), dict):
                           valid_structured_data_list.append(item.data)
                      # else: # Optional: log items that don't fit the expected structured format
                      #      print(f"Warning: Linkup Structured list item lacks valid 'data' attribute: {item}")

                 if valid_structured_data_list:
                      print(f"Linkup Structured Search returned LinkupSearchResults list, extracted {len(valid_structured_data_list)} structured data items.")
                      # If there's only one structured data item in the list, return it directly as a dict for consistency
                      if len(valid_structured_data_list) == 1:
                           return valid_structured_data_list[0]
                      else:
                           # If there are multiple structured data items, return the list
                           return valid_structured_data_list
                 else:
                      print(f"Linkup Structured Search returned LinkupSearchResults list, but no valid structured data items found within.")
                      return None

            elif isinstance(response, str):
                 # If the response is a string, attempt to parse it as JSON
                 print(f"Linkup Structured Search returned string. Attempting JSON parse.")
                 try:
                      parsed_response = json.loads(response)
                      if isinstance(parsed_response, dict):
                          return parsed_response
                      elif isinstance(parsed_response, list) and parsed_response and isinstance(parsed_response[0], dict):
                           # If it's a list of dictionaries, validate and return it
                           print(f"Linkup Structured Search string parsed into a list of dictionaries ({len(parsed_response)} items).")
                           valid_list = [item for item in parsed_response if isinstance(item, dict)]
                           if len(valid_list) != len(parsed_response):
                                print(f"Warning: Parsed list from string contained non-dictionary items. Kept {len(valid_list)} out of {len(parsed_response)}.")
                           # If there's only one item in the list, return the dict directly
                           if len(valid_list) == 1:
                                return valid_list[0]
                           else:
                                return valid_list
                      else:
                          print(f"Warning: Parsed Linkup Structured response string is not a dictionary or list of dicts. Type: {type(parsed_response)}. Response: {response[:200]}...")
                          return None
                 except json.JSONDecodeError:
                      print(f"ERROR decoding JSON from Linkup Structured response string: {response[:200]}...")
                      return None
            else:
                 # Handle any other unexpected response types
                 print(f"Linkup Structured Search returned unexpected type: {type(response)}. Response: {response}")
                 return None
        else:
            print("Linkup Structured Search returned None response.")
            return None

    except Exception as e:
        print(f"ERROR during Linkup structured search for '{query[:50]}...': {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

def check_wayback_machine(url_to_check: str) -> dict:
    """Check Wayback Machine Availability API."""
    if not url_to_check or not isinstance(url_to_check, str) or not url_to_check.strip():
        # print("Skipping Wayback check for invalid or empty URL.")
        return {"status": "skipped_invalid_url", "original_url": url_to_check, "message": "Invalid URL provided"}
    api_url = f"https://archive.org/wayback/available?url={requests.utils.quote(url_to_check)}"
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        closest = data.get("archived_snapshots", {}).get("closest", {})
        if closest and closest.get("available") and closest.get("status", "").startswith("2"):
            return {"status": "found", "original_url": url_to_check, "wayback_url": closest.get("url"), "timestamp": closest.get("timestamp")}
        else: return {"status": "not_found", "original_url": url_to_check, "message": "No archive found"}
    except requests.exceptions.Timeout: print(f"Timeout checking Wayback: {url_to_check[:100]}..."); return {"status": "error", "original_url": url_to_check, "message": "Timeout checking Wayback Machine."}
    except requests.exceptions.RequestException as req_e: print(f"Request Error checking Wayback: {url_to_check[:100]}... Error: {req_e}"); return {"status": "error", "original_url": url_to_check, "message": f"Request Error checking Wayback: {req_e}"}
    except Exception as e: print(f"Unexpected Error checking Wayback: {url_to_check[:100]}... Error: {e}"); return {"status": "error", "original_url": url_to_check, "message": f"Unexpected error checking Wayback: {str(e)}"}

def search_for_ownership_docs(entity_name: str,
                              num_per_query: int = 10, # This is still passed to individual search functions that support it
                              country_code: str = 'cn', # This influences query generation and search engine parameters
                              lang_code: str = 'en'
                              ) -> List[Dict[str, Any]]:
    """
    Performs targeted searches to find potential ownership documents/mentions
    related to a specific entity. Prioritizes Linkup snippet search if enabled,
    then Google CSE, then Serpapi. Returns standardized snippet results.
    Collects all unique results found across enabled sources.
    """
    print(f"\n--- Searching for ownership docs (snippets) for: {entity_name} (Focus Country: {country_code}, Lang: {lang_code}) ---")
    if not entity_name or not isinstance(entity_name, str) or not entity_name.strip():
        print("Skipping ownership docs search: Invalid entity name.")
        return []

    queries = [
        f'"{entity_name}" "subsidiaries" OR "affiliates" list',
        f'"{entity_name}" "ownership structure" OR "shareholding"',
        f'"{entity_name}" annual report OR filing "equity method investments"',
        f'"{entity_name}" "controlling interest" OR "majority stake"',
        f'"{entity_name}" "joint venture" OR "partnership structure"',
        f'"{entity_name}" parent company OR subsidiary news',
        # Add a broader query for Linkup specifically, if not using the combined approach below
    ]
    # Add country context to queries for web search engines
    country_name = country_code.upper()
    try:
         import pycountry
         country = pycountry.countries.get(alpha_2=country_code.upper())
         if country: country_name = country.name
    except Exception: pass # pycountry_available should handle this check higher up

    web_search_queries = [f"{q} {country_name}" for q in queries]
    web_search_queries = list(set([q.strip() for q in web_search_queries if q.strip()]))


    all_raw_results_map = {} # Use a map to store results by URL for deduplication
    search_source_prefix = "ownership_multi"


    google_cse_available_bool = bool(google_api_client_available_and_configured and hasattr(search_google_official, '__call__'))
    serpapi_available_bool = bool(serpapi_available and hasattr(search_via_serpapi, '__call__'))
    linkup_search_enabled_bool = bool(linkup_search_enabled and hasattr(search_linkup_snippets, '__call__'))


    # Define a reasonable target for this specific type of search
    # This target is NOT used to truncate, but to decide if we need to continue to the next search engine
    overall_results_threshold = max(num_per_query * 2, 20) # e.g., stop after finding 20+ results across engines


    linkup_attempted = False
    if linkup_search_enabled_bool:
         linkup_attempted = True
         print(f"Using Linkup Snippet Search for ownership docs (q='{entity_name} ownership', num={num_per_query}, country={country_code})...")
         # Use a combined query approach for Linkup snippet search
         linkup_query_combined = f'"{entity_name}" ownership OR subsidiary OR affiliate OR stake OR filing OR "joint venture" OR "acquired" OR parent OR "equity method" {country_code}' # Add country code to query
         try:
              # search_linkup_snippets now returns all unique results found for the queries it runs
              # Pass the combined query and num to search_linkup_snippets. Removed country_code parameter here.
              linkup_results_list = search_linkup_snippets(linkup_query_combined, num=num_per_query)
              if linkup_results_list:
                   print(f"    Linkup Snippet Search found {len(linkup_results_list)} unique results.")
                   for r in linkup_results_list:
                       # standardize_result handles LinkupSDK objects and dicts
                       standardized = standardize_result(r, source='linkup_snippet_search')
                       if standardized and standardized.get('url') and standardized['url'] not in all_raw_results_map:
                            all_raw_results_map[standardized['url']] = standardized # Store standardized result directly
         except Exception as e:
              print(f"ERROR during Linkup snippet ownership search: {type(e).__name__}: {e}")
              traceback.print_exc()
         # No sleep here as search_linkup_snippets has internal delays if multiple queries are passed (though here it's one combined query)


    # Check if we have enough results before hitting other, potentially more expensive APIs
    if google_cse_available_bool:
        if len(all_raw_results_map) < overall_results_threshold:
            google_api_attempted = True
            print(f"Using Google Official API (lang={lang_code}) for ownership search...")

            # Iterate through web search queries
            for q_idx, q in enumerate(web_search_queries):
                if len(all_raw_results_map) >= overall_results_threshold:
                     print(f"Stopping Google CSE search early due to sufficient total results ({len(all_raw_results_map)} >= {overall_results_threshold}).")
                     break # Stop if we've hit our overall threshold

                try:
                     # num_per_query is capped at 10 by search_google_official
                     google_results_list = search_google_official(q, lang=lang_code, num=num_per_query);
                     for r in google_results_list:
                         standardized = standardize_result(r, source=f'google_cse_ownership_q{q_idx+1}')
                         if standardized and standardized.get('url') and standardized['url'] not in all_raw_results_map:
                              all_raw_results_map[standardized['url']] = standardized
                     time.sleep(0.3) # Small delay between Google CSE calls
                except Exception as e: print(f"    Google CSE call failed for query '{q[:50]}...': {type(e).__name__}: {e}"); traceback.print_exc()
        elif not google_cse_available_bool:
            print("Skipping Google CSE queries - not configured or function not available.")
        elif len(all_raw_results_map) >= overall_results_threshold:
             print(f"Skipping Google CSE. Preceding steps returned {len(all_raw_results_map)} results (>= {overall_results_threshold} threshold).")


    if serpapi_available_bool:
        # Use all generated queries for Serpapi, including Chinese ones if cn, and general ones with country context
        serpapi_queries = queries + web_search_queries
        # Attempt to get Chinese keywords via NLP only if country is China and NLP is available
        if country_code.lower() == 'cn' and nlp_processor_available_search and hasattr(nlp_processor, 'translate_keywords_for_context') and callable(nlp_processor.translate_keywords_for_context):
             try:
                  # Use a default LLM config for this internal call if main config is not available
                  # Need to ensure config is available and has the default models before using them
                  nlp_llm_provider = 'openai' # Default fallback provider name
                  nlp_llm_model = 'gpt-4o-mini' # Default fallback model name

                  if config and hasattr(config, 'OPENAI_API_KEY') and getattr(config, 'OPENAI_API_KEY', None):
                       nlp_llm_provider = 'openai'
                       nlp_llm_model = getattr(config, 'DEFAULT_OPENAI_MODEL', 'gpt-4o-mini')
                  elif config and hasattr(config, 'OPENROUTER_API_KEY') and getattr(config, 'OPENROUTER_API_KEY', None):
                       nlp_llm_provider = 'openrouter'
                       nlp_llm_model = getattr(config, 'DEFAULT_OPENROUTER_MODEL', 'google/gemini-flash-1.5')
                  elif config and hasattr(config, 'GOOGLE_AI_API_KEY') and getattr(config, 'GOOGLE_AI_API_KEY', None):
                       nlp_llm_provider = 'google_ai'
                       nlp_llm_model = getattr(config, 'DEFAULT_GOOGLE_AI_MODEL', 'models/gemini-1.5-flash-latest')
                  else:
                       # No LLM keys found via config for NLP translation
                       print("Warning: No LLM API keys found in config for NLP translation in search_for_ownership_docs.")
                       nlp_llm_provider = None # Explicitly set to None to skip NLP call

                  # Add check if the chosen NLP LLM config is valid before calling NLP
                  if nlp_llm_provider:
                       try:
                            # Check availability and configure transiently if needed
                            nlp_processor._get_llm_client_and_model(nlp_llm_provider, nlp_llm_model)
                            chinese_keywords = nlp_processor.translate_keywords_for_context(f'"{entity_name}" ownership structure', f"Baidu search for {entity_name} ownership in {country_code}", nlp_llm_provider, nlp_llm_model) # Use specific context for translation
                            serpapi_queries.extend(chinese_keywords)
                       except Exception as nlp_llm_e:
                            print(f"Warning: Failed to initialize/use NLP LLM for Chinese keyword generation for Serpapi ownership search: {nlp_llm_e}")


             except Exception as e:
                  print(f"Warning: Failed to get Chinese keywords for Serpapi ownership search: {e}")


        serpapi_queries = list(set([q.strip() for q in serpapi_queries if q.strip()])) # Deduplicate again


        if len(all_raw_results_map) < overall_results_threshold:
            serpapi_attempted = True
            serpapi_engine = 'baidu' if country_code.lower() == 'cn' else 'google'
            print(f"Using SerpApi ({serpapi_engine}, gl={country_code}, hl={lang_code}) for ownership search...")

            # Iterate through SerpApi queries
            for q_idx, query_text in enumerate(serpapi_queries):
                 if len(all_raw_results_map) >= overall_results_threshold:
                      print(f"Stopping Serpapi search early due to sufficient total results ({len(all_raw_results_map)} >= {overall_results_threshold}).")
                      break # Stop if we've hit our overall threshold

                 print(f"  SerpApi ({serpapi_engine}) Query {q_idx+1}/{len(serpapi_queries)}: '{query_text[:50]}...'")
                 try:
                      # num_per_query is passed to search_via_serpapi
                      serpapi_results_list = search_via_serpapi(query_text, serpapi_engine, country_code, lang_code='en', num=num_per_query);
                      if serpapi_results_list:
                           print(f"    SerpApi {serpapi_engine} found {len(serpapi_results_list)} raw results.")
                           for r in serpapi_results_list:
                               standardized = standardize_result(r, source=f'serpapi_{serpapi_engine}_ownership_q{q_idx+1}')
                               # Attempt to set original language if not already set by standardize_result
                               if standardized and standardized.get('original_language') is None:
                                   # For Baidu, assume Chinese unless it clearly wasn't the search language
                                   if serpapi_engine == 'baidu': standardized['original_language'] = 'zh'
                                   # For Google, it's less certain, can leave as None or default to 'en'
                                   # The NLP translation step will handle non-English snippets anyway.

                               if standardized and standardized.get('url') and standardized['url'] not in all_raw_results_map:
                                    all_raw_results_map[standardized['url']] = standardized
                           else: print(f"    SerpApi {serpapi_engine} returned no results for this query.")
                      time.sleep(0.5) # Small delay between SerpApi calls
                 except Exception as e: print(f"    SerpApi ({serpapi_engine}) call failed for query '{query_text[:50]}...': {type(e).__name__}: {e}"); traceback.print_exc()
        elif not serpapi_available_bool:
            print("Skipping SerpApi search - not configured or function not available.")
        elif len(all_raw_results_map) >= overall_results_threshold:
             print(f"Skipping Serpapi. Preceding steps returned {len(all_raw_results_map)} results (>= {overall_results_threshold} threshold).")


    final_results = list(all_raw_results_map.values())

    # No final truncation here, return all unique results found across sources up to their individual limits/thresholds.

    print(f"Found {len(final_results)} unique potential ownership documents across sources.")
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

    # Check for Linkup SDK object first
    if linkup_library_available and isinstance(item, (LinkupSearchTextResult, LinkupSearchResults)):
         # Handle LinkupSearchTextResult (from searchResults output)
         if isinstance(item, LinkupSearchTextResult):
              title = getattr(item, 'name', '')
              link = getattr(item, 'url', '')
              snippet = getattr(item, 'content', '')
              date_str = getattr(item, 'date', '') # Linkup TextResult might have a date attribute
              original_language = getattr(item, 'language', None)
              source_type = 'snippet'

         # Handle LinkupSearchResults (which contains a list of TextResults) - Although search_linkup_snippets
         # extracts items from the list, this is a safeguard if LinkupSearchResults object itself is passed.
         # It's unlikely this branch will be hit with current search_linkup_snippets logic.
         elif isinstance(item, LinkupSearchResults) and hasattr(item, 'results') and isinstance(item.results, list) and item.results:
             # Process the first item as a representative
             first_item = item.results[0]
             if isinstance(first_item, LinkupSearchTextResult):
                  title = getattr(first_item, 'name', '') + " (from SearchResults list)"
                  link = getattr(first_item, 'url', '')
                  snippet = getattr(first_item, 'content', '')
                  date_str = getattr(first_item, 'date', '')
                  original_language = getattr(first_item, 'language', None)
                  source_type = 'snippet_list_item'
             # Add handling for structured results within LinkupSearchResults if needed
             # elif hasattr(first_item, 'data') and isinstance(first_item.data, dict):
             #     title = "Linkup Structured Result"
             #     link = getattr(first_item, 'url', '') # Structured results might not have a direct URL
             #     snippet = json.dumps(getattr(first_item, 'data', {}))[:200] + '...' # Summarize data
             #     source_type = 'structured_list_item'


    elif isinstance(item, dict):
        # Handle dictionary-based results from Google CSE, SerpApi, or raw Linkup JSON/dictionary outputs
        title = item.get("title", item.get("Title", ""))
        link = item.get("link", item.get("url", item.get("displayed_link", item.get("Link", item.get("Url", item.get("sourceUrl", item.get("source_url", ""))))))) # Added source_url from Linkup structured
        # Corrected the unmatched parenthesis in the snippet get chain by ensuring balance
        snippet = item.get("snippet", item.get("description", item.get("Snippet", item.get("Description", item.get("Summary", item.get("textSnippet", item.get("excerpt", item.get("content", ""))))))))
        date_str = item.get("date", item.get("published_date", item.get("Date", item.get("PublishedDate", item.get("publish_date", item.get("timestamp", item.get("source_date", ""))))))) # Added source_date from Linkup structured
        # Attempt to get language from known keys if available
        original_language = item.get("language", item.get("lang", item.get("original_language", None))) # Added original_language

        # Check if this dict item looks like structured data rather than a snippet
        if item.get("schema") and item.get("data") is not None:
             source_type = 'structured_item'
             # For structured items, use a summary of the data as the snippet
             snippet = f"Structured Data ({item['schema']}): {json.dumps(item['data'])[:200]}..."
             # Title might be less relevant, maybe use entity name if available
             title = item.get("entity", "Structured Data")
             # Link might be a source_url within the data itself, extracted later in processing

        elif item.get("ownership_relationships") is not None or item.get("key_risks_identified") is not None:
             # This dict might be the 'data' part of a structured item itself
             source_type = 'structured_data_content'
             snippet = f"Structured Data Content: {json.dumps(item)[:200]}..."
             title = item.get("company_name", "Structured Data Content")
             # Link/date would be extracted from items within these lists by process_linkup_structured_data


        else:
             # Default to snippet if it doesn't look like known structured data formats
             source_type = 'snippet'


    else:
         print(f"Warning: Standardize_result received unexpected item type: {type(item)}. Item: {item}")
         return None

    # Basic validation for critical fields after determining type
    if not link or not isinstance(link, str) or not link.strip():
        # print(f"Skipping item from source '{source}' due to missing or invalid URL: {item}")
        # Decide whether to keep items without URLs. For now, dropping them as they can't be sourced/verified.
        return None
    # Allow items with just a link and either title OR snippet OR date
    if not title.strip() and not snippet.strip() and not date_str.strip():
         # print(f"Skipping item from source '{source}' with URL '{link}' due to missing title, snippet, and date.")
         # Decide whether to keep items with just a URL. Dropping for now unless they have some minimal info.
         # return None # Keep for now, let NLP handle empty snippets

        pass # Keep if URL exists, even if other fields are empty

    # Ensure title and snippet are strings
    title = str(title).strip() if title is not None else ""
    snippet = str(snippet).strip() if snippet is not None else ""
    link = str(link).strip()

    # Ensure date_str is a string if not None
    if date_str is not None and not isinstance(date_str, str):
        date_str = str(date_str).strip()
    elif date_str is None:
        date_str = ""
    else:
         date_str = date_str.strip()


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
    print("\n--- Running Local Search Engine Tests ---")
    print("NOTE: Requires API keys for Google CSE, SerpApi, Linkup configured in .env")

    print("\nTesting Linkup Credit Check...")
    test_balance = check_linkup_balance()
    if test_balance is not None:
        print(f"Linkup Available Credits during test: {test_balance:.4f}")
    else:
        print("Linkup credit check test failed or key is missing.")

    if google_api_client_available_and_configured:
        print("\nTesting Google Official Search...")
        google_results = search_google_official("example search query China", num=10)
        print(f"Found {len(google_results)} Google Official results (max 10 per query).")
        # print(json.dumps(google_results[:2], indent=2)) # Print sample
    else:
        print("\nSkipping Google Official Search test: Configuration missing or client not available.")

    if serpapi_available:
        print("\nTesting SerpApi Google Search...")
        serpapi_google_results = search_via_serpapi("example search query China", engine='google', country_code='cn', num=20)
        print(f"Found {len(serpapi_google_results)} SerpApi Google results (requested num 20).")
        # print(json.dumps(serpapi_google_results[:2], indent=2)) # Print sample
    else:
        print("\nSkipping SerpApi Google Search test: Configuration missing or client not available.")

    if serpapi_available:
         print("\nTesting SerpApi Baidu Search...")
         serpapi_baidu_results = search_via_serpapi("企业合规风险", engine='baidu', country_code='cn', num=20)
         print(f"Found {len(serpapi_baidu_results)} SerpApi Baidu results (requested num 20).")
         # print(json.dumps(serpapi_baidu_results[:2], indent=2)) # Print sample
    else:
         print("\nSkipping SerpApi Baidu Search test: Configuration missing or client not available.")

    if linkup_search_enabled:
         print("\nTesting Linkup Snippet Search (Single Query)...")
         try:
             # Pass a query that should yield more than 20 results if possible
             # Removed num=20 from here to rely on Linkup's default per query
             linkup_snippets_results = search_linkup_snippets("Apple Inc OR Google OR Microsoft OR Amazon financial news")
             print(f"Found {len(linkup_snippets_results)} Linkup Snippet Search unique results (single query).")
             # Print sample to verify no truncation occurred in the function itself
             print(json.dumps(linkup_snippets_results[:5], indent=2))
         except Exception as e:
             print(f"Error during Linkup Snippet Search (Single) test: {type(e).__name__}: {e}")
             traceback.print_exc()
    else:
        print("\nSkipping Linkup Snippet Search test: Linkup search is not enabled.")

    if linkup_search_enabled:
         print("\nTesting Linkup Snippet Search (Multiple Queries)...")
         try:
             # Pass multiple queries that, when combined, might yield more than 20 unique results
             multi_queries = ["Apple Inc China", "Apple compliance China", "Apple tax China", "Apple regulatory China"]
             # Removed num=20 from here to rely on Linkup's default per query
             linkup_snippets_multi_results = search_linkup_snippets(multi_queries)
             print(f"Found {len(linkup_snippets_multi_results)} Linkup Snippet Search unique results (multiple queries).")
             # Print sample to verify no truncation occurred in the function itself
             print(json.dumps(linkup_snippets_multi_results[:5], indent=2))
         except Exception as e:
             print(f"Error during Linkup Snippet Search (Multiple) test: {type(e).__name__}: {e}")
             traceback.print_exc()
    else:
        print("\nSkipping Linkup Snippet Search (Multiple) test: Linkup search is not enabled.")


    if linkup_search_enabled:
         print("\nTesting Linkup Structured Search (with dummy schema)...")
         dummy_structured_schema = json.dumps({"type": "object", "properties": {"company_name": {"type": "string"}}})
         try:
             # Pass country code to structured search as well, just in case SDK uses it. Removed country_code parameter.
             linkup_structured_results = search_linkup_structured("Apple Inc ownership structure China", structured_output_schema=dummy_structured_schema, depth="deep")
             print(f"Linkup Structured Search Result Type: {type(linkup_structured_results)}")
             if isinstance(linkup_structured_results, dict):
                  print(f"Linkup Structured Search found data with keys: {list(linkup_structured_results.keys())}")
             elif isinstance(linkup_structured_results, list) and linkup_structured_results and isinstance(linkup_structured_results[0], dict):
                  print(f"Linkup Structured Search found list of data items. Count: {len(linkup_structured_results)}. First item keys: {list(linkup_structured_results[0].keys())}")
             else:
                  print("Linkup Structured Search did not return expected type (dict or list of dict).")
             # print(json.dumps(linkup_structured_results, indent=2)) # Print full result
         except Exception as e:
             print(f"Error during Linkup Structured Search test: {type(e).__name__}: {e}")
             traceback.print_exc()
    else:
         print("\nSkipping Linkup Structured Search test: Linkup search is not enabled.")


    print("\nTesting Targeted Ownership Docs Search (Combines sources)...")
    # This function calls the individual search functions. num_per_query is passed to those functions.
    # Removed country_code parameter from this call as search_linkup_snippets no longer accepts it.
    # Rely on the query generation within search_for_ownership_docs to add country context.
    ownership_docs_results = search_for_ownership_docs("Example Chinese Company", num_per_query=10, country_code='cn')
    print(f"Found {len(ownership_docs_results)} unique potential ownership documents across sources.") # Report total unique found
    # print(json.dumps(ownership_docs_results[:5], indent=2)) # Print sample


    print("\n--- Local Search Engine Tests Complete ---")