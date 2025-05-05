# orchestrator.py
import time
from typing import Dict, List, Any, Tuple, Optional, Mapping
import traceback
import json
from datetime import datetime, timezone
import sys
import re # Import re for the Chinese character check

# Import standard libraries
try:
    from zoneinfo import ZoneInfo
    zoneinfo_available = True
except ImportError:
    zoneinfo_available = False
    print("Warning: zoneinfo not available. Timestamps will be UTC.")

# Import Google Sheets libraries
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    import os
    google_sheets_library_available = True
except ImportError:
    print("Warning: Google API libraries not installed. Saving to Google Sheets disabled.")
    google_sheets_library_available = False
    service_account = None
    build = None

# Import custom modules
import search_engines
import nlp_processor
import knowledge_graph

# Import config
import config

# Import pycountry at module level but check availability in functions
try:
    import pycountry
    pycountry_available = True
except ImportError:
    pycountry = None
    pycountry_available = False
    print("Warning: 'pycountry' not installed. Full country name resolution limited.")


SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
# Initialize Google Sheets related variables to None or False outside the config block
SHEET_ID = None
GCP_SERVICE_ACCOUNT_JSON_STR = None
SERVICE_ACCOUNT_INFO = None
google_sheets_configured = False # Flag indicating if config was successfully loaded and parsed
google_sheets_available = False # Combined flag: library available AND config is valid

# Process config if it was imported successfully
if config:
    SHEET_ID = config.GOOGLE_SHEET_ID
    GCP_SERVICE_ACCOUNT_JSON_STR = config.GCP_SERVICE_ACCOUNT_JSON_STR

    # Check if essential config values are present
    if SHEET_ID and isinstance(SHEET_ID, str) and GCP_SERVICE_ACCOUNT_JSON_STR and isinstance(GCP_SERVICE_ACCOUNT_JSON_STR, str):
         try:
             SERVICE_ACCOUNT_INFO = json.loads(GCP_SERVICE_ACCOUNT_JSON_STR)
             if isinstance(SERVICE_ACCOUNT_INFO, dict):
                 google_sheets_configured = True
             else:
                  print("Warning: GCP_SERVICE_ACCOUNT_JSON_STR does not contain a valid JSON object for service account info.")
                  SERVICE_ACCOUNT_INFO = None
         except Exception as e:
              print(f"ERROR parsing GCP JSON from config: {e}")
              SERVICE_ACCOUNT_INFO = None
    elif config.GOOGLE_SHEET_ID:
         print("Warning: GCP_SERVICE_ACCOUNT_JSON_STR environment variable not set or empty, but GOOGLE_SHEET_ID is. Google Sheets saving will be disabled.")

    DEFAULT_PROVIDER = "openai"
    DEFAULT_MODEL = config.DEFAULT_OPENAI_MODEL if config and hasattr(config, 'DEFAULT_OPENAI_MODEL') else "gpt-4o-mini"
    FALLBACK_LLM_CONFIG = (DEFAULT_PROVIDER, DEFAULT_MODEL)

    if config and (getattr(config, 'OPENAI_API_KEY', None) or getattr(config, 'OPENROUTER_API_KEY', None) or getattr(config, 'GOOGLE_AI_API_KEY', None)):
         if DEFAULT_PROVIDER == "openai" and not getattr(config, 'OPENAI_API_KEY', None):
              if getattr(config, 'OPENROUTER_API_KEY', None) and hasattr(config, 'DEFAULT_OPENROUTER_MODEL'): FALLBACK_LLM_CONFIG = ("openrouter", config.DEFAULT_OPENROUTER_MODEL)
              elif getattr(config, 'GOOGLE_AI_API_KEY', None) and hasattr(config, 'DEFAULT_GOOGLE_AI_MODEL'): FALLBACK_LLM_CONFIG = ("google_ai", config.DEFAULT_GOOGLE_AI_MODEL)

    if FALLBACK_LLM_CONFIG == (None, None):
         print("ERROR: No valid default LLM configuration found. Automation script will not be able to run queries if LLM columns in sheet are empty.")

# The overall availability depends on library available AND configured credentials are valid
google_sheets_available = google_sheets_library_available and google_sheets_configured

if not google_sheets_available:
    print("Warning: Google Sheets saving will be disabled due to missing libraries or configuration.")


SHEET_NAME_RUNS = 'Runs'
SHEET_NAME_ENTITIES = 'Entities'
SHEET_NAME_RISKS = 'Risks'
SHEET_NAME_RELATIONSHIPS = 'Relationships'
SHEET_NAME_EXPOSURES = 'High Risk Exposures'

gsheet_service = None

def _get_gsheet_service():
    """Initializes and returns the Google Sheets service. Checks for library and config availability."""
    global gsheet_service
    if not google_sheets_library_available: return None
    if not google_sheets_configured: return None

    if gsheet_service is None:
        if SERVICE_ACCOUNT_INFO is None: return None
        try:
            creds = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
            gsheet_service = build('sheets', 'v4', credentials=creds, cache_discovery=False)
            print("Google Sheets service authenticated successfully.")
        except Exception as e:
             print(f"ERROR: Failed to authenticate/build Google Sheets service: {e}")
             gsheet_service = None
    return gsheet_service

def _append_to_gsheet(service, sheet_name: str, values: List[List[Any]]):
    """Appends a list of rows (values) to the specified sheet tab. Checks for service."""
    if service is None: # Only check service here, SHEET_ID checked higher up in save function
        print("GSheet service unavailable for append."); return False
    if not values: return True
    try:
        body = {'values': values}
        result = service.spreadsheets().values().append( spreadsheetId=SHEET_ID, range=f"{sheet_name}!A1", valueInputOption='USER_ENTERED', insertDataOption='INSERT_ROWS', body=body ).execute()
        return True
    except Exception as e: print(f"ERROR appending data to Google Sheet '{sheet_name}': {e}"); return False

# --- NEW HELPER FUNCTION ---
def contains_chinese(text):
    """Simple check if a string contains Chinese characters."""
    if not isinstance(text, str):
        return False
    # Check for common CJK Unified Ideographs ranges
    # This is a basic check, not comprehensive language detection
    return any('\u4e00' <= char <= '\u9fff' for char in text)


def _save_analysis_to_gsheet(run_results: Dict,
                             entities_to_save: List[Dict],
                             risks_to_save: List[Dict],
                             relationships_to_save: List[Dict],
                             exposures_to_save: List[Dict],
                             llm_provider_for_translation: str, # Pass LLM config for translation
                             llm_model_for_translation: str): # Pass LLM config for translation
    """
    Saves filtered analysis results to Google Sheets.
    Accepts pre-filtered lists of entities, risks, relationships, and exposures.
    Checks for sheet availability.
    Translates Chinese entity names in Exposure sheet rows if NLP is available.
    """
    print("Attempting to save analysis results to Google Sheets...")
    service = _get_gsheet_service()
    if service is None: print("Aborting save: GSheet service unavailable or config missing."); return

    # Check NLP availability for translation specifically
    nlp_available_for_translation = nlp_processor is not None and hasattr(nlp_processor, 'translate_text') and callable(nlp_processor.translate_text)
    if not nlp_available_for_translation:
        print("Warning: NLP processor or translate_text function not available. Chinese entity names in Exposures sheet will not be translated.")

    run_timestamp_iso = "Timestamp Error"
    try:
        dt_utc = datetime.now(timezone.utc)
        if zoneinfo_available: mountain_tz = ZoneInfo("America/Denver"); dt_mt = dt_utc.astimezone(mountain_tz); run_timestamp_iso = dt_mt.isoformat(timespec='seconds')
        else: run_timestamp_iso = dt_utc.isoformat(timespec='seconds'); print(f"Using UTC timestamp: {run_timestamp_iso} (zoneinfo not available)")
    except Exception as e: print(f"ERROR generating timestamp: {e}"); run_timestamp_iso = datetime.now().isoformat(timespec='seconds') + "_Error"

    run_summary_row = [
        run_timestamp_iso,
        run_results.get('query'),
        run_results.get('run_duration_seconds'),
        run_results.get('kg_update_status'),
        run_results.get('error'),
        run_results.get('analysis_summary', '')[:500]
    ]
    _append_to_gsheet(service, SHEET_NAME_RUNS, [run_summary_row])

    print(f"Saving {len(entities_to_save)} entities to sheet '{SHEET_NAME_ENTITIES}'.")
    if entities_to_save:
        entity_rows = []
        for e in entities_to_save:
            if isinstance(e, dict) and e.get('name'):
                 # Remove the '_source_type' key if it exists, not needed in sheet
                 entity_copy = e.copy()
                 entity_copy.pop('_source_type', None)
                 mentions_str = json.dumps(list(set(entity_copy.get('mentions', []))))
                 entity_rows.append([run_timestamp_iso, entity_copy.get('name'), entity_copy.get('type'), mentions_str])
            else: print(f"Skipping invalid entity format for sheet save: {e}") # Should not happen with pre-filtered list
        _append_to_gsheet(service, SHEET_NAME_ENTITIES, entity_rows)
    else: print(f"No valid entities to save to sheet '{SHEET_NAME_ENTITIES}'.")

    print(f"Saving {len(risks_to_save)} risks to sheet '{SHEET_NAME_RISKS}'.")
    if risks_to_save:
         risk_rows = []
         for r in risks_to_save:
              if isinstance(r, dict) and r.get('description'):
                   # Remove the '_source_type' key if it exists, not needed in sheet
                   risk_copy = r.copy()
                   risk_copy.pop('_source_type', None)
                   related_entities_str = json.dumps(list(set(risk_copy.get('related_entities', []))))
                   source_urls_str = json.dumps(list(set(risk_copy.get('source_urls', []))))
                   risk_rows.append([run_timestamp_iso, risk_copy.get('description'), risk_copy.get('severity', 'UNKNOWN'), risk_copy.get('risk_category', 'UNKNOWN'), related_entities_str, source_urls_str]) # Added risk_category
              else: print(f"Skipping invalid risk format for sheet save: {r}") # Should not happen with pre-filtered list
         _append_to_gsheet(service, SHEET_NAME_RISKS, risk_rows)
    else: print(f"No valid risks to save to sheet '{SHEET_NAME_RISKS}'.")

    print(f"Saving {len(relationships_to_save)} relationships to sheet '{SHEET_NAME_RELATIONSHIPS}'.")
    if relationships_to_save:
        rel_rows = []
        for rel in relationships_to_save:
             if isinstance(rel, dict) and rel.get('entity1') and rel.get('relationship_type') and rel.get('entity2'):
                  # Remove the '_source_type' key if it exists, not needed in sheet
                  rel_copy = rel.copy()
                  rel_copy.pop('_source_type', None)
                  context_urls_str = json.dumps(list(set(rel_copy.get('context_urls', []))))
                  rel_rows.append([run_timestamp_iso, rel_copy.get('entity1'), rel_copy.get('relationship_type'), rel_copy.get('entity2'), context_urls_str])
             else: print(f"Skipping invalid relationship format for sheet save: {rel}") # Should not happen with pre-filtered list
        _append_to_gsheet(service, SHEET_NAME_RELATIONSHIPS, rel_rows)
    else: print(f"No valid relationships to save to sheet '{SHEET_NAME_RELATIONSHIPS}'.")

    print(f"Saving {len(exposures_to_save)} exposure rows to sheet '{SHEET_NAME_EXPOSURES}'.")
    if exposures_to_save:
        exposure_rows_to_save_sheet = []

        # --- NEW HELPER FUNCTION FOR EXPOSURE NAME FORMATTING ---
        def format_entity_name_for_sheet(name: Any, original_field_key: str) -> str:
            if not name or not isinstance(name, str):
                return str(name) if name is not None else '' # Return as string or empty string if invalid or empty

            name_str = name.strip()

            # Check if it contains Chinese characters AND NLP translation is available
            if contains_chinese(name_str) and nlp_available_for_translation:
                 # Use the LLM config from the main run_analysis function scope
                 # Need to ensure nlp_processor is not None before calling its methods
                 if nlp_processor is not None:
                      print(f"[Sheet Save] Translating Chinese entity name for '{original_field_key}' column: '{name_str}'")
                      try:
                           translated_name = nlp_processor.translate_text(name_str, 'en', llm_provider_for_translation, llm_model_for_translation)
                           if translated_name and isinstance(translated_name, str) and translated_name.strip():
                                # Combine translated English with original Chinese in brackets
                                return f"{translated_name.strip()} ({name_str})"
                           else:
                                print(f"[Sheet Save] Translation failed for '{name_str}'. Using original.")
                                return name_str # Use original if translation fails

                      except Exception as e:
                           print(f"[Sheet Save] Error during translation for '{name_str}' ({original_field_key}): {type(e).__name__}: {e}. Using original.")
                           traceback.print_exc() # Log the traceback for the translation error
                           return name_str # Use original if error occurs
                 else:
                      print(f"[Sheet Save] Skipping translation for '{name_str}' (NLP processor not available). Using original.")
                      return name_str

            else:
                # Not Chinese, or NLP not available, just use the original name
                return name_str


        for exp in exposures_to_save:
             if isinstance(exp, dict):
                 # Process each name field in the exposure dictionary
                 entity_col_value = format_entity_name_for_sheet(exp.get('Entity'), 'Entity')
                 sub_aff_col_value = format_entity_name_for_sheet(exp.get('Subsidiary/Affiliate'), 'Subsidiary/Affiliate')
                 parent_col_value = format_entity_name_for_sheet(exp.get('Parent Company'), 'Parent Company')


                 # Columns: Timestamp, Entity, Subsidiary/Affiliate, Parent Company, Risk_Severity, Risk_Type, Explanation, Main_Source(s)
                 # Ensure required keys are present, default to empty string or N/A if not
                 row_8_cols = [
                     run_timestamp_iso,
                     entity_col_value, # Use formatted value
                     sub_aff_col_value, # Use formatted value
                     parent_col_value, # Use formatted value
                     exp.get('Risk_Severity', 'N/A'),
                     exp.get('Risk_Type', 'N/A'), # This is the derived label, e.g., "Subsidiary Risk"
                     exp.get('Explanation', 'N/A'),
                     json.dumps(list(exp.get('Main_Sources', []))) # Ensure sources is a list before json dumping
                 ]
                 exposure_rows_to_save_sheet.append(row_8_cols)
             else:
                  print(f"Skipping invalid exposure format during sheet save prep: {exp}") # Should not happen with pre-filtered list

        if exposure_rows_to_save_sheet:
            _append_to_gsheet(service, SHEET_NAME_EXPOSURES, exposure_rows_to_save_sheet)
    else:
        print(f"No exposures to save to sheet '{SHEET_NAME_EXPOSURES}'.")

    print("Google Sheets save process finished.")

LINKUP_SCHEMA_OWNERSHIP = json.dumps({
    "type": "object",
    "properties": {
        "company_name": {"type": "string", "description": "The main company being described"},
        "ownership_relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parent_company": {"type": "string", "description": "Name of parent company"},
                    "subsidiary_affiliate": {"type": "string", "description": "Name of subsidiary or affiliate company"},
                    "relation_type": {"type": "string", "description": "Type of relationship (e.g., 'subsidiary', 'affiliate', 'joint venture', 'private company', 'investee', 'parent', 'acquired')", "enum": ["subsidiary", "affiliate", "joint venture", "private company", "investee", "parent", "acquired"]},
                    "stake_percentage": {"type": ["number", "null"], "description": "Ownership percentage if available"},
                    "source_date": {"type": ["string", "null"], "description": "Date of the information"},
                    "source_description": {"type": ["string", "null"], "description": "Brief description of source (e.g., 'annual report', 'press release')"},
                    "source_url": {"type": ["string", "null"], "description": "URL of the source document"}
                },
                "required": ["parent_company", "subsidiary_affiliate", "relation_type"]
            }
        }
    },
    "required": ["company_name", "ownership_relationships"]
})

LINKUP_SCHEMA_KEY_RISKS = json.dumps({
    "type": "object",
    "properties": {
        "company_name": {"type": "string", "description": "The main company being described"},
        "key_risks_identified": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "risk_description": {"type": "string", "description": "Description of the risk"},
                    "risk_category": {"type": "string", "description": "Category of the risk (e.g., 'compliance', 'financial', 'environmental', 'governance', 'supply chain')"},
                    "reported_severity": {"type": ["string", "null"], "description": "Severity as reported (e.g., 'high', 'material')", "enum": ["low", "medium", "high", "severe", "material"]},
                    "source_date": {"type": ["string", "null"], "description": "Date of the information"},
                    "source_description": {"type": ["string", "null"], "description": "Brief description of source (e.e.g., 'SEC filing', 'news report')"},
                    "source_url": {"type": ["string", "null"], "description": "URL of the source document"}
                },
                "required": ["risk_description", "risk_category"]
            }
        }
    },
     "required": ["company_name", "key_risks_identified"]
})

# Ensure nlp_processor is checked for availability at module level
nlp_processor_available = True
if nlp_processor is None or not hasattr(nlp_processor, '_get_llm_client_and_model') or not callable(nlp_processor._get_llm_client_and_model):
     print("Warning: NLP processor module or essential functions not available.")
     nlp_processor_available = False


def run_analysis(initial_query: str,
                 llm_provider: str,
                 llm_model: str,
                 global_search_context: str = "General global news and filings",
                 specific_search_context: str = "Search for specific company examples and details",
                 specific_country_code: str = 'cn',
                 max_global_results: int = 20,
                 max_specific_results: int = 20
                 ) -> Dict[str, Any]:
    start_run_time = time.time()

    IS_LOCAL_TEST_RUN = False # Set to False to use UI selection
    if IS_LOCAL_TEST_RUN:
        print("--- LOCAL TEST OVERRIDE: Forcing Default OpenAI LLM ---")
        llm_provider_to_use = "openai"
        llm_model_to_use = config.DEFAULT_OPENAI_MODEL if config and hasattr(config, 'DEFAULT_OPENAI_MODEL') else "gpt-4o-mini"
        if not config or not getattr(config, 'OPENAI_API_KEY', None): print("WARNING: OpenAI API Key missing for local OpenAI test!")
    else:
        llm_provider_to_use = llm_provider
        llm_model_to_use = llm_model
        print(f"--- Using Selected LLM: {llm_provider_to_use} ({llm_model_to_use}) ---")

    results: Dict[str, Any] = {
        "query": initial_query, "steps": [], "llm_used": f"{llm_provider_to_use} ({llm_model_to_use})",
        "final_extracted_data": {"entities": [], "risks": [], "relationships": []}, # Raw data from all steps before filtering
        "high_risk_exposures": [], # This will hold the generated exposures before filtering for sheet/KG
        "analysis_summary": "Summary not generated.",
        "wayback_results": [], "kg_update_status": "not_run",
        "linkup_structured_data": [] # Raw structured data collected
    }
    print(f"\n--- Orchestrator Start --- Query: '{initial_query}', Using LLM: {llm_provider_to_use}/{llm_model_to_use}, SpecificCountry: {specific_country_code} ---")

    kg_driver_available = False
    kg_driver = None # Initialize driver to None
    if hasattr(knowledge_graph, 'get_driver') and callable(knowledge_graph.get_driver):
         # Attempt to get the driver immediately to check connection status early
         try:
             kg_driver = knowledge_graph.get_driver()
             kg_driver_available = bool(kg_driver)
             if not kg_driver_available:
                  results["kg_update_status"] = "skipped_no_connection"
                  print("KG driver not available after initial connection attempt.")
         except Exception as e:
             print(f"Error getting KG driver at start: {type(e).__name__}: {e}")
             traceback.print_exc()
             kg_driver_available = False
             kg_driver = None # Ensure driver is None if connection failed
             results["kg_update_status"] = "skipped_connection_error"
             print("KG driver connection failed at start.")
    else:
         print("Warning: Knowledge graph module or get_driver function not available.")
         results["kg_update_status"] = "skipped_module_unavailable"


    nlp_processor_available = True
    if not hasattr(nlp_processor, '_get_llm_client_and_model') or not callable(nlp_processor._get_llm_client_and_model):
         print("Warning: NLP processor module or essential functions (_get_llm_client_and_model) not available.")
         nlp_processor_available = False

    if nlp_processor_available:
        # Check for all required NLP functions
        required_nlp_funcs = ['translate_text', 'translate_snippets', 'translate_keywords_for_context',
                              'extract_entities_only', 'extract_risks_only', 'link_entities_to_risk',
                              'extract_relationships_only', 'extract_regulatory_sanction_relationships',
                              'process_linkup_structured_data', 'generate_analysis_summary']
        for func_name in required_nlp_funcs:
             if not hasattr(nlp_processor, func_name) or not callable(getattr(nlp_processor, func_name)):
                  print(f"Warning: nlp_processor.{func_name} not available.")
                  nlp_processor_available = False
                  # No break here, report all missing functions


    if not nlp_processor_available:
         results["error"] = "NLP processor module or essential functions not available."; print(f"--- Orchestrator ERROR: {results['error']} ---")
         results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
         # Pass empty lists to _save_analysis_to_gsheet since no data was extracted
         if google_sheets_available: _save_analysis_to_gsheet(results, [], [], [], [])
         if kg_driver_available and kg_driver is not None: knowledge_graph.close_driver() # Close if it was successfully obtained
         return results

    try:
         nlp_processor._get_llm_client_and_model(llm_provider_to_use, llm_model_to_use)
         print("LLM client initialized successfully for NLP processing.")
    except Exception as e:
         results["error"] = f"LLM configuration/initialization failed: {e}"; print(f"--- Orchestrator ERROR: {results['error']} ---")
         results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
         # Pass empty lists to _save_analysis_to_gsheet since no data was extracted
         if google_sheets_available: _save_analysis_to_gsheet(results, [], [], [], [])
         if kg_driver_available and kg_driver is not None: knowledge_graph.close_driver() # Close if it was successfully obtained
         return results


    google_cse_available = hasattr(search_engines, 'google_api_client_available_and_configured') and search_engines.google_api_client_available_and_configured and hasattr(search_engines, 'search_google_official') and callable(search_engines.search_google_official)
    serpapi_available = hasattr(search_engines, 'serpapi_available') and search_engines.serpapi_available and hasattr(search_engines, 'search_via_serpapi') and callable(search_engines.search_via_serpapi)
    linkup_search_enabled_orchestrator = hasattr(search_engines, 'linkup_search_enabled') and search_engines.linkup_search_enabled # Use a distinct name to avoid conflict
    linkup_snippet_search_available = linkup_search_enabled_orchestrator and hasattr(search_engines, 'search_linkup_snippets') and callable(search_engines.search_linkup_snippets)
    linkup_structured_search_available = linkup_search_enabled_orchestrator and hasattr(search_engines, 'search_linkup_structured') and callable(search_engines.search_linkup_structured)

    wayback_available = hasattr(search_engines, 'check_wayback_machine') and callable(search_engines.check_wayback_machine)
    search_engines_available_any = google_cse_available or serpapi_available or linkup_snippet_search_available or linkup_structured_search_available # Check availability of at least one search method


    # Define the threshold for stopping subsequent search engines early
    search_threshold_results = 15

    all_search_results_map: Dict[str, Dict] = {} # Use map for O(1) URL lookup


    # Initialize empty lists within try block scope for finally access
    entities_to_save_to_sheet = []
    risks_to_save_to_sheet = []
    relationships_to_save_to_sheet = []
    exposures_to_save_to_sheet = [] # This will hold the final list of exposure dictionaries

    # Initialize raw structured data list within try block
    raw_linkup_structured_data_collected = []


    try:
        print(f"\n--- Running Step 1: Initial Search ---")
        step1_start = time.time()
        step1_search_results = [] # This will hold the final, deduplicated list for Step 1 extraction


        step1_english_queries = []
        # Check if nlp_processor and specific functions are available before calling
        if nlp_processor_available and hasattr(nlp_processor, 'translate_keywords_for_context') and callable(nlp_processor.translate_keywords_for_context):
             step1_english_queries = nlp_processor.translate_keywords_for_context(
                  initial_query, global_search_context,
                  llm_provider_to_use, llm_model_to_use
                  )
        if not step1_english_queries: step1_english_queries = [initial_query]


        step1_chinese_queries = []
        if specific_country_code.lower() == 'cn' and nlp_processor_available and hasattr(nlp_processor, 'translate_text') and callable(nlp_processor.translate_text):
            print("[Step 1 Search] Translating queries to Chinese for Baidu search...")
            for query_en in step1_english_queries:
                # Use a potentially different LLM for translation if configured, though using the main one is fine
                translated_query_zh = nlp_processor.translate_text(query_en, 'zh', llm_provider_to_use, llm_model_to_use)
                if translated_query_zh and isinstance(translated_query_zh, str) and translated_query_zh.strip():
                    step1_chinese_queries.append(translated_query_zh.strip())
                    print(f"  Translated '{query_en[:50]}...' to '{translated_query_zh[:50]}...'")
                else:
                    print(f"Warning: Failed to translate query '{query_en[:50]}...' to Chinese.")
            if step1_english_queries: time.sleep(len(step1_english_queries) * 0.5) # Add a small delay after translation calls

        step1_all_queries = list(set(step1_english_queries + step1_chinese_queries))
        step1_all_queries = [q.strip() for q in step1_all_queries if q.strip()]

        print(f"[Step 1 Search] Running searches for {len(step1_all_queries)} query variants (max num per search: {max_global_results})...")

        if linkup_snippet_search_available:
             print(f"[Step 1 Search] Attempting broad Linkup snippet search: q='{initial_query}' (and variants)") # Removed num= param from log
             try:
                  # Linkup search_linkup_snippets handles multiple queries and returns combined unique results.
                  # It no longer accepts 'num' or 'country_code' as parameters causing TypeError.
                  # Pass the combined query to search_linkup_snippets.
                  linkup_global_results = search_engines.search_linkup_snippets(query=step1_all_queries) # Removed num and country_code params from the call
                  if linkup_global_results:
                      print(f"    Linkup Snippet Search returned {len(linkup_global_results)} unique results after internal deduplication.")
                      # **REMOVED TRUNCATION:** We keep all unique results returned by search_engines.search_linkup_snippets
                      # linkup_global_results_limited = linkup_global_results[:max_global_results]
                      # if len(linkup_global_results) > len(linkup_global_results_limited):
                      #     print(f"    Truncating Linkup results to max_global_results: {len(linkup_global_results_limited)} results kept.")

                      for r in linkup_global_results: # Iterate through the *full* list returned by search_linkup_snippets
                           if isinstance(r, dict) and r.get('url') and isinstance(r.get('url'), str) and r['url'] not in all_search_results_map:
                               # Preserve source if already set by search_engines, default to 'linkup_snippet_step1'
                               r['source'] = r.get('source', 'linkup_snippet_step1')
                               step1_search_results.append(r) # Add directly to final list for step 1
                               all_search_results_map[r['url']] = r
                  else: print("    Linkup Snippet Search returned no results.")
                  time.sleep(0.8)
             except Exception as e:
                  print(f"Error during broad Linkup snippet search: {type(e).__name__}: {e}")
                  traceback.print_exc()


        # Check threshold after the first search engine results are collected
        if google_cse_available and len(step1_search_results) < max_global_results and len(step1_search_results) < search_threshold_results:
            print("[Step 1 Search] Running Google CSE queries...")
            queries_for_google = step1_english_queries
            for q_idx, query_text in enumerate(queries_for_google):
                if len(step1_search_results) >= max_global_results:
                     print("Stopping Google CSE search early due to sufficient total results.")
                     break
                lang_to_use = None # Let Google detect or use default
                print(f"  Google Query {q_idx+1}/{len(queries_for_google)}: '{query_text}'")
                try:
                    # search_google_official respects num=10 limit internally
                    raw_results = search_engines.search_google_official(query=query_text, lang=lang_to_use, num=max_global_results);
                    if raw_results:
                        print(f"    Google CSE found {len(raw_results)} raw results.")
                        for r in raw_results:
                            standardized = search_engines.standardize_result(r, f"google_cse_q{q_idx+1}")
                            if standardized and isinstance(standardized, dict) and standardized.get('url') and isinstance(standardized.get('url'), str) and standardized['url'] not in all_search_results_map:
                                 step1_search_results.append(standardized)
                                 all_search_results_map[standardized['url']] = standardized
                    else: print("    Google CSE returned no results.")
                    time.sleep(0.5)
                except Exception as e: print(f"    Google CSE call failed for query '{query_text}': {type(e).__name__}: {e}"); traceback.print_exc()
        elif not google_cse_available:
            print("[Step 1 Search] Skipping Google CSE queries - not configured.")
        elif len(step1_search_results) >= search_threshold_results:
             print(f"[Step 1 Search] Skipping Google CSE. Preceding steps returned {len(step1_search_results)} results (>= {search_threshold_results} threshold).")


        # Check threshold again before running the next engine
        if serpapi_available and len(step1_search_results) < max_global_results and len(step1_search_results) < search_threshold_results:
            print("[Step 1 Search] Running Serpapi queries...")
            serpapi_engine = 'baidu' if specific_country_code.lower() == 'cn' else 'google'
            queries_for_serpapi = step1_all_queries
            for q_idx, query_text in enumerate(queries_for_serpapi):
                 if len(step1_search_results) >= max_global_results:
                      print("Stopping Serpapi search early due to sufficient total results.")
                      break
                 print(f"  Serpapi ({serpapi_engine}) Query {q_idx+1}/{len(queries_for_serpapi)}: '{query_text[:50]}...'")
                 try:
                      # search_via_serpapi accepts num and country_code
                      serpapi_results_list = search_engines.search_via_serpapi(query_text, serpapi_engine, specific_country_code, lang_code='en', num=max_global_results);
                      if serpapi_results_list:
                           print(f"    SerpApi {serpapi_engine} found {len(serpapi_results_list)} raw results.")
                           for r in serpapi_results_list:
                               standardized = search_engines.standardize_result(r, source=f'serpapi_{serpapi_engine}_q{q_idx+1}')
                               # standardize_result attempts to set original_language. If not set, default based on engine
                               if serpapi_engine == 'baidu': standardized['original_language'] = 'zh'
                               elif serpapi_engine == 'google': standardized['original_language'] = 'en'

                               if standardized and isinstance(standardized, dict) and standardized.get('url') and isinstance(standardized.get('url'), str) and standardized['url'] not in all_search_results_map:
                                    step1_search_results.append(standardized)
                                    all_search_results_map[standardized['url']] = standardized
                           else: print(f"    SerpApi {serpapi_engine} returned no results for this query.")

                      time.sleep(0.5)
                 except Exception as e: print(f"    SerpApi ({serpapi_engine}) call failed for query '{query_text[:50]}...': {type(e).__name__}: {e}"); traceback.print_exc()
        elif serpapi_available and not config.SERPAPI_KEY:
             print("[Step 1 Search] Skipping Serpapi search - not configured.")
        elif len(step1_search_results) >= search_threshold_results:
             print(f"[Step 1 Search] Skipping Serpapi. Preceding steps returned {len(step1_search_results)} results (>= {search_threshold_results} threshold).")


        # NO FINAL TRUNCATION HERE. We keep all unique results found up to the limits of individual search calls.

        print(f"[Step 1 Search] Total standardized results from Linkup/Google/SerpApi after combining & deduplication: {len(step1_search_results)}")

        snippets_to_translate = []
        if nlp_processor_available and hasattr(nlp_processor, 'translate_snippets') and callable(nlp_processor.translate_snippets):
             # Only translate if original language is known and not English, and snippet is available
             snippets_to_translate = [s for s in step1_search_results if isinstance(s, dict) and s.get('snippet') and isinstance(s.get('snippet'), str) and s.get('original_language') and s['original_language'].lower() not in ['en', 'english']]

        if snippets_to_translate:
             print(f"\n--- Translating {len(snippets_to_translate)} non-English snippets from Step 1 ---")
             # Store original search results temporarily to update them with translations
             original_step1_search_results = list(step1_search_results)
             step1_search_results = [] # Reset to rebuild with translations

             translated_step1_snippets = nlp_processor.translate_snippets(snippets_to_translate, 'en', llm_provider_to_use, llm_model_to_use)

             translated_urls = {s.get('url') for s in translated_step1_snippets if isinstance(s, dict) and s.get('url')}
             # Rebuild the list, replacing originals with translations where available
             for original_s in original_step1_search_results:
                  if isinstance(original_s, dict) and original_s.get('url') in translated_urls:
                       # Find the translated version
                       translated_s = next((ts for ts in translated_step1_snippets if isinstance(ts, dict) and ts.get('url') == original_s.get('url')), original_s)
                       step1_search_results.append(translated_s)
                  else:
                       # Ensure original_s is valid before appending
                       if isinstance(original_s, dict):
                           step1_search_results.append(original_s)
                       else: print(f"Warning: Skipping invalid original snippet during translation merge: {original_s}")


             print(f"--- Step 1 snippets updated with translations. Total snippets: {len(step1_search_results)} ---")
             # Update the map with the translated snippets as well
             all_search_results_map.update({s['url']: s for s in step1_search_results if isinstance(s, dict) and s.get('url')})


        print(f"\n--- Running Step 1 Extraction (Multi-Call) ---")
        step1_extracted_data = {"entities": [], "risks": [], "relationships": []}
        step1_entities = []
        step1_risks_initial = []
        step1_risks = []
        step1_relationships = []
        step1_entity_names = []
        # Check for all required NLP extraction functions
        required_extract_funcs = ['extract_entities_only', 'extract_risks_only', 'link_entities_to_risk',
                                  'extract_relationships_only', 'extract_regulatory_sanction_relationships']
        nlp_extraction_available = nlp_processor_available and all(hasattr(nlp_processor, func) and callable(getattr(nlp_processor, func)) for func in required_extract_funcs)


        if step1_search_results and nlp_extraction_available:
            step1_context = f"Analyze search results for query '{initial_query}'. Extract relevant entities, risks, and relationships based ONLY on the provided text snippets."
            print(f"[Step 1 Extract] Calling NLP processor (Multi-Call)... Context: {step1_context[:100]}...")

            step1_entities = nlp_processor.extract_entities_only(step1_search_results, step1_context, llm_provider_to_use, llm_model_to_use)
            # Filter entities to only include valid ones with names
            step1_entities_filtered = [e for e in step1_entities if isinstance(e, dict) and e.get('name')]
            step1_entity_names = [e['name'] for e in step1_entities_filtered if isinstance(e,dict) and e.get('name')] # Get names from validated entities
            print(f"[Step 1 Extract] Extracted {len(step1_entities_filtered)} entities of various types.")
            time.sleep(1.0)

            step1_risks_initial = nlp_processor.extract_risks_only(step1_search_results, step1_context, llm_provider_to_use, llm_model_to_use)
            time.sleep(1.0)

            step1_risks = []
            if step1_risks_initial and step1_entity_names:
                 print(f"[Step 1 Linker] Starting entity linking for {len(step1_risks_initial)} risks against {len(step1_entity_names)} filtered entities...")
                 # Ensure all_snippets_map is passed
                 step1_risks = nlp_processor.link_entities_to_risk(
                     risks=step1_risks_initial,
                     list_of_entity_names=step1_entity_names,
                     all_snippets_map=all_search_results_map, # Pass the combined map
                     llm_provider=llm_provider_to_use,
                     llm_model=llm_model_to_use
                 )
            else:
                 print("[Step 1 Linker] Skipping entity linking (no initial risks or filtered entities).")
                 step1_risks = step1_risks_initial
            time.sleep(1.0)

            # Relationships extraction uses the *filtered* list of entities from this step
            step1_ownership_relationships = []
            step1_entities_company_org = [e for e in step1_entities_filtered if e.get('type') in ["COMPANY", "ORGANIZATION"]]
            if step1_entities_company_org:
                 print(f"[Step 1 Extract] Calling NLP for Ownership Relationships...")
                 step1_ownership_relationships = nlp_processor.extract_relationships_only(step1_search_results, step1_context, step1_entities_company_org, llm_provider_to_use, llm_model_to_use)
            else: print("[Step 1 Extract] Skipping ownership relationship extraction - no COMPANY/ORGANIZATION entities found.")
            time.sleep(1.0)

            step1_reg_sanc_relationships = []
            # Regulatory/Sanction relationships can involve any relevant entity type
            if step1_entities_filtered:
                 print(f"[Step 1 Extract] Calling NLP for Regulatory/Sanction Relationships...")
                 step1_reg_sanc_relationships = nlp_processor.extract_regulatory_sanction_relationships(step1_search_results, step1_context, step1_entities_filtered, llm_provider_to_use, llm_model_to_use)
            else: print("[Step 1 Extract] Skipping regulatory/sanction relationship extraction - no entities found.")
            time.sleep(1.0)

            step1_relationships = step1_ownership_relationships + step1_reg_sanc_relationships

            print(f"[Step 1 Extract] Multi-Call Results: E:{len(step1_entities_filtered)}, R: {len(step1_risks)}, Rel:{len(step1_relationships)} (Ownership: {len(step1_ownership_relationships)}, Reg/Sanc: {len(step1_reg_sanc_relationships)})")

            # Assign the extracted data for this step
            step1_extracted_data["entities"] = step1_entities_filtered
            step1_extracted_data["risks"] = step1_risks
            step1_extracted_data["relationships"] = step1_relationships

        else:
             print("[Step 1 Extract] Skipping NLP extraction (no search results or NLP functions unavailable).")
             step1_extracted_data = {"entities": [], "risks": [], "relationships": []}

        # Accumulate extracted data into the final results dictionary
        if step1_extracted_data["entities"]: results["final_extracted_data"]["entities"].extend(step1_extracted_data["entities"])
        if step1_extracted_data["risks"]: results["final_extracted_data"]["risks"].extend(step1_extracted_data["risks"])
        if step1_extracted_data["relationships"]: results["final_extracted_data"]["relationships"].extend(step1_extracted_data["relationships"])

        results["steps"].append({ "name": "Initial Search & Snippet Extraction", "duration": round(time.time() - step1_start, 2), "search_results_count": len(step1_search_results), "extracted_data_counts": {k: len(v) for k,v in step1_extracted_data.items()}, "status": "OK" if step1_search_results else "No Results/Error" })

        print(f"\n--- Running Step 1.5: Targeted Linkup Structured Search ---")
        step1_5_start = time.time()
        step1_5_status = "Skipped (Linkup not available)"
        structured_data_status = "not_run"
        # raw_linkup_structured_data_collected = [] # Moved initialization to before try block

        # Check if Linkup structured search and processing functions are available
        linkup_structured_available_orchestrator = linkup_structured_search_available and nlp_processor_available and hasattr(nlp_processor, 'process_linkup_structured_data') and callable(nlp_processor.process_linkup_structured_data)


        if linkup_structured_available_orchestrator:
             step1_5_status = "OK"
             structured_data_status = "attempted"
             # Use the entities accumulated so far (from Step 1 snippet extraction)
             all_entities_accumulated_before_targeted_structured = list(results["final_extracted_data"].get("entities", []))

             if all_entities_accumulated_before_targeted_structured:
                 print(f"[Step 1.5 Structured] Found {len(all_entities_accumulated_before_targeted_structured)} entities so far. Attempting structured searches for Companies/Organizations.")
                 # Only run structured searches for entities identified as COMPANIES or ORGANIZATIONS
                 company_org_entities = [e for e in all_entities_accumulated_before_targeted_structured if isinstance(e, dict) and e.get('type') in ["COMPANY", "ORGANIZATION"] and e.get('name')]

                 if company_org_entities:
                     for entity in company_org_entities:
                          entity_name = entity.get('name')
                          if not entity_name or not isinstance(entity_name, str): continue

                          print(f"  Attempting structured search for entity: '{entity_name}'")

                          # Ownership search doesn't need country code as Linkup handles it internally based on query
                          ownership_query = f'"{entity_name}" ownership structure subsidiary affiliate parent acquired joint venture'


                          try:
                               # search_linkup_structured no longer accepts 'country_code' parameter causing TypeError
                               structured_ownership_data_item = search_engines.search_linkup_structured(
                                    query=ownership_query,
                                    structured_output_schema=LINKUP_SCHEMA_OWNERSHIP,
                                    depth="deep" # Use deep for potentially more results
                               )
                               # search_linkup_structured can return dict, list of dicts, or None
                               if structured_ownership_data_item is not None:
                                    if isinstance(structured_ownership_data_item, dict):
                                         print(f"    Structured ownership data found for '{entity_name}'.")
                                         structured_data_status = "found_data" # Set status if any data is found
                                         raw_linkup_structured_data_collected.append({"entity": entity_name, "schema": "ownership", "data": structured_ownership_data_item})
                                    elif isinstance(structured_ownership_data_item, list):
                                         print(f"    Found {len(structured_ownership_data_item)} structured ownership items for '{entity_name}'.")
                                         structured_data_status = "found_data"
                                         for item_data in structured_ownership_data_item:
                                               if isinstance(item_data, dict):
                                                    raw_linkup_structured_data_collected.append({"entity": entity_name, "schema": "ownership", "data": item_data})
                                               else: print(f"Warning: Skipping invalid structured ownership list item: {item_data}")

                                    else:
                                         print(f"    Structured ownership search for '{entity_name}' returned unexpected valid type: {type(structured_ownership_data_item)}")
                               else: print(f"    No structured ownership data found for '{entity_name}'.")

                               time.sleep(1.0) # Delay between calls

                          except Exception as e:
                               print(f"Error during structured ownership search for '{entity_name}': {type(e).__name__}: {e}")
                               traceback.print_exc()
                               structured_data_status = "error" if structured_data_status != "found_data" else structured_data_status # Don't overwrite found_data with error

                          # Risks search
                          risks_query = f'"{entity_name}" key risks compliance environmental governance supply chain legal regulatory financial esg'

                          try:
                               # search_linkup_structured no longer accepts 'country_code' parameter causing TypeError
                               structured_risks_data_item = search_engines.search_linkup_structured(
                                    query=risks_query,
                                    structured_output_schema=LINKUP_SCHEMA_KEY_RISKS,
                                    depth="deep" # Use deep for potentially more results
                               )
                               # search_linkup_structured can return dict, list of dicts, or None
                               if structured_risks_data_item is not None:
                                    if isinstance(structured_risks_data_item, dict):
                                         print(f"    Structured risks data found for '{entity_name}'.")
                                         structured_data_status = "found_data" # Set status if any data is found
                                         raw_linkup_structured_data_collected.append({"entity": entity_name, "schema": "key_risks", "data": structured_risks_data_item})
                                    elif isinstance(structured_risks_data_item, list):
                                         print(f"    Found {len(structured_risks_data_item)} structured risks items for '{entity_name}'.")
                                         structured_data_status = "found_data"
                                         for item_data in structured_risks_data_item:
                                              if isinstance(item_data, dict):
                                                   raw_linkup_structured_data_collected.append({"entity": entity_name, "schema": "key_risks", "data": item_data})
                                              else: print(f"Warning: Skipping invalid structured risks list item: {item_data}")

                                    else:
                                         print(f"    Structured risks search for '{entity_name}' returned unexpected valid type: {type(structured_risks_data_item)}")

                               else: print(f"    No structured risks data found for '{entity_name}'.")
                               time.sleep(1.0) # Delay between calls

                          except Exception as e:
                                print(f"Error during structured risks search for '{entity_name}': {type(e).__name__}: {e}")
                                traceback.print_exc()
                                structured_data_status = "error" if structured_data_status != "found_data" else structured_data_status


                 else: print("[Step 1.5 Structured] No COMPANY or ORGANIZATION entities found to perform structured searches.")


                 if raw_linkup_structured_data_collected:
                      print(f"[Step 1.5 Structured] Processing {len(raw_linkup_structured_data_collected)} raw structured results.")
                      processed_structured = nlp_processor.process_linkup_structured_data(raw_linkup_structured_data_collected, initial_query)
                      # Extend the final extracted data lists
                      if processed_structured.get("entities"): results["final_extracted_data"]["entities"].extend(processed_structured["entities"])
                      if processed_structured.get("risks"): results["final_extracted_data"]["risks"].extend(processed_structured["risks"])
                      if processed_structured.get("relationships"): results["final_extracted_data"]["relationships"].extend(processed_structured["relationships"])
                      # Store the raw structured data results
                      results["linkup_structured_data"] = raw_linkup_structured_data_collected

                      print(f"[Step 1.5 Structured] Merged processed data. Final counts after merging: E:{len(results['final_extracted_data']['entities'])}, R:{len(results['final_extracted_data']['risks'])}, Rel:{len(results['final_extracted_data']['relationships'])}")

                 else:
                      print("[Step 1.5 Structured] No raw structured results collected from Linkup.")
                      if structured_data_status == "attempted": structured_data_status = "no_data" # Update status if attempted but found nothing

             else: print("[Step 1.5 Structured] No entities found so far to perform structured searches.")
        else:
            print("[Step 1.5 Structured] Skipping Linkup structured search - Linkup is not enabled or functions unavailable.")

        results["steps"].append({
            "name": "Targeted Linkup Structured Search",
            "duration": round(time.time() - step1_5_start, 2),
            "status": step1_5_status,
            "structured_data_status": structured_data_status,
            "structured_results_count": len(raw_linkup_structured_data_collected)
         })

        print(f"\n--- Running Step 2: Translating Keywords ---")
        step2_start = time.time()
        # Check if nlp_processor and specific functions are available before calling
        if nlp_processor_available and hasattr(nlp_processor, 'translate_keywords_for_context') and callable(nlp_processor.translate_keywords_for_context):
             translated_keywords = nlp_processor.translate_keywords_for_context(
                  initial_query, f"{specific_search_context} relevant for {specific_country_code}",
                  llm_provider_to_use, llm_model_to_use
                  )
        if not translated_keywords: translated_keywords = [initial_query]


        specific_query_base = translated_keywords[0] if translated_keywords else initial_query
        print(f"[Step 2 Translate] Base specific query for Step 3: '{specific_query_base}'")
        results["steps"].append({"name": "Keyword Translation", "duration": round(time.time() - step2_start, 2), "translated_keywords": translated_keywords, "status": "OK"})
        time.sleep(1.0)

        print(f"\n--- Running Step 3: Specific Country Search ({specific_query_base} in {specific_country_code}) ---")
        step3_start = time.time()
        step3_search_results = [] # This will hold the final, deduplicated list for Step 3 extraction

        # Generate search queries for step 3
        step3_english_queries_base = [f"{specific_query_base} report OR filing OR case OR lawsuit"]
        if len(translated_keywords) > 1:
             # Use the second translated keyword as another query base
             step3_english_queries_base.append(f"{translated_keywords[1]} company examples OR scandals")
        # Always include the base query itself
        step3_english_queries_base.append(specific_query_base)
        step3_english_queries_base = list(set([q.strip() for q in step3_english_queries_base if q.strip()])) # Deduplicate


        country_name_display = specific_country_code.upper()
        if pycountry_available and pycountry is not None:
            try:
                 country = pycountry.countries.get(alpha_2=specific_country_code.upper())
                 if country: country_name_display = country.name
            except Exception: # Catch any pycountry error
                 pass


        step3_english_queries_web = [f"{q} {country_name_display}" for q in step3_english_queries_base] # Add country name for web searches
        step3_english_queries_web = list(set([q.strip() for q in step3_english_queries_web if q.strip()])) # Deduplicate

        step3_chinese_queries = []
        if nlp_processor_available and hasattr(nlp_processor, 'translate_text') and callable(nlp_processor.translate_text) and specific_country_code.lower() == 'cn':
            print("[Step 3 Search] Translating queries to Chinese for Baidu search...")
            for query_en in step3_english_queries_base:
                translated_query_zh = nlp_processor.translate_text(query_en, 'zh', llm_provider_to_use, llm_model_to_use)
                if translated_query_zh and isinstance(translated_query_zh, str) and translated_query_zh.strip():
                    step3_chinese_queries.append(translated_query_zh.strip())
                    print(f"  Translated '{query_en[:50]}...' to '{translated_query_zh[:50]}...'")
                else:
                    print(f"Warning: Failed to translate query '{query_en[:50]}...' to Chinese.")
            if step3_english_queries_base: time.sleep(len(step3_english_queries_base) * 0.5) # Small delay


        step3_all_queries = list(set(step3_english_queries_base + step3_chinese_queries)) # Use base queries and Chinese translations
        step3_all_queries = [q.strip() for q in step3_all_queries if q.strip()]


        print(f"[Step 3 Search] Running searches for {len(step3_all_queries)} query variants (max num per search: {max_specific_results})...")

        linkup_specific_results = []
        if linkup_snippet_search_available:
             print(f"[Step 3 Search] Attempting Linkup snippet search: q='{specific_query_base}' (and variants)") # Removed num= param from log
             try:
                  # Linkup search_linkup_snippets handles multiple queries and returns combined unique results
                  # It no longer accepts 'num' or 'country_code' as parameters causing TypeError.
                  # Pass the combined query to search_linkup_snippets.
                  linkup_specific_results = search_engines.search_linkup_snippets(query=step3_all_queries) # Removed num and country_code
                  if linkup_specific_results:
                       print(f"    Linkup Snippet Search returned {len(linkup_specific_results)} unique results after internal deduplication.")
                       # **REMOVED TRUNCATION:** We keep all unique results returned by search_engines.search_linkup_snippets
                       # linkup_specific_results_limited = linkup_specific_results[:max_specific_results]
                       # if len(linkup_specific_results) > len(linkup_specific_results_limited):
                       #      print(f"    Truncating Linkup results to max_specific_results: {len(linkup_specific_results_limited)} results kept.")

                       for r in linkup_specific_results: # Iterate through the *full* list returned by search_linkup_snippets
                            if isinstance(r, dict) and r.get('url') and isinstance(r.get('url'), str) and r['url'] not in all_search_results_map:
                                r['source'] = r.get('source', 'linkup_snippet_step3') # Preserve source
                                step3_search_results.append(r) # Add directly to final list for step 3
                                all_search_results_map[r['url']] = r
                  else: print("    Linkup Snippet Search returned no results.")
                  time.sleep(0.8)
             except Exception as e:
                  print(f"Error during Linkup specific search: {type(e).__name__}: {e}")
                  traceback.print_exc()


        required_for_threshold = 15 # Specific threshold for results before skipping next engine

        # Check threshold after the first search engine results are collected
        if google_cse_available and len(step3_search_results) < max_specific_results and len(step3_search_results) < required_for_threshold:
            print("[Step 3 Search] Running Google CSE queries...")
            queries_for_google = step3_english_queries_web # Use queries with country name appended
            for q_idx, query_text in enumerate(queries_for_google):
                 if len(step3_search_results) >= max_specific_results:
                      print("Stopping Google CSE search early due to sufficient total results.")
                      break
                 lang_to_use = None # Let Google detect or use default
                 print(f"  Google Query {q_idx+1}/{len(queries_for_google)}: '{query_text}'")
                 try:
                     # search_google_official respects num=10 limit internally
                     raw_results = search_engines.search_google_official(query_text, lang=lang_to_use, num=max_specific_results);
                     if raw_results:
                         print(f"    Google CSE found {len(raw_results)} raw results.")
                         for r in raw_results:
                             standardized = search_engines.standardize_result(r, source=f"google_cse_q{q_idx+1}")
                             if standardized and isinstance(standardized, dict) and standardized.get('url') and isinstance(standardized.get('url'), str) and standardized['url'] not in all_search_results_map:
                                 step3_search_results.append(standardized)
                                 all_search_results_map[standardized['url']] = standardized
                     else: print(f"    Google CSE returned no results.")
                     time.sleep(0.5)
                 except Exception as e: print(f"    Google CSE call failed for query '{query_text}': {type(e).__name__}: {e}"); traceback.print_exc()
        elif not google_cse_available:
            print("[Step 3 Search] Skipping Google CSE queries - not configured.")
        elif len(step3_search_results) >= required_for_threshold:
             print(f"[Step 3 Search] Skipping Google CSE. Preceding steps returned {len(step3_search_results)} results (>= {required_for_threshold} threshold).")


        # Check threshold again before running the next engine
        if serpapi_available and len(step3_search_results) < max_specific_results and len(step3_search_results) < required_for_threshold:
            print("[Step 3 Search] Running Serpapi queries...")
            serpapi_engine = 'baidu' if specific_country_code.lower() == 'cn' else 'google'
            queries_for_serpapi = step3_all_queries # Use all queries including Chinese ones for Serpapi (especially Baidu)
            for q_idx, query_text in enumerate(queries_for_serpapi):
                 if len(step3_search_results) >= max_specific_results:
                      print("Stopping Serpapi search early due to sufficient total results.")
                      break
                 print(f"  Serpapi ({serpapi_engine}) Query {q_idx+1}/{len(queries_for_serpapi)}: '{query_text[:50]}...'")
                 try:
                      # search_via_serpapi accepts num and country_code
                      serpapi_results_list = search_engines.search_via_serpapi(query_text, serpapi_engine, specific_country_code, lang_code='en', num=max_specific_results);
                      if serpapi_results_list:
                           print(f"    SerpApi {serpapi_engine} found {len(serpapi_results_list)} raw results.")
                           for r in serpapi_results_list:
                               standardized = search_engines.standardize_result(r, source=f'serpapi_{serpapi_engine}_q{q_idx+1}')
                               # standardize_result attempts to set original_language. If not set, default based on engine
                               if serpapi_engine == 'baidu': standardized['original_language'] = 'zh'
                               elif serpapi_engine == 'google': standardized['original_language'] = 'en'

                               if standardized and isinstance(standardized, dict) and standardized.get('url') and isinstance(standardized.get('url'), str) and standardized['url'] not in all_search_results_map:
                                    step3_search_results.append(standardized)
                                    all_search_results_map[standardized['url']] = standardized
                           else: print(f"    SerpApi {serpapi_engine} returned no results for this query.")

                      time.sleep(0.5)
                 except Exception as e: print(f"    SerpApi ({serpapi_engine}) call failed for query '{query_text[:50]}...': {type(e).__name__}: {e}"); traceback.print_exc()
        elif serpapi_available and not config.SERPAPI_KEY:
             print("[Step 3 Search] Skipping Serpapi search - not configured.")
        elif len(step3_search_results) >= required_for_threshold:
             print(f"[Step 3 Search] Skipping Serpapi. Preceding steps returned {len(step3_search_results)} results (>= {required_for_threshold} threshold).")


        # NO FINAL TRUNCATION HERE. We keep all unique results found up to the limits of individual search calls.

        print(f"[Step 3 Search] Total standardized results from Linkup/Google/SerpApi after combining & deduplication: {len(step3_search_results)}")


        snippets_to_translate = []
        if nlp_processor_available and hasattr(nlp_processor, 'translate_snippets') and callable(nlp_processor.translate_snippets):
             # Only translate if original language is known and not English, and snippet is available
             snippets_to_translate = [s for s in step3_search_results if isinstance(s, dict) and s.get('snippet') and isinstance(s.get('snippet'), str) and s.get('original_language') and s['original_language'].lower() not in ['en', 'english']]


        if snippets_to_translate:
             print(f"\n--- Translating {len(snippets_to_translate)} non-English snippets from Step 3 ---")
             # Store original search results temporarily to update them with translations
             original_step3_search_results = list(step3_search_results)
             step3_search_results = [] # Reset to rebuild with translations

             translated_step3_snippets = nlp_processor.translate_snippets(snippets_to_translate, 'en', llm_provider_to_use, llm_model_to_use)

             translated_urls = {s.get('url') for s in translated_step3_snippets if isinstance(s, dict) and s.get('url')}
             # Rebuild the list, replacing originals with translations where available
             for original_s in original_step3_search_results:
                  if isinstance(original_s, dict) and original_s.get('url') in translated_urls:
                       translated_s = next((ts for ts in translated_step3_snippets if isinstance(ts, dict) and ts.get('url') == original_s.get('url')), original_s)
                       step3_search_results.append(translated_s)
                  else:
                       # Ensure original_s is valid before appending
                       if isinstance(original_s, dict):
                           step3_search_results.append(original_s)
                       else: print(f"Warning: Skipping invalid original snippet during translation merge: {original_s}")

             print(f"--- Step 3 snippets updated with translations. Total snippets: {len(step3_search_results)} ---")
             # Update the map with the translated snippets as well
             all_search_results_map.update({s['url']: s for s in step3_search_results if isinstance(s, dict) and s.get('url')})


        print(f"\n--- Running Step 3 Extraction (Multi-Call) ---")
        step3_extracted_data = {"entities": [], "risks": [], "relationships": []}
        step3_entities = []
        step3_risks_initial = []
        step3_risks = []
        step3_relationships = []
        step3_entity_names = []

        # Check if all required NLP extraction functions are available
        required_extract_funcs = ['extract_entities_only', 'extract_risks_only', 'link_entities_to_risk',
                                  'extract_relationships_only', 'extract_regulatory_sanction_relationships']
        nlp_extraction_available = nlp_processor_available and all(hasattr(nlp_processor, func) and callable(getattr(nlp_processor, func)) for func in required_extract_funcs)


        if step3_search_results and nlp_extraction_available:
            step3_context = specific_search_context
            print(f"[Step 3 Extract] Calling NLP processor (Multi-Call)... Context: {step3_context[:100]}...")

            step3_entities = nlp_processor.extract_entities_only(step3_search_results, step3_context, llm_provider_to_use, llm_model_to_use)
            # Filter entities to only include valid ones with names
            step3_entities_filtered = [e for e in step3_entities if isinstance(e, dict) and e.get('name')]
            step3_entity_names = [e['name'] for e in step3_entities_filtered if isinstance(e,dict) and e.get('name')] # Get names from validated entities
            print(f"[Step 3 Extract] Extracted {len(step3_entities_filtered)} entities of various types.")
            time.sleep(1.0)

            step3_risks_initial = nlp_processor.extract_risks_only(step3_search_results, step3_context, llm_provider_to_use, llm_model_to_use)
            time.sleep(1.0)

            step3_risks = []
            if step3_risks_initial and step3_entity_names:
                 print(f"[Step 3 Linker] Starting entity linking for {len(step3_risks_initial)} risks against {len(step3_entity_names)} filtered entities...")
                 # Ensure all_snippets_map is passed
                 step3_risks = nlp_processor.link_entities_to_risk(
                     risks=step3_risks_initial,
                     list_of_entity_names=step3_entity_names,
                     all_snippets_map=all_search_results_map, # Pass the combined map
                     llm_provider=llm_provider_to_use,
                     llm_model=llm_model_to_use
                 )
            else:
                 print("[Step 3 Linker] Skipping entity linking (no initial risks or filtered entities).")
                 step3_risks = step3_risks_initial
            time.sleep(1.0)

            # Relationships extraction uses the *filtered* list of entities from this step
            step3_ownership_relationships = []
            step3_entities_company_org = [e for e in step3_entities_filtered if e.get('type') in ["COMPANY", "ORGANIZATION"]]
            if step3_entities_company_org:
                 print(f"[Step 3 Extract] Calling NLP for Ownership Relationships...")
                 step3_ownership_relationships = nlp_processor.extract_relationships_only(step3_search_results, step3_context, step3_entities_company_org, llm_provider_to_use, llm_model_to_use)
            else: print("[Step 3 Extract] Skipping ownership relationship extraction - no COMPANY/ORGANIZATION entities found.")
            time.sleep(1.0)

            step3_reg_sanc_relationships = []
            # Regulatory/Sanction relationships can involve any relevant entity type
            if step3_entities_filtered:
                 print(f"[Step 3 Extract] Calling NLP for Regulatory/Sanction Relationships...")
                 step3_reg_sanc_relationships = nlp_processor.extract_regulatory_sanction_relationships(step3_search_results, step3_context, step3_entities_filtered, llm_provider_to_use, llm_model_to_use)
            else: print("[Step 3 Extract] Skipping regulatory/sanction relationship extraction - no entities found.")
            time.sleep(1.0)

            step3_relationships = step3_ownership_relationships + step3_reg_sanc_relationships

            print(f"[Step 3 Extract] Multi-Call Results: E: {len(step3_entities_filtered)}, R:{len(step3_risks)}, Rel:{len(step3_relationships)} (Ownership: {len(step3_ownership_relationships)}, Reg/Sanc: {len(step3_reg_sanc_relationships)})")

            # Assign the extracted data for this step
            step3_extracted_data["entities"] = step3_entities_filtered
            step3_extracted_data["risks"] = step3_risks
            step3_extracted_data["relationships"] = step3_relationships

        else:
             print("[Step 3 Extract] Skipping NLP extraction (no search results or NLP functions unavailable).")
             step3_extracted_data = {"entities": [], "risks": [], "relationships": []}

        # Accumulate extracted data into the final results dictionary
        if step3_extracted_data["entities"]: results["final_extracted_data"]["entities"].extend(step3_extracted_data["entities"])
        if step3_extracted_data["risks"]: results["final_extracted_data"]["risks"].extend(step3_extracted_data["risks"])
        if step3_extracted_data["relationships"]: results["final_extracted_data"]["relationships"].extend(step3_extracted_data["relationships"])


        results["steps"].append({ "name": f"Specific Search ({specific_country_code}) & Snippet Extraction", "duration": round(time.time() - step3_start, 2), "search_results_count": len(step3_search_results), "extracted_data_counts": {k: len(v) for k,v in step3_extracted_data.items()}, "status": "OK" if step3_search_results else "No Results/Error" })


        print(f"\n--- Running Step 3.5: Generating High Risk Exposures ---")
        step3_5_start = time.time()

        all_entities_accumulated = results["final_extracted_data"].get("entities", [])
        all_risks_accumulated = results["final_extracted_data"].get("risks", [])
        all_collected_relationships = list(results["final_extracted_data"].get("relationships", []))

        # Recalculate likely Chinese company/org names based on ALL accumulated entities
        # This is crucial for filtering consistently across sheets and KG
        likely_chinese_company_org_names = {e.get('name','') for e in all_entities_accumulated if isinstance(e, dict) and e.get('name') and e.get('type') in ["COMPANY", "ORGANIZATION"]}
        likely_chinese_company_org_names_lower = {name.lower() for name in likely_chinese_company_org_names}
        print(f"[Step 3.5 Exposures] Considering entities and relationships involving {len(likely_chinese_company_org_names_lower)} likely Chinese Company/Organization entities.")


        print(f"[Step 3.5 Exposures] Analyzing {len(all_risks_accumulated)} accumulated risks and {len(all_entities_accumulated)} entities.")

        # Identify Chinese Regulatory Agencies and Sanctions from the accumulated entities
        # These will be used to identify the 'subject_to' relationships involving Chinese entities
        chinese_reg_agency_names = {e.get('name','') for e in all_entities_accumulated if isinstance(e, dict) and e.get('name') and e.get('type') == "REGULATORY_AGENCY"}
        chinese_sanction_names = {e.get('name','') for e in all_entities_accumulated if isinstance(e, dict) and e.get('name') and e.get('type') == "SANCTION"}
        chinese_reg_agency_names_lower = {name.lower() for name in chinese_reg_agency_names}
        chinese_sanction_names_lower = {name.lower() for name in chinese_sanction_names}
        chinese_reg_sanc_names_lower = chinese_reg_agency_names_lower.union(chinese_sanction_names_lower)


        # --- Identify potential exposure triggers ---
        # A trigger is a Chinese Company/Org that is either:
        # 1. SUBJECT_TO a Chinese Regulator/Sanction
        # 2. Has a HIGH/SEVERE risk related to Compliance (based on risk description/category) or Financial (based on category)

        # 1. Entities SUBJECT_TO Chinese Reg/Sanc
        companies_subject_to_reg_sanc_rels = [
            rel for rel in all_collected_relationships
            if isinstance(rel, dict)
            and rel.get('relationship_type') == "SUBJECT_TO"
            and isinstance(rel.get('entity1'), str) and isinstance(rel.get('entity2'), str) # Ensure both entity names are strings
            and rel.get('entity1').lower() in likely_chinese_company_org_names_lower # Entity1 is a Chinese Company/Org
            and rel.get('entity2').lower() in chinese_reg_sanc_names_lower # Entity2 is a Chinese Regulator/Sanction
        ]
        companies_subject_to_names_lower = {rel['entity1'].strip().lower() for rel in companies_subject_to_reg_sanc_rels}

        # 2. Entities with HIGH/SEVERE Compliance or Financial Risks
        companies_with_high_severe_compliance_or_financial_risks = set()
        high_severe_risks = [r for r in all_risks_accumulated if isinstance(r, dict) and r.get('severity') in ["HIGH", "SEVERE"]]
        for risk in high_severe_risks:
             risk_desc = risk.get('description', '').lower()
             risk_category = risk.get('risk_category', '').lower() # Use extracted category

             # Check for "compliance" in description OR category is 'Compliance' or 'Financial'
             if "compliance" in risk_desc or risk_category in ['compliance', 'financial']:
                  for entity_name in risk.get('related_entities', []):
                       if isinstance(entity_name, str) and entity_name.strip().lower() in likely_chinese_company_org_names_lower:
                            companies_with_high_severe_compliance_or_financial_risks.add(entity_name.strip().lower())


        # Combine all Chinese Company/Org entities that are triggers
        # A trigger is a Chinese Company/Org that is either SUBJECT_TO or has a HIGH/SEVERE Compliance/Financial Risk
        triggering_chinese_company_org_names_lower = companies_subject_to_names_lower.union(companies_with_high_severe_compliance_or_financial_risks)


        print(f"[Step 3.5 Exposures] Identified {len(triggering_chinese_company_org_names_lower)} Chinese Companies/Orgs that are potential exposure triggers (Subject To OR High/Severe Compliance/Financial Risk).")


        consolidated_exposures: Dict[Tuple, Dict] = {} # Key will be (Triggering Chinese Co/Org Name, Related Chinese Co/Org Name, Ownership/Affiliate/JV Relationship Type)

        # Iterate through the Chinese Company/Org entities that are triggers
        for triggering_company_name_lower in triggering_chinese_company_org_names_lower:
             triggering_company_name = next((name for name in likely_chinese_company_org_names if name.lower() == triggering_company_name_lower), triggering_company_name_lower) # Get original casing if possible


             # Find relationships where this triggering Chinese Company/Org is involved in the required Ownership/Affiliate relationships
             required_ownership_relationships_involving_triggering_entity = [
                 rel for rel in all_collected_relationships
                 if isinstance(rel, dict)
                 and rel.get('relationship_type') in ["PARENT_COMPANY_OF", "SUBSIDIARY_OF", "AFFILIATE_OF", "JOINT_VENTURE_PARTNER"] # Added Joint Venture here
                 and isinstance(rel.get('entity1'), str) and isinstance(rel.get('entity2'), str)
                 and (rel.get('entity1').lower() == triggering_company_name_lower or rel.get('entity2').lower() == triggering_company_name_lower)
                 # Ensure the OTHER entity in the relationship is also a Chinese Company/Org
                 and (
                      (rel.get('entity1').lower() == triggering_company_name_lower and rel.get('entity2').lower() in likely_chinese_company_org_names_lower) or
                      (rel.get('entity2').lower() == triggering_company_name_lower and rel.get('entity1').lower() in likely_chinese_company_org_names_lower)
                     )
             ]

             if not required_ownership_relationships_involving_triggering_entity:
                  # If a triggering Chinese Company/Org has NO required Ownership/Affiliate relationship with *another* Chinese Co/Org, it does NOT trigger a sheet exposure row.
                  # print(f"Skipping exposure for triggering Chinese Company/Org '{triggering_company_name}': No PARENT_COMPANY_OF, SUBSIDIARY_OF, AFFILIATE_OF, or JOINT_VENTURE_PARTNER relationship found with another Chinese Company/Org.")
                  continue # Skip to the next triggering entity


             # Process the valid required ownership relationships found for the triggering entity
             for ownership_rel in required_ownership_relationships_involving_triggering_entity:
                 e1_name = ownership_rel.get('entity1'); e2_name = ownership_rel.get('entity2'); r_type_raw = ownership_rel.get('relationship_type'); rel_urls = set(ownership_rel.get('context_urls', []))

                 if not isinstance(r_type_raw, str) or not isinstance(e1_name, str) or not isinstance(e2_name, str): continue
                 r_type_upper = r_type_raw.upper()

                 # Determine the OTHER Chinese Company/Org involved in this relationship
                 other_chinese_company_name = None
                 if e1_name.lower() == triggering_company_name_lower:
                     other_chinese_company_name = e2_name.strip()
                 elif e2_name.lower() == triggering_company_name_lower:
                     other_chinese_company_name = e1_name.strip()
                 else:
                     continue # Should not happen if logic is correct

                 # Determine the roles for the Exposure columns (Parent, Subsidiary/Affiliate)
                 parent_col_value = ""
                 sub_aff_col_value = ""
                 exposure_risk_type_label = r_type_raw.replace('_', ' ').title() # Default label based on relationship type


                 if r_type_upper == "PARENT_COMPANY_OF":
                      # If triggering_entity is entity1 (parent) -> triggering_entity is Parent, entity2 is Sub/Aff
                      if e1_name.lower() == triggering_company_name_lower:
                           parent_col_value = triggering_company_name
                           sub_aff_col_value = other_chinese_company_name
                           exposure_risk_type_label = "Parent Company Risk" # Risk on parent entity affects subsidiary
                      # If triggering_entity is entity2 (subsidiary) -> entity1 is Parent, triggering_entity is Sub/Aff
                      elif e2_name.lower() == triggering_company_name_lower:
                           parent_col_value = other_chinese_company_name
                           sub_aff_col_value = triggering_company_name
                           exposure_risk_type_label = "Subsidiary Risk" # Risk on subsidiary affects parent
                      else: continue # Should not happen


                 elif r_type_upper == "SUBSIDIARY_OF":
                      # Inverse of PARENT_COMPANY_OF
                      # If triggering_entity is entity1 (subsidiary) -> entity2 is Parent, triggering_entity is Sub/Aff
                      if e1_name.lower() == triggering_company_name_lower:
                           parent_col_value = other_chinese_company_name
                           sub_aff_col_value = triggering_company_name
                           exposure_risk_type_label = "Subsidiary Risk" # Risk on subsidiary affects parent
                      # If triggering_entity is entity2 (parent) -> triggering_entity is Parent, entity1 is Sub/Aff
                      elif e2_name.lower() == triggering_company_name_lower:
                           parent_col_value = triggering_company_name
                           sub_aff_col_value = e1_name.strip() # The other entity is the sub/affiliate
                           exposure_risk_type_label = "Parent Company Risk" # Risk on parent entity affects subsidiary
                      else: continue

                 elif r_type_upper == "AFFILIATE_OF":
                      # For Affiliate, the triggering entity is one affiliate, the other is the other affiliate/partner
                      parent_col_value = "" # No strict parent/sub in affiliate
                      sub_aff_col_value = f"{triggering_company_name} (Affiliate) / {other_chinese_company_name} (Partner)"
                      exposure_risk_type_label = "Affiliate Risk" # Risk on one affiliate/partner affects the other

                 elif r_type_upper == "JOINT_VENTURE_PARTNER":
                       # For Joint Venture, the triggering entity is one partner, the other is the other partner
                       parent_col_value = ""
                       sub_aff_col_value = f"{triggering_company_name} (JV Partner) / {other_chinese_company_name} (Partner)"
                       exposure_risk_type_label = "Joint Venture Risk" # Risk on one partner affects the other

                 else:
                      # This should not be hit based on the 'ONLY these three relationships' filter + JV above,
                      # but including a safety print
                      print(f"Warning: Encountered unexpected relationship type for exposure generation (should be Parent/Sub/Affiliate/JV): '{r_type_raw}'. Skipping relationship {ownership_rel}")
                      continue


                 # Define the key for consolidating exposures
                 # Key will be (Triggering Chinese Co/Org Name, Related Chinese Co/Org Name, Ownership/Affiliate/JV Relationship Type)
                 # Example: ('alibaba group', 'aliexpress', 'SUBSIDIARY_OF') --> sanctioned_entity is AliExpress, related is Alibaba
                 # Example: ('alibaba group', 'aliexpress', 'PARENT_COMPANY_OF') --> sanctioned_entity is Alibaba, related is AliExpress
                 # Use sorted names to make the key consistent regardless of which entity came first in the relationship tuple
                 # Ensure both names are strings and not empty before lower() and strip()
                 triggering_name_for_key = triggering_company_name.strip().lower() if isinstance(triggering_company_name, str) else ''
                 other_name_for_key = other_chinese_company_name.strip().lower() if isinstance(other_chinese_company_name, str) else ''

                 entity_pair_sorted = tuple(sorted((triggering_name_for_key, other_name_for_key)))

                 # Ensure the tuple key is valid (no empty strings from names)
                 if not all(entity_pair_sorted):
                      print(f"Warning: Skipping exposure key due to invalid entity names: {entity_pair_sorted}. Relationship: {ownership_rel}")
                      continue

                 relationship_exposure_key = (entity_pair_sorted[0], entity_pair_sorted[1], r_type_upper)


                 # Consolidate data if the same relationship between the same entities has multiple risks/sanctions apply
                 if relationship_exposure_key not in consolidated_exposures:
                      consolidated_exposures[relationship_exposure_key] = {
                          "Entity": triggering_company_name, # The Chinese Co/Org that is SUBJECT_TO or has Compliance Risk
                          "Subsidiary/Affiliate": sub_aff_col_value,
                          "Parent Company": parent_col_value,
                          "Risk_Severity": "MEDIUM", # Start with MEDIUM, update based on found risks/sanctions
                          "Risk_Type_Label": exposure_risk_type_label, # Use the derived label (Parent, Subsidiary, Affiliate, JV Risk)
                          "Explanation_Details": set(), # Set to collect details (e.g., which sanctions/risks apply)
                          "Main_Sources": set() # Set to collect unique sources
                      }

                 # Find ALL relevant risks (High/Severe) and SUBJECT_TO relationships involving the triggering entity
                 # Add details and sources to the consolidated entry for this key

                 # Find relevant risks for this triggering entity
                 relevant_risks = [
                     r for r in all_risks_accumulated
                     if isinstance(r, dict) and r.get('severity') in ["HIGH", "SEVERE"] # Only High/Severe risks
                     and any(isinstance(e_name, str) and e_name.lower() == triggering_company_name_lower for e_name in r.get('related_entities', []))
                 ]
                 for risk in relevant_risks:
                      if isinstance(risk.get('description'), str):
                           # Include risk category in the detail
                           risk_desc_detail = risk['description'].strip()
                           risk_category_detail = risk.get('risk_category', 'UNKNOWN').strip()
                           detail_string = f"Risk ({risk_category_detail}): {risk_desc_detail}"
                           consolidated_exposures[relationship_exposure_key]["Explanation_Details"].add(detail_string)

                      if isinstance(risk.get('severity'), str) and risk.get('severity') in ["HIGH", "SEVERE"]:
                           # Update severity if the current risk is higher
                           current_severity = consolidated_exposures[relationship_exposure_key]["Risk_Severity"]
                           severity_order = ["LOW", "MEDIUM", "HIGH", "SEVERE"]
                           if severity_order.index(risk['severity']) > severity_order.index(current_severity):
                                consolidated_exposures[relationship_exposure_key]["Risk_Severity"] = risk['severity']
                      consolidated_exposures[relationship_exposure_key]["Main_Sources"].update(risk.get('source_urls', []))


                 # Find relevant SUBJECT_TO relationships for this triggering entity
                 relevant_subject_to_rels = [
                     rel for rel in all_collected_relationships
                     if isinstance(rel, dict)
                     and rel.get('relationship_type') == "SUBJECT_TO"
                     and isinstance(rel.get('entity1'), str) and isinstance(rel.get('entity2'), str) # Ensure both entity names are strings
                     and rel.get('entity1').lower() == triggering_company_name_lower # Entity1 is the triggering company
                     and rel.get('entity2').lower() in chinese_reg_sanc_names_lower # Entity2 is a Chinese Regulator/Sanction
                 ]
                 for rel in relevant_subject_to_rels:
                      if isinstance(rel.get('entity2'), str):
                           # Add sanction/regulator info to details
                           subject_to_entity_name = rel['entity2'].strip()
                           consolidated_exposures[relationship_exposure_key]["Explanation_Details"].add(f"SUBJECT_TO: {subject_to_entity_name}")
                           # A SUBJECT_TO relationship implies at least High severity, often Severe
                           current_severity = consolidated_exposures[relationship_exposure_key]["Risk_Severity"]
                           severity_order = ["LOW", "MEDIUM", "HIGH", "SEVERE"]
                           if severity_order.index("HIGH") > severity_order.index(current_severity):
                                consolidated_exposures[relationship_exposure_key]["Risk_Severity"] = "HIGH" # At least High
                           # Check if the entity2 is a SANCTION node explicitly for Severe
                           subject_to_entity_name_lower = subject_to_entity_name.lower()
                           subject_to_entity_obj = next((e for e in all_entities_accumulated if isinstance(e, dict) and e.get('name','') and e['name'].lower() == subject_to_entity_name_lower), None)
                           if subject_to_entity_obj and subject_to_entity_obj.get('type') == "SANCTION":
                                if severity_order.index("SEVERE") > severity_order.index(consolidated_exposures[relationship_exposure_key]["Risk_Severity"]):
                                     consolidated_exposures[relationship_exposure_key]["Risk_Severity"] = "SEVERE" # Sanction often implies Severe

                      consolidated_exposures[relationship_exposure_key]["Main_Sources"].update(rel.get('context_urls', []))


                 # Add the ownership relationship sources as well
                 consolidated_exposures[relationship_exposure_key]["Main_Sources"].update(rel_urls)


        generated_exposures = []
        # Convert the consolidated exposures dictionary back into a list of dictionaries
        for key, data in consolidated_exposures.items():
             # Ensure details are sorted and joined for the Explanation field
             explanation_details_list = sorted(list(data["Explanation_Details"]))
             explanation_details_str = "; ".join(explanation_details_list)

             # Construct the Explanation text
             # Ensure the explanation clearly states the triggers found (Subject To, Risk)
             trigger_phrases = []
             if any("SUBJECT_TO:" in detail for detail in explanation_details_list):
                  # Get the list of things it's subject to for the summary phrase
                  subject_to_items = [detail.replace("SUBJECT_TO: ", "") for detail in explanation_details_list if "SUBJECT_TO:" in detail]
                  trigger_phrases.append(f"is SUBJECT_TO: {'; '.join(subject_to_items)}")
             if any("Risk (" in detail for detail in explanation_details_list): # Check for the "Risk (Category):" format
                   # Get the list of risk descriptions/categories for the summary phrase
                   risk_details_only = [d.replace("Risk (", "").replace("):", ":").strip() for d in explanation_details_list if "Risk (" in d] # Clean up format for summary phrase
                   if risk_details_only:
                        # Reformat risk details slightly for the summary phrase if needed
                        formatted_risks_for_phrase = [item.split(':', 1) for item in risk_details_only] # Split into [Category, Description]
                        risk_phrase_parts = [f"{desc.strip()} ({cat.strip()})" for cat, desc in formatted_risks_for_phrase if cat and desc] # Format as "Description (Category)"
                        trigger_phrases.append(f"has High/Severe risks: {'; '.join(risk_phrase_parts)}")


             trigger_summary = " and ".join(trigger_phrases)


             explanation = f"Chinese Company/Organization '{data['Entity']}' {trigger_summary}. It is involved in a {data['Risk_Type_Label'].lower().replace('risk','').strip()} relationship with another Chinese Company/Org. Full Details: {explanation_details_str}."

             # Clean up Risk_Type label for display (e.g., remove " Risk")
             display_risk_type = data["Risk_Type_Label"].replace(" Risk", "").strip()

             exposure_row = {
                 "Entity": data["Entity"],
                 "Subsidiary/Affiliate": data["Subsidiary/Affiliate"],
                 "Parent Company": data["Parent Company"],
                 "Risk_Severity": data["Risk_Severity"],
                 "Risk_Type": display_risk_type, # Use the derived label (Parent Company, Subsidiary, Affiliate, Joint Venture)
                 "Explanation": explanation,
                 "Main_Sources": list(data["Main_Sources"]) # Convert set back to list
             }
             generated_exposures.append(exposure_row)

        # Assign the generated exposures to the results dictionary
        results["high_risk_exposures"] = generated_exposures


        print(f"[Step 3.5 Exposures] Generated {len(generated_exposures)} unique high/severe risk exposure rows after consolidation.")
        results["steps"].append({"name": "High Risk Exposure Generation", "duration": round(time.time() - step3_5_start, 2), "exposures_found": len(generated_exposures), "status": "OK" if generated_exposures else "No Exposures Generated" })

        print(f"\n--- Running Step 4: Wayback Machine Check ---")
        step4_start = time.time()
        # Collect URLs from all search results (Step 1 and Step 3) and Linkup structured data sources
        all_snippet_urls_combined = list(set( [r.get('url') for r in step1_search_results if isinstance(r, dict) and r.get('url') and isinstance(r.get('url'), str)] + [r.get('url') for r in step3_search_results if isinstance(r, dict) and r.get('url') and isinstance(r.get('url'), str)] ))
        # Add source URLs from processed Linkup structured data
        structured_data_urls = set()
        for item in results.get("linkup_structured_data", []):
             # structured_data_list contains items like {"entity": "...", "schema": "...", "data": {...}}
             if isinstance(item, dict) and isinstance(item.get("data"), dict):
                  structured_data_content = item.get("data")
                  # Check common places for URLs within structured data content dicts
                  if isinstance(structured_data_content.get("ownership_relationships"), list):
                       for rel in structured_data_content["ownership_relationships"]:
                            if isinstance(rel, dict) and isinstance(rel.get("source_url"), str):
                                structured_data_urls.add(rel["source_url"])
                  if isinstance(structured_data_content.get("key_risks_identified"), list):
                       for risk_item in structured_data_content["key_risks_identified"]:
                            if isinstance(risk_item, dict) and isinstance(risk_item.get("source_url"), str):
                                structured_data_urls.add(risk_item["source_url"])
                  # Add similar logic for other structured schemas (like regulations, sanctions) if they are implemented
                  # Example for a hypothetical "sanctions_regulations_found" list with "source_url":
                  if isinstance(structured_data_content.get("sanctions_regulations_found"), list):
                       for item_data in structured_data_content["sanctions_regulations_found"]:
                            if isinstance(item_data, dict) and isinstance(item_data.get("source_url"), str):
                                structured_data_urls.add(item_data["source_url"])

        # Add sources from the generated exposures as well
        exposure_source_urls = set()
        for exp in results.get("high_risk_exposures", []):
             if isinstance(exp, dict) and isinstance(exp.get("Main_Sources"), list):
                  exposure_source_urls.update(exp["Main_Sources"])


        urls_to_check = list(set(all_snippet_urls_combined + list(structured_data_urls) + list(exposure_source_urls))) # Combine and deduplicate all unique URLs

        wayback_checks = []
        if wayback_available:
            print(f"[Step 4 Wayback] Checking {len(urls_to_check)} URLs via Wayback Machine...");
            if urls_to_check:
                for url in urls_to_check:
                    # Add a small delay between Wayback checks if needed to avoid rate limits
                    # time.sleep(0.2) # Uncomment if needed
                    # print(f"  Checking: {url}") # Too verbose for many URLs
                    wayback_result = search_engines.check_wayback_machine(url);
                    wayback_checks.append(wayback_result);
                print(f"Finished checking {len(urls_to_check)} URLs via Wayback Machine.")
            else: print("[Step 4 Wayback] No URLs found to check.")
        else:
             print("[Step 4 Wayback] Skipping Wayback Machine check: search_engines.check_wayback_machine not available.")
             wayback_checks = [{"status": "skipped", "message": "check_wayback_machine not available", "original_url": "N/A"}] # Add placeholder if skipped

        results["wayback_results"] = wayback_checks
        # Update status based on whether checks were attempted and successful
        wayback_status = "OK" if wayback_checks and not any(chk.get("status", "").startswith("Error") or chk.get("status") == "skipped_invalid_url" for chk in wayback_checks) else "Error/Skipped"
        if not wayback_checks: wayback_status = "No URLs to Check"

        results["steps"].append({ "name": "Wayback Machine Check", "duration": round(time.time() - step4_start, 2), "urls_checked": len(urls_to_check), "status": wayback_status })

        print(f"\n--- Running Step 5: Prepare Data for Sheet & KG Update ---")
        step5_prep_start = time.time()

        # --- Filtering Logic (Moved here from _save_analysis_to_gsheet) ---
        all_entities_from_run = results.get("final_extracted_data", {}).get("entities", [])
        all_risks_from_run = results.get("final_extracted_data", {}).get("risks", [])
        all_relationships_from_run = results.get("final_extracted_data", {}).get("relationships", [])
        all_exposures_from_run = results.get("high_risk_exposures", []) # Use the list generated in step 3.5

        # Recalculate likely Chinese company/org names based on ALL accumulated entities
        # This is crucial for filtering consistently across sheets and KG
        likely_chinese_company_org_names = {e.get('name','') for e in all_entities_from_run if isinstance(e, dict) and e.get('name') and e.get('type') in ["COMPANY", "ORGANIZATION", "ORGANIZATION_NON_PROFIT", "GOVERNMENT_BODY"]} # Added potential other types for completeness in Chinese list
        likely_chinese_company_org_names_lower = {name.lower() for name in likely_chinese_company_org_names}
        print(f"[Step 5 Prep] Identified {len(likely_chinese_company_org_names_lower)} likely Chinese Company/Organization entities for filtering.")

        # Identify Chinese Regulatory Agencies and Sanctions based on filtering rules for sheet/KG
        chinese_reg_agency_names = {e.get('name','') for e in all_entities_from_run if isinstance(e, dict) and e.get('name') and e.get('type') == "REGULATORY_AGENCY"}
        chinese_sanction_names = {e.get('name','') for e in all_entities_from_run if isinstance(e, dict) and e.get('name') and e.get('type') == "SANCTION"}

        chinese_reg_agency_names_lower = {name.lower() for name in chinese_reg_agency_names}
        chinese_sanction_names_lower = {name.lower() for name in chinese_sanction_names}


        # Filter entities for the sheet/KG: ONLY Entities identified as likely Chinese Companies/Organizations,
        # AND ONLY Regulatory Agencies and Sanctions identified as likely Chinese (simplified check).
        entities_to_save_to_sheet = [] # Use a different variable name
        # List of common known non-Chinese regulators/sanctions/orgs/countries for filtering
        common_non_chinese_entities_lower = {
            "sec", "securities and exchange commission", "ofac", "office of foreign assets control",
            "hmrc", "irs", "fbi", "eu commission", "european commission", "us department of justice", "doj",
            "uk hmrc", "uk government", "united states", "us", "united kingdom", "uk",
            "germany", "de", "india", "in", "france", "fr", "japan", "jp", "canada", "ca", "australia", "au",
            "nato", "un", "world bank", "imf", "oecd", "international monetary fund", "world trade organization", "wto" # Added more international orgs
        } # Added more country/org names that might be misidentified


        for e in all_entities_from_run:
            if isinstance(e, dict) and e.get('name'):
                 entity_name = e.get('name')
                 entity_name_lower = entity_name.lower()
                 entity_type = e.get('type')

                 # Include Companies or Organizations if they are in the likely Chinese list
                 if entity_type in ["COMPANY", "ORGANIZATION", "ORGANIZATION_NON_PROFIT", "GOVERNMENT_BODY"]: # Include related ORG types if they were identified
                      if entity_name_lower in likely_chinese_company_org_names_lower:
                           entities_to_save_to_sheet.append(e)
                       # else: print(f"[Step 5 Prep] Filtering out non-Chinese Company/Org '{entity_name}'.") # Optional logging
                 # Include Regulators or Sanctions IF their name is NOT in the common non-Chinese list
                 # AND does NOT contain common non-Chinese country indicators.
                 elif entity_type in ["REGULATORY_AGENCY", "SANCTION"]:
                      if entity_name_lower not in common_non_chinese_entities_lower:
                           non_chinese_country_indicators = ["us", "uk", "eu", "usa", "united states", "united kingdom", "european", "germany", "india", "france", "japan", "canada", "australia"]
                           if not any(indicator in entity_name_lower for indicator in non_chinese_country_indicators):
                                # Assuming any regulator/sanction not explicitly filtered or containing foreign indicators is Chinese for now.
                                # A more robust approach would use country identified during NLP extraction if possible.
                                entities_to_save_to_sheet.append(e)
                           else:
                                print(f"[Step 5 Prep] Filtering out likely non-Chinese regulator/sanction '{entity_name}' based on country indicator.")
                      else:
                           print(f"[Step 5 Prep] Filtering out likely non-Chinese regulator/sanction '{entity_name}' from sheet/KG save (common list).")
                 elif entity_type == "ORGANIZATION":
                      # Include ORGANIZATION type only if it's in the likely Chinese company/org list
                      # (This captures Chinese state-owned enterprises often classified as ORGANIZATIONS)
                      if entity_name_lower in likely_chinese_company_org_names_lower:
                           entities_to_save_to_sheet.append(e)
                      # else: print(f"[Step 5 Prep] Filtering out non-Chinese or non-relevant ORGANIZATION '{entity_name}'.") # Optional logging
                 # else: print(f"[Step 5 Prep] Filtering out entity '{entity_name}' with type '{entity_type}' not explicitly allowed for sheet/KG save.")


        # Filter risks: only save risks that are related to at least one Chinese Company/Organization entity
        risks_to_save_to_sheet = [
            r for r in all_risks_from_run
            if isinstance(r, dict) and r.get('description')
            and any(isinstance(entity_name, str) and entity_name.lower() in likely_chinese_company_org_names_lower for entity_name in r.get('related_entities', []))
        ]

        # Filter relationships: Only include relationships where BOTH entities were deemed worthy of saving to the Entity sheet/KG
        relationships_to_save_to_sheet = []
        # Rebuild the set of entity names that *will* be saved based on the filtering above
        entity_names_that_will_be_saved_lower = {e.get('name','').lower() for e in entities_to_save_to_sheet if isinstance(e, dict) and e.get('name')}

        # Updated allowed relationship types for the Relationships sheet (Removing ACQUIRED, RELATED_COMPANY as requested)
        allowed_sheet_rel_types = ["PARENT_COMPANY_OF", "SUBSIDIARY_OF", "AFFILIATE_OF", "JOINT_VENTURE_PARTNER", "REGULATED_BY", "ISSUED_BY", "SUBJECT_TO", "MENTIONED_WITH"]


        for rel in all_relationships_from_run:
            if isinstance(rel, dict) and rel.get('entity1') and rel.get('relationship_type') and rel.get('entity2'):
                 e1_name = rel.get('entity1'); e2_name = rel.get('entity2'); r_type_raw = rel.get('relationship_type')
                 if isinstance(r_type_raw, str) and isinstance(e1_name, str) and isinstance(e2_name, str):

                     e1_name_lower = e1_name.lower()
                     e2_name_lower = e2_name.lower()
                     r_type_upper = r_type_raw.upper()

                     # Only include relationships where BOTH entities were deemed worthy of saving to the Entity sheet/KG
                     if e1_name_lower in entity_names_that_will_be_saved_lower and e2_name_lower in entity_names_that_will_be_saved_lower:
                          # Check if the relationship type is allowed for the sheet
                          if r_type_upper in allowed_sheet_rel_types:
                               relationships_to_save_to_sheet.append(rel)
                          else:
                                print(f"[Step 5 Prep] Filtering out relationship type '{r_type_raw}' not explicitly allowed for sheet/KG save between saved entities.")
                     # else: print(f"[Step 5 Prep] Filtering out relationship {rel} because one or both entities ('{e1_name}', '{e2_name}') were not saved to the entity list.") # Optional logging


        # Filter exposures: The list results["high_risk_exposures"] already contains only exposures matching the specified criteria from Step 3.5
        # So, we just need to assign it to the variable used for saving.
        exposures_to_save_to_sheet = list(results.get("high_risk_exposures", []))


        print(f"[Step 5 Prep] Finished filtering data: Entities:{len(entities_to_save_to_sheet)}, Risks:{len(risks_to_save_to_sheet)}, Relationships:{len(relationships_to_save_to_sheet)}, Exposures:{len(exposures_to_save_to_sheet)}.")

        results["steps"].append({"name": "Prepare Data for Sheet & KG", "duration": round(time.time() - step5_prep_start, 2), "filtered_counts": {"entities": len(entities_to_save_to_sheet), "risks": len(risks_to_save_to_sheet), "relationships": len(relationships_to_save_to_sheet), "exposures": len(exposures_to_save_to_sheet)}, "status": "OK"})


        print(f"\n--- Running Step 5.1: Knowledge Graph Update ---") # Renamed to 5.1 as prep is 5.0
        step5_1_start = time.time()
        update_success = False
        kg_status_message = results["kg_update_status"] # Start with the initial connection status

        # Only attempt KG update if driver is available and update function exists
        if kg_driver_available and kg_driver is not None and hasattr(knowledge_graph, 'update_knowledge_graph') and callable(knowledge_graph.update_knowledge_graph):
            # Deduplication logic
            unique_entities_dict = {}
            # Use the already filtered list for KG entities
            entities_for_kg = entities_to_save_to_sheet

            for e in entities_for_kg:
                 if isinstance(e, dict) and e.get('name') and e.get('type'):
                      key = (e['name'].strip().lower(), e['type'].upper())
                      if key not in unique_entities_dict:
                           # Create a new entry, copying the dictionary to avoid modifying the original lists
                           unique_entities_dict[key] = e.copy()
                           unique_entities_dict[key]['mentions'] = list(set(e.get('mentions', []))) # Ensure mentions is a unique list
                      else:
                           # Merge mentions
                           existing_mentions = unique_entities_dict[key].get('mentions', [])
                           new_mentions = e.get('mentions', [])
                           merged_mentions = list(set(existing_mentions + new_mentions))
                           unique_entities_dict[key]['mentions'] = merged_mentions

            unique_relationships_dict = {}
            # Use the already filtered list for KG relationships
            relationships_for_kg = relationships_to_save_to_sheet

            # Note: The KG update function itself already filters by allowed types and checks node labels,
            # so this filtering step here is partially redundant with KG's internal checks,
            # but it aligns the data sent to KG with the data saved to the sheet.
            # No need to explicitly list allowed_rel_types here, KG update function handles that check internally.


            for r in relationships_for_kg:
                 if isinstance(r, dict) and isinstance(r.get('entity1'), str) and isinstance(r.get('relationship_type'), str) and isinstance(r.get('entity2'), str):
                      entity1_name_lower = r['entity1'].strip().lower()
                      entity2_name_lower = r['entity2'].strip().lower()
                      rel_type = r['relationship_type'].upper()

                      # Use sorted entity names for the key regardless of entity1/entity2 order
                      entity_pair_sorted = tuple(sorted((entity1_name_lower, entity2_name_lower)))
                      key = (entity_pair_sorted[0], entity_pair_sorted[1], rel_type)

                      if key not in unique_relationships_dict:
                           # Create a new entry, copying the dictionary
                           unique_relationships_dict[key] = r.copy()
                           unique_relationships_dict[key]['context_urls'] = list(set(r.get('context_urls', []))) # Ensure URLs is unique list
                      else:
                           # Merge context URLs
                           existing_urls = unique_relationships_dict[key].get('context_urls', [])
                           new_urls = r.get('context_urls', [])
                           merged_urls = list(set(existing_urls + new_urls))
                           unique_relationships_dict[key]['context_urls'] = merged_urls


            unique_risks_dict = {}
            # Use the already filtered list for KG risks
            risks_for_kg = risks_to_save_to_sheet

            for r in risks_for_kg:
                 if isinstance(r, dict) and isinstance(r.get('description'), str):
                      key = r['description'].strip().lower()
                      if key not in unique_risks_dict:
                           # Create a new entry, copying the dictionary
                           unique_risks_dict[key] = r.copy()
                           unique_risks_dict[key]['related_entities'] = list(set(r.get('related_entities', []))) # Ensure unique related entities
                           unique_risks_dict[key]['source_urls'] = list(set(r.get('source_urls', []))) # Ensure unique source urls
                      else:
                           # Merge related entities and sources
                           existing_entities = unique_risks_dict[key].get('related_entities', [])
                           new_entities = r.get('related_entities', [])
                           merged_entities = list(set(existing_entities + new_entities))
                           unique_risks_dict[key]['related_entities'] = merged_entities

                           existing_urls = unique_risks_dict[key].get('source_urls', [])
                           new_urls = r.get('source_urls', [])
                           merged_urls = list(set(existing_urls + new_urls))
                           unique_risks_dict[key]['source_urls'] = merged_urls

            unique_entities = list(unique_entities_dict.values())
            unique_relationships = list(unique_relationships_dict.values())
            unique_risks = list(unique_risks_dict.values())

            deduped_data_for_kg_update = {"entities": unique_entities, "risks": unique_risks, "relationships": unique_relationships}
            print(f"[Step 5.1 KG] Data AFTER deduplication, FILTERING, and PREP for KG: E:{len(unique_entities)}, R:{len(unique_risks)}, Rel:{len(unique_relationships)}.")


            if not any([unique_entities, unique_risks, unique_relationships]):
                 kg_status_message = "skipped_no_data_for_kg" # Specific status if no data remains after filtering/dedup
                 print("[Step 5.1 KG] No unique data for KG update after filtering.")
            else:
                 print(f"[Step 5.1 KG] Sending {len(unique_entities) + len(unique_risks) + len(unique_relationships)} items to KG update...")
                 # Pass the *filtered* and *deduplicated* data to the KG update function
                 update_success = knowledge_graph.update_knowledge_graph(deduped_data_for_kg_update)
                 kg_status_message = "success" if update_success else "error"
        else:
            print("[Step 5.1 KG] Skipping KG update: driver or update function not available.")
            # kg_status_message was already set based on initial connection attempt


        results["kg_update_status"] = kg_status_message # Update results dict with final status
        results["steps"].append({"name": "Knowledge Graph Update", "duration": round(time.time() - step5_1_start, 2), "status": results["kg_update_status"]})


        print(f"\n--- Running Step 5.5: Generating Analysis Summary ---")
        step5_5_start = time.time()
        summary = "Summary generation skipped or failed."
        # Use the FINAL, accumulated data for the summary
        final_data_for_summary = results.get("final_extracted_data", {})
        exposures_for_summary_count = len(results.get("high_risk_exposures", [])) # Use the count of generated exposures
        structured_data_list = results.get("linkup_structured_data", [])
        structured_data_present = bool(structured_data_list)

        print(f"[Step 5.5 Summary] Data for summary: E:{len(final_data_for_summary.get('entities',[]))}, R:{len(final_data_for_summary.get('risks',[]))}, Rel:{len(final_data_for_summary.get('relationships',[]))}, Exp:{exposures_for_summary_count}, Structured:{structured_data_present}.")

        # Check if there is ANY data collected before attempting summary
        if final_data_for_summary.get("entities") or final_data_for_summary.get("risks") or final_data_for_summary.get("relationships") or exposures_for_summary_count > 0 or structured_data_present:
             summary_data_payload = {
                 "entities": final_data_for_summary.get("entities", []),
                 "risks": final_data_for_summary.get("risks", []),
                 "relationships": final_data_for_summary.get("relationships", []),
                 "linkup_structured_data": structured_data_list,
                 "high_risk_exposures": results.get("high_risk_exposures", []) # Include exposures for summary context
             }
             if nlp_processor_available and hasattr(nlp_processor, 'generate_analysis_summary') and callable(nlp_processor.generate_analysis_summary):
                 # Pass the full results dictionary here
                 summary = nlp_processor.generate_analysis_summary(
                     results, # Pass the full results dict
                     initial_query,
                     exposures_for_summary_count,
                     llm_provider_to_use,
                     llm_model_to_use
                     )
             else:
                  print("Warning: NLP processor or generate_analysis_summary function not available.")
                  summary = "Summary generation skipped: NLP processor or generate_analysis_summary function missing."
        else:
             summary = "No significant data extracted or exposures identified across steps to generate a summary."
             print("[Step 5.5 Summary] Skipping summary generation due to lack of data.")

        print(f"[Step 5.5 Summary] Generated Summary: {summary[:150]}...")
        results["analysis_summary"] = summary
        summary_status = "OK"
        if "Could not generate" in summary or "No significant" in summary or "failed due to" in summary:
            summary_status = "Skipped/Failed"

        results["steps"].append({"name": "Analysis Summary Generation", "duration": round(time.time() - step5_5_start, 2), "status": summary_status})


        print("\n--- Running Step 6: Learning/Adapting (Deferred) ---")
        # Placeholder for future learning/adaptation steps
        results["steps"].append({"name": "Learning/Adapting", "duration": 0, "status": "deferred"})


    except Exception as e:
        print(f"\n--- Orchestrator ERROR ---")
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else "No message"
        error_traceback = traceback.format_exc()
        print(f"Error Type: {error_type}\nError Message: {error_msg}\nTraceback:\n{error_traceback}")

        # Ensure results is a dictionary and update error details
        if isinstance(results, dict):
             results["error"] = f"{error_type}: {error_msg}"
             # Find the last step or append a new one to log the error
             if results["steps"]:
                  # Check if the last step is a dictionary before modifying
                  if isinstance(results["steps"][-1], dict):
                      # Add error info to the last step that was running or just finished
                      last_step = results["steps"][-1]
                      last_step_name = last_step.get("name", "Unknown Step")
                      if "status" in last_step and not str(last_step["status"]).startswith("Error"):
                           last_step["status"] = f"Error: {error_type}" # Update status if not already an error
                      last_step["error_message"] = error_msg # Add error message
                      if "duration" not in last_step: # Ensure duration is logged if error occurred mid-step
                            # Calculate approximate duration for the step that failed
                            total_elapsed = time.time() - start_run_time
                            previous_steps_duration = sum(s.get("duration", 0) for s in results["steps"][:-1] if isinstance(s, dict))
                            last_step["duration"] = round(total_elapsed - previous_steps_duration, 2) if total_elapsed > previous_steps_duration else 0.0

                  else:
                      # Last step is not a dictionary, append a new error step
                      print(f"Warning: Last step in results['steps'] is not a dictionary ({type(results['steps'][-1])}). Appending new error step.")
                      results["steps"].append({"name": "Orchestrator Error (Appended)", "status": f"Error: {error_type}", "error_message": error_msg, "duration": round(time.time() - start_run_time, 2)}) # Log duration up to error
             else:
                  # No steps logged yet, append the first error step
                  results["steps"].append({"name": "Orchestrator Error", "status": f"Error: {error_type}", "error_message": error_msg, "duration": round(time.time() - start_run_time, 2)}) # Log duration up to error

             # Ensure extracted_data_counts is added to steps if it wasn't already
             for step_data in results["steps"]:
                  if isinstance(step_data, dict) and "extracted_data" in step_data:
                       if isinstance(step_data.get("extracted_data"), dict):
                            step_data["extracted_data_counts"] = {k: len(v) for k,v in step_data["extracted_data"].items()}
                       del step_data["extracted_data"] # Remove large list of data


        else:
             print("ERROR: 'results' variable is not a dictionary during error handling. Cannot log error details to results object.")
             # Attempt to save basic error info to sheet if possible
             try:
                 if google_sheets_available:
                      exc_type, exc_value, exc_traceback = sys.exc_info()
                      basic_error_results = {
                         "query": initial_query,
                         "run_duration_seconds": round(time.time() - start_run_time, 2),
                         "kg_update_status": "error",
                         "error": f"Critical Orchestrator Failure: {type(exc_value).__name__}: {exc_value}",
                         "analysis_summary": "Analysis failed due to critical internal error."
                     }
                      # Pass empty lists as no data was filtered/prepared
                      _save_analysis_to_gsheet(basic_error_results, [], [], [], [])
             except Exception as save_e:
                  print(f"CRITICAL ERROR: Failed to save even basic error info to sheets: {save_e}")
                  traceback.print_exc()


    finally:
        # Ensure KG driver is closed if it was successfully obtained and is not None
        if kg_driver_available and kg_driver is not None:
            try:
                 driver.close()
                 driver = None
                 print("Neo4j connection closed.")
            except Exception as e:
                 print(f"Error closing Neo4j driver: {e}")

        # Update final duration and attempt to save results to Google Sheets
        if isinstance(results, dict):
             results["run_duration_seconds"] = round(time.time() - start_run_time, 2)

             # Remove large data structures from final results to avoid issues with JSON serialization if not needed
             # Example: Remove raw search results or redundant data from step logs
             for step_data in results.get("steps", []):
                  if isinstance(step_data, dict) and "extracted_data" in step_data:
                       if isinstance(step_data.get("extracted_data"), dict):
                            step_data["extracted_data_counts"] = {k: len(v) for k,v in step_data["extracted_data"].items()}
                       del step_data["extracted_data"] # Remove large list of data

             # Pass the filtered data (now defined in the main try block scope) to the save function
             if google_sheets_available:
                 # Ensure lists exist even if filtering failed partially by checking locals()
                 _save_analysis_to_gsheet(
                     results,
                     entities_to_save_to_sheet if 'entities_to_save_to_sheet' in locals() else [],
                     risks_to_save_to_sheet if 'risks_to_save_to_sheet' in locals() else [],
                     relationships_to_save_to_sheet if 'relationships_to_save_to_sheet' in locals() else [],
                     exposures_to_save_to_sheet if 'exposures_to_save_to_sheet' in locals() else []
                 )
             else: print("Skipping save to Google Sheets: Configuration missing or invalid.")
        else:
             print("ERROR: 'results' variable is not a dictionary in the finally block. Cannot update final metrics or save to sheets.")

        print(f"\n--- Analysis Complete ({results.get('run_duration_seconds', 'N/A')}s) ---")
        if isinstance(results, dict) and results.get("error"): print(f"--- Run finished with ERROR: {results['error']} ---")
        elif not isinstance(results, dict): print("--- Run finished with CRITICAL ERROR (results variable corrupted) ---")
        else: print(f"--- Run finished successfully (KG Status: {results.get('kg_update_status','?')}, Exposures: {len(results.get('high_risk_exposures', []))}) ---")


    return results

if __name__ == "__main__":
    print("\n--- Running Local Orchestrator Tests ---")
    print("NOTE: Requires LLM API keys and search API keys in .env.")
    print("Ensure Neo4j is running if KG update is enabled.")
    print("Ensure Google Sheets is configured if saving is enabled.")

    test_query = "Corporate tax evasion cases in China 2023"
    test_country = "cn"

    test_llm_provider = "openai"
    test_llm_model = config.DEFAULT_OPENAI_MODEL

    print(f"\nRunning analysis for query: '{test_query}' in country: '{test_country}'")

    try:
        test_run_results = run_analysis(
            initial_query=test_query,
            llm_provider=test_llm_provider,
            llm_model=test_llm_model,
            specific_country_code=test_country,
            max_global_results=20,
            max_specific_results=20
        )

        print("\n--- Test Run Results ---")
        # Print results nicely, but avoid printing huge lists directly
        printable_results = test_run_results.copy()
        if 'linkup_structured_data' in printable_results:
             printable_results['linkup_structured_data_count'] = len(printable_results['linkup_structured_data'])
             del printable_results['linkup_structured_data']
        if 'wayback_results' in printable_results:
             printable_results['wayback_results_count'] = len(printable_results['wayback_results'])
             # Optionally truncate or remove details if the list is very long
             # printable_results['wayback_results_sample'] = printable_results['wayback_results'][:3]
             # del printable_results['wayback_results']
        if 'final_extracted_data' in printable_results:
             # Summarize counts within final_extracted_data
             printable_results['final_extracted_data_counts'] = {
                 k: len(v) for k, v in printable_results['final_extracted_data'].items()
             }
             # Optionally remove the full lists if they are large
             # del printable_results['final_extracted_data']
        if 'high_risk_exposures' in printable_results:
             printable_results['high_risk_exposures_count'] = len(printable_results['high_risk_exposures'])
             # Optionally truncate or remove details if the list is very long
             # printable_results['high_risk_exposures_sample'] = printable_results['high_risk_exposures'][:3]
             del printable_results['high_risk_exposures']

        # Clean up steps data to show counts rather than full extracted_data lists
        if 'steps' in printable_results:
             for step in printable_results['steps']:
                  if isinstance(step, dict) and 'extracted_data' in step:
                       if isinstance(step['extracted_data'], dict):
                            step['extracted_data_counts'] = {k: len(v) for k,v in step['extracted_data'].items()}
                       del step['extracted_data']


        print(json.dumps(printable_results, indent=2))

    except Exception as e:
        print(f"\n--- Test Run Exception ---")
        print(f"An exception occurred during the test run: {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\n--- Local Orchestrator Tests Complete ---")