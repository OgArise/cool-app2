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
        # --- ENSURE service is returned even if exception occurred ---
        # The global gsheet_service variable is updated inside the try/except block
        # The return statement needs to access the global variable.
        # The logic below relies on gsheet_service being None if the build failed.
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
        print(f"--- Using Selected LLM: {llm_provider_to_use}/{llm_model_to_use} ---")

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


    # Re-evaluate nlp_extraction_available with all required functions
    nlp_extraction_available = nlp_processor_available and \
        hasattr(nlp_processor, 'translate_keywords_for_context') and callable(nlp_processor.translate_keywords_for_context) and \
        hasattr(nlp_processor, 'translate_text') and callable(nlp_processor.translate_text) and \
        hasattr(nlp_processor, 'translate_snippets') and callable(nlp_processor.translate_snippets) and \
        hasattr(nlp_processor, 'extract_entities_only') and callable(nlp_processor.extract_entities_only) and \
        hasattr(nlp_processor, 'extract_risks_only') and callable(nlp_processor.extract_risks_only) and \
        hasattr(nlp_processor, 'link_entities_to_risk') and callable(nlp_processor.link_entities_to_risk) and \
        hasattr(nlp_processor, 'extract_relationships_only') and callable(nlp_processor.extract_relationships_only) and \
        hasattr(nlp_processor, 'extract_regulatory_sanction_relationships') and callable(nlp_processor.extract_regulatory_sanction_relationships) and \
        hasattr(nlp_processor, 'process_linkup_structured_data') and callable(nlp_processor.process_linkup_structured_data) and \
        hasattr(nlp_processor, 'generate_analysis_summary') and callable(nlp_processor.generate_analysis_summary)


    if not nlp_extraction_available:
         results["error"] = "NLP processor module or essential functions not available."; print(f"--- Orchestrator ERROR: {results['error']} ---")
         results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
         # Pass empty lists to _save_analysis_to_gsheet since no data was extracted
         # Pass LLM config used in the run for potential error logging in sheet save
         if google_sheets_available: _save_analysis_to_gsheet(results, [], [], [], [], llm_provider_to_use, llm_model_to_use)
         if kg_driver_available and kg_driver is not None: knowledge_graph.close_driver() # Close if it was successfully obtained
         return results

    try:
         # Attempt to initialize the LLM client early to catch config errors
         # Pass provider and model from the run request
         nlp_processor._get_llm_client_and_model(llm_provider_to_use, llm_model_to_use)
         print("LLM client initialized successfully for NLP processing.")
    except Exception as e:
         results["error"] = f"LLM configuration/initialization failed: {type(e).__name__}: {e}"; print(f"--- Orchestrator ERROR: {results['error']} ---")
         results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
         # Pass empty lists to _save_analysis_to_gsheet since no data was extracted
         # Pass LLM config used in the run for potential error logging in sheet save
         if google_sheets_available: _save_analysis_to_gsheet(results, [], [], [], [], llm_provider_to_use, llm_model_to_use)
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


    # Initialize variables that need to persist through the try block
    # These are populated *within* the try block but potentially accessed in `finally` or later steps
    all_entities_accumulated: List[Dict] = []
    all_risks_accumulated: List[Dict] = []
    all_collected_relationships: List[Dict] = []
    raw_linkup_structured_data_collected: List[Dict] = []

    # Variables used for filtering and exposure generation, populated *within* the try block
    likely_chinese_company_org_names_set: set[str] = set()
    likely_chinese_company_org_names: set[str] = set()
    likely_chinese_company_org_names_lower: set[str] = set()
    triggering_chinese_company_org_names_lower_initial: set[str] = set()
    all_reg_agency_names: set[str] = set()
    all_sanction_names: set[str] = set()
    all_reg_sanc_names_lower: set[str] = set()
    consolidated_exposures: Dict[Tuple, Dict] = {} # Used internally for exposure generation deduplication
    generated_exposures: List[Dict] = [] # Holds the list of generated exposure dictionaries

    # Final lists for saving to sheet/KG - populated *after* filtering in Step 5
    entities_to_save_to_sheet: List[Dict] = []
    risks_to_save_to_sheet: List[Dict] = []
    relationships_to_save_to_sheet: List[Dict] = []
    exposures_to_save_to_sheet: List[Dict] = [] # Holds exposure dictionaries filtered for saving to sheet


    try:
        print(f"\n--- Running Step 1: Initial Search ---")
        step1_start = time.time()
        step1_search_results = [] # This will hold the final, deduplicated list for Step 1 extraction


        step1_english_queries = []
        # Check if nlp_processor and specific functions are available before calling
        # The check for nlp_extraction_available covers these
        if nlp_extraction_available:
             step1_english_queries = nlp_processor.translate_keywords_for_context(
                  initial_query, global_search_context,
                  llm_provider_to_use, llm_model_to_use
                  )
        if not step1_english_queries: step1_english_queries = [initial_query]


        step1_chinese_queries = []
        if specific_country_code.lower() == 'cn' and nlp_extraction_available:
            print("[Step 1 Search] Translating queries to Chinese for Baidu search...")
            for query_en in step1_english_queries:
                # Use the LLM config from the run request for translation
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
                  # Removed num and country_code params from the call based on previous TypeErrors.
                  # The number of results per query is handled by Linkup's defaults or internal logic.
                  linkup_global_results = search_engines.search_linkup_snippets(query=step1_all_queries)
                  if linkup_global_results:
                       # Ensure Linkup results are standardized before adding to the map/list
                       standardized_linkup_results = [search_engines.standardize_result(r, source='linkup_snippet_step1') for r in linkup_global_results]
                       standardized_linkup_results = [r for r in standardized_linkup_results if r is not None] # Filter out invalid ones

                       print(f"    Linkup Snippet Search returned {len(standardized_linkup_results)} unique *standardized* results after internal deduplication.")

                       for r in standardized_linkup_results:
                           if isinstance(r, dict) and r.get('url') and isinstance(r.get('url'), str) and r['url'] not in all_search_results_map:
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
                 except Exception as e:
                      print(f"    SerpApi ({serpapi_engine}) call failed for query '{query_text[:50]}...': {type(e).__name__}: {e}") # Added closing parenthesis
                      traceback.print_exc()

        elif serpapi_available and not config.SERPAPI_KEY:
             print("[Step 1 Search] Skipping Serpapi search - not configured.")
        elif len(step1_search_results) >= search_threshold_results:
             print(f"[Step 1 Search] Skipping Serpapi. Preceding steps returned {len(step1_search_results)} results (>= {search_threshold_results} threshold).")


        # NO FINAL TRUNCATION HERE. We keep all unique results found up to the limits of individual search calls.

        print(f"[Step 1 Search] Total standardized results from Linkup/Google/SerpApi after combining & deduplication: {len(step1_search_results)}")

        snippets_to_translate = []
        # nlp_extraction_available check includes translate_snippets
        if nlp_extraction_available:
             # Only translate if original language is known and not English, and snippet is available
             snippets_to_translate = [s for s in step1_search_results if isinstance(s, dict) and s.get('snippet') and isinstance(s.get('snippet'), str) and s.get('original_language') and s['original_language'].lower() not in ['en', 'english']]

        if snippets_to_translate:
             print(f"\n--- Translating {len(snippets_to_translate)} non-English snippets from Step 1 ---")
             # Store original search results temporarily to update them with translations
             original_step1_search_results = list(step1_search_results)
             step1_search_results = [] # Reset to rebuild with translations

             # Pass LLM config from the run request for translation
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
        # nlp_extraction_available covers these
        nlp_extraction_step1_ok = nlp_extraction_available


        if step1_search_results and nlp_extraction_step1_ok:
            step1_context = f"Analyze search results for query '{initial_query}'. Extract relevant entities, risks, and relationships based ONLY on the provided text snippets."
            print(f"[Step 1 Extract] Calling NLP processor (Multi-Call)... Context: {step1_context[:100]}...")

            # Pass LLM config from the run request
            step1_entities = nlp_processor.extract_entities_only(step1_search_results, step1_context, llm_provider_to_use, llm_model_to_use)
            # Filter entities to only include valid ones with names
            step1_entities_filtered = [e for e in step1_entities if isinstance(e, dict) and e.get('name')]
            step1_entity_names = [e['name'] for e in step1_entities_filtered if isinstance(e,dict) and e.get('name')] # Get names from validated entities
            print(f"[Step 1 Extract] Extracted {len(step1_entities_filtered)} entities of various types.")
            time.sleep(1.0)

            # Pass LLM config from the run request
            step1_risks_initial = nlp_processor.extract_risks_only(step1_search_results, step1_context, llm_provider_to_use, llm_model_to_use)
            time.sleep(1.0)

            step1_risks = []
            if step1_risks_initial and step1_entity_names:
                 print(f"[Step 1 Linker] Starting entity linking for {len(step1_risks_initial)} risks against {len(step1_entity_names)} filtered entities...")
                 # Ensure all_snippets_map is passed
                 # Pass LLM config from the run request
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
                 # Pass LLM config from the run request
                 step1_ownership_relationships = nlp_processor.extract_relationships_only(step1_search_results, step1_context, step1_entities_company_org, llm_provider_to_use, llm_model_to_use)
            else: print("[Step 1 Extract] Skipping ownership relationship extraction - no COMPANY/ORGANIZATION entities found.")
            time.sleep(1.0)

            step1_reg_sanc_relationships = []
            # Regulatory/Sanction relationships can involve any relevant entity type
            if step1_entities_filtered:
                 print(f"[Step 1 Extract] Calling NLP for Regulatory/Sanction Relationships...")
                 # Pass LLM config from the run request
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
        # raw_linkup_structured_data_collected is initialized before try


        # Check if Linkup structured search and processing functions are available
        linkup_structured_available_orchestrator = linkup_structured_search_available and nlp_extraction_available # nlp_processor.process_linkup_structured_data is included in nlp_extraction_available


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
                               # Pass the configured LLM provider and model to search_linkup_structured as it might use NLP internally
                               structured_ownership_data_item = search_engines.search_linkup_structured(
                                    query=ownership_query,
                                    structured_output_schema=LINKUP_SCHEMA_OWNERSHIP,
                                    depth="deep", # Use deep for potentially more results
                                    # Removed country_code param
                                    # Added LLM config params
                                    llm_provider=llm_provider_to_use,
                                    llm_model=llm_model_to_use
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
                               # Pass the configured LLM provider and model to search_linkup_structured as it might use NLP internally
                               structured_risks_data_item = search_engines.search_linkup_structured(
                                    query=risks_query,
                                    structured_output_schema=LINKUP_SCHEMA_KEY_RISKS,
                                    depth="deep", # Use deep for potentially more results
                                    # Removed country_code param
                                     # Added LLM config params
                                    llm_provider=llm_provider_to_use,
                                    llm_model=llm_model_to_use
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
                      # Process the raw structured data into internal format
                      # Pass LLM config from the run request to nlp_processor
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
        # The check for nlp_extraction_available covers these
        if nlp_extraction_available:
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
        # nlp_extraction_available check includes translate_text
        if specific_country_code.lower() == 'cn' and nlp_extraction_available:
            print("[Step 3 Search] Translating queries to Chinese for Baidu search...")
            for query_en in step3_english_queries_base:
                # Pass LLM config from the run request for translation
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
                  # Pass the combined query to search_linkup_snippets. Removed num and country_code params.
                  linkup_specific_results = search_engines.search_linkup_snippets(query=step3_all_queries)
                  if linkup_specific_results:
                       # Ensure Linkup results are standardized before adding to the map/list
                       standardized_linkup_results = [search_engines.standardize_result(r, source='linkup_snippet_step3') for r in linkup_specific_results]
                       standardized_linkup_results = [r for r in standardized_linkup_results if r is not None] # Filter out invalid ones

                       print(f"    Linkup Snippet Search returned {len(standardized_linkup_results)} unique *standardized* results after internal deduplication.")

                       for r in standardized_linkup_results:
                            if isinstance(r, dict) and r.get('url') and isinstance(r.get('url'), str) and r['url'] not in all_search_results_map:
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
                             standardized = search_engines.standardize_result(r, f"google_cse_q{q_idx+1}")
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
                 except Exception as e:
                      print(f"    SerpApi ({serpapi_engine}) call failed for query '{query_text[:50]}...': {type(e).__name__}: {e}") # Added closing parenthesis
                      traceback.print_exc()

        elif serpapi_available and not config.SERPAPI_KEY:
             print("[Step 3 Search] Skipping Serpapi search - not configured.")
        elif len(step3_search_results) >= required_for_threshold:
             print(f"[Step 3 Search] Skipping Serpapi. Preceding steps returned {len(step3_search_results)} results (>= {required_for_threshold} threshold).")


        # NO FINAL TRUNCATION HERE. We keep all unique results found up to the limits of individual search calls.

        print(f"[Step 3 Search] Total standardized results from Linkup/Google/SerpApi after combining & deduplication: {len(step3_search_results)}")


        snippets_to_translate = []
        # nlp_extraction_available check includes translate_snippets
        if nlp_extraction_available:
             # Only translate if original language is known and not English, and snippet is available
             snippets_to_translate = [s for s in step3_search_results if isinstance(s, dict) and s.get('snippet') and isinstance(s.get('snippet'), str) and s.get('original_language') and s['original_language'].lower() not in ['en', 'english']]


        if snippets_to_translate:
             print(f"\n--- Translating {len(snippets_to_translate)} non-English snippets from Step 3 ---")
             # Store original search results temporarily to update them with translations
             original_step3_search_results = list(step3_search_results)
             step3_search_results = [] # Reset to rebuild with translations

             # Pass LLM config from the run request for translation
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
        # nlp_extraction_available covers these
        nlp_extraction_step3_ok = nlp_extraction_available


        if step3_search_results and nlp_extraction_step3_ok:
            step3_context = specific_search_context
            print(f"[Step 3 Extract] Calling NLP processor (Multi-Call)... Context: {step3_context[:100]}...")

            # Pass LLM config from the run request
            step3_entities = nlp_processor.extract_entities_only(step3_search_results, step3_context, llm_provider_to_use, llm_model_to_use)
            # Filter entities to only include valid ones with names
            step3_entities_filtered = [e for e in step3_entities if isinstance(e, dict) and e.get('name')]
            step3_entity_names = [e['name'] for e in step3_entities_filtered if isinstance(e,dict) and e.get('name')] # Get names from validated entities
            print(f"[Step 3 Extract] Extracted {len(step3_entities_filtered)} entities of various types.")
            time.sleep(1.0)

            # Pass LLM config from the run request
            step3_risks_initial = nlp_processor.extract_risks_only(step3_search_results, step3_context, llm_provider_to_use, llm_model_to_use)
            time.sleep(1.0)

            step3_risks = []
            if step3_risks_initial and step3_entity_names:
                 print(f"[Step 3 Linker] Starting entity linking for {len(step3_risks_initial)} risks against {len(step3_entity_names)} filtered entities...")
                 # Ensure all_snippets_map is passed
                 # Pass LLM config from the run request
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
                 # Pass LLM config from the run request
                 step3_ownership_relationships = nlp_processor.extract_relationships_only(step3_search_results, step3_context, step3_entities_company_org, llm_provider_to_use, llm_model_to_use)
            else: print("[Step 3 Extract] Skipping ownership relationship extraction - no COMPANY/ORGANIZATION entities found.")
            time.sleep(1.0)

            step3_reg_sanc_relationships = []
            # Regulatory/Sanction relationships can involve any relevant entity type
            if step3_entities_filtered:
                 print(f"[Step 3 Extract] Calling NLP for Regulatory/Sanction Relationships...")
                 # Pass LLM config from the run request
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

        # Retrieve the *final* accumulated lists from the results dict after all extraction steps
        all_entities_accumulated = results["final_extracted_data"].get("entities", [])
        all_risks_accumulated = results["final_extracted_data"].get("risks", [])
        all_collected_relationships = results["final_extracted_data"].get("relationships", []) # Use .get() just in case


        # Recalculate likely Chinese company/org names based on ALL accumulated entities
        # This is crucial for filtering consistently across sheets and KG
        # Include names from Linkup structured data and derived entities tagged as COMPANY/ORGANIZATION
        likely_chinese_company_org_names_set = {e.get('name','') for e in all_entities_accumulated if isinstance(e, dict) and e.get('name') and e.get('type') in ["COMPANY", "ORGANIZATION", "ORGANIZATION_NON_PROFIT", "GOVERNMENT_BODY", "SANCTION"]} # Added SANCTION as they can be subjects
        structured_companies_from_raw_data = {item.get('data',{}).get('company_name','') for item in results.get("linkup_structured_data", []) if isinstance(item, dict) and item.get("schema") in ["ownership", "key_risks"]}
        likely_chinese_company_org_names_set.update(structured_companies_from_raw_data)
        # Add names from entities derived from structured data processing (should have _source_type: linkup_structured)
        structured_derived_entities = {e.get('name','') for e in all_entities_accumulated if isinstance(e, dict) and e.get('_source_type') == 'linkup_structured' and e.get('type') in ["COMPANY", "ORGANIZATION", "ORGANIZATION_NON_PROFIT", "GOVERNMENT_BODY", "SANCTION"]}
        likely_chinese_company_org_names_set.update(structured_derived_entities)

        likely_chinese_company_org_names_set.discard('') # Remove empty strings


        # Recalculate the set of triggering entities based on all accumulated data
        # Find ALL relevant risks (High/Severe) and SUBJECT_TO relationships involving the triggering entity (any entity type for now)
        high_severe_risks = [r for r in all_risks_accumulated if isinstance(r, dict) and r.get('severity') in ["HIGH", "SEVERE"]]
        relevant_risk_categories_lower = ['compliance', 'financial', 'legal', 'regulatory', 'governance'] # Added more categories
        companies_with_high_severe_relevant_risks = set()
        for risk in high_severe_risks:
             risk_desc = risk.get('description', '').lower()
             risk_category = risk.get('risk_category', '').lower() # Use extracted category
             if risk_category in relevant_risk_categories_lower or (risk_category == "UNKNOWN" and any(keyword in risk_desc for keyword in relevant_risk_categories_lower + ['sanction', 'fine', 'penalty', 'violation', 'lawsuit', 'investigation', 'fraud', 'corruption'])):
                   for entity_name in risk.get('related_entities', []):
                        # Check if the related entity is in the set of *all* accumulated names initially deemed potentially Chinese
                        if isinstance(entity_name, str) and entity_name.strip().lower() in likely_chinese_company_org_names_set: # Use set of all potential names here
                             companies_with_high_severe_relevant_risks.add(entity_name.strip().lower())

        # Find SUBJECT_TO relationships where entity1 is in the set of *all* accumulated names initially deemed potentially Chinese
        companies_subject_to_reg_sanc_rels = [
            rel for rel in all_collected_relationships
            if isinstance(rel, dict)
            and rel.get('relationship_type') == "SUBJECT_TO"
            and isinstance(rel.get('entity1'), str) # Ensure entity1 is a string
            and rel.get('entity1').strip().lower() in likely_chinese_company_org_names_set # Entity1 is a potentially Chinese Company/Org
            # No need to check entity2 type/origin here for the trigger list calculation, just need entity1 to be the subject.
        ]
        companies_subject_to_names_lower = {rel['entity1'].strip().lower() for rel in companies_subject_to_reg_sanc_rels}

        # Combine all potentially Chinese entities that are triggers
        triggering_chinese_company_org_names_lower_initial = companies_subject_to_names_lower.union(companies_with_high_severe_relevant_risks)


        if specific_country_code.lower() == 'cn':
             # Keep names that were identified AND contain Chinese characters OR are already translated English names that were linked to Chinese sources/context (more complex check)
             # For simplicity now, let's just assume entities with Chinese characters are "Chinese" for the sheet/KG filtering in the CN run.
             # A more robust approach would involve language detection on the entity name or source snippet.
             # Keep triggers even if name is English translation if the trigger logic identified them based on relationships/risks involving Chinese entities
             likely_chinese_company_org_names = {name for name in likely_chinese_company_org_names_set if contains_chinese(name) or name.lower() in triggering_chinese_company_org_names_lower_initial}
        else:
             # If not a CN run, this filtering logic might be less relevant or need different criteria.
             # For now, assume we save/KG entities identified if not in common non-country specific list.
             # Revert to a simpler filter or adapt based on non-CN requirements.
             # For this code version focused on the CN example, we'll stick to the CN-centric filter if specific_country_code is 'cn'.
             # If not 'cn', we'll skip this Chinese-character based filtering for entity names and use all identified as potentially relevant.
             likely_chinese_company_org_names = likely_chinese_company_org_names_set # In non-CN run, keep all initially identified as potential


        # Make sure triggering_chinese_company_org_names_lower_initial is a set of lowercase strings
        # It already is from the calculation above, but adding a safety check
        triggering_chinese_company_org_names_lower_initial = {name.lower() for name in triggering_chinese_company_org_names_lower_initial if isinstance(name, str)}


        likely_chinese_company_org_names_lower = {name.lower() for name in likely_chinese_company_org_names}

        # Identify Regulatory Agencies and Sanctions based on ALL accumulated entities
        all_reg_agency_names = {e.get('name','') for e in all_entities_accumulated if isinstance(e, dict) and e.get('name') and e.get('type') == "REGULATORY_AGENCY"}
        all_sanction_names = {e.get('name','') for e in all_entities_accumulated if isinstance(e, dict) and e.get('name') and e.get('type') == "SANCTION"}

        # Also include Regulatory Agencies and Sanctions potentially identified from structured data
        structured_derived_reg_sanc_entities = {e.get('name','') for e in all_entities_accumulated if isinstance(e, dict) and e.get('_source_type') == 'linkup_structured' and e.get('type') in ["REGULATORY_AGENCY", "SANCTION"]}
        all_reg_agency_names.update(structured_derived_reg_sanc_entities) # Just lump them here for now, refinement might need specific type check

        all_reg_agency_names.discard('')
        all_sanction_names.discard('')

        all_reg_agency_names_lower = {name.lower() for name in all_reg_agency_names}
        all_sanction_names_lower = {name.lower() for name in all_sanction_names}
        all_reg_sanc_names_lower = all_reg_agency_names_lower.union(all_sanction_names_lower)


        # --- Placeholder Exposure Generation Logic ---
        # This section takes the accumulated data and identifies specific "exposures"
        # that are high risk and relevant (e.g., involving triggered entities).
        # A real implementation would involve more complex analysis combining risks, relationships, and entity types.
        # The output populates the `generated_exposures` list.

        consolidated_exposures = {} # Reset or ensure this is clear before starting

        # Define the set of entity names that should trigger exposure generation if involved in risks/relationships
        triggering_and_likely_chinese_entities_lower = likely_chinese_company_org_names_lower.union(triggering_chinese_company_org_names_lower_initial)

        print(f"[Step 3.5 GenExp] Triggering/Likely Entities ({len(triggering_and_likely_chinese_entities_lower)}): {list(triggering_and_likely_chinese_entities_lower)[:10]}...")

        # Generate exposures from Risks
        high_severe_risks_for_exposure = [
            r for r in all_risks_accumulated
            if isinstance(r, dict) and r.get('severity') in ["HIGH", "SEVERE"]
        ]
        print(f"[Step 3.5 GenExp] Found {len(high_severe_risks_for_exposure)} high/severe risks.")

        for risk in high_severe_risks_for_exposure:
            risk_desc = risk.get('description', '')
            risk_severity = risk.get('severity', 'N/A')
            risk_category = risk.get('risk_category', 'UNKNOWN')
            sources = risk.get('source_urls', [])
            related_entities = risk.get('related_entities', [])

            for entity_name in related_entities:
                if isinstance(entity_name, str) and entity_name.strip().lower() in triggering_and_likely_chinese_entities_lower:
                     # This entity is a trigger/likely Chinese company with a high risk
                     # Create a basic exposure for this entity
                     exposure_key = (entity_name.strip().lower(), risk_desc.strip().lower(), 'Risk') # Use risk description as part of key

                     if exposure_key not in consolidated_exposures:
                         # Find if this entity is a subsidiary/affiliate or parent from relationships
                         parent_name = "N/A"
                         sub_aff_name = "N/A"
                         # Iterate through ALL relationships to find relevant ownership connections for this entity
                         for rel in all_collected_relationships:
                              if isinstance(rel, dict):
                                   e1 = rel.get('entity1')
                                   e2 = rel.get('entity2')
                                   rel_type = rel.get('relationship_type')

                                   if isinstance(e1, str) and isinstance(e2, str) and isinstance(rel_type, str):
                                         if rel_type.upper() == "SUBSIDIARY_OF" and e1.strip().lower() == entity_name.strip().lower():
                                             parent_name = e2.strip() # e2 is the parent of e1
                                             break # Found parent, move on (simplification: assumes one primary parent)
                                         # Add check if the trigger entity is e2 in a PARENT_COMPANY_OF relationship
                                         if rel_type.upper() == "PARENT_COMPANY_OF" and e2.strip().lower() == entity_name.strip().lower():
                                             # If the trigger entity is e2, then e1 is its parent
                                             parent_name = e1.strip()
                                             break # Found parent


                         # Create the exposure dictionary
                         exposure_dict = {
                             'Entity': entity_name.strip(),
                             'Subsidiary/Affiliate': sub_aff_name, # Need logic to populate this correctly
                             'Parent Company': parent_name,
                             'Risk_Severity': risk_severity,
                             'Risk_Type': f"Risk ({risk_category})", # Indicate it's a risk-based exposure
                             'Explanation': risk_desc.strip(),
                             'Main_Sources': list(set(sources)), # Ensure unique sources
                             '_source_type': 'derived_from_risk'
                         }
                         consolidated_exposures[exposure_key] = exposure_dict
                         # print(f"  [Step 3.5 GenExp] Created risk exposure for '{entity_name}'")


        # Generate exposures from Relationships (e.g., Regulatory/Sanction relationships)
        reg_sanc_relationships_for_exposure = [
            rel for rel in all_collected_relationships
            if isinstance(rel, dict)
            and isinstance(rel.get('entity1'), str)
            and rel.get('entity1').strip().lower() in triggering_and_likely_chinese_entities_lower # Entity1 is the triggering/likely Chinese subject
            and isinstance(rel.get('relationship_type'), str)
            and rel.get('relationship_type').upper() in ["SUBJECT_TO", "ISSUED_BY"] # Relevant relationship types
            and isinstance(rel.get('entity2'), str)
            and rel.get('entity2').strip().lower() in all_reg_sanc_names_lower # Entity2 is a regulator or sanction
        ]
        print(f"[Step 3.5 GenExp] Found {len(reg_sanc_relationships_for_exposure)} relevant regulatory/sanction relationships.")

        for rel in reg_sanc_relationships_for_exposure:
            entity1 = rel.get('entity1', '').strip()
            entity2 = rel.get('entity2', '').strip()
            rel_type = rel.get('relationship_type', '').strip()
            context_urls = rel.get('context_urls', [])

            # Create a basic exposure for this relationship
            exposure_key = (entity1.lower(), f"{rel_type.lower()}_{entity2.lower()}", 'Relationship') # Use relationship details in key

            if exposure_key not in consolidated_exposures:
                 # Find parent/subsidiary info for entity1 (the subject)
                 parent_name = "N/A"
                 sub_aff_name = "N/A"
                 for own_rel in all_collected_relationships:
                      if isinstance(own_rel, dict):
                           e1_own = own_rel.get('entity1')
                           e2_own = own_rel.get('entity2')
                           rel_type_own = own_rel.get('relationship_type')

                           if isinstance(e1_own, str) and isinstance(e2_own, str) and isinstance(rel_type_own, str):
                                if rel_type_own.upper() == "SUBSIDIARY_OF" and e1_own.strip().lower() == entity1.lower():
                                     parent_name = e2_own.strip()
                                     break
                                if rel_type_own.upper() == "PARENT_COMPANY_OF" and e2_own.strip().lower() == entity1.lower():
                                     parent_name = e1_own.strip()
                                     break

                 # Create the exposure dictionary
                 exposure_dict = {
                     'Entity': entity1,
                     'Subsidiary/Affiliate': sub_aff_name, # Need logic to populate this correctly
                     'Parent Company': parent_name,
                     'Risk_Severity': 'HIGH', # Assume regulatory/sanction relationship implies high risk
                     'Risk_Type': f"Regulatory/Sanction ({rel_type.replace('_', ' ').title()} {entity2})", # Describe the relationship
                     'Explanation': f"{entity1} is {rel_type.lower().replace('_', ' ')} {entity2}", # Basic explanation
                     'Main_Sources': list(set(context_urls)), # Ensure unique sources
                     '_source_type': 'derived_from_relationship'
                 }
                 consolidated_exposures[exposure_key] = exposure_dict
                 # print(f"  [Step 3.5 GenExp] Created relationship exposure for '{entity1}'")


        # The list for the sheet needs to be derived from the consolidated dictionary's values
        # This implicitly handles deduplication by key used in consolidated_exposures
        generated_exposures = list(consolidated_exposures.values())
        results["high_risk_exposures"] = generated_exposures # Store generated exposures in results dict
        print(f"[Step 3.5 GenExp] Generated {len(generated_exposures)} unique exposure candidates.")

        # --- End Placeholder Exposure Generation ---


        results["steps"].append({"name": "Generate High Risk Exposures", "duration": round(time.time() - step3_5_start, 2), "generated_exposures_count": len(generated_exposures), "status": "OK" if generated_exposures else "No Exposures Generated" })


        print(f"\n--- Running Step 5: Prepare Data for Sheet & KG ---") # Renamed step to align with process
        step5_prep_start = time.time() # Renamed variable

        # The filtering logic for entities, risks, relationships for sheet/KG now follows,
        # using the sets calculated above (triggering_..., likely_...).

        # Recalculate likely Chinese company/org names based on ALL accumulated entities
        # This logic was already done in Step 3.5 - the sets are available here.
        # Identify Regulatory Agencies and Sanctions to save based on filtering rules for sheet/KG
        # This logic was also done in Step 3.5 - the sets are available here.


        # --- Corrected the variable names causing the NameError ---
        print(f"[Step 5 Prep] Analyzing {len(all_risks_accumulated)} raw risks and {len(all_entities_accumulated)} raw entities for Sheet/KG filtering.")


        # Filter entities for the sheet/KG: ONLY Entities identified as likely Chinese Companies/Organizations,
        # AND ONLY Regulatory Agencies and Sanctions identified as likely Chinese (simplified check).
        entities_to_save_to_sheet = []
        # List of common known non-Chinese regulators/sanctions/orgs/countries for filtering
        common_non_chinese_entities_lower = {
            "sec", "securities and exchange commission", "ofac", "office of foreign assets control",
            "hmrc", "irs", "fbi", "eu commission", "european commission", "us department of justice", "doj",
            "uk hmrc", "uk government", "united states", "us", "united kingdom", "uk",
            "germany", "de", "india", "in", "france", "fr", "japan", "jp", "canada", "ca", "australia", "au",
            "nato", "un", "world bank", "imf", "oecd", "international monetary fund", "world trade organization", "wto", # Added more international orgs
            "google", "apple", "microsoft", "amazon", "facebook", "meta", "twitter", "x", "alibaba", "tencent", "baidu", "huawei", # Add common tech/large company names if they get misclassified
            "shanghai", "beijing", "shenzhen", "guangzhou", "hong kong", "taiwan", # Add cities/regions if misclassified
            "people's republic of china", "prc" # Sometimes PRC itself gets extracted as an entity
        }

        for e in all_entities_accumulated: # Filter from ALL accumulated entities
            if isinstance(e, dict) and e.get('name'):
                 entity_name = e.get('name')
                 entity_name_lower = entity_name.lower().strip()
                 entity_type = e.get('type')

                 # Skip common non-Chinese entities regardless of type if they match the filter list
                 if entity_name_lower in common_non_chinese_entities_lower:
                      # print(f"[Step 5 Prep] Filtering out common non-Chinese entity '{entity_name}'.") # Optional logging
                      continue


                 # Include Companies or Organizations if they are in the likely Chinese list (calculated above)
                 if entity_type in ["COMPANY", "ORGANIZATION", "ORGANIZATION_NON_PROFIT", "GOVERNMENT_BODY"]: # Include related ORG types if they were identified
                      if entity_name_lower in likely_chinese_company_org_names_lower:
                           entities_to_save_to_sheet.append(e)
                       # else: print(f"[Step 5 Prep] Filtering out non-Chinese Company/Org '{entity_name}'.") # Optional logging
                 # Include Regulators or Sanctions IF their name is NOT in the common non-Chinese list (already checked above)
                 # And if they are in the list of all_reg_agency_names or all_sanction_names (i.e., were actually extracted)
                 elif entity_type in ["REGULATORY_AGENCY", "SANCTION"]:
                      if entity_name_lower in all_reg_agency_names_lower or entity_name_lower in all_sanction_names_lower:
                          # Keep any remaining regulator/sanction found that was actually extracted
                          entities_to_save_to_sheet.append(e)
                 # Consider adding other types like PERSON, LOCATION, REGULATION if they are relevant for the KG structure
                 # else: print(f"[Step 5 Prep] Filtering out entity '{entity_name}' with type '{entity_type}' not explicitly allowed for sheet/KG save.")


        # Filter risks: only save risks that are related to at least one entity that will be saved to the sheet/KG
        entity_names_saved_lower = {e.get('name','').lower() for e in entities_to_save_to_sheet if isinstance(e, dict) and e.get('name')}

        risks_to_save_to_sheet = [
            r for r in all_risks_accumulated # Filter from ALL accumulated risks
            if isinstance(r, dict) and r.get('description')
            # Check if *any* related entity for this risk is in the list of entities that will be saved
            and any(isinstance(entity_name, str) and entity_name.strip().lower() in entity_names_saved_lower for entity_name in r.get('related_entities', []))
        ]

        # Filter relationships: Only include relationships where BOTH entities were deemed worthy of saving to the Entity sheet/KG
        relationships_to_save_to_sheet = []

        # Updated allowed relationship types for the Relationships sheet
        allowed_sheet_rel_types = ["PARENT_COMPANY_OF", "SUBSIDIARY_OF", "AFFILIATE_OF", "JOINT_VENTURE_PARTNER", "REGULATED_BY", "ISSUED_BY", "SUBJECT_TO", "MENTIONED_WITH", "ACQUIRED", "RELATED_COMPANY"]

        for rel in all_collected_relationships: # Filter from ALL collected relationships
            if isinstance(rel, dict) and rel.get('entity1') and rel.get('relationship_type') and rel.get('entity2'):
                 e1_name = rel.get('entity1'); e2_name = rel.get('entity2'); r_type_raw = rel.get('relationship_type')
                 if isinstance(r_type_raw, str) and isinstance(e1_name, str) and isinstance(e2_name, str):

                     e1_name_lower = e1_name.strip().lower()
                     e2_name_lower = e2_name.strip().lower()
                     r_type_upper = r_type_raw.upper()

                     # Only include relationships where BOTH entities were deemed worthy of saving to the Entity sheet/KG
                     if e1_name_lower in entity_names_saved_lower and e2_name_lower in entity_names_saved_lower:
                          # Check if the relationship type is allowed for the sheet
                          if r_type_upper in allowed_sheet_rel_types:
                               relationships_to_save_to_sheet.append(rel)
                          # else: print(f"[Step 5 Prep] Filtering out relationship type '{r_type_raw}' not explicitly allowed for sheet/KG save between saved entities.")
                     # else: print(f"[Step 5 Prep] Filtering out relationship {rel} because one or both entities ('{e1_name}', '{e2_name}') were not saved to the entity list.") # Optional logging

        # Filter exposures for the sheet (currently just using all generated exposures, could add more filtering here)
        exposures_to_save_to_sheet = list(generated_exposures) # Save all generated exposures from Step 3.5

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
                           unique_risks_dict[key]['risk_category'] = r.get('risk_category', 'UNKNOWN') # Include category
                      else:
                           # Merge related entities and sources and update category if UNKNOWN
                           existing_entities = unique_risks_dict[key].get('related_entities', [])
                           new_entities = r.get('related_entities', [])
                           merged_entities = list(set(existing_entities + new_entities))
                           unique_risks_dict[key]['related_entities'] = merged_entities

                           existing_urls = unique_risks_dict[key].get('source_urls', [])
                           new_urls = r.get('source_urls', [])
                           merged_urls = list(set(existing_urls + new_urls))
                           unique_risks_dict[key]['source_urls'] = merged_urls
                           # Update category only if existing is UNKNOWN and new is known
                           if unique_risks_dict[key].get('risk_category', 'UNKNOWN') == 'UNKNOWN' and r.get('risk_category') != 'UNKNOWN':
                                unique_risks_dict[key]['risk_category'] = r.get('risk_category')


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

        print(f"[Step 5.5 Summary] Data for summary: E:{len(final_data_for_summary.get('entities',[]))}, R:{len(final_data_for_summary.get('risks',[]))}, Rel:{len(final_data_for_summary.get('relationships',[]))}, Exp:{exposures_for_summary_count}, Structured: {structured_data_present}.")

        # Check if there is ANY data collected before attempting summary
        if final_data_for_summary.get("entities") or final_data_for_summary.get("risks") or final_data_for_summary.get("relationships") or exposures_for_summary_count > 0 or structured_data_present:
             # Pass the full results dictionary here
             # nlp_extraction_available check includes generate_analysis_summary
             if nlp_extraction_available:
                 # Pass LLM config from the run request
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
             for step_data in results.get("steps", []):
                  if isinstance(step_data, dict) and "extracted_data" in step_data:
                       # Ensure extracted_data_counts is added if it wasn't by the error handler
                       if "extracted_data_counts" not in step_data:
                            if isinstance(step_data.get("extracted_data"), dict):
                                  step_data["extracted_data_counts"] = {k: len(v) for k,v in step_data["extracted_data"].items()}
                            else: step_data["extracted_data_counts"] = {}
                       del step_data["extracted_data"] # Remove large data from step log


        else:
             print("ERROR: 'results' variable is not a dictionary during error handling. Cannot log error details to results object.")
             # Attempt to save basic error info to sheet if possible
             try:
                 if google_sheets_available:
                      exc_type, exc_value, exc_traceback_obj = sys.exc_info()
                      basic_error_results = {
                         "query": initial_query,
                         "run_duration_seconds": round(time.time() - start_run_time, 2),
                         "kg_update_status": "error",
                         "error": f"Critical Orchestrator Failure: {type(exc_value).__name__}: {exc_value}",
                         "analysis_summary": "Analysis failed due to critical internal error."
                     }
                      # Pass empty lists as no data was filtered/prepared
                      # Pass LLM config used in the run for potential error logging in sheet save
                      # These lists are initialized before the try, so they are available, even if empty
                      _save_analysis_to_gsheet(basic_error_results, entities_to_save_to_sheet, risks_to_save_to_sheet, relationships_to_save_to_sheet, exposures_to_save_to_sheet, llm_provider_to_use, llm_model_to_use)
             except Exception as save_e:
                  print(f"CRITICAL ERROR: Failed to save even basic error info to sheets: {save_e}")
                  traceback.print_exc()


    finally:
        # Ensure KG driver is closed if it was successfully obtained and is not None
        if kg_driver_available and kg_driver is not None:
            try:
                 knowledge_graph.close_driver()
            except Exception as close_e:
                 print(f"Error closing KG driver in finally block: {close_e}")
                 traceback.print_exc()

        # Update final duration and attempt to save results to Google Sheets
        if isinstance(results, dict):
             results["run_duration_seconds"] = round(time.time() - start_run_time, 2)

             # Remove large data structures from final results to avoid issues with JSON serialization if not needed
             # Example: Remove raw search results or redundant data from step logs
             for step_data in results.get("steps", []):
                  if isinstance(step_data, dict) and "extracted_data" in step_data:
                       # Ensure extracted_data_counts is added if it wasn't by the error handler
                       if "extracted_data_counts" not in step_data:
                            if isinstance(step_data.get("extracted_data"), dict):
                                  step_data["extracted_data_counts"] = {k: len(v) for k,v in step_data["extracted_data"].items()}
                            else: step_data["extracted_data_counts"] = {}
                       del step_data["extracted_data"] # Remove large list of data

             # Pass the filtered data (now defined before the try block and populated within) to the save function
             # Pass LLM config used in the run for potential error logging in sheet save
             if google_sheets_available:
                 # These lists are initialized before the try block, so they are always available
                 _save_analysis_to_gsheet(
                     results,
                     entities_to_save_to_sheet,
                     risks_to_save_to_sheet,
                     relationships_to_save_to_sheet,
                     exposures_to_save_to_sheet, # This list is populated in Step 3.5/5
                     llm_provider_to_use, # Pass LLM config
                     llm_model_to_use # Pass LLM config
                 )
             else: print("Skipping save to Google Sheets: Configuration missing or invalid.")
        else:
             print("ERROR: 'results' variable is not a dictionary in the finally block. Cannot update final metrics or save to sheets.")

        print(f"\n--- Analysis Complete ({results.get('run_duration_seconds', 'N/A')}s) ---")
        if isinstance(results, dict) and results.get("error"): print(f"--- Run finished with ERROR: {results['error']} ---")
        elif not isinstance(results, dict): print("--- Run finished with CRITICAL ERROR (results variable corrupted) ---")
        else: print(f"--- Run finished successfully (KG Status: {results.get('kg_update_status','?')}, Exposures: {len(results.get('high_risk_exposures', []))}) ---")


    return results

# Check nlp_processor availability at module level for helper functions
# This block seems redundant as it's done near the top, but keeping it for now.
# Let's remove the redundancy and rely on the check at the top.


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
             # Optionally truncate or remove details if the list is very long
             # printable_results['linkup_structured_data_sample'] = printable_results['linkup_structured_data'][:3]
             del printable_results['linkup_structured_data']
        if 'wayback_results' in printable_results:
             printable_results['wayback_results_count'] = len(printable_results['wayback_results'])
             # Optionally truncate or remove details if the list is very long
             # printable_results['wayback_results_sample'] = printable_results['wayback_results'][:3]
             del printable_results['wayback_results']
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