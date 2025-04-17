# orchestrator.py

import time
from typing import Dict, List, Any
import traceback
import json
from datetime import datetime

# Google Sheets Imports
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    import os # To read env var for credentials
    google_sheets_available = True
except ImportError:
    print("Warning: Google API libraries not installed (`google-api-python-client google-auth-httplib2 google-auth-oauthlib`). Saving to Google Sheets disabled.")
    google_sheets_available = False


# Import your custom modules
import search_engines
import nlp_processor # Updated NLP processor
import knowledge_graph
import config # Now used for Search/DB/Sheets config and LLM keys

# --- Google Sheets Configuration & Functions ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SHEET_ID = config.GOOGLE_SHEET_ID
SERVICE_ACCOUNT_INFO = None
if google_sheets_available and config.GCP_SERVICE_ACCOUNT_JSON_STR:
    try:
        SERVICE_ACCOUNT_INFO = json.loads(config.GCP_SERVICE_ACCOUNT_JSON_STR)
    except Exception as e:
        print(f"ERROR parsing GCP JSON environment variable: {e}")
else:
     if google_sheets_available: # Only warn if library is installed but config is missing
        print("Warning: GCP_SERVICE_ACCOUNT_JSON_STR environment variable not set or empty.")

SHEET_NAME_RUNS = 'Runs'
SHEET_NAME_ENTITIES = 'Entities'
SHEET_NAME_RISKS = 'Risks'
SHEET_NAME_RELATIONSHIPS = 'Relationships'

gsheet_service = None # Global variable for the service

def _get_gsheet_service():
    """Authenticates and builds the Google Sheets service object (singleton)."""
    global gsheet_service
    if not google_sheets_available: return None # Skip if library missing
    if gsheet_service is None: # Only initialize if not already done
        creds = None
        if not SERVICE_ACCOUNT_INFO:
            print("Cannot authenticate to Google Sheets: Service account info missing.")
            return None
        try:
            creds = service_account.Credentials.from_service_account_info(
                SERVICE_ACCOUNT_INFO, scopes=SCOPES)
            # Use cache_discovery=False for serverless/ephemeral environments like Render free tier
            gsheet_service = build('sheets', 'v4', credentials=creds, cache_discovery=False)
            print("Google Sheets service authenticated successfully.")
        except Exception as e:
            print(f"ERROR: Failed to authenticate/build Google Sheets service: {e}")
            gsheet_service = None # Ensure it's None on failure
    return gsheet_service

def _append_to_gsheet(service, sheet_name: str, values: List[List[Any]]):
    """Appends rows of values to a specific sheet."""
    if not service or not SHEET_ID:
        print(f"Skipping append to {sheet_name}: Missing GSheet service or Sheet ID.")
        return False
    if not values: # Don't try to append empty data
        print(f"Skipping append to {sheet_name}: No data provided.")
        return True # Consider it "successful" if there was nothing to write

    try:
        body = {'values': values}
        result = service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range=f"{sheet_name}!A1",
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body=body
        ).execute()
        print(f"Successfully appended {result.get('updates', {}).get('updatedRows', 0)} rows to sheet '{sheet_name}'.")
        return True
    except Exception as e:
        print(f"ERROR appending data to Google Sheet '{sheet_name}': {e}")
        # Optionally log the data that failed to append for debugging
        # print(f"Failed data: {values}")
        return False

def _save_analysis_to_gsheet(run_results: Dict):
    """Saves the analysis results to different tabs in Google Sheets."""
    print("Attempting to save analysis results to Google Sheets...")
    service = _get_gsheet_service() # Get or initialize service
    if not service:
        print("Aborting save to Google Sheets due to authentication/initialization failure.")
        return

    run_timestamp = datetime.now().isoformat()

    # --- 1. Save Run Summary ---
    run_summary_row = [
        run_timestamp,
        run_results.get('query'),
        run_results.get('run_duration_seconds'),
        run_results.get('kg_update_status'),
        run_results.get('error')
    ]
    _append_to_gsheet(service, SHEET_NAME_RUNS, [run_summary_row])

    # --- 2. Save Extracted Data ---
    extracted = run_results.get("final_extracted_data", {})

    # Save Entities
    entities = extracted.get("entities", [])
    if entities:
        entity_rows = [
            [run_timestamp, e.get('name'), e.get('type'), json.dumps(e.get('mentions', []))]
            for e in entities if e.get('name')
        ]
        if entity_rows: _append_to_gsheet(service, SHEET_NAME_ENTITIES, entity_rows)

    # Save Risks
    risks = extracted.get("risks", [])
    if risks:
        risk_rows = [
            [run_timestamp, r.get('description'), r.get('severity'),
             json.dumps(r.get('related_entities', [])), json.dumps(r.get('source_urls', []))]
            for r in risks if r.get('description')
        ]
        if risk_rows: _append_to_gsheet(service, SHEET_NAME_RISKS, risk_rows)

    # Save Relationships
    relationships = extracted.get("relationships", [])
    if relationships:
        rel_rows = [
            [run_timestamp, rel.get('entity1'), rel.get('relationship_type'), rel.get('entity2'), # Using corrected keys
             json.dumps(rel.get('context_urls', []))]
            for rel in relationships if rel.get('entity1') and rel.get('relationship_type') and rel.get('entity2')
        ]
        if rel_rows: _append_to_gsheet(service, SHEET_NAME_RELATIONSHIPS, rel_rows)

    print("Google Sheets save process finished.")


# --- Main Orchestration Logic ---
def run_analysis(initial_query: str,
                 llm_provider: str,
                 llm_model: str,
                 global_search_context: str = "global financial news and legal filings",
                 specific_search_context: str = "Baidu search in China for specific company supply chain info",
                 specific_country_code: str = 'cn',
                 max_global_results: int = 10,
                 max_specific_results: int = 10
                 ) -> Dict[str, Any]:
    """
    Main orchestration function for the AI Analyst Agent.
    Coordinates calls, includes enhanced error logging, and saves results to Google Sheets.
    """
    start_run_time = time.time()
    results = { # Initialize results structure
        "query": initial_query, "steps": [],
        "llm_used": f"{llm_provider} ({llm_model})",
        "final_extracted_data": {"entities": [], "risks": [], "relationships": []},
        "wayback_results": [], "kg_update_status": "not_run",
        "run_duration_seconds": 0, "error": None,
    }

    # Initialize Neo4j driver
    kg_driver_available = bool(knowledge_graph.get_driver())
    if not kg_driver_available:
         print("WARNING: Failed to initialize Neo4j driver. KG updates will be skipped.")
         results["kg_update_status"] = "skipped_no_connection"

    # Basic check if essential LLM params are missing (key checked within nlp_processor now)
    if not llm_provider or not llm_model:
        results["error"] = "Missing LLM configuration (provider or model name)."
        print(f"ERROR: {results['error']}")
        results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
        # Try saving error state to GSheet before returning
        if config.GOOGLE_SHEET_ID and SERVICE_ACCOUNT_INFO: # Check if GSheet saving might be possible
            _save_analysis_to_gsheet(results)
        knowledge_graph.close_driver() # Ensure driver is closed
        return results

    try:
        # --- Step 1: Global Insights ---
        print(f"\n--- Running Step 1: Global Search ({initial_query}) ---")
        step1_start = time.time()
        global_raw_results = []; global_search_source = "unknown"
        if config.GOOGLE_API_KEY_SEARCH and config.GOOGLE_CX: global_raw_results = search_engines.search_google_official(initial_query, lang='en', num=max_global_results); global_search_source = 'google_api'
        if not global_raw_results and config.SERPAPI_KEY: print("Trying global search via SerpApi (Google)..."); global_raw_results = search_engines.search_via_serpapi(initial_query, engine='google', country_code='us', lang_code='en', num=max_global_results); global_search_source = 'serpapi_google'
        elif not global_raw_results: print("Warning: No suitable global search engine configured or search failed.")
        global_search_results = [search_engines.standardize_result(r, global_search_source) for r in global_raw_results]
        print(f"Found {len(global_search_results)} global results.")

        # Call NLP Processor (key read internally)
        global_extracted_data = nlp_processor.extract_data_from_results(
            global_search_results, global_search_context,
            llm_provider, llm_model
        )
        print(f"Extracted {len(global_extracted_data.get('entities',[]))} entities, {len(global_extracted_data.get('risks',[]))} risks globally.")
        results["steps"].append({"name": "Global Search & Extraction", "duration": round(time.time() - step1_start, 2),"search_results_count": len(global_search_results), "extracted_data": global_extracted_data, "status": "OK"})
        results["final_extracted_data"]["entities"].extend(global_extracted_data.get('entities',[]))
        results["final_extracted_data"]["risks"].extend(global_extracted_data.get('risks',[]))
        results["final_extracted_data"]["relationships"].extend(global_extracted_data.get('relationships',[]))


        # --- Step 2: Translate Keywords ---
        print(f"\n--- Running Step 2: Translating Keywords ---")
        step2_start = time.time()
        # Call NLP Processor (key read internally)
        translated_keywords = nlp_processor.translate_keywords_for_context(
            initial_query, specific_search_context,
            llm_provider, llm_model
        )
        specific_query = translated_keywords[0] if translated_keywords else initial_query
        print(f"Using specific query: '{specific_query}'")
        results["steps"].append({"name": "Keyword Translation", "duration": round(time.time() - step2_start, 2), "translated_keywords": translated_keywords, "status": "OK"})


        # --- Step 3: Specific Country Data ---
        print(f"\n--- Running Step 3: Specific Search ({specific_query} in {specific_country_code}) ---")
        step3_start = time.time(); specific_raw_results = []; specific_search_source = 'baidu'
        if config.SERPAPI_KEY: specific_raw_results = search_engines.search_via_serpapi(specific_query, engine='baidu', country_code=specific_country_code, num=max_specific_results)
        else: print("Warning: SerpApi key missing, skipping Baidu search.")
        specific_search_results = [search_engines.standardize_result(r, specific_search_source) for r in specific_raw_results]
        print(f"Found {len(specific_search_results)} specific results.")

        # Call NLP Processor (key read internally)
        specific_extracted_data = nlp_processor.extract_data_from_results(
            specific_search_results, specific_search_context,
            llm_provider, llm_model
        )
        print(f"Extracted {len(specific_extracted_data.get('entities',[]))} entities, {len(specific_extracted_data.get('risks',[]))} risks specifically.")
        results["steps"].append({"name": "Specific Search & Extraction", "duration": round(time.time() - step3_start, 2),"search_results_count": len(specific_search_results), "extracted_data": specific_extracted_data, "status": "OK"})
        results["final_extracted_data"]["entities"].extend(specific_extracted_data.get('entities',[]))
        results["final_extracted_data"]["risks"].extend(specific_extracted_data.get('risks',[]))
        results["final_extracted_data"]["relationships"].extend(specific_extracted_data.get('relationships',[]))


        # --- Step 4: Wayback Machine ---
        print(f"\n--- Running Step 4: Wayback Machine Check ---")
        step4_start = time.time(); all_urls = list(set([r.get('url') for r in global_search_results + specific_search_results if r.get('url')])); urls_to_check = all_urls[:5]; wayback_checks = []
        print(f"Checking {len(urls_to_check)} URLs...");
        for url in urls_to_check: wayback_checks.append(search_engines.check_wayback_machine(url)); time.sleep(0.5)
        results["wayback_results"] = wayback_checks; results["steps"].append({"name": "Wayback Machine Check", "duration": round(time.time() - step4_start, 2),"urls_checked": len(urls_to_check), "status": "OK"})


        # --- Step 5: Update Knowledge Graph ---
        print(f"\n--- Running Step 5: Update Knowledge Graph ---")
        step5_start = time.time(); update_success = False; kg_status_message = results["kg_update_status"]
        if kg_driver_available:
            unique_entities = list({f"{e.get('name', '')}_{e.get('type', '')}".lower(): e for e in results["final_extracted_data"]["entities"] if e.get('name')}.values())
            unique_relationships = list({f"{r.get('entity1', '')}_{r.get('relationship_type', '')}_{r.get('entity2', '')}".lower(): r for r in results["final_extracted_data"]["relationships"] if r.get('entity1') and r.get('relationship_type') and r.get('entity2')}.values())
            unique_risks = list({r.get('description', ''): r for r in results["final_extracted_data"]["risks"] if r.get('description')}.values())
            deduped_data = {"entities": unique_entities, "risks": unique_risks, "relationships": unique_relationships}
            if not any([unique_entities, unique_risks, unique_relationships]): print("No unique data extracted to update KG."); kg_status_message = "skipped_no_data"
            else: print(f"Attempting KG update..."); update_success = knowledge_graph.update_knowledge_graph(deduped_data); kg_status_message = "success" if update_success else "error"
        else: print("Skipping KG update: driver not available.")
        results["kg_update_status"] = kg_status_message; results["steps"].append({"name": "Knowledge Graph Update", "duration": round(time.time() - step5_start, 2), "status": results["kg_update_status"]})


        # --- Step 6: Learning / Adapting (Placeholder) ---
        print("\n--- Step 6: Learning/Adapting (Deferred) ---")
        results["steps"].append({"name": "Learning/Adapting", "status": "deferred"})

    # --- Error Handling ---
    except Exception as e:
        print(f"\n--- Orchestration Error ---")
        error_type = type(e).__name__; error_msg = str(e) if str(e) else "No error message provided."; error_traceback = traceback.format_exc()
        print(f"Error Type: {error_type}\nError Message: {error_msg}\nTraceback:\n{error_traceback}")
        results["error"] = f"{error_type}: {error_msg}"

    finally:
        knowledge_graph.close_driver() # Close Neo4j
        results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
        # --- Save results to Google Sheet ---
        # Use SERVICE_ACCOUNT_INFO existence check which depends on the env var being set AND parsed
        if config.GOOGLE_SHEET_ID and SERVICE_ACCOUNT_INFO:
            _save_analysis_to_gsheet(results) # Call the save function
        else:
             print("Skipping save to Google Sheets: Configuration missing or invalid.")
        # --- End Save ---
        print(f"\n--- Analysis Complete ({results['run_duration_seconds']}s) ---")

    return results # Return the results dictionary