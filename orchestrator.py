# orchestrator.py

import time
from typing import Dict, List, Any
import traceback
import json
from datetime import datetime

# Google Sheets Imports (wrapped in try/except)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    import os # To read env var for credentials
    google_sheets_available = True
except ImportError:
    print("Warning: Google API libraries not installed (`google-api-python-client google-auth-httplib2 google-auth-oauthlib`). Saving to Google Sheets disabled.")
    google_sheets_available = False
    service_account = None # Define for checks later
    build = None # Define for checks later

# Import your custom modules
import search_engines
import nlp_processor # Assumes this uses the dynamic LLM approach
import knowledge_graph
import config # Used for Search/DB/Sheets config and LLM keys

# --- Google Sheets Configuration & Functions ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SHEET_ID = config.GOOGLE_SHEET_ID
SERVICE_ACCOUNT_INFO = None
GCP_JSON_STR = config.GCP_SERVICE_ACCOUNT_JSON_STR

if google_sheets_available and GCP_JSON_STR:
    try:
        SERVICE_ACCOUNT_INFO = json.loads(GCP_JSON_STR)
        # print("Successfully parsed GCP Service Account JSON from environment.") # Optional: uncomment for debug
    except Exception as e:
        print(f"ERROR parsing GCP JSON environment variable: {e}")
        SERVICE_ACCOUNT_INFO = None
elif google_sheets_available:
    # Only warn if library installed but config missing
    if not GCP_JSON_STR:
        print("Warning: GCP_SERVICE_ACCOUNT_JSON_STR environment variable not set or empty.")

# Define Sheet Names (must match your tabs)
SHEET_NAME_RUNS = 'Runs'
SHEET_NAME_ENTITIES = 'Entities'
SHEET_NAME_RISKS = 'Risks'
SHEET_NAME_RELATIONSHIPS = 'Relationships'
SHEET_NAME_EXPOSURES = 'Supply Chain Exposures' # New Sheet Name

gsheet_service = None # Global variable for the service

def _get_gsheet_service():
    """Authenticates and builds the Google Sheets service object (singleton)."""
    global gsheet_service
    if not google_sheets_available: return None
    if gsheet_service is None:
        if not SERVICE_ACCOUNT_INFO:
            print("Cannot authenticate to Google Sheets: Service account info missing or invalid.")
            return None
        try:
            creds = service_account.Credentials.from_service_account_info( SERVICE_ACCOUNT_INFO, scopes=SCOPES)
            gsheet_service = build('sheets', 'v4', credentials=creds, cache_discovery=False)
            print("Google Sheets service authenticated successfully.")
        except Exception as e:
            print(f"ERROR: Failed to authenticate/build Google Sheets service: {e}")
            gsheet_service = None
    return gsheet_service

def _append_to_gsheet(service, sheet_name: str, values: List[List[Any]]):
    """Appends rows of values to a specific sheet."""
    if not service or not SHEET_ID:
        print(f"Skipping append to {sheet_name}: Missing GSheet service or Sheet ID.")
        return False
    if not values: return True # Nothing to write is not an error

    try:
        body = {'values': values}
        result = service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID, range=f"{sheet_name}!A1",
            valueInputOption='USER_ENTERED', insertDataOption='INSERT_ROWS', body=body
        ).execute()
        updated_rows = result.get('updates', {}).get('updatedRows', 0)
        if updated_rows > 0: # Only log if rows were actually added
             print(f"Successfully appended {updated_rows} rows to sheet '{sheet_name}'.")
        return True
    except Exception as e:
        print(f"ERROR appending data to Google Sheet '{sheet_name}': {e}")
        return False

def _save_analysis_to_gsheet(run_results: Dict):
    """Saves the analysis results, including supply chain exposures, to Google Sheets."""
    print("Attempting to save analysis results to Google Sheets...")
    service = _get_gsheet_service()
    if not service: print("Aborting save: GSheet service unavailable."); return

    run_timestamp = datetime.now().isoformat()

    # --- 1. Save Run Summary ---
    run_summary_row = [ run_timestamp, run_results.get('query'), run_results.get('run_duration_seconds'), run_results.get('kg_update_status'), run_results.get('error') ]
    _append_to_gsheet(service, SHEET_NAME_RUNS, [run_summary_row])

    # --- 2. Save Extracted Data (Entities, Risks, Relationships) ---
    extracted = run_results.get("final_extracted_data", {})
    entities = extracted.get("entities", []); risks = extracted.get("risks", []); relationships = extracted.get("relationships", [])
    if entities: entity_rows = [[run_timestamp, e.get('name'), e.get('type'), json.dumps(e.get('mentions', []))] for e in entities if isinstance(e, dict) and e.get('name')]; _append_to_gsheet(service, SHEET_NAME_ENTITIES, entity_rows)
    if risks: risk_rows = [[run_timestamp, r.get('description'), r.get('severity'), json.dumps(r.get('related_entities', [])), json.dumps(r.get('source_urls', []))] for r in risks if isinstance(r, dict) and r.get('description')]; _append_to_gsheet(service, SHEET_NAME_RISKS, risk_rows)
    if relationships: rel_rows = [[run_timestamp, rel.get('entity1'), rel.get('relationship_type'), rel.get('entity2'), json.dumps(rel.get('context_urls', []))] for rel in relationships if isinstance(rel, dict) and rel.get('entity1') and rel.get('relationship_type') and rel.get('entity2')]; _append_to_gsheet(service, SHEET_NAME_RELATIONSHIPS, rel_rows)

    # --- 3. Save Supply Chain Exposures ---
    exposures = run_results.get("supply_chain_exposures", [])
    if exposures:
        exposure_rows = [
            [   run_timestamp,
                exp.get('problematic_sub_affiliate'), # (a) Name
                exp.get('relationship_type'),         # (b) Description (SUBSIDIARY/AFFILIATE)
                exp.get('parent_company'),            # (c) Name
                exp.get('reason_involved'),           # (d) Summary why sub/aff involved
                exp.get('risk_source_url'),           # (e) Link to relevant risk article
                exp.get('ownership_source_url')       # (f) Link to ownership evidence doc
            ]
            for exp in exposures if isinstance(exp, dict) and exp.get('problematic_sub_affiliate')
        ]
        if exposure_rows:
            _append_to_gsheet(service, SHEET_NAME_EXPOSURES, exposure_rows)

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
    Coordinates calls, includes enhanced error logging, handles supply chain checks,
    and saves results to Google Sheets.
    Accepts LLM provider/model, reads key from config via nlp_processor.
    """
    start_run_time = time.time()
    results = {
        "query": initial_query, "steps": [],
        "llm_used": f"{llm_provider} ({llm_model})",
        "final_extracted_data": {"entities": [], "risks": [], "relationships": []},
        "supply_chain_exposures": [], # Initialize new key
        "wayback_results": [], "kg_update_status": "not_run",
        "run_duration_seconds": 0, "error": None,
    }

    # Initialize Neo4j driver
    kg_driver_available = bool(knowledge_graph.get_driver())
    if not kg_driver_available:
         print("WARNING: Failed to initialize Neo4j driver. KG updates will be skipped.")
         results["kg_update_status"] = "skipped_no_connection"

    # Basic check if essential LLM params are missing
    if not llm_provider or not llm_model:
        results["error"] = "Missing LLM configuration (provider or model name)."
        print(f"ERROR: {results['error']}")
        results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
        # Try saving error state to GSheet before returning
        if config.GOOGLE_SHEET_ID and SERVICE_ACCOUNT_INFO:
            _save_analysis_to_gsheet(results)
        knowledge_graph.close_driver()
        return results

    try:
        # --- Step 1: Global Insights ---
        print(f"\n--- Running Step 1: Global Search ({initial_query}) ---")
        step1_start = time.time()
        global_raw_results = []; global_search_source = "unknown"
        if config.GOOGLE_API_KEY_SEARCH and config.GOOGLE_CX:
             global_raw_results = search_engines.search_google_official(initial_query, lang='en', num=max_global_results); global_search_source = 'google_api'
        if not global_raw_results and config.SERPAPI_KEY:
             print("Trying global search via SerpApi (Google)..."); global_raw_results = search_engines.search_via_serpapi(initial_query, engine='google', country_code='us', lang_code='en', num=max_global_results); global_search_source = 'serpapi_google'
        elif not global_raw_results: print("Warning: No suitable global search engine configured or search failed.")
        global_search_results = [res for res in (search_engines.standardize_result(r, global_search_source) for r in global_raw_results) if res is not None]
        print(f"Found {len(global_search_results)} standardized global results.")

        global_extracted_data = nlp_processor.extract_data_from_results(global_search_results, global_search_context, llm_provider, llm_model)
        print(f"Extracted {len(global_extracted_data.get('entities',[]))} entities, {len(global_extracted_data.get('risks',[]))} risks globally.")
        results["steps"].append({"name": "Global Search & Extraction", "duration": round(time.time() - step1_start, 2),"search_results_count": len(global_search_results), "extracted_data": global_extracted_data, "status": "OK"})
        results["final_extracted_data"]["entities"].extend(global_extracted_data.get('entities',[]))
        results["final_extracted_data"]["risks"].extend(global_extracted_data.get('risks',[]))
        results["final_extracted_data"]["relationships"].extend(global_extracted_data.get('relationships',[]))

        # --- Step 2: Translate Keywords ---
        print(f"\n--- Running Step 2: Translating Keywords ---")
        step2_start = time.time()
        translated_keywords = nlp_processor.translate_keywords_for_context(initial_query, specific_search_context, llm_provider, llm_model)
        specific_query = translated_keywords[0] if translated_keywords else initial_query
        print(f"Using specific query: '{specific_query}'")
        results["steps"].append({"name": "Keyword Translation", "duration": round(time.time() - step2_start, 2), "translated_keywords": translated_keywords, "status": "OK"})

        # --- Step 3: Specific Country Data ---
        print(f"\n--- Running Step 3: Specific Search ({specific_query} in {specific_country_code}) ---")
        step3_start = time.time(); specific_raw_results = []; specific_search_source = 'baidu'
        if config.SERPAPI_KEY: specific_raw_results = search_engines.search_via_serpapi(specific_query, engine='baidu', country_code=specific_country_code, num=max_specific_results)
        else: print("Warning: SerpApi key missing, skipping Baidu search.")
        specific_search_results = [res for res in (search_engines.standardize_result(r, specific_search_source) for r in specific_raw_results) if res is not None]
        print(f"Found {len(specific_search_results)} standardized specific results.")

        specific_extracted_data = nlp_processor.extract_data_from_results(specific_search_results, specific_search_context, llm_provider, llm_model)
        print(f"Extracted {len(specific_extracted_data.get('entities',[]))} entities, {len(specific_extracted_data.get('risks',[]))} risks specifically.")
        results["steps"].append({"name": "Specific Search & Extraction", "duration": round(time.time() - step3_start, 2),"search_results_count": len(specific_search_results), "extracted_data": specific_extracted_data, "status": "OK"})
        results["final_extracted_data"]["entities"].extend(specific_extracted_data.get('entities',[]))
        results["final_extracted_data"]["risks"].extend(specific_extracted_data.get('risks',[]))
        results["final_extracted_data"]["relationships"].extend(specific_extracted_data.get('relationships',[]))

        # ===> Step 3.5: Identify Supply Chain Exposures <===
        print(f"\n--- Running Step 3.5: Identifying Supply Chain Exposures ---")
        step3_5_start = time.time()
        identified_exposures = []
        high_risk_parent_entities = {} # Store parent_name -> risk_info dict

        # 1. Identify entities associated with medium or high risks from *all* results
        all_risks = results["final_extracted_data"].get("risks", [])
        for risk in all_risks:
            severity = risk.get("severity", "LOW")
            if severity in ["HIGH", "MEDIUM"]:
                for entity_name in risk.get("related_entities", []):
                    if entity_name and entity_name not in high_risk_parent_entities: # Store first encountered high/medium risk per entity
                        high_risk_parent_entities[entity_name] = {
                            "description": risk.get('description', 'N/A'),
                            "source_urls": risk.get('source_urls', [])
                        }

        print(f"Found {len(high_risk_parent_entities)} unique entities involved in medium/high risks: {list(high_risk_parent_entities.keys())}")

        # 2. Get all unique entity names extracted in this run
        all_entity_names_in_run = {e.get('name') for e in results["final_extracted_data"].get("entities", []) if e.get('name')}

        # 3. For each high-risk entity, check its relationship to *other* entities from the run
        for parent_name, risk_info in high_risk_parent_entities.items():
            potential_related_names = all_entity_names_in_run - {parent_name}
            if not potential_related_names: continue

            print(f"\nAnalyzing potential ownership for high-risk entity: {parent_name}")
            # Search for docs related to the potential parent
            ownership_docs = search_engines.search_for_ownership_docs(parent_name, num_per_query=2)
            if not ownership_docs: print(f"No potential ownership documents found for {parent_name}."); continue

            # Check relationship between parent and each potential sub/affiliate
            for related_name in potential_related_names:
                 print(f"---> Checking ownership: '{parent_name}' owning '{related_name}'?")
                 ownership_details = nlp_processor.extract_ownership_relationships(
                     parent_entity_name=parent_name,
                     related_entity_name=related_name,
                     text_snippets=ownership_docs, # Use snippets from targeted search
                     llm_provider=llm_provider,
                     llm_model=llm_model
                 )

                 if ownership_details and ownership_details.get("relationship_type") in ["SUBSIDIARY", "AFFILIATE"]:
                      print(f"  ==> Found Exposure: {parent_name} -> {ownership_details['relationship_type']} -> {related_name}")
                      exposure_record = {
                          "problematic_sub_affiliate": related_name, # (a) Name
                          "relationship_type": ownership_details["relationship_type"], # (b) Description
                          "parent_company": parent_name, # (c) Name
                          "reason_involved": f"Parent ({parent_name}) linked to risk: {risk_info['description'][:150]}...", # (d) Summary
                          "risk_source_url": risk_info['source_urls'][0] if risk_info['source_urls'] else "N/A", # (e) Link to parent risk article
                          "ownership_source_url": ownership_details.get("source_url", "Not Stated") # (f) Link to ownership doc
                      }
                      identified_exposures.append(exposure_record)
                 elif ownership_details: print(f"  Relationship is {ownership_details['relationship_type']}") # Log if unrelated/other
                 else: print(f"  Could not determine ownership relationship.")
                 time.sleep(0.5) # Small delay between LLM calls per pair

        results["supply_chain_exposures"] = identified_exposures
        results["steps"].append({
            "name": "Supply Chain Exposure Analysis", "duration": round(time.time() - step3_5_start, 2),
            "exposures_found": len(identified_exposures), "status": "OK"
        })
        # ===> END Step 3.5 <===


        # --- Step 4: Wayback Machine ---
        print(f"\n--- Running Step 4: Wayback Machine Check ---")
        step4_start = time.time(); all_urls = list(set([r.get('url') for r in global_search_results + specific_search_results if r and r.get('url')])); urls_to_check = all_urls[:5]; wayback_checks = []
        print(f"Checking {len(urls_to_check)} URLs...");
        for url in urls_to_check: wayback_checks.append(search_engines.check_wayback_machine(url)); time.sleep(0.5)
        results["wayback_results"] = wayback_checks; results["steps"].append({"name": "Wayback Machine Check", "duration": round(time.time() - step4_start, 2),"urls_checked": len(urls_to_check), "status": "OK"})


        # --- Step 5: Update Knowledge Graph ---
        print(f"\n--- Running Step 5: Update Knowledge Graph ---")
        step5_start = time.time(); update_success = False; kg_status_message = results["kg_update_status"]
        if kg_driver_available:
            final_data = results.get("final_extracted_data", {})
            entities_list = final_data.get("entities", []); relationships_list = final_data.get("relationships", []); risks_list = final_data.get("risks", [])
            unique_entities = list({ f"{e.get('name', '')}_{e.get('type', '')}".lower(): e for e in entities_list if isinstance(e, dict) and e.get('name') }.values())
            unique_relationships = list({ f"{r.get('entity1', '')}_{r.get('relationship_type', '')}_{r.get('entity2', '')}".lower(): r for r in relationships_list if isinstance(r, dict) and r.get('entity1') and r.get('relationship_type') and r.get('entity2') }.values())
            unique_risks = list({ r.get('description', ''): r for r in risks_list if isinstance(r, dict) and r.get('description') }.values())
            deduped_data = {"entities": unique_entities, "risks": unique_risks, "relationships": unique_relationships}

            if not any([unique_entities, unique_risks, unique_relationships]): print("No unique data extracted to update knowledge graph."); kg_status_message = "skipped_no_data"
            else: print(f"Attempting KG update..."); update_success = knowledge_graph.update_knowledge_graph(deduped_data); kg_status_message = "success" if update_success else "error"
        else: print("Skipping KG update: driver not available.")
        results["kg_update_status"] = kg_status_message; results["steps"].append({"name": "Knowledge Graph Update", "duration": round(time.time() - step5_start, 2), "status": results["kg_update_status"]})


        # --- Step 6: Learning / Adapting (Placeholder) ---
        print("\n--- Step 6: Learning/Adapting (Deferred) ---")
        results["steps"].append({"name": "Learning/Adapting", "status": "deferred"})

    # --- Error Handling ---
    except Exception as e:
        print(f"\n--- Orchestration Error ---"); error_type = type(e).__name__; error_msg = str(e) if str(e) else "No msg"; error_traceback = traceback.format_exc()
        print(f"Type: {error_type}\nMsg: {error_msg}\nTraceback:\n{error_traceback}"); results["error"] = f"{error_type}: {error_msg}"

    finally:
        # Close Neo4j driver
        knowledge_graph.close_driver()
        results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
        # --- Save results to Google Sheet ---
        # Use SERVICE_ACCOUNT_INFO check to see if auth might work
        if config.GOOGLE_SHEET_ID and SERVICE_ACCOUNT_INFO:
            _save_analysis_to_gsheet(results) # Call the save function
        else:
             print("Skipping save to Google Sheets: Configuration missing or invalid.")
        print(f"\n--- Analysis Complete ({results['run_duration_seconds']}s) ---")

    return results # Return the results dictionary