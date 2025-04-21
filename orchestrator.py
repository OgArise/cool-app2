# orchestrator.py

import time
from typing import Dict, List, Any
import traceback
import json
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    zoneinfo_available = True
except ImportError:
    from datetime import timezone # Fallback
    zoneinfo_available = False
    print("Warning: zoneinfo not available. Timestamps will be UTC.")


# Google Sheets Imports
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    import os
    google_sheets_available = True
except ImportError:
    print("Warning: Google API libraries not installed. Saving to GSheets disabled.")
    google_sheets_available = False; service_account = None; build = None


# Custom Modules
import search_engines
import nlp_processor
import knowledge_graph
import config

# --- Google Sheets Config & Functions ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SHEET_ID = config.GOOGLE_SHEET_ID
SERVICE_ACCOUNT_INFO = None
GCP_JSON_STR = config.GCP_SERVICE_ACCOUNT_JSON_STR
if google_sheets_available and GCP_JSON_STR:
    try: SERVICE_ACCOUNT_INFO = json.loads(GCP_JSON_STR)
    except Exception as e: print(f"ERROR parsing GCP JSON: {e}"); SERVICE_ACCOUNT_INFO = None
elif google_sheets_available and not GCP_JSON_STR: print("Warning: GCP_SERVICE_ACCOUNT_JSON_STR missing.")

SHEET_NAME_RUNS = 'Runs'; SHEET_NAME_ENTITIES = 'Entities'; SHEET_NAME_RISKS = 'Risks'
SHEET_NAME_RELATIONSHIPS = 'Relationships'; SHEET_NAME_EXPOSURES = 'Supply Chain Exposures'
gsheet_service = None

def _get_gsheet_service():
    # ... (keep as before) ...
    global gsheet_service
    if not google_sheets_available: return None
    if gsheet_service is None and SERVICE_ACCOUNT_INFO:
        try:
            creds = service_account.Credentials.from_service_account_info( SERVICE_ACCOUNT_INFO, scopes=SCOPES)
            gsheet_service = build('sheets', 'v4', credentials=creds, cache_discovery=False)
            print("Google Sheets service authenticated successfully.")
        except Exception as e: print(f"ERROR auth GSheets: {e}"); gsheet_service = None
    return gsheet_service


def _append_to_gsheet(service, sheet_name: str, values: List[List[Any]]):
    # ... (keep as before) ...
    if not service or not SHEET_ID: return False
    if not values: return True
    try:
        body = {'values': values}
        result = service.spreadsheets().values().append(spreadsheetId=SHEET_ID, range=f"{sheet_name}!A1", valueInputOption='USER_ENTERED', insertDataOption='INSERT_ROWS', body=body).execute()
        updated_rows = result.get('updates', {}).get('updatedRows', 0)
        if updated_rows > 0: print(f"Appended {updated_rows} rows to sheet '{sheet_name}'.")
        return True
    except Exception as e: print(f"ERROR appending to GSheet '{sheet_name}': {e}"); return False


def _save_analysis_to_gsheet(run_results: Dict):
    """Saves the analysis results, including summary, to Google Sheets using Mountain Time."""
    print("Attempting to save analysis results to Google Sheets...")
    service = _get_gsheet_service()
    if not service: print("Aborting save: GSheet service unavailable."); return

    # --- Generate Timestamp in Mountain Time ---
    run_timestamp_iso = "Timestamp Error"
    try:
        dt_utc = datetime.now(timezone.utc)
        if zoneinfo_available:
            mountain_tz = ZoneInfo("America/Denver")
            dt_mt = dt_utc.astimezone(mountain_tz)
            run_timestamp_iso = dt_mt.isoformat()
            print(f"Generated Mountain Time timestamp: {run_timestamp_iso}")
        else:
            run_timestamp_iso = dt_utc.isoformat() # Fallback to UTC
            print(f"Using UTC timestamp (zoneinfo unavailable): {run_timestamp_iso}")
    except Exception as e:
        print(f"ERROR generating timestamp: {e}")
        run_timestamp_iso = datetime.now().isoformat() + "_Error"


    # --- 1. Save Run Summary (Including new Analysis Summary) ---
    run_summary_row = [
        run_timestamp_iso,
        run_results.get('query'),
        run_results.get('run_duration_seconds'),
        run_results.get('kg_update_status'),
        run_results.get('error'),
        run_results.get('analysis_summary', '')[:500] # Truncate summary for sheet cell limit
    ]
    _append_to_gsheet(service, SHEET_NAME_RUNS, [run_summary_row])

    # --- 2. Save Extracted Data (Entities, Risks, Relationships) ---
    extracted = run_results.get("final_extracted_data", {})
    entities = extracted.get("entities", []); risks = extracted.get("risks", []); relationships = extracted.get("relationships", [])
    if entities: entity_rows = [[run_timestamp_iso, e.get('name'), e.get('type'), json.dumps(e.get('mentions', []))] for e in entities if isinstance(e, dict) and e.get('name')]; _append_to_gsheet(service, SHEET_NAME_ENTITIES, entity_rows)
    if risks: risk_rows = [[run_timestamp_iso, r.get('description'), r.get('severity'), json.dumps(r.get('related_entities', [])), json.dumps(r.get('source_urls', []))] for r in risks if isinstance(r, dict) and r.get('description')]; _append_to_gsheet(service, SHEET_NAME_RISKS, risk_rows)
    if relationships: rel_rows = [[run_timestamp_iso, rel.get('entity1'), rel.get('relationship_type'), rel.get('entity2'), json.dumps(rel.get('context_urls', []))] for rel in relationships if isinstance(rel, dict) and rel.get('entity1') and rel.get('relationship_type') and rel.get('entity2')]; _append_to_gsheet(service, SHEET_NAME_RELATIONSHIPS, rel_rows)

    # --- 3. Save Supply Chain Exposures ---
    exposures = run_results.get("supply_chain_exposures", [])
    if exposures:
        exposure_rows = [ [run_timestamp_iso, exp.get('problematic_sub_affiliate'), exp.get('relationship_type'), exp.get('parent_company'), exp.get('reason_involved'), exp.get('risk_source_url'), exp.get('ownership_source_url') ] for exp in exposures if isinstance(exp, dict) and exp.get('problematic_sub_affiliate')]
        _append_to_gsheet(service, SHEET_NAME_EXPOSURES, exposure_rows)

    print("Google Sheets save process finished.")


# --- Main Orchestration Logic ---
def run_analysis(initial_query: str,
                 llm_provider: str, llm_model: str,
                 global_search_context: str = "global financial news and legal filings",
                 specific_search_context: str = "search for specific company examples and regulatory actions",
                 specific_country_code: str = 'us',
                 max_global_results: int = 10,
                 max_specific_results: int = 10
                 ) -> Dict[str, Any]:
    start_run_time = time.time()
    results = { # Initialize results structure
        "query": initial_query, "steps": [], "llm_used": f"{llm_provider} ({llm_model})",
        "final_extracted_data": {"entities": [], "risks": [], "relationships": []},
        "supply_chain_exposures": [], "analysis_summary": "Summary not generated.", # Init summary
        "wayback_results": [], "kg_update_status": "not_run",
        "run_duration_seconds": 0, "error": None,
    }
    # Init Neo4j driver
    kg_driver_available = bool(knowledge_graph.get_driver())
    if not kg_driver_available: results["kg_update_status"] = "skipped_no_connection"
    # Check LLM params
    if not llm_provider or not llm_model:
        results["error"] = "Missing LLM configuration."; print(f"ERROR: {results['error']}")
        results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
        if config.GOOGLE_SHEET_ID and SERVICE_ACCOUNT_INFO: _save_analysis_to_gsheet(results)
        knowledge_graph.close_driver(); return results

    try:
        # --- Step 1: Global Insights ---
        print(f"\n--- Running Step 1: Global Search ({initial_query}) ---")
        step1_start = time.time(); global_raw_results = []; global_search_source = "unknown"
        if config.GOOGLE_API_KEY_SEARCH and config.GOOGLE_CX and search_engines.google_api_client_available: global_raw_results = search_engines.search_google_official(initial_query, lang='en', num=max_global_results); global_search_source = 'google_api'
        elif config.SERPAPI_KEY and search_engines.serpapi_available: print("Using SerpApi Google for global search..."); global_raw_results = search_engines.search_via_serpapi(initial_query, engine='google', country_code='us', lang_code='en', num=max_global_results); global_search_source = 'serpapi_google'
        else: print("Warning: No suitable global search engine available.")
        global_search_results = [res for res in (search_engines.standardize_result(r, global_search_source) for r in global_raw_results) if res]; print(f"Found {len(global_search_results)} standardized global results.")
        global_extracted_data = nlp_processor.extract_data_from_results(global_search_results, global_search_context, llm_provider, llm_model); print(f"Extracted {len(global_extracted_data.get('entities',[]))} entities, {len(global_extracted_data.get('risks',[]))} risks globally.")
        results["steps"].append({"name": "Global Search & Extraction", "duration": round(time.time() - step1_start, 2),"search_results_count": len(global_search_results), "extracted_data": global_extracted_data, "status": "OK"})
        results["final_extracted_data"]["entities"].extend(global_extracted_data.get('entities',[])); results["final_extracted_data"]["risks"].extend(global_extracted_data.get('risks',[])); results["final_extracted_data"]["relationships"].extend(global_extracted_data.get('relationships',[]))

        # --- Step 2: Translate Keywords ---
        print(f"\n--- Running Step 2: Translating Keywords ---")
        step2_start = time.time()
        translated_keywords = nlp_processor.translate_keywords_for_context( initial_query, specific_search_context, llm_provider, llm_model )
        specific_query = translated_keywords[0] if translated_keywords else initial_query; print(f"Using specific query: '{specific_query}'")
        results["steps"].append({"name": "Keyword Translation", "duration": round(time.time() - step2_start, 2), "translated_keywords": translated_keywords, "status": "OK"})

        # --- Step 3: Specific Country Data ---
        print(f"\n--- Running Step 3: Specific Search ({specific_query} in {specific_country_code}) ---")
        step3_start = time.time(); specific_raw_results = []; specific_search_source = 'unknown_specific'; specific_engine = 'baidu' if specific_country_code == 'cn' else 'google'; specific_lang = 'zh-CN' if specific_country_code == 'cn' else 'en'
        # ===> CORRECTED Search Logic to use country code <===
        if specific_engine == 'baidu' and config.SERPAPI_KEY and search_engines.serpapi_available:
            print(f"Using SerpApi Baidu (cc={specific_country_code})...")
            specific_raw_results = search_engines.search_via_serpapi(specific_query, engine='baidu', country_code=specific_country_code, num=max_specific_results); specific_search_source = 'serpapi_baidu'
        elif specific_engine == 'google': # If not Baidu/cn, use Google
            # Prioritize Google Official API if available
            if config.GOOGLE_API_KEY_SEARCH and config.GOOGLE_CX and search_engines.google_api_client_available:
                print(f"Using Google Official API for specific search (lr=lang_{specific_lang})...")
                specific_raw_results = search_engines.search_google_official(specific_query, lang=specific_lang, num=max_specific_results); specific_search_source = 'google_api'
            # Fallback to SerpApi Google
            elif config.SERPAPI_KEY and search_engines.serpapi_available:
                print(f"Using SerpApi Google for specific search (gl={specific_country_code}, hl={specific_lang})...")
                specific_raw_results = search_engines.search_via_serpapi(specific_query, engine='google', country_code=specific_country_code, lang_code=specific_lang, num=max_specific_results); specific_search_source = 'serpapi_google'
            else: print(f"Warning: No Google search engine available for specific search in {specific_country_code}.")
        else: print(f"Warning: Cannot perform Baidu search for {specific_country_code} without SerpApi key/library.")
        # ===> END CORRECTION <===
        specific_search_results = [res for res in (search_engines.standardize_result(r, specific_search_source) for r in specific_raw_results) if res]; print(f"Found {len(specific_search_results)} standardized specific results.")
        specific_extracted_data = nlp_processor.extract_data_from_results(specific_search_results, specific_search_context, llm_provider, llm_model); print(f"Extracted {len(specific_extracted_data.get('entities',[]))} entities, {len(specific_extracted_data.get('risks',[]))} risks specifically.")
        results["steps"].append({"name": "Specific Search & Extraction", "duration": round(time.time() - step3_start, 2),"search_results_count": len(specific_search_results), "extracted_data": specific_extracted_data, "status": "OK"})
        results["final_extracted_data"]["entities"].extend(specific_extracted_data.get('entities',[])); results["final_extracted_data"]["risks"].extend(specific_extracted_data.get('risks',[])); results["final_extracted_data"]["relationships"].extend(specific_extracted_data.get('relationships',[]))

        # --- Step 3.5: Identify Supply Chain Exposures ---
        print(f"\n--- Running Step 3.5: Identifying Supply Chain Exposures ---")
        step3_5_start = time.time(); identified_exposures = []; high_risk_parent_entities = {}
        all_risks = results["final_extracted_data"].get("risks", []); all_entity_names_in_run = {e.get('name') for e in results["final_extracted_data"].get("entities", []) if e.get('name')}
        for risk in all_risks:
            severity = risk.get("severity", "LOW");
            if severity in ["HIGH", "MEDIUM"]:
                for entity_name in risk.get("related_entities", []):
                    if entity_name and entity_name not in high_risk_parent_entities: high_risk_parent_entities[entity_name] = {"description": risk.get('description', 'N/A'), "source_urls": risk.get('source_urls', [])}
        print(f"Found {len(high_risk_parent_entities)} unique entities in medium/high risks: {list(high_risk_parent_entities.keys())}")
        OWNERSHIP_CHECK_DELAY = 4.0 # Increased delay
        for parent_name, risk_info in high_risk_parent_entities.items():
            potential_related_names = all_entity_names_in_run - {parent_name};
            if not potential_related_names: continue
            print(f"\nAnalyzing ownership for high-risk entity: {parent_name}")
            ownership_docs = search_engines.search_for_ownership_docs(parent_name, num_per_query=2)
            if not ownership_docs: print(f"No ownership documents found for {parent_name}."); continue
            for related_name in potential_related_names:
                 print(f"---> Checking ownership: '{parent_name}' owning '{related_name}'?")
                 ownership_details = nlp_processor.extract_ownership_relationships( parent_entity_name=parent_name, related_entity_name=related_name, text_snippets=ownership_docs, llm_provider=llm_provider, llm_model=llm_model )
                 if ownership_details and ownership_details.get("relationship_type") in ["SUBSIDIARY", "AFFILIATE"]:
                      print(f"  ==> Found Exposure: {parent_name} -> {ownership_details['relationship_type']} -> {related_name}")
                      exposure_record = { "problematic_sub_affiliate": related_name, "relationship_type": ownership_details["relationship_type"], "parent_company": parent_name, "reason_involved": f"Parent ({parent_name}) linked to risk: {risk_info['description'][:150]}...", "risk_source_url": risk_info['source_urls'][0] if risk_info['source_urls'] else "N/A", "ownership_source_url": ownership_details.get("source_url", "Not Stated") }
                      identified_exposures.append(exposure_record)
                 elif ownership_details: print(f"  Relationship is {ownership_details['relationship_type']}")
                 else: print(f"  Could not determine ownership relationship.")
                 print(f"Pausing {OWNERSHIP_CHECK_DELAY}s before next ownership check..."); time.sleep(OWNERSHIP_CHECK_DELAY)
        results["supply_chain_exposures"] = identified_exposures
        results["steps"].append({ "name": "Supply Chain Exposure Analysis", "duration": round(time.time() - step3_5_start, 2), "exposures_found": len(identified_exposures), "status": "OK" })

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
            final_data = results.get("final_extracted_data", {}); entities_list = final_data.get("entities", []); relationships_list = final_data.get("relationships", []); risks_list = final_data.get("risks", [])
            unique_entities = list({ f"{e.get('name', '')}_{e.get('type', '')}".lower(): e for e in entities_list if isinstance(e, dict) and e.get('name') }.values())
            unique_relationships = list({ f"{r.get('entity1', '')}_{r.get('relationship_type', '')}_{r.get('entity2', '')}".lower(): r for r in relationships_list if isinstance(r, dict) and r.get('entity1') and r.get('relationship_type') and r.get('entity2') }.values())
            unique_risks = list({ r.get('description', ''): r for r in risks_list if isinstance(r, dict) and r.get('description') }.values())
            deduped_data = {"entities": unique_entities, "risks": unique_risks, "relationships": unique_relationships}
            if not any([unique_entities, unique_risks, unique_relationships]): print("No unique data extracted to update KG."); kg_status_message = "skipped_no_data"
            else: print(f"Attempting KG update..."); update_success = knowledge_graph.update_knowledge_graph(deduped_data); kg_status_message = "success" if update_success else "error"
        else: print("Skipping KG update: driver not available.")
        results["kg_update_status"] = kg_status_message; results["steps"].append({"name": "Knowledge Graph Update", "duration": round(time.time() - step5_start, 2), "status": results["kg_update_status"]})

        # ===> Step 5.5: Generate Final Summary <===
        print(f"\n--- Running Step 5.5: Generating Analysis Summary ---")
        step5_5_start = time.time()
        summary = "Summary generation skipped or failed."
        # Check if there's anything worth summarizing
        if results["final_extracted_data"]["entities"] or results["final_extracted_data"]["risks"] or results["final_extracted_data"]["relationships"] or results["supply_chain_exposures"]:
             summary = nlp_processor.generate_analysis_summary(
                 extracted_data=results["final_extracted_data"],
                 query=initial_query,
                 exposures_count=len(results["supply_chain_exposures"]), # Pass exposure count
                 llm_provider=llm_provider,
                 llm_model=llm_model
             )
        else: summary = "No significant data extracted to generate a summary."
        results["analysis_summary"] = summary # Store the summary
        results["steps"].append({"name": "Analysis Summary Generation", "duration": round(time.time() - step5_5_start, 2), "status": "OK" if not summary.startswith("Could not generate") else "Error"})
        # ===> END Step 5.5 <===


        # --- Step 6: Learning / Adapting (Placeholder) ---
        print("\n--- Step 6: Learning/Adapting (Deferred) ---")
        results["steps"].append({"name": "Learning/Adapting", "status": "deferred"})

    # --- Error Handling ---
    except Exception as e:
        print(f"\n--- Orchestration Error ---"); error_type = type(e).__name__; error_msg = str(e) if str(e) else "No msg"; error_traceback = traceback.format_exc()
        print(f"Type: {error_type}\nMsg: {error_msg}\nTraceback:\n{error_traceback}"); results["error"] = f"{error_type}: {error_msg}"

    finally:
        knowledge_graph.close_driver() # Close Neo4j
        results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
        # --- Save results to Google Sheet ---
        if config.GOOGLE_SHEET_ID and SERVICE_ACCOUNT_INFO:
            _save_analysis_to_gsheet(results) # Call the save function
        else:
             print("Skipping save to Google Sheets: Configuration missing or invalid.")
        print(f"\n--- Analysis Complete ({results['run_duration_seconds']}s) ---")

    return results # Return the results dictionary