# orchestrator.py

import time
from typing import Dict, List, Any
import traceback
import json
from datetime import datetime, timezone # Import timezone directly
try:
    from zoneinfo import ZoneInfo
    zoneinfo_available = True
except ImportError:
    zoneinfo_available = False
    print("Warning: zoneinfo not available. Timestamps will be UTC.")


# Google Sheets Imports
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    import os
    google_sheets_available = True
except ImportError:
    print("Warning: Google API libraries not installed (`google-api-python-client google-auth-httplib2 google-auth-oauthlib`). Saving to Google Sheets disabled.")
    google_sheets_available = False
    service_account = None
    build = None


# Custom Modules
import search_engines
import nlp_processor
import knowledge_graph
import config

# --- Google Sheets Configuration & Functions ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SHEET_ID = config.GOOGLE_SHEET_ID
SERVICE_ACCOUNT_INFO = None
GCP_JSON_STR = config.GCP_SERVICE_ACCOUNT_JSON_STR
if google_sheets_available and GCP_JSON_STR:
    try:
        SERVICE_ACCOUNT_INFO = json.loads(GCP_JSON_STR)
        # print("Successfully parsed GCP Service Account JSON from environment.")
    except Exception as e:
        print(f"ERROR parsing GCP JSON environment variable: {e}")
        SERVICE_ACCOUNT_INFO = None
elif google_sheets_available and not GCP_JSON_STR:
    print("Warning: GCP_SERVICE_ACCOUNT_JSON_STR environment variable not set or empty.")

SHEET_NAME_RUNS = 'Runs'
SHEET_NAME_ENTITIES = 'Entities'
SHEET_NAME_RISKS = 'Risks'
SHEET_NAME_RELATIONSHIPS = 'Relationships'
SHEET_NAME_EXPOSURES = 'Supply Chain Exposures'
gsheet_service = None

def _get_gsheet_service():
    """Authenticates and builds the Google Sheets service object (singleton)."""
    global gsheet_service
    if not google_sheets_available:
        return None
    if gsheet_service is None:
        if not SERVICE_ACCOUNT_INFO:
            print("Cannot authenticate to Google Sheets: Service account info missing or invalid.")
            return None
        try:
            creds = service_account.Credentials.from_service_account_info(
                SERVICE_ACCOUNT_INFO, scopes=SCOPES)
            gsheet_service = build('sheets', 'v4', credentials=creds, cache_discovery=False)
            print("Google Sheets service authenticated successfully.")
        except Exception as e:
            print(f"ERROR: Failed to authenticate/build Google Sheets service: {e}")
            gsheet_service = None
    return gsheet_service

def _append_to_gsheet(service, sheet_name: str, values: List[List[Any]]):
    """Appends rows of values to a specific sheet."""
    if not service or not SHEET_ID:
        # print(f"Skipping append to {sheet_name}: Missing GSheet service or Sheet ID.")
        return False
    if not values:
        return True # Nothing to write is not an error

    try:
        body = {'values': values}
        result = service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID, range=f"{sheet_name}!A1",
            valueInputOption='USER_ENTERED', insertDataOption='INSERT_ROWS', body=body
        ).execute()
        updated_rows = result.get('updates', {}).get('updatedRows', 0)
        if updated_rows > 0:
             print(f"Successfully appended {updated_rows} rows to sheet '{sheet_name}'.")
        return True
    except Exception as e:
        print(f"ERROR appending data to Google Sheet '{sheet_name}': {e}")
        return False

def _save_analysis_to_gsheet(run_results: Dict):
    """Saves the analysis results to Google Sheets using Mountain Time."""
    print("Attempting to save analysis results to Google Sheets...")
    service = _get_gsheet_service()
    if not service: print("Aborting save: GSheet service unavailable."); return

    # Generate Timestamp in Mountain Time
    run_timestamp_iso = "Timestamp Error"
    try:
        dt_utc = datetime.now(timezone.utc) # Use imported timezone
        if zoneinfo_available:
            mountain_tz = ZoneInfo("America/Denver")
            dt_mt = dt_utc.astimezone(mountain_tz)
            run_timestamp_iso = dt_mt.isoformat(timespec='seconds')
            print(f"Generated Mountain Time timestamp: {run_timestamp_iso}")
        else:
            run_timestamp_iso = dt_utc.isoformat(timespec='seconds')
            print(f"Using UTC timestamp (zoneinfo unavailable): {run_timestamp_iso}")
    except Exception as e:
        print(f"ERROR generating timestamp: {e}")
        run_timestamp_iso = datetime.now().isoformat(timespec='seconds') + "_Error"

    # --- 1. Save Run Summary ---
    run_summary_row = [ run_timestamp_iso, run_results.get('query'), run_results.get('run_duration_seconds'), run_results.get('kg_update_status'), run_results.get('error'), run_results.get('analysis_summary', '')[:500] ]
    _append_to_gsheet(service, SHEET_NAME_RUNS, [run_summary_row])

    # --- 2. Save Extracted Data ---
    # Use final_extracted_data which accumulates results from all steps
    extracted = run_results.get("final_extracted_data", {})
    entities = extracted.get("entities", []); risks = extracted.get("risks", []); relationships = extracted.get("relationships", [])

    # Save Entities (Note: These are all entities found across steps before KG deduplication)
    if entities:
        entity_rows = [
            [run_timestamp_iso, e.get('name'), e.get('type'), json.dumps(e.get('mentions', []))]
            for e in entities if isinstance(e, dict) and e.get('name')
        ]
        if entity_rows: _append_to_gsheet(service, SHEET_NAME_ENTITIES, entity_rows)

    # Save Risks (Note: These are all risks found across steps before KG deduplication)
    if risks:
        risk_rows_to_append = []
        for r in risks:
            if isinstance(r, dict) and r.get('description'):
                desc = r.get('description')
                sev = r.get('severity', 'UNKNOWN') # Default if missing
                rels_json = json.dumps(r.get('related_entities', []))
                urls_json = json.dumps(r.get('source_urls', []))
                current_risk_row = [run_timestamp_iso, desc, sev, rels_json, urls_json]
                risk_rows_to_append.append(current_risk_row)
            else:
                print(f"Warning: Skipping invalid risk item during GSheet save: {r}")
        if risk_rows_to_append: _append_to_gsheet(service, SHEET_NAME_RISKS, risk_rows_to_append)

    # Save Relationships (Note: These are all relationships found across steps before KG deduplication)
    if relationships:
        rel_rows = [
            [run_timestamp_iso, rel.get('entity1'), rel.get('relationship_type'), rel.get('entity2'), json.dumps(rel.get('context_urls', []))]
            for rel in relationships if isinstance(rel, dict) and rel.get('entity1') and rel.get('relationship_type') and rel.get('entity2')
        ]
        if rel_rows: _append_to_gsheet(service, SHEET_NAME_RELATIONSHIPS, rel_rows)

    # --- 3. Save Supply Chain Exposures ---
    exposures = run_results.get("supply_chain_exposures", [])
    if exposures:
        exposure_rows = [
            [run_timestamp_iso, exp.get('problematic_sub_affiliate'), exp.get('relationship_type'), exp.get('parent_company'), exp.get('reason_involved'), exp.get('risk_source_url'), exp.get('ownership_source_url') ]
            for exp in exposures if isinstance(exp, dict) and exp.get('problematic_sub_affiliate')
        ]
        if exposure_rows: _append_to_gsheet(service, SHEET_NAME_EXPOSURES, exposure_rows)

    print("Google Sheets save process finished.")


# --- Main Orchestration Logic ---
def run_analysis(initial_query: str,
                 llm_provider: str, llm_model: str,
                 # Context parameters are less critical now focus comes from search/extraction flags
                 global_search_context: str = "China-related financial news and legal filings",
                 specific_search_context: str = "search for specific company examples and regulatory actions",
                 # Default specific country changed to 'cn'
                 specific_country_code: str = 'cn',
                 max_global_results: int = 10,
                 max_specific_results: int = 10
                 ) -> Dict[str, Any]:
    start_run_time = time.time()
    results = { # Initialize results structure
        "query": initial_query, "steps": [], "llm_used": f"{llm_provider} ({llm_model})",
        "final_extracted_data": {"entities": [], "risks": [], "relationships": []}, # Accumulates data
        "supply_chain_exposures": [], "analysis_summary": "Summary not generated.",
        "wayback_results": [], "kg_update_status": "not_run",
        "run_duration_seconds": 0, "error": None,
    }
    kg_driver_available = bool(knowledge_graph.get_driver())
    if not kg_driver_available: results["kg_update_status"] = "skipped_no_connection"
    if not llm_provider or not llm_model:
        results["error"] = "Missing LLM configuration."; print(f"ERROR: {results['error']}")
        results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
        # Attempt save even on config error to log the attempt
        if config.GOOGLE_SHEET_ID and SERVICE_ACCOUNT_INFO: _save_analysis_to_gsheet(results)
        knowledge_graph.close_driver(); return results

    try:
        # === Step 1: Initial China-Focused Search (REVISED) ===
        print(f"\n--- Running Step 1: Initial China-Focused Search (Query: '{initial_query}') ---")
        step1_start = time.time(); step1_raw_results = []; step1_search_source = "unknown"
        # Don't modify original query here, use parameters or context
        # china_focused_query = f"{initial_query} China" # Removed: Rely on search params/extraction focus

        # --- Prioritize SerpApi Google search from China (gl=cn) for English results (hl=en) ---
        if config.SERPAPI_KEY and search_engines.serpapi_available:
            print(f"Using SerpApi Google (gl=cn, hl=en) for China-focused search: '{initial_query}'")
            step1_raw_results = search_engines.search_via_serpapi(
                query=initial_query, # Use original query text
                engine='google',
                country_code='cn',  # Set geographical location to China
                lang_code='en',     # Request English language results
                num=max_global_results
            )
            step1_search_source = 'serpapi_google_cn_en'
            if not step1_raw_results:
                 print("SerpApi Google (cn, en) returned no results. Trying query modification fallback...")
                 # Fallback: Try again with modified query if first attempt yielded nothing
                 china_focused_query_fallback = f"{initial_query} China"
                 step1_raw_results = search_engines.search_via_serpapi(china_focused_query_fallback, 'google', 'cn', 'en', max_global_results)
                 if step1_raw_results: step1_search_source = 'serpapi_google_cn_en_mod' # Indicate fallback used


        # --- Fallback 1: Official Google API with modified query ---
        # Use this if SerpApi is unavailable OR if SerpApi (cn,en) returned no results
        if not step1_raw_results and config.GOOGLE_API_KEY_SEARCH and config.GOOGLE_CX and search_engines.google_api_client_available:
            china_focused_query_fallback = f"{initial_query} China"
            print(f"Fallback: Using Google Official API with modified query: '{china_focused_query_fallback}'")
            # Official API doesn't have a reliable geo-location param like SerpApi's 'gl'
            # We rely on adding "China" to the query string.
            step1_raw_results = search_engines.search_google_official(
                query=china_focused_query_fallback,
                lang='en',
                num=max_global_results
            )
            step1_search_source = 'google_api_query_mod'

        # --- Fallback 2: SerpApi Google (US region) with modified query ---
        # Use this if only SerpApi is available AND the gl=cn attempts failed
        elif not step1_raw_results and config.SERPAPI_KEY and search_engines.serpapi_available:
             china_focused_query_fallback = f"{initial_query} China"
             print(f"Fallback: Using SerpApi Google (US region) with modified query: '{china_focused_query_fallback}'")
             step1_raw_results = search_engines.search_via_serpapi(china_focused_query_fallback, 'google', 'us', 'en', max_global_results)
             step1_search_source = 'serpapi_google_us_query_mod'

        else:
            # Only print warning if nothing worked AND we never got results
            if not step1_raw_results:
                print("Warning: No suitable search engine available or no results found for Step 1.")

        step1_search_results = [res for res in (search_engines.standardize_result(r, step1_search_source) for r in step1_raw_results) if res];
        print(f"Found {len(step1_search_results)} standardized results for Step 1 (China-focused search).")

        # --- Step 1 Extraction: Use focus_on_china=True ---
        # The context message can be simpler now, assuming results are relevant
        step1_extraction_context = f"Analyze the following search results regarding '{initial_query}' focusing primarily on China-related entities, risks, and relationships described in the text."
        print(f"Applying China-focused extraction (focus_on_china=True) to Step 1 results.")

        step1_extracted_data = nlp_processor.extract_data_from_results(
            search_results=step1_search_results,
            extraction_context=step1_extraction_context, # Use the simpler context
            llm_provider=llm_provider,
            llm_model=llm_model,
            focus_on_china=True # Set the flag to True for Step 1 extraction
        )

        print(f"Extracted {len(step1_extracted_data.get('entities',[]))} entities, {len(step1_extracted_data.get('risks',[]))} risks, {len(step1_extracted_data.get('relationships',[]))} relationships from Step 1 (China-focused).")
        results["steps"].append({
            "name": "Initial China-Focused Search & Extraction", # Renamed step
            "duration": round(time.time() - step1_start, 2),
            "search_results_count": len(step1_search_results),
            "extracted_data": step1_extracted_data, # Log data specific to this step
            "status": "OK" if step1_search_results else "No Results"
        })
        # Add extracted data to final results accumulator
        if step1_extracted_data.get("entities"): results["final_extracted_data"]["entities"].extend(step1_extracted_data["entities"])
        if step1_extracted_data.get("risks"): results["final_extracted_data"]["risks"].extend(step1_extracted_data["risks"])
        if step1_extracted_data.get("relationships"): results["final_extracted_data"]["relationships"].extend(step1_extracted_data["relationships"])


        # --- Step 2: Translate Keywords ---
        # Translates original query for Step 3's specific context target (e.g., Baidu)
        print(f"\n--- Running Step 2: Translating Keywords ---")
        step2_start = time.time()
        # Use the specific_search_context to guide keyword translation for Step 3
        translated_keywords = nlp_processor.translate_keywords_for_context(
            original_query=initial_query,
            target_context=specific_search_context, # Context for the *next* step's search
            llm_provider=llm_provider,
            llm_model=llm_model
        )
        # Use the *first* translated keyword for the specific search, or original query if translation fails
        specific_query_base = translated_keywords[0] if translated_keywords else initial_query
        print(f"Base specific query from translation (for Step 3): '{specific_query_base}'")
        results["steps"].append({"name": "Keyword Translation", "duration": round(time.time() - step2_start, 2), "translated_keywords": translated_keywords, "status": "OK"})

        # --- Step 3: Specific Country Data ---
        # Performs search based on specific_country_code using translated query if available
        print(f"\n--- Running Step 3: Specific Country Search ({specific_query_base} in {specific_country_code}) ---")
        step3_start = time.time(); step3_raw_results = []; step3_search_source = 'unknown_specific';
        # Determine engine/language based on the specific_country_code for this step
        specific_engine = 'baidu' if specific_country_code == 'cn' else 'google';
        specific_lang = 'zh-CN' if specific_country_code == 'cn' else 'en'
        # Construct the query for this specific step (using the potentially translated query)
        specific_query_step3 = specific_query_base

        if specific_engine == 'baidu' and config.SERPAPI_KEY and search_engines.serpapi_available:
            print(f"Using SerpApi Baidu (cc={specific_country_code}). Query: '{specific_query_step3}'");
            step3_raw_results = search_engines.search_via_serpapi(specific_query_step3, engine='baidu', country_code=specific_country_code, num=max_specific_results);
            step3_search_source = 'serpapi_baidu'
        elif specific_engine == 'google':
            # Prefer SerpApi Google for specific country targeting if available
            if config.SERPAPI_KEY and search_engines.serpapi_available:
                 print(f"Using SerpApi Google (gl={specific_country_code}, hl={specific_lang}). Query: '{specific_query_step3}'");
                 step3_raw_results = search_engines.search_via_serpapi(specific_query_step3, engine='google', country_code=specific_country_code, lang_code=specific_lang, num=max_specific_results);
                 step3_search_source = 'serpapi_google'
            # Fallback to Official Google API
            elif config.GOOGLE_API_KEY_SEARCH and config.GOOGLE_CX and search_engines.google_api_client_available:
                 print(f"Fallback: Using Google Official API (lang={specific_lang}). Query: '{specific_query_step3}'");
                 # Note: Official API 'lang' is less precise than SerpApi 'hl' and 'gl'
                 step3_raw_results = search_engines.search_google_official(specific_query_step3, lang=specific_lang, num=max_specific_results);
                 step3_search_source = 'google_api'
            else: print(f"Warning: No Google search engine available for specific search in {specific_country_code}.")
        else: print(f"Warning: Cannot perform Baidu search for {specific_country_code} without SerpApi key/library.")

        step3_search_results = [res for res in (search_engines.standardize_result(r, step3_search_source) for r in step3_raw_results) if res];
        print(f"Found {len(step3_search_results)} standardized specific results for Step 3.")

        # Determine if extraction needs China focus based on specific_country_code
        is_china_focused_step3 = (specific_country_code == 'cn')
        # Context here remains the user-provided specific_search_context from input
        step3_extraction_context = specific_search_context
        step3_extracted_data = nlp_processor.extract_data_from_results(
            search_results=step3_search_results,
            extraction_context=step3_extraction_context, # Use the specific context parameter
            llm_provider=llm_provider,
            llm_model=llm_model,
            focus_on_china=is_china_focused_step3 # Use the boolean flag here based on country code
        );
        print(f"Extracted {len(step3_extracted_data.get('entities',[]))} entities, {len(step3_extracted_data.get('risks',[]))} risks, {len(step3_extracted_data.get('relationships',[]))} relationships from Step 3 (China focus flag: {is_china_focused_step3}).")
        results["steps"].append({
            "name": f"Specific Search ({specific_country_code}) & Extraction",
            "duration": round(time.time() - step3_start, 2),
            "search_results_count": len(step3_search_results),
            "extracted_data": step3_extracted_data, # Log data specific to this step
            "status": "OK" if step3_search_results else "No Results"
        })
        # Add extracted data to final results accumulator
        if step3_extracted_data.get("entities"): results["final_extracted_data"]["entities"].extend(step3_extracted_data["entities"])
        if step3_extracted_data.get("risks"): results["final_extracted_data"]["risks"].extend(step3_extracted_data["risks"])
        if step3_extracted_data.get("relationships"): results["final_extracted_data"]["relationships"].extend(step3_extracted_data["relationships"])


        # --- Step 3.5: Identify Supply Chain Exposures ---
        # Operates on the combined data accumulated in results["final_extracted_data"]
        print(f"\n--- Running Step 3.5: Identifying Supply Chain Exposures ---")
        step3_5_start = time.time(); identified_exposures = []; high_risk_parent_entities = {}
        # Use the consolidated data collected so far
        all_final_entities = results["final_extracted_data"].get("entities", [])
        all_final_risks = results["final_extracted_data"].get("risks", [])
        # Create a set of all unique entity names found in the entire run
        all_entity_names_in_run = {e.get('name') for e in all_final_entities if isinstance(e, dict) and e.get('name')}
        print(f"Total unique entity names identified across steps: {len(all_entity_names_in_run)}")

        # Find entities mentioned in medium/high risks
        for risk_item in all_final_risks:
            if not isinstance(risk_item, dict): print(f"Warning: Skipping non-dict risk item in Step 3.5: {risk_item}"); continue
            severity = risk_item.get("severity", "LOW")
            if severity in ["HIGH", "MEDIUM"]:
                related_entities_for_risk = risk_item.get("related_entities", [])
                if isinstance(related_entities_for_risk, list):
                    for entity_name in related_entities_for_risk:
                         # Ensure entity_name is valid and not already tracked
                         if entity_name and isinstance(entity_name, str) and entity_name not in high_risk_parent_entities:
                             # Store risk description and source URLs associated with this first mention
                             high_risk_parent_entities[entity_name] = {
                                 "description": risk_item.get('description', 'N/A'),
                                 "source_urls": risk_item.get('source_urls', [])
                              }
                else: print(f"Warning: related_entities is not a list in risk: {risk_item.get('description')}")

        print(f"Found {len(high_risk_parent_entities)} unique entities involved in medium/high risks: {list(high_risk_parent_entities.keys())}")

        OWNERSHIP_CHECK_DELAY = 3.5 # Slightly reduced delay
        if high_risk_parent_entities: # Only proceed if potential parents found
            print(f"Analyzing ownership for {len(high_risk_parent_entities)} high-risk entities...")
            parent_check_count = 0
            for parent_name, risk_info in high_risk_parent_entities.items():
                parent_check_count += 1
                print(f"\n[{parent_check_count}/{len(high_risk_parent_entities)}] Checking parent: {parent_name}")
                # Identify potential related entities *excluding the parent itself*
                potential_related_names = all_entity_names_in_run - {parent_name}
                if not potential_related_names:
                     print(f"  No other unique entities found in run to check ownership against '{parent_name}'.")
                     continue

                print(f"-- Searching ownership docs for: {parent_name}")
                # Get potential ownership docs related to the parent
                ownership_docs = search_engines.search_for_ownership_docs(parent_name, num_per_query=3) # Can adjust num_per_query
                if not ownership_docs:
                    print(f"  No relevant ownership documents found for {parent_name}.")
                    continue
                else:
                     print(f"  Found {len(ownership_docs)} potential ownership docs for {parent_name}. Analyzing against {len(potential_related_names)} other entities...")

                # Check relationship between this parent and *other* entities found in the run
                related_check_count = 0
                for related_name in potential_related_names:
                     related_check_count +=1
                     print(f"  -> [{related_check_count}/{len(potential_related_names)}] Checking: '{parent_name}' owns '{related_name}'?")
                     # Use NLP processor to check the relationship based on found docs
                     ownership_details = nlp_processor.extract_ownership_relationships(
                         parent_entity_name=parent_name,
                         related_entity_name=related_name,
                         text_snippets=ownership_docs, # Use the docs found for the parent
                         llm_provider=llm_provider,
                         llm_model=llm_model
                     )

                     # Check the result from NLP processor
                     if ownership_details and ownership_details.get("relationship_type") in ["SUBSIDIARY", "AFFILIATE"]:
                          print(f"    ==> Found Exposure: {parent_name} -> {ownership_details['relationship_type']} -> {related_name}")
                          exposure_record = {
                              "problematic_sub_affiliate": related_name,
                              "relationship_type": ownership_details["relationship_type"],
                              "parent_company": parent_name,
                              "reason_involved": f"Parent ({parent_name}) linked to risk: {risk_info['description'][:150]}...",
                              "risk_source_url": risk_info['source_urls'][0] if risk_info.get('source_urls') else "N/A", # Safer access
                              "ownership_source_url": ownership_details.get("source_url", "Not Stated")
                          }
                          # Avoid adding duplicate exposures (same parent-sub pair)
                          if exposure_record not in identified_exposures:
                               identified_exposures.append(exposure_record)
                          else: print("    (Duplicate exposure record avoided)")

                     # Optional: Log other relationship types found for debugging, but can be verbose
                     # elif ownership_details and ownership_details.get("relationship_type") == "UNRELATED":
                     #     print(f"    Relationship determined as UNRELATED.")
                     # elif ownership_details:
                     #      print(f"    Relationship is {ownership_details.get('relationship_type', 'Unknown')}. Evidence: {ownership_details.get('evidence_snippet', 'N/A')[:60]}...")
                     elif not ownership_details:
                          print(f"    Could not determine ownership relationship (NLP Processor returned None).")

                     # Avoid hitting LLM rate limits aggressively during checks
                     # print(f"    Pausing {OWNERSHIP_CHECK_DELAY}s...") # Pause removed for brevity unless hitting limits
                     time.sleep(OWNERSHIP_CHECK_DELAY) # Still pause between checks
        else:
            print("No medium/high risk parent entities identified to check for exposures.")

        results["supply_chain_exposures"] = identified_exposures
        results["steps"].append({ "name": "Supply Chain Exposure Analysis", "duration": round(time.time() - step3_5_start, 2), "exposures_found": len(identified_exposures), "status": "OK" })


        # --- Step 4: Wayback Machine ---
        # Operates on combined URLs from Step 1 and Step 3 search results
        print(f"\n--- Running Step 4: Wayback Machine Check ---")
        step4_start = time.time();
        # Combine URLs from both steps' search results for checking
        all_urls_combined = list(set(
            [r.get('url') for r in step1_search_results if r and r.get('url')] +
            [r.get('url') for r in step3_search_results if r and r.get('url')]
        ));
        urls_to_check = all_urls_combined[:5]; # Limit checks to avoid excessive calls
        wayback_checks = []
        print(f"Checking {len(urls_to_check)} URLs via Wayback Machine (out of {len(all_urls_combined)} unique URLs found)...");
        if urls_to_check:
            for url in urls_to_check:
                wayback_checks.append(search_engines.check_wayback_machine(url));
                time.sleep(0.6) # Be nice to Wayback API
        else:
            print("No URLs found in search results to check with Wayback Machine.")
        results["wayback_results"] = wayback_checks;
        results["steps"].append({
            "name": "Wayback Machine Check",
            "duration": round(time.time() - step4_start, 2),
            "urls_checked": len(urls_to_check),
            "status": "OK"
        })


        # --- Step 5: Update Knowledge Graph ---
        # Deduplication happens here based on the combined data before update
        print(f"\n--- Running Step 5: Update Knowledge Graph ---")
        step5_start = time.time(); update_success = False; kg_status_message = results["kg_update_status"]
        if kg_driver_available:
            # Use the consolidated data from results["final_extracted_data"]
            final_data_for_kg = results.get("final_extracted_data", {});
            entities_list = final_data_for_kg.get("entities", []);
            relationships_list = final_data_for_kg.get("relationships", []);
            risks_list = final_data_for_kg.get("risks", [])

            # Deduplicate based on relevant identifiers before sending to KG
            # Use simple dict trick for deduplication based on a composite key
            unique_entities_dict = { f"{e.get('name', '')}_{e.get('type', '')}".lower(): e for e in entities_list if isinstance(e, dict) and e.get('name') }
            unique_relationships_dict = { f"{r.get('entity1', '')}_{r.get('relationship_type', '')}_{r.get('entity2', '')}".lower(): r for r in relationships_list if isinstance(r, dict) and r.get('entity1') and r.get('relationship_type') and r.get('entity2') }
            # Use description as key for risks - might miss nuances if descriptions slightly differ
            unique_risks_dict = { r.get('description', '').strip().lower(): r for r in risks_list if isinstance(r, dict) and r.get('description') }

            # Get the values (the unique items) from the deduplication dictionaries
            unique_entities = list(unique_entities_dict.values())
            unique_relationships = list(unique_relationships_dict.values())
            unique_risks = list(unique_risks_dict.values())

            deduped_data = {"entities": unique_entities, "risks": unique_risks, "relationships": unique_relationships}
            print(f"Attempting KG Update with Deduplicated Data: {len(unique_entities)} E, {len(unique_risks)} R, {len(unique_relationships)} Rel.")

            if not any([unique_entities, unique_risks, unique_relationships]):
                print("No unique data extracted/remaining after deduplication to update KG.");
                kg_status_message = "skipped_no_unique_data"
            else:
                print(f"Sending {len(unique_entities) + len(unique_risks) + len(unique_relationships)} items to KG update function...");
                update_success = knowledge_graph.update_knowledge_graph(deduped_data);
                kg_status_message = "success" if update_success else "error"
        else:
            print("Skipping KG update: driver not available.")

        results["kg_update_status"] = kg_status_message
        results["steps"].append({"name": "Knowledge Graph Update", "duration": round(time.time() - step5_start, 2), "status": results["kg_update_status"]})


        # --- Step 5.5: Generate Final Summary ---
        # Uses combined data from final_extracted_data and exposures count
        print(f"\n--- Running Step 5.5: Generating Analysis Summary ---")
        step5_5_start = time.time(); summary = "Summary generation skipped or failed."
        # Check if there's anything meaningful to summarize from the *entire run*
        final_data_for_summary = results.get("final_extracted_data", {})
        exposures_for_summary = results.get("supply_chain_exposures", [])
        if final_data_for_summary.get("entities") or final_data_for_summary.get("risks") or final_data_for_summary.get("relationships") or exposures_for_summary:
             summary = nlp_processor.generate_analysis_summary(
                 extracted_data=final_data_for_summary, # Pass the combined data
                 query=initial_query, # Use the original query for context
                 exposures_count=len(exposures_for_summary), # Pass the count of identified exposures
                 llm_provider=llm_provider,
                 llm_model=llm_model
             )
        else: summary = "No significant data extracted or exposures identified across steps to generate a summary."
        results["analysis_summary"] = summary
        results["steps"].append({"name": "Analysis Summary Generation", "duration": round(time.time() - step5_5_start, 2), "status": "OK" if not summary.startswith("Could not generate") and not summary.startswith("No significant") else "Generated/Skipped"})


        # --- Step 6: Learning / Adapting (Placeholder) ---
        print("\n--- Step 6: Learning/Adapting (Deferred) ---")
        results["steps"].append({"name": "Learning/Adapting", "status": "deferred"})

    # --- Error Handling ---
    except Exception as e:
        print(f"\n--- Orchestration Error ---"); error_type = type(e).__name__; error_msg = str(e) if str(e) else "No message"; error_traceback = traceback.format_exc()
        print(f"Error Type: {error_type}\nError Message: {error_msg}\nTraceback:\n{error_traceback}")
        # Ensure error is added to the results dictionary
        results["error"] = f"{error_type}: {error_msg}"
        # Optionally add traceback to results if needed for debugging, but can be long
        # results["traceback"] = error_traceback

    finally:
        # --- Final Steps ---
        knowledge_graph.close_driver() # Ensure Neo4j connection is closed
        results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
        # --- Save results to Google Sheet ---
        if config.GOOGLE_SHEET_ID and SERVICE_ACCOUNT_INFO:
            _save_analysis_to_gsheet(results) # Save final results including any errors
        else:
             print("Skipping save to Google Sheets: Configuration missing or invalid.")
        print(f"\n--- Analysis Complete ({results['run_duration_seconds']}s) ---")
        # Log final status
        if results.get("error"): print(f"--- Run finished with ERROR: {results['error']} ---")
        else: print("--- Run finished successfully ---")


    return results # Return the results dictionary