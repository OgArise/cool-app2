# orchestrator.py

import time
from typing import Dict, List, Any
import traceback
import sqlite3 # Import sqlite
import json    # Import json for serializing lists/dicts
from datetime import datetime # For timestamping runs

# Import your custom modules
import search_engines
import nlp_processor
import knowledge_graph
import config

# --- Database Configuration ---
DB_FILE = "analysis_history.db"

def _init_db():
    """Initializes the SQLite database and creates tables if they don't exist."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Table for overall analysis runs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                duration_seconds REAL,
                kg_update_status TEXT,
                error_message TEXT
            )
        ''')
        # Table for extracted entities (linked to a run)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extracted_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                type TEXT,
                mentions_json TEXT, -- Store list of URLs as JSON string
                FOREIGN KEY (run_id) REFERENCES analysis_runs (id)
            )
        ''')
        # Table for extracted risks (linked to a run)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extracted_risks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                description TEXT NOT NULL,
                severity TEXT,
                related_entities_json TEXT, -- Store list of names as JSON string
                source_urls_json TEXT, -- Store list of URLs as JSON string
                FOREIGN KEY (run_id) REFERENCES analysis_runs (id)
            )
        ''')
        # Table for extracted relationships (linked to a run)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extracted_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                entity1_name TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                entity2_name TEXT NOT NULL,
                context_urls_json TEXT, -- Store list of URLs as JSON string
                FOREIGN KEY (run_id) REFERENCES analysis_runs (id)
            )
        ''')
        conn.commit()
        conn.close()
        print(f"Database '{DB_FILE}' initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize database '{DB_FILE}': {e}")

def _save_analysis_to_db(run_results: Dict):
    """Saves the summary and detailed extracted data of an analysis run to SQLite."""
    print("Attempting to save analysis results to database...")
    run_id = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # --- 1. Insert the main run record ---
        run_timestamp = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO analysis_runs (
                run_timestamp, query, duration_seconds, kg_update_status, error_message
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            run_timestamp,
            run_results.get('query'),
            run_results.get('run_duration_seconds'),
            run_results.get('kg_update_status'),
            run_results.get('error') # Will be None if no error
        ))
        run_id = cursor.lastrowid # Get the ID of the inserted run record
        print(f"Saved run summary with ID: {run_id}")

        # --- 2. Insert Extracted Data (if run_id exists) ---
        if run_id:
            extracted = run_results.get("final_extracted_data", {})

            # Insert Entities
            entities = extracted.get("entities", [])
            if entities:
                entity_data = [
                    (run_id, e.get('name'), e.get('type'), json.dumps(e.get('mentions', [])))
                    for e in entities if e.get('name') # Ensure name exists
                ]
                if entity_data:
                    cursor.executemany('''
                        INSERT INTO extracted_entities (run_id, name, type, mentions_json)
                        VALUES (?, ?, ?, ?)
                    ''', entity_data)
                    print(f"Saved {len(entity_data)} entities.")

            # Insert Risks
            risks = extracted.get("risks", [])
            if risks:
                risk_data = [
                    (run_id, r.get('description'), r.get('severity'),
                     json.dumps(r.get('related_entities', [])), json.dumps(r.get('source_urls', [])))
                    for r in risks if r.get('description') # Ensure description exists
                ]
                if risk_data:
                    cursor.executemany('''
                        INSERT INTO extracted_risks (run_id, description, severity, related_entities_json, source_urls_json)
                        VALUES (?, ?, ?, ?, ?)
                    ''', risk_data)
                    print(f"Saved {len(risk_data)} risks.")

            # Insert Relationships
            relationships = extracted.get("relationships", [])
            if relationships:
                rel_data = [
                    (run_id, rel.get('entity1_name'), rel.get('relationship_type'), rel.get('entity2_name'),
                     json.dumps(rel.get('context_urls', [])))
                    for rel in relationships if rel.get('entity1_name') and rel.get('relationship_type') and rel.get('entity2_name') # Ensure core fields exist
                ]
                if rel_data:
                     cursor.executemany('''
                        INSERT INTO extracted_relationships (run_id, entity1_name, relationship_type, entity2_name, context_urls_json)
                        VALUES (?, ?, ?, ?, ?)
                    ''', rel_data)
                     print(f"Saved {len(rel_data)} relationships.")

        conn.commit()
        conn.close()
        print("Database save complete.")

    except Exception as e:
        print(f"ERROR saving results to database (Run ID: {run_id}): {e}")
        # Optionally rollback if needed, though commit might have partially happened
        # if conn: conn.rollback()
        # if conn: conn.close()


# --- Main Orchestration Logic ---
def run_analysis(initial_query: str,
                 global_search_context: str = "global financial news and legal filings",
                 specific_search_context: str = "Baidu search in China for specific company supply chain info",
                 specific_country_code: str = 'cn',
                 max_global_results: int = 10,
                 max_specific_results: int = 10
                 ) -> Dict[str, Any]:
    """
    Main orchestration function for the AI Analyst Agent.
    Coordinates calls, includes enhanced error logging, and saves results to SQLite.
    """
    start_run_time = time.time()
    # Initialize results structure
    results = {
        "query": initial_query,
        "steps": [],
        "final_extracted_data": {"entities": [], "risks": [], "relationships": []},
        "wayback_results": [],
        "kg_update_status": "not_run",
        "run_duration_seconds": 0,
        "error": None,
    }

    # Initialize DB schema if needed (safe to call multiple times)
    _init_db()

    # Initialize Neo4j driver
    kg_driver_available = bool(knowledge_graph.get_driver()) # Check if driver initialized successfully
    if not kg_driver_available:
         print("WARNING: Failed to initialize Neo4j driver. KG updates will be skipped.")
         results["kg_update_status"] = "skipped_no_connection"

    try:
        # --- Step 1: Global Insights ---
        print(f"\n--- Running Step 1: Global Search ({initial_query}) ---")
        step1_start = time.time()
        global_raw_results = []
        global_search_source = "unknown"

        if config.GOOGLE_API_KEY and config.GOOGLE_CX:
            print("Using Google Official API...")
            global_raw_results = search_engines.search_google_official(initial_query, lang='en', num=max_global_results)
            global_search_source = 'google_api'
        if not global_raw_results and config.SERPAPI_KEY:
            print("Trying global search via SerpApi (Google)...")
            global_raw_results = search_engines.search_via_serpapi(initial_query, engine='google', country_code='us', lang_code='en', num=max_global_results)
            global_search_source = 'serpapi_google'
        elif not global_raw_results:
            print("Warning: No suitable global search engine configured or search failed.")

        global_search_results = [search_engines.standardize_result(r, global_search_source) for r in global_raw_results]
        print(f"Found {len(global_search_results)} global results.")

        global_extracted_data = {"entities": [], "risks": [], "relationships": []}
        if global_search_results and config.LLM_PROVIDER:
            global_extracted_data = nlp_processor.extract_data_from_results(global_search_results, global_search_context)
            print(f"Extracted {len(global_extracted_data.get('entities',[]))} entities, {len(global_extracted_data.get('risks',[]))} risks globally.")
        elif not config.LLM_PROVIDER:
             print("Warning: LLM Provider not configured, skipping global NLP extraction.")
        else:
             print("No global results to process for NLP.")

        results["steps"].append({
            "name": "Global Search & Extraction", "duration": round(time.time() - step1_start, 2),
            "search_results_count": len(global_search_results), "extracted_data": global_extracted_data, "status": "OK"
        })
        results["final_extracted_data"]["entities"].extend(global_extracted_data.get('entities',[]))
        results["final_extracted_data"]["risks"].extend(global_extracted_data.get('risks',[]))
        results["final_extracted_data"]["relationships"].extend(global_extracted_data.get('relationships',[]))


        # --- Step 2: Translate Keywords ---
        print(f"\n--- Running Step 2: Translating Keywords ---")
        step2_start = time.time()
        translated_keywords = []
        specific_query = initial_query
        if config.LLM_PROVIDER:
            translated_keywords = nlp_processor.translate_keywords_for_context(initial_query, specific_search_context)
            specific_query = translated_keywords[0] if translated_keywords else initial_query
            print(f"Using specific query: '{specific_query}' (Translated from '{initial_query}')")
        else:
            print("Warning: LLM Provider not configured, skipping keyword translation.")
        results["steps"].append({
            "name": "Keyword Translation", "duration": round(time.time() - step2_start, 2),
            "translated_keywords": translated_keywords, "status": "OK"
        })

        # --- Step 3: Specific Country Data ---
        print(f"\n--- Running Step 3: Specific Search ({specific_query} in {specific_country_code}) ---")
        step3_start = time.time()
        specific_raw_results = []
        specific_search_source = 'baidu' # Assuming Baidu for specific
        if config.SERPAPI_KEY:
            specific_raw_results = search_engines.search_via_serpapi(
                specific_query, engine='baidu', country_code=specific_country_code, num=max_specific_results
            )
        else:
            print("Warning: SerpApi key missing, skipping Baidu search.")

        specific_search_results = [search_engines.standardize_result(r, specific_search_source) for r in specific_raw_results]
        print(f"Found {len(specific_search_results)} specific results.")

        specific_extracted_data = {"entities": [], "risks": [], "relationships": []}
        if specific_search_results and config.LLM_PROVIDER:
             specific_extracted_data = nlp_processor.extract_data_from_results(specific_search_results, specific_search_context)
             print(f"Extracted {len(specific_extracted_data.get('entities',[]))} entities, {len(specific_extracted_data.get('risks',[]))} risks specifically.")
        elif not config.LLM_PROVIDER:
             print("Warning: LLM Provider not configured, skipping specific NLP extraction.")
        else:
             print("No specific results to process for NLP.")

        results["steps"].append({
            "name": "Specific Search & Extraction", "duration": round(time.time() - step3_start, 2),
            "search_results_count": len(specific_search_results), "extracted_data": specific_extracted_data, "status": "OK"
        })
        results["final_extracted_data"]["entities"].extend(specific_extracted_data.get('entities',[]))
        results["final_extracted_data"]["risks"].extend(specific_extracted_data.get('risks',[]))
        results["final_extracted_data"]["relationships"].extend(specific_extracted_data.get('relationships',[]))

        # --- Step 4: Wayback Machine ---
        print(f"\n--- Running Step 4: Wayback Machine Check ---")
        step4_start = time.time()
        all_urls = list(set([r.get('url') for r in global_search_results + specific_search_results if r.get('url')]))
        urls_to_check = all_urls[:5]
        wayback_checks = []
        print(f"Checking {len(urls_to_check)} URLs in Wayback Machine...")
        for url in urls_to_check:
            wayback_checks.append(search_engines.check_wayback_machine(url))
            time.sleep(0.5) # Avoid hitting API too rapidly
        results["wayback_results"] = wayback_checks
        results["steps"].append({
            "name": "Wayback Machine Check", "duration": round(time.time() - step4_start, 2),
            "urls_checked": len(urls_to_check), "status": "OK"
        })

        # --- Step 5: Update Knowledge Graph ---
        print(f"\n--- Running Step 5: Update Knowledge Graph ---")
        step5_start = time.time()
        update_success = False
        kg_status_message = results["kg_update_status"] # Keep initial status if driver failed

        if kg_driver_available: # Proceed only if driver was successfully initialized
            # De-duplicate extracted data before sending to KG
            unique_entities = list({f"{e.get('name', '')}_{e.get('type', '')}".lower(): e for e in results["final_extracted_data"]["entities"] if e.get('name')}.values())
            unique_relationships = list({f"{r.get('entity1_name', '')}_{r.get('relationship_type', '')}_{r.get('entity2_name', '')}".lower(): r for r in results["final_extracted_data"]["relationships"] if r.get('entity1_name') and r.get('relationship_type') and r.get('entity2_name')}.values())
            unique_risks = list({r.get('description', ''): r for r in results["final_extracted_data"]["risks"] if r.get('description')}.values())

            deduped_data = {
                "entities": unique_entities,
                "risks": unique_risks,
                "relationships": unique_relationships
            }
            if not any([unique_entities, unique_risks, unique_relationships]):
                 print("No unique data extracted to update knowledge graph.")
                 kg_status_message = "skipped_no_data"
            else:
                print(f"Attempting to update KG with {len(unique_entities)} entities, {len(unique_risks)} risks, {len(unique_relationships)} relationships.")
                update_success = knowledge_graph.update_knowledge_graph(deduped_data)
                kg_status_message = "success" if update_success else "error"
        else:
            print("Skipping KG update because driver is not available.")
            # kg_status_message remains 'skipped_no_connection' from above

        results["kg_update_status"] = kg_status_message
        results["steps"].append({
            "name": "Knowledge Graph Update", "duration": round(time.time() - step5_start, 2),
            "status": results["kg_update_status"]
        })

        # --- Step 6: Learning / Adapting (Placeholder) ---
        print("\n--- Step 6: Learning/Adapting (Deferred) ---")
        results["steps"].append({"name": "Learning/Adapting", "status": "deferred"})

    # --- Error Handling ---
    except Exception as e:
        print(f"\n--- Orchestration Error ---")
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else "No error message provided."
        error_traceback = traceback.format_exc()
        print(f"Error Type: {error_type}")
        print(f"Error Message: {error_msg}")
        print(f"Traceback:\n{error_traceback}")
        results["error"] = f"{error_type}: {error_msg}"

    finally:
        # Close Neo4j driver
        knowledge_graph.close_driver()
        # Calculate final duration
        results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
        # --- Save results to SQLite DB ---
        _save_analysis_to_db(results) # Call the save function here
        # --- End Save ---
        print(f"\n--- Analysis Complete ({results['run_duration_seconds']}s) ---")

    return results

# Example of how to run it directly (for local testing)
# if __name__ == "__main__":
#     test_query = "environmental regulations impact on semiconductor industry 2024"
#     analysis_result = run_analysis(test_query)
#     # import json # Already imported
#     print("\n--- FINAL RESULT ---")
#     print(json.dumps(analysis_result, indent=2, default=str))