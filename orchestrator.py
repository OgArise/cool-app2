import time
from typing import Dict, List, Any
import search_engines
import nlp_processor
import knowledge_graph
import config # Import config to potentially check if keys exist before calling

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
    Coordinates calls to search, NLP, Wayback, and Knowledge Graph modules.
    """
    start_run_time = time.time()
    # Initialize the results structure
    results = {
        "query": initial_query,
        "steps": [],
        "final_extracted_data": {"entities": [], "risks": [], "relationships": []},
        "wayback_results": [],
        "kg_update_status": "not_run",
        "run_duration_seconds": 0,
        "error": None # Field to store any errors
    }

    # Initialize Neo4j driver early - helps catch config errors sooner
    if not knowledge_graph.get_driver():
         print("ERROR: Failed to initialize Neo4j driver. KG updates will be skipped.")
         results["kg_update_status"] = "skipped_no_connection"
         # Depending on requirements, you might want to stop the whole process here
         # return results # Or continue without KG steps

    try:
        # --- Step 1: Global Insights ---
        print(f"\n--- Running Step 1: Global Search ({initial_query}) ---")
        step1_start = time.time()
        global_raw_results = []
        global_search_source = "unknown"

        # Try official Google API first if configured
        if config.GOOGLE_API_KEY and config.GOOGLE_CX:
            print("Using Google Official API...")
            global_raw_results = search_engines.search_google_official(initial_query, lang='en', num=max_global_results)
            global_search_source = 'google_api'
        # Fallback to SerpApi Google if official failed or isn't configured AND SerpApi is configured
        if not global_raw_results and config.SERPAPI_KEY:
            print("Trying global search via SerpApi (Google)...")
            global_raw_results = search_engines.search_via_serpapi(initial_query, engine='google', country_code='us', lang_code='en', num=max_global_results)
            global_search_source = 'serpapi_google'
        elif not global_raw_results:
            print("Warning: No suitable global search engine configured or search failed.")

        global_search_results = [search_engines.standardize_result(r, global_search_source) for r in global_raw_results]
        print(f"Found {len(global_search_results)} global results.")

        # Perform NLP extraction only if we have results and OpenAI key
        global_extracted_data = {"entities": [], "risks": [], "relationships": []}
        if global_search_results and config.OPENAI_API_KEY:
            global_extracted_data = nlp_processor.extract_data_from_results(global_search_results, global_search_context)
            print(f"Extracted {len(global_extracted_data['entities'])} entities, {len(global_extracted_data['risks'])} risks globally.")
        elif not config.OPENAI_API_KEY:
             print("Warning: OpenAI API Key missing, skipping global NLP extraction.")
        else:
             print("No global results to process for NLP.")


        results["steps"].append({
            "name": "Global Search & Extraction",
            "duration": round(time.time() - step1_start, 2),
            "search_results_count": len(global_search_results),
            "extracted_data": global_extracted_data
        })
        # Accumulate extracted data
        results["final_extracted_data"]["entities"].extend(global_extracted_data['entities'])
        results["final_extracted_data"]["risks"].extend(global_extracted_data['risks'])
        results["final_extracted_data"]["relationships"].extend(global_extracted_data['relationships'])


        # --- Step 2: Translate Keywords ---
        print(f"\n--- Running Step 2: Translating Keywords ---")
        step2_start = time.time()
        translated_keywords = []
        specific_query = initial_query # Default if translation fails

        if config.OPENAI_API_KEY:
            translated_keywords = nlp_processor.translate_keywords_for_context(initial_query, specific_search_context)
            specific_query = translated_keywords[0] if translated_keywords else initial_query # Use first or default
            print(f"Using specific query: '{specific_query}' (Translated from '{initial_query}')")
        else:
            print("Warning: OpenAI API Key missing, skipping keyword translation. Using original query.")


        results["steps"].append({
            "name": "Keyword Translation",
            "duration": round(time.time() - step2_start, 2),
            "translated_keywords": translated_keywords
        })

        # --- Step 3: Specific Country Data (Baidu via SerpApi) ---
        print(f"\n--- Running Step 3: Specific Search ({specific_query} in {specific_country_code}) ---")
        step3_start = time.time()
        specific_raw_results = []
        specific_search_source = 'baidu'

        if config.SERPAPI_KEY:
            specific_raw_results = search_engines.search_via_serpapi(
                specific_query, engine='baidu', country_code=specific_country_code, num=max_specific_results
            )
        else:
            print("Warning: SerpApi key missing, skipping Baidu search.")


        specific_search_results = [search_engines.standardize_result(r, specific_search_source) for r in specific_raw_results]
        print(f"Found {len(specific_search_results)} specific results.")

        # Perform NLP extraction
        specific_extracted_data = {"entities": [], "risks": [], "relationships": []}
        if specific_search_results and config.OPENAI_API_KEY:
             specific_extracted_data = nlp_processor.extract_data_from_results(specific_search_results, specific_search_context)
             print(f"Extracted {len(specific_extracted_data['entities'])} entities, {len(specific_extracted_data['risks'])} risks specifically.")
        elif not config.OPENAI_API_KEY:
             print("Warning: OpenAI API Key missing, skipping specific NLP extraction.")
        else:
             print("No specific results to process for NLP.")


        results["steps"].append({
            "name": "Specific Search & Extraction",
            "duration": round(time.time() - step3_start, 2),
            "search_results_count": len(specific_search_results),
            "extracted_data": specific_extracted_data
        })
         # Accumulate extracted data
        results["final_extracted_data"]["entities"].extend(specific_extracted_data['entities'])
        results["final_extracted_data"]["risks"].extend(specific_extracted_data['risks'])
        results["final_extracted_data"]["relationships"].extend(specific_extracted_data['relationships'])

        # --- Step 4: Wayback Machine ---
        print(f"\n--- Running Step 4: Wayback Machine Check ---")
        step4_start = time.time()
        # Example: Check top N URLs from combined results
        # Combine unique, valid URLs first
        all_urls = list(set([r.get('url') for r in global_search_results + specific_search_results if r.get('url')]))
        urls_to_check = all_urls[:5] # Limit checks to avoid long run times/API limits
        wayback_checks = []
        print(f"Checking {len(urls_to_check)} URLs in Wayback Machine...")
        for url in urls_to_check:
            wayback_checks.append(search_engines.check_wayback_machine(url))
            time.sleep(0.5) # Be nice to the API

        results["wayback_results"] = wayback_checks
        results["steps"].append({
            "name": "Wayback Machine Check",
            "duration": round(time.time() - step4_start, 2),
            "urls_checked": len(urls_to_check)
        })

        # --- Step 5: Update Knowledge Graph ---
        print(f"\n--- Running Step 5: Update Knowledge Graph ---")
        step5_start = time.time()
        update_success = False
        # Attempt KG update only if driver was initialized
        if knowledge_graph.driver:
            # De-duplicate extracted data before sending to KG
            # This is basic; production might need better ID generation or merging logic
            unique_entities = list({f"{e.get('name', '')}_{e.get('type', '')}".lower(): e for e in results["final_extracted_data"]["entities"] if e.get('name')}.values())
            unique_relationships = list({f"{r.get('entity1', '')}_{r.get('relationship_type', '')}_{r.get('entity2', '')}".lower(): r for r in results["final_extracted_data"]["relationships"] if r.get('entity1') and r.get('relationship_type') and r.get('entity2')}.values())
            unique_risks = list({r.get('description', ''): r for r in results["final_extracted_data"]["risks"] if r.get('description')}.values())

            deduped_data = {
                "entities": unique_entities,
                "risks": unique_risks,
                "relationships": unique_relationships
            }
            print(f"Attempting to update KG with {len(unique_entities)} entities, {len(unique_risks)} risks, {len(unique_relationships)} relationships.")
            update_success = knowledge_graph.update_knowledge_graph(deduped_data)
        else:
            print("Skipping KG update because driver is not available.")


        results["kg_update_status"] = "success" if update_success else ("skipped" if not knowledge_graph.driver else "error")
        results["steps"].append({
            "name": "Knowledge Graph Update",
            "duration": round(time.time() - step5_start, 2),
            "status": results["kg_update_status"]
        })

        # --- Step 6: Learning / Adapting (Placeholder) ---
        print("\n--- Step 6: Learning/Adapting (Deferred) ---")
        results["steps"].append({"name": "Learning/Adapting", "status": "deferred"})

    except Exception as e:
        print(f"\n--- Orchestration Error ---")
        print(f"An error occurred during the analysis: {e}")
        # Include traceback for debugging if possible/desired
        # import traceback
        # results["error_traceback"] = traceback.format_exc()
        results["error"] = str(e) # Add error information to results

    finally:
        # Ensure the Neo4j driver is closed when the analysis is done
        knowledge_graph.close_driver()

    results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
    print(f"\n--- Analysis Complete ({results['run_duration_seconds']}s) ---")
    return results