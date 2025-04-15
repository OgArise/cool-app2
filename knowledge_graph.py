import time
from typing import Dict, List, Any
import search_engines
import nlp_processor
import knowledge_graph

def run_analysis(initial_query: str,
                 global_search_context: str = "global financial news and legal filings",
                 specific_search_context: str = "Baidu search in China for specific company supply chain info",
                 specific_country_code: str = 'cn',
                 max_global_results: int = 10,
                 max_specific_results: int = 10
                 ) -> Dict[str, Any]:
    """
    Main orchestration function for the AI Analyst Agent.
    """
    start_run_time = time.time()
    results = {
        "query": initial_query,
        "steps": [],
        "final_extracted_data": {"entities": [], "risks": [], "relationships": []},
        "wayback_results": [],
        "kg_update_status": "not_run",
        "run_duration_seconds": 0
    }

    try:
        # --- Step 1: Global Insights ---
        print(f"\n--- Running Step 1: Global Search ({initial_query}) ---")
        step1_start = time.time()
        # Prioritize official Google API, fallback to SerpApi if needed (adjust logic if desired)
        global_raw_results = search_engines.search_google_official(initial_query, lang='en', num=max_global_results)
        if not global_raw_results: # Fallback if official fails or isn't configured
             print("Trying global search via SerpApi...")
             global_raw_results = search_engines.search_via_serpapi(initial_query, engine='google', country_code='us', lang_code='en', num=max_global_results)

        global_search_results = [search_engines.standardize_result(r, 'google') for r in global_raw_results]
        print(f"Found {len(global_search_results)} global results.")

        global_extracted_data = nlp_processor.extract_data_from_results(global_search_results, global_search_context)
        print(f"Extracted {len(global_extracted_data['entities'])} entities, {len(global_extracted_data['risks'])} risks globally.")
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
        translated_keywords = nlp_processor.translate_keywords_for_context(initial_query, specific_search_context)
        # Use the first translated keyword for the specific search for simplicity
        specific_query = translated_keywords[0] if translated_keywords else initial_query
        print(f"Using specific query: '{specific_query}'")
        results["steps"].append({
            "name": "Keyword Translation",
            "duration": round(time.time() - step2_start, 2),
            "translated_keywords": translated_keywords
        })

        # --- Step 3: Specific Country Data (Baidu via SerpApi) ---
        print(f"\n--- Running Step 3: Specific Search ({specific_query} in {specific_country_code}) ---")
        step3_start = time.time()
        specific_raw_results = search_engines.search_via_serpapi(
            specific_query, engine='baidu', country_code=specific_country_code, num=max_specific_results
        )
        specific_search_results = [search_engines.standardize_result(r, 'baidu') for r in specific_raw_results]
        print(f"Found {len(specific_search_results)} specific results.")

        specific_extracted_data = nlp_processor.extract_data_from_results(specific_search_results, specific_search_context)
        print(f"Extracted {len(specific_extracted_data['entities'])} entities, {len(specific_extracted_data['risks'])} risks specifically.")
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
        # Example: Check top N URLs from combined results - choose relevant ones
        urls_to_check = [r['url'] for r in global_search_results[:2] + specific_search_results[:2] if r.get('url')]
        wayback_checks = []
        for url in urls_to_check:
            print(f"Checking Wayback for: {url}")
            wayback_checks.append(search_engines.check_wayback_machine(url))
            time.sleep(0.5) # Avoid hitting API too rapidly

        results["wayback_results"] = wayback_checks
        results["steps"].append({
            "name": "Wayback Machine Check",
            "duration": round(time.time() - step4_start, 2),
            "urls_checked": len(urls_to_check)
        })

        # --- Step 5: Update Knowledge Graph ---
        print(f"\n--- Running Step 5: Update Knowledge Graph ---")
        step5_start = time.time()
        # Optional: De-duplicate entities/relationships before sending to KG
        # Simple deduplication example (more robust needed for production)
        unique_entities = list({f"{e['name']}_{e['type']}": e for e in results["final_extracted_data"]["entities"]}.values())
        unique_relationships = list({f"{r['entity1']}_{r['relationship_type']}_{r['entity2']}": r for r in results["final_extracted_data"]["relationships"]}.values())
        unique_risks = list({r['description']: r for r in results["final_extracted_data"]["risks"]}.values())

        deduped_data = {
            "entities": unique_entities,
            "risks": unique_risks,
            "relationships": unique_relationships
        }

        update_success = knowledge_graph.update_knowledge_graph(deduped_data)
        results["kg_update_status"] = "success" if update_success else "error"
        results["steps"].append({
            "name": "Knowledge Graph Update",
            "duration": round(time.time() - step5_start, 2),
            "status": results["kg_update_status"]
        })

        # --- Step 6: Learning / Adapting (Placeholder) ---
        # This step requires significant ML infrastructure and is deferred.
        # You could add logging of results/feedback here for later analysis.
        print("\n--- Step 6: Learning/Adapting (Deferred) ---")
        results["steps"].append({"name": "Learning/Adapting", "status": "deferred"})

    except Exception as e:
        print(f"\n--- Orchestration Error ---")
        print(f"An error occurred during the analysis: {e}")
        results["error"] = str(e) # Add error information to results

    finally:
        knowledge_graph.close_driver() # Ensure DB connection is closed

    results["run_duration_seconds"] = round(time.time() - start_run_time, 2)
    print(f"\n--- Analysis Complete ({results['run_duration_seconds']}s) ---")
    return results