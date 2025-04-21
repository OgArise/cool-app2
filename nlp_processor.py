# nlp_processor.py

import json
from typing import List, Dict, Any, Optional
import time
import re

# Import config variables needed
from config import (
    GOOGLE_AI_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL, OPENROUTER_HEADERS,
    DEFAULT_GOOGLE_AI_MODEL, DEFAULT_OPENAI_MODEL, DEFAULT_OPENROUTER_MODEL
)

# Attempt to import LLM libraries
try: import openai
except ImportError: openai = None

try: import google.generativeai as genai
except ImportError: genai = None


# --- Helper to dynamically get client AND key ---
def _get_llm_client_and_model(provider: str, model_name: str):
    """
    Initializes LLM client, returns client/lib, type, and model name.
    Fetches API key from config based on provider.
    """
    api_key = None; client_or_lib = None; client_type = None
    effective_model_name = model_name

    print(f"Attempting to initialize LLM client for provider: {provider}...") # Added log

    if provider == "openai":
        api_key = OPENAI_API_KEY
        if not openai: raise ImportError("OpenAI library not installed.")
        if not api_key: raise ValueError("OpenAI API Key is missing in environment.")
        try:
            client_or_lib = openai.OpenAI(api_key=api_key)
            client_type = "openai_compatible"
            print(f"Initialized OpenAI client.")
        except Exception as e: print(f"ERROR init OpenAI client: {e}"); raise e

    elif provider == "openrouter":
        api_key = OPENROUTER_API_KEY
        if not openai: raise ImportError("OpenAI library not installed (for OpenRouter).")
        if not api_key: raise ValueError("OpenRouter API Key is missing.")
        try:
            client_or_lib = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL, default_headers=OPENROUTER_HEADERS)
            client_type = "openai_compatible"
            print(f"Initialized OpenRouter client (Model: {model_name})")
        except Exception as e: print(f"ERROR init OpenRouter client: {e}"); raise e

    elif provider == "google_ai":
        api_key = GOOGLE_AI_API_KEY
        if not genai: raise ImportError("Google Generative AI library not installed.")
        if not api_key: raise ValueError("Google AI API Key is missing.")
        try:
            genai.configure(api_key=api_key)
            client_or_lib = genai.GenerativeModel(model_name=model_name)
            client_type = "google_ai"
            print(f"Initialized Google AI client (Model: {model_name})")
        except Exception as e: print(f"ERROR init Google AI client: {e}"); raise e
    else:
        raise ValueError(f"Unsupported LLM provider specified: {provider}")

    if client_or_lib is None: # Add final check
        raise ConnectionError(f"Failed to create LLM client for provider {provider}")

    return client_or_lib, client_type, effective_model_name


# --- Keyword Generation ---
def translate_keywords_for_context(original_query: str, target_context: str,
                                   llm_provider: str, llm_model: str) -> List[str]:
    """Generates ENGLISH keywords using the specified LLM."""
    print(f"\n--- Attempting Keyword Generation via {llm_provider} (Model: {llm_model}) ---")
    if not llm_provider or not llm_model: return [original_query]

    prompt = f"""You are an expert keyword generator... IMPORTANT: Your entire response must contain ONLY the comma-separated list...""" # Keep improved prompt

    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        raw_content = ""
        if client_type == "openai_compatible": response = client_or_lib.chat.completions.create( model=model_name_used, messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=150 ); raw_content = response.choices[0].message.content.strip()
        elif client_type == "google_ai": response = client_or_lib.generate_content(prompt); raw_content = response.text.strip() if response.candidates else "" # Check candidates
        else: raise ValueError("Unknown client type.")

        print(f"\nRaw {llm_provider} Response (Keywords):\n>>>\n{raw_content}\n<<<")
        # Cleaning logic
        prefixes_to_remove = ["Sure, here are the keywords:", "Here are the keywords:", "Okay, here is the list:", "Here is the list:", "Of course! Here are the keywords:"]; cleaned_content = raw_content
        for prefix in prefixes_to_remove:
            if cleaned_content.startswith(prefix): cleaned_content = cleaned_content[len(prefix):].strip(); break
        if '\n' in cleaned_content: cleaned_content = cleaned_content.split('\n')[0]
        keywords = [kw.strip() for kw in cleaned_content.split(',') if kw.strip()]
        if not keywords or any(word in cleaned_content.lower() for word in ["sure", "here", "keywords", "list", "okay", "provide", "prompt"]): print("Warning: Keyword response looks conversational/empty."); return [original_query]
        print(f"Parsed Keywords: {keywords}"); return keywords
    except Exception as e: print(f"ERROR during keyword generation via {llm_provider}: {e}"); return [original_query]


# --- Entity / Risk Extraction ---
# Schema definition remains the same
extraction_schema = { "type": "object", "properties": { "entities": { "type": "array", "description": "List of identified entities (companies, people, locations, organizations). Provide names in English.", "items": { "type": "object", "properties": { "name": {"type": "string"}, "type": {"type": "string", "enum": ["COMPANY", "PERSON", "LOCATION", "ORGANIZATION", "REGULATION", "OTHER"]}, "mentions": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "type"]} }, "risks": { "type": "array", "description": "List of potential risks identified (supply chain, compliance, financial, regulatory, etc.). Describe risks in English.", "items": { "type": "object", "properties": { "description": {"type": "string"}, "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]}, "related_entities": {"type": "array", "items": {"type": "string"}}, "source_urls": {"type": "array", "items": {"type": "string"}}}, "required": ["description"]} }, "relationships": { "type": "array", "description": "Identified relationships between entities (e.g., supplier, regulator). Use English names and relationship types.", "items": { "type": "object", "properties": { "entity1": {"type": "string"}, "relationship_type": {"type": "string"}, "entity2": {"type": "string"}, "context_urls": {"type": "array", "items": {"type": "string"}}}, "required": ["entity1", "relationship_type", "entity2"]} } }, "required": ["entities", "risks", "relationships"] }

def extract_data_from_results(search_results: List[Dict[str, Any]], extraction_context: str,
                              llm_provider: str, llm_model: str,
                              focus_on_china: bool = False) -> Dict[str, List]: # Added focus parameter
    """
    Extracts structured data using the specified LLM.
    Can optionally focus extraction on Chinese entities/context.
    """
    print(f"\n--- Attempting Data Extraction via {llm_provider} (Model: {llm_model}) ---")
    if not llm_provider or not llm_model: return {"entities": [], "risks": [], "relationships": []}
    if not search_results: return {"entities": [], "risks": [], "relationships": []}

    # Prepare context (same logic)
    context_text = f"Extraction Context: {extraction_context}\n\nRelevant Information (from search results):\n"
    max_chars = 8000; char_count = len(context_text); added_snippets = 0
    print(f"\nPreparing context for extraction (Max Chars: {max_chars})...")
    for result in search_results:
        snippet = result.get('snippet', ''); title = result.get('title', ''); url = result.get('url', '')
        if snippet and not snippet.isspace():
            entry = f"\n---\nTitle: {title}\nURL: {url}\nSnippet: {snippet}\n"
            if char_count + len(entry) <= max_chars: context_text += entry; char_count += len(entry); added_snippets += 1
            else: print(f"Stopping context prep early ({char_count} chars)."); break
        else: print(f"Skipping empty snippet: {title} ({url})")
    if added_snippets == 0: print("No valid snippets for context."); return {"entities": [], "risks": [], "relationships": []}
    print(f"Prepared context using {added_snippets} snippets.")

    # --- Prepare Prompt with Schema Description and Optional Focus ---
    schema_string_for_prompt = json.dumps(extraction_schema, indent=2) # Keep schema for reference

    prompt_base = f"""Analyze the following text snippets based on the extraction context: "{extraction_context}".
Extract key entities, potential risks, and relationships found ONLY in the provided text.

IMPORTANT:
1. Provide all names, types, and descriptions in ENGLISH.
2. Your entire response MUST be a single valid JSON object containing three keys: "entities", "risks", and "relationships".
3. The value for each key must be an array of objects, even if empty ([]).
4. Follow these structures:
   - entities: `[{{ "name": "...", "type": "COMPANY|PERSON|LOCATION|ORGANIZATION|REGULATION|OTHER", "mentions": ["url1", ...] }}, ...]`
   - risks: `[{{ "description": "...", "severity": "LOW|MEDIUM|HIGH", "related_entities": ["name1", ...], "source_urls": ["url1", ...] }}, ...]`
   - relationships: `[{{ "entity1": "name1", "relationship_type": "TYPE_NAME", "entity2": "name2", "context_urls": ["url1", ...] }}, ...]`
5. Do NOT include any text before or after the JSON object. Do not use markdown formatting like ```json.
"""
    # Add focus instruction if needed
    if focus_on_china:
        prompt_focus = """
SPECIAL INSTRUCTION: Focus ONLY on extracting entities that are Chinese companies, Chinese government/regulatory bodies, locations within China, or international entities explicitly discussed in the context of their operations, compliance, or risks *within China*. Filter out entities primarily relevant only outside of China unless their Chinese operations are the specific topic."""
        prompt = prompt_base + prompt_focus + f"\n\nBegin analysis of the following text:\n{context_text}"
    else:
        prompt = prompt_base + f"\n\nBegin analysis of the following text:\n{context_text}"
    # --- End Prompt Preparation ---


    raw_content = None; extracted_json = None; api_error = None

    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        print(f"\n--- Sending Request to {llm_provider} for Data Extraction (Model: {model_name_used}) ---")

        if client_type == "openai_compatible":
            request_params = {"model": model_name_used, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "response_format": {"type": "json_object"}}
            try:
                print("Attempting extraction with JSON mode...")
                response = client_or_lib.chat.completions.create(**request_params)
                raw_content = response.choices[0].message.content
            except Exception as e_json_mode:
                 print(f"WARNING: {llm_provider} possibly failed JSON mode ({e_json_mode}). Retrying without forcing JSON format...")
                 del request_params["response_format"]
                 response = client_or_lib.chat.completions.create(**request_params)
                 raw_content = response.choices[0].message.content
        elif client_type == "google_ai":
            generation_config = genai.types.GenerationConfig(temperature=0.1)
            print("Attempting extraction with Google AI...")
            response = client_or_lib.generate_content(prompt, generation_config=generation_config)
            if not response.candidates: raise ValueError(f"Google AI response blocked. Feedback: {response.prompt_feedback}")
            raw_content = response.text
        else: raise ValueError("Unknown client type")

        # --- Attempt to parse the response ---
        print(f"\nRaw {llm_provider} Response (Extraction):\n>>>\n{raw_content}\n<<<")
        try:
            cleaned_content = raw_content.strip(); json_start = cleaned_content.find('{'); json_end = cleaned_content.rfind('}') + 1
            if json_start != -1 and json_end != -1: json_str = cleaned_content[json_start:json_end]
            else: json_str = cleaned_content # Fallback
            print("Attempting to parse extracted JSON block...")
            extracted_json = json.loads(json_str)
        except json.JSONDecodeError as json_e: print(f"ERROR: Failed to decode JSON from {llm_provider} response: {json_e}"); api_error = json_e # Store error

    except Exception as e: print(f"ERROR during data extraction API call via {llm_provider}: {e}"); api_error = e

    # --- Process Final Result ---
    validated_data = {"entities": [], "risks": [], "relationships": []} # Default empty
    if extracted_json is not None:
        validated_data = { "entities": extracted_json.get("entities", []) if isinstance(extracted_json.get("entities"), list) else [], "risks": extracted_json.get("risks", []) if isinstance(extracted_json.get("risks"), list) else [], "relationships": extracted_json.get("relationships", []) if isinstance(extracted_json.get("relationships"), list) else [] }
        if any(validated_data.values()): print("\nSuccessfully Parsed Extracted Data.")
        else: print("\nWarning: LLM returned valid JSON structure, but all data arrays were empty.")
    elif api_error: print(f"\nERROR: LLM extraction failed. No valid JSON obtained. Error: {api_error}")
    else: print("\nERROR: Unknown state in LLM extraction. No JSON obtained.")
    print(f"\nFinal Parsed Extracted Data:\n{json.dumps(validated_data, indent=2)}")
    return validated_data


# --- Ownership Extraction ---
def extract_ownership_relationships(parent_entity_name: str, related_entity_name: str,
                                    text_snippets: List[Dict[str, Any]],
                                    llm_provider: str, llm_model: str) -> Dict | None:
    """Uses LLM to analyze text snippets for evidence of ownership/control."""
    # ... (Keep this function exactly as defined in response #58 - it's independent of China focus) ...
    print(f"\n--- Analyzing ownership: '{parent_entity_name}' owning '{related_entity_name}'? ---")
    if not llm_provider or not llm_model: return None
    if not text_snippets: print("No text snippets provided for ownership check."); return None
    context = f"Analyze ONLY the following text snippets...where '{parent_entity_name}' owns/invests in '{related_entity_name}'.\n\nFocus *only* on stated ownership percentages or clear descriptions of control/significant influence...\n\nText Snippets:\n"
    max_chars = 7000; char_count = len(context); added_snippets = 0; relevant_snippets_for_prompt = []
    for snip in text_snippets:
        snippet_text = snip.get('snippet', ''); url = snip.get('url', 'N/A')
        if snippet_text and not snippet_text.isspace() and parent_entity_name.lower() in snippet_text.lower() and related_entity_name.lower() in snippet_text.lower():
            entry = f"---\nSource URL: {url}\nSnippet: {snippet_text}\n"
            if char_count + len(entry) <= max_chars: context += entry; relevant_snippets_for_prompt.append({"url": url, "snippet": snippet_text}); char_count += len(entry); added_snippets += 1
            else: break
    if added_snippets == 0: print(f"No snippets found mentioning both entities."); return None
    print(f"Prepared ownership context using {added_snippets} relevant snippets.")
    ownership_schema_desc = """{"relationship_type": "SUBSIDIARY | AFFILIATE | UNRELATED", "evidence_snippet": "Exact snippet text (max 200 chars) supporting conclusion, or 'No specific evidence'.", "source_url": "Source URL of snippet, or 'N/A'."}"""
    prompt = f"""{context}\nBased ONLY on the snippets, determine ownership relationship... Definitions:\n- SUBSIDIARY: >50% or control.\n- AFFILIATE: 5-50% or sig. influence.\n- UNRELATED: <5% or other relationship or no evidence.\nResponse MUST be ONLY valid JSON matching:\n```json\n{ownership_schema_desc}\n```"""
    raw_content = None; parsed_json = None; api_error = None
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        print(f"Sending ownership request to {llm_provider} ({model_name_used})...")
        if client_type == "openai_compatible":
             request_params = {"model": model_name_used, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "response_format": {"type": "json_object"}}
             try: response = client_or_lib.chat.completions.create(**request_params); raw_content = response.choices[0].message.content
             except Exception as e_json_mode: print(f"WARNING: {llm_provider} ownership failed JSON mode ({e_json_mode}). Retrying..."); del request_params["response_format"]; response = client_or_lib.chat.completions.create(**request_params); raw_content = response.choices[0].message.content
        elif client_type == "google_ai":
            generation_config = genai.types.GenerationConfig(temperature=0.1); response = client_or_lib.generate_content(prompt, generation_config=generation_config)
            if not response.candidates: raise ValueError(f"Google AI ownership response blocked: {response.prompt_feedback}")
            raw_content = response.text
        else: raise ValueError("Unknown client type")
        print(f"Raw {llm_provider} ownership response:\n>>>\n{raw_content}\n<<<")
        cleaned_content = raw_content.strip().replace("```json", "").replace("```", "").strip(); parsed_json = json.loads(cleaned_content)
    except json.JSONDecodeError as json_e: print(f"ERROR: Failed decode JSON ownership: {json_e}\nRaw: {raw_content}"); api_error = json_e
    except Exception as e: print(f"ERROR during ownership extraction call via {llm_provider}: {e}"); api_error = e
    if parsed_json and isinstance(parsed_json, dict) and parsed_json.get("relationship_type") in ["SUBSIDIARY", "AFFILIATE", "UNRELATED"]:
        print("Successfully parsed ownership info."); parsed_json.setdefault("evidence_snippet", "N/A"); parsed_json.setdefault("source_url", "N/A"); return parsed_json
    else: print(f"Failed valid ownership structure. Error: {api_error}"); return {"relationship_type": "UNRELATED", "evidence_snippet": f"LLM Error/Invalid JSON: {api_error}", "source_url": "N/A"} if raw_content else None


# --- Analysis Summary Generation ---
def generate_analysis_summary(extracted_data: Dict, query: str, exposures_count: int,
                              llm_provider: str, llm_model: str) -> str:
    """Generates a concise summary of the analysis findings."""
    # ... (keep this function exactly as defined in response #57) ...
    print(f"\n--- Attempting Analysis Summary via {llm_provider} (Model: {llm_model}) ---");
    if not llm_provider or not llm_model: return "Summary generation skipped: Missing LLM config."
    summary_context = f"Original Query: {query}\n\nAnalysis Findings:\n"; entities = extracted_data.get("entities", []); risks = extracted_data.get("risks", []); relationships = extracted_data.get("relationships", [])
    if entities: summary_context += f"- Key Entities ({len(entities)}): " + ", ".join([f"{e.get('name')} ({e.get('type')})" for e in entities[:5]]) + ("..." if len(entities)>5 else "") + "\n"
    if risks: high_risks = [r for r in risks if r.get('severity') == 'HIGH']; med_risks = [r for r in risks if r.get('severity') == 'MEDIUM']; summary_context += f"- Potential Risks ({len(risks)}): ";
    if high_risks: summary_context += f"{len(high_risks)} High (e.g., '{high_risks[0].get('description')[:80]}...'); ";
    if med_risks: summary_context += f"{len(med_risks)} Medium (e.g., '{med_risks[0].get('description')[:80]}...'); "; summary_context += "\n"
    if relationships: summary_context += f"- Key Relationships ({len(relationships)}): " + "; ".join([f"{rel.get('entity1')} {rel.get('relationship_type')} {rel.get('entity2')}" for rel in relationships[:3]]) + ("..." if len(relationships)>3 else "") + "\n"
    if exposures_count > 0: summary_context += f"- Supply Chain Exposures Identified: {exposures_count}\n"
    if not any([entities, risks, relationships]): return "No significant data extracted to summarize."
    prompt = f"""You are a professional geopolitical and financial analyst summarizing research findings... Make it clear, direct, and easy to understand...\n\n{summary_context}\n\nGenerate ONLY the summary paragraph."""
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model); raw_content = ""
        print(f"Sending summary request to {llm_provider} ({model_name_used})...")
        if client_type == "openai_compatible": response = client_or_lib.chat.completions.create( model=model_name_used, messages=[{"role": "user", "content": prompt}], temperature=0.6, max_tokens=300 ); raw_content = response.choices[0].message.content.strip()
        elif client_type == "google_ai": safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in [ "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT", ] ]; generation_config = genai.types.GenerationConfig(temperature=0.6, max_output_tokens=300); response = client_or_lib.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings); raw_content = response.text.strip() if response.candidates else ""
        else: raise ValueError("Unknown client type")
        print(f"\nRaw {llm_provider} Response (Summary):\n>>>\n{raw_content}\n<<<")
        cleaned_summary = raw_content; prefixes_to_remove = ["Okay, here's a summary:", "Here is a summary:", "Summary:", "Based on the findings:", "Based on the analysis:", "Here's a summary:"]
        for prefix in prefixes_to_remove:
             if cleaned_summary.lower().startswith(prefix.lower()): cleaned_summary = cleaned_summary[len(prefix):].strip()
        return cleaned_summary if cleaned_summary else "LLM returned an empty summary."
    except Exception as e: print(f"ERROR during summary generation via {llm_provider}: {e}"); return f"Could not generate summary due to error: {type(e).__name__}"


# --- Local Testing Block ---
if __name__ == "__main__":
    # ... (keep local testing block as before - it will now test the updated functions) ...
    pass # Add pass or keep previous test calls


# --- Local Testing Block ---
if __name__ == "__main__":
   print("\n--- Running Local NLP Processor Tests ---")
   print("NOTE: Local testing requires LLM API keys in .env and will use the configured keys.")
   # Determine which provider to test based on available keys in config
   provider_to_test = None
   model_to_test = None
   if config.GOOGLE_AI_API_KEY:
       provider_to_test = "google_ai"; model_to_test = config.DEFAULT_GOOGLE_AI_MODEL; print("--> Testing with Google AI")
   elif config.OPENAI_API_KEY:
       provider_to_test = "openai"; model_to_test = config.DEFAULT_OPENAI_MODEL; print("--> Testing with OpenAI")
   elif config.OPENROUTER_API_KEY:
       provider_to_test = "openrouter"; model_to_test = config.DEFAULT_OPENROUTER_MODEL; print("--> Testing with OpenRouter")

   if not provider_to_test:
       print("No LLM API keys configured in .env for testing. Exiting.")
   else:
       # --- Test Keyword Translation ---
       print("\nTesting Keyword Translation...")
       sample_query_kw = "supply chain compliance issues 2023"
       sample_context_kw = "Baidu search in China for specific company supply chain info"
       translated_keywords_test = translate_keywords_for_context(sample_query_kw, sample_context_kw, provider_to_test, model_to_test)
       print(f"\nTranslated Keywords Result: {translated_keywords_test}")

       # --- Test Data Extraction ---
       print("\nTesting Data Extraction...")
       sample_results_test = [
           {"title": "New Supply Chain Regulations Impact Global Trade - GovReport", "url": "https://example.gov/report-2023-supply-chain", "snippet": "The Regulatory Body C announced new compliance measures affecting manufacturers like Corporation A, especially those operating in Manufacturing Hub B. Financial penalties may apply from 2024.", "source": "google", "published_date": "2023-11-01"},
           {"title": "Corporation A Faces Scrutiny Over Supplier Ethics - Finance News", "url": "https://example.finance/corp-a-ethics", "snippet": "Concerns are rising about Corporation A's sourcing from suppliers in Manufacturing Hub B, potentially violating international labor standards. This poses a significant reputational risk.", "source": "google", "published_date": "2023-10-15"},
           {"title": "Baidu Search Results Snippet (Example)", "url": "https://example.baidu.com/search-result-3", "snippet": "Analysis shows shifting trade patterns in Manufacturing Hub B. Regulatory Body C is monitoring closely. Investors are watching Corporation A.", "source": "baidu", "published_date": "2023-12-01"},
           {"title": "Empty Snippet Example", "url": "https://example.com/empty", "snippet": " ", "source": "google", "published_date": "2023-09-01"}
       ]
       sample_extraction_context_test = "supply chain risks and regulations involving Corporation A and Manufacturing Hub B"
       extracted_data_test = extract_data_from_results(sample_results_test, sample_extraction_context_test, provider_to_test, model_to_test)
       print("\nFinal Extracted Data Result:")
       print(json.dumps(extracted_data_test, indent=2))

   # ===> ADD Local Test for Ownership Extraction <===
   print("\nTesting Ownership Extraction...")
   provider_to_test = None; model_to_test = None # Determine provider as before
   if config.GOOGLE_AI_API_KEY: provider_to_test = "google_ai"; model_to_test = config.DEFAULT_GOOGLE_AI_MODEL; print("--> Using Google AI")
   elif config.OPENAI_API_KEY: provider_to_test = "openai"; model_to_test = config.DEFAULT_OPENAI_MODEL; print("--> Using OpenAI")
   elif config.OPENROUTER_API_KEY: provider_to_test = "openrouter"; model_to_test = config.DEFAULT_OPENROUTER_MODEL; print("--> Using OpenRouter")

   if provider_to_test:
       sample_ownership_snippets = [
           {'url': 'http://example.com/report1', 'snippet': 'In 2022, ParentCorp acquired a 60% controlling stake in SubCo Inc.'},
           {'url': 'http://example.com/news1', 'snippet': 'ParentCorp holds a significant minority investment (approx 30%) in AffiliateCorp.'},
           {'url': 'http://example.com/report2', 'snippet': 'ParentCorp and Unrelated Inc entered into a strategic partnership.'},
           {'url': 'http/example.com/sec', 'snippet': 'Exhibit 21 lists SubCo Inc as a subsidiary.'}
       ]
       ownership_result = extract_ownership_relationships(
           parent_entity_name="ParentCorp",
           related_entity_name="SubCo Inc",
           text_snippets=sample_ownership_snippets,
           llm_provider=provider_to_test,
           llm_model=model_to_test
       )
       print("\nOwnership Result (ParentCorp owning SubCo Inc):")
       print(json.dumps(ownership_result, indent=2))

       ownership_result_2 = extract_ownership_relationships(
           parent_entity_name="ParentCorp",
           related_entity_name="AffiliateCorp",
           text_snippets=sample_ownership_snippets,
           llm_provider=provider_to_test,
           llm_model=model_to_test
       )
       print("\nOwnership Result (ParentCorp owning AffiliateCorp):")
       print(json.dumps(ownership_result_2, indent=2))

       ownership_result_3 = extract_ownership_relationships(
           parent_entity_name="ParentCorp",
           related_entity_name="Unrelated Inc",
           text_snippets=sample_ownership_snippets,
           llm_provider=provider_to_test,
           llm_model=model_to_test
       )
       print("\nOwnership Result (ParentCorp owning Unrelated Inc):")
       print(json.dumps(ownership_result_3, indent=2))
   else:
       print("Skipping ownership tests - No LLM configured.")

   print("\n--- Local Tests Complete ---")