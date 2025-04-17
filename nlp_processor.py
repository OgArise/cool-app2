# nlp_processor.py

import json
from typing import List, Dict, Any, Optional
import time
import re # Import regex for cleaning

# Import config variables
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
    """Initializes LLM client, returns client/lib, type, and model name."""
    api_key = None; client_or_lib = None; client_type = None
    effective_model_name = model_name # Start with user-provided model

    if provider == "openai":
        api_key = OPENAI_API_KEY
        if not openai: raise ImportError("OpenAI library not installed.")
        if not api_key: raise ValueError("OpenAI API Key is missing.")
        try:
            client_or_lib = openai.OpenAI(api_key=api_key)
            client_type = "openai_compatible"
            print(f"Initialized OpenAI client for provider: {provider}")
        except Exception as e: print(f"ERROR init OpenAI client: {e}"); raise e

    elif provider == "openrouter":
        api_key = OPENROUTER_API_KEY
        if not openai: raise ImportError("OpenAI library not installed (for OpenRouter).")
        if not api_key: raise ValueError("OpenRouter API Key is missing.")
        try:
            client_or_lib = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL, default_headers=OPENROUTER_HEADERS)
            client_type = "openai_compatible"
            print(f"Initialized OpenAI client for provider: {provider} (Model: {model_name})")
        except Exception as e: print(f"ERROR init OpenRouter client: {e}"); raise e

    elif provider == "google_ai":
        api_key = GOOGLE_AI_API_KEY
        if not genai: raise ImportError("Google Generative AI library not installed.")
        if not api_key: raise ValueError("Google AI API Key is missing.")
        try:
            genai.configure(api_key=api_key)
            # Use the provided model name directly
            client_or_lib = genai.GenerativeModel(model_name=model_name)
            client_type = "google_ai"
            print(f"Initialized Google AI client for provider: {provider} (Model: {model_name})")
        except Exception as e: print(f"ERROR init Google AI client: {e}"); raise e
    else:
        raise ValueError(f"Unsupported LLM provider specified: {provider}")

    return client_or_lib, client_type, effective_model_name


# --- Keyword Generation with Fallback ---
def translate_keywords_for_context(original_query: str, target_context: str,
                                   llm_provider: str, llm_model: str) -> List[str]:
    """Generates ENGLISH keywords using the specified LLM."""
    print(f"\n--- Attempting Keyword Generation via {llm_provider} (Model: {llm_model}) ---")
    if not llm_provider or not llm_model: return [original_query]

    # ===> STRONGER PROMPT INSTRUCTION <===
    prompt = f"""You are an expert keyword generator. Given the query and context below, provide a list of 3-5 relevant ENGLISH search keywords suitable for the context.

    Initial Query: {original_query}
    Target Search Context: {target_context}

    IMPORTANT: Your entire response must contain ONLY the comma-separated list of keywords. Do NOT include any other text, explanation, or formatting."""
    # ===> END OF CHANGE <===

    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        raw_content = "" # Initialize raw_content

        if client_type == "openai_compatible":
            response = client_or_lib.chat.completions.create( model=model_name_used, messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=150 ) # Slightly lower temp
            raw_content = response.choices[0].message.content.strip()
        elif client_type == "google_ai":
            response = client_or_lib.generate_content(prompt)
            if not response.candidates: raise ValueError(f"Google AI response blocked. Feedback: {response.prompt_feedback}")
            raw_content = response.text.strip()
        else: raise ValueError("Internal error: Unknown client type.")

        print(f"\nRaw {llm_provider} Response (Keywords):\n>>>\n{raw_content}\n<<<") # Clearer logging

        # Attempt to clean potential conversational filler before splitting
        # Remove common conversational prefixes if they appear alone or at the start
        prefixes_to_remove = ["Sure, here are the keywords:", "Here are the keywords:", "Okay, here is the list:", "Here is the list:", "Of course! Here are the keywords:"]
        cleaned_content = raw_content
        for prefix in prefixes_to_remove:
            if cleaned_content.startswith(prefix):
                cleaned_content = cleaned_content[len(prefix):].strip()
                print(f"Cleaned prefix '{prefix}'")
                break # Only remove one prefix

        # Check if ONLY keywords remain (simple check: no newline after cleaning)
        if '\n' in cleaned_content:
             print("Warning: Detected multiple lines in keyword response after cleaning. Attempting split on first line.")
             cleaned_content = cleaned_content.split('\n')[0] # Take only the first line

        keywords = [kw.strip() for kw in cleaned_content.split(',') if kw.strip()]

        # Check if the result looks like keywords or still conversational
        if not keywords or any(word in cleaned_content.lower() for word in ["sure", "here", "keywords", "list", "okay", "provide", "prompt"]):
            print("Warning: LLM response for keywords still looks conversational or empty. Returning original query.")
            return [original_query]

        print(f"Parsed Keywords: {keywords}")
        return keywords

    except Exception as e:
        print(f"ERROR during keyword generation via {llm_provider}: {e}")
        return [original_query] # Fallback

# --- Entity / Risk Extraction with Fallback ---
# Schema definition remains the same
extraction_schema = { "type": "object", "properties": { "entities": { "type": "array", "description": "List of identified entities (companies, people, locations, organizations). Provide names in English.", "items": { "type": "object", "properties": { "name": {"type": "string"}, "type": {"type": "string", "enum": ["COMPANY", "PERSON", "LOCATION", "ORGANIZATION", "REGULATION", "OTHER"]}, "mentions": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "type"]} }, "risks": { "type": "array", "description": "List of potential risks identified (supply chain, compliance, financial, regulatory, etc.). Describe risks in English.", "items": { "type": "object", "properties": { "description": {"type": "string"}, "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]}, "related_entities": {"type": "array", "items": {"type": "string"}}, "source_urls": {"type": "array", "items": {"type": "string"}}}, "required": ["description"]} }, "relationships": { "type": "array", "description": "Identified relationships between entities (e.g., supplier, regulator). Use English names and relationship types.", "items": { "type": "object", "properties": { "entity1": {"type": "string"}, "relationship_type": {"type": "string"}, "entity2": {"type": "string"}, "context_urls": {"type": "array", "items": {"type": "string"}}}, "required": ["entity1", "relationship_type", "entity2"]} } }, "required": ["entities", "risks", "relationships"] }


def extract_data_from_results(search_results: List[Dict[str, Any]], extraction_context: str,
                              llm_provider: str, llm_model: str) -> Dict[str, List]:
    """Extracts structured data using the specified LLM, prioritizing JSON mode."""
    print(f"\n--- Attempting Data Extraction via {llm_provider} (Model: {llm_model}) ---")
    if not llm_provider or not llm_model: return {"entities": [], "risks": [], "relationships": []}
    if not search_results: return {"entities": [], "risks": [], "relationships": []}

    # Prepare context (same logic)
    context_text = f"Extraction Context: {extraction_context}\n\nRelevant Information:\n"
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

    # ===> SIMPLIFIED JSON PROMPT <===
    # Focus on telling the model the *task* and to *use JSON*, rather than embedding the full schema
    # The schema is still useful if the `response_format` parameter works
    prompt = f"""Analyze the following text snippets based on the extraction context: "{extraction_context}".
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

    Begin analysis of the following text:
    {context_text}"""
    # ===> END OF SIMPLIFIED PROMPT <===

    raw_content = None
    extracted_json = None
    api_error = None # To store error for logging

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
            # generation_config = genai.types.GenerationConfig(response_mime_type="application/json", temperature=0.1) # Might work for some models
            generation_config = genai.types.GenerationConfig(temperature=0.1)
            print("Attempting extraction with Google AI...")
            response = client_or_lib.generate_content(prompt, generation_config=generation_config)
            if not response.candidates: raise ValueError(f"Google AI response blocked. Feedback: {response.prompt_feedback}")
            raw_content = response.text
        else:
             raise ValueError("Internal error: Unknown client type.")

        # --- Attempt to parse the response ---
        print(f"\nRaw {llm_provider} Response (Extraction):\n>>>\n{raw_content}\n<<<")
        try:
            # Clean potential markdown wrappers/text before parsing
            cleaned_content = raw_content.strip()
            # Find the first '{' and the last '}'
            json_start = cleaned_content.find('{')
            json_end = cleaned_content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = cleaned_content[json_start:json_end]
                print("Attempting to parse extracted JSON block...")
                extracted_json = json.loads(json_str)
            else:
                # If no brackets found, maybe the whole thing is JSON (or junk)
                print("No JSON block markers found, attempting to parse raw content...")
                extracted_json = json.loads(cleaned_content) # This might fail

        except json.JSONDecodeError as json_e:
            print(f"ERROR: Failed to decode JSON from {llm_provider} response: {json_e}")
            api_error = json_e # Store error

    except Exception as e:
        print(f"ERROR during data extraction API call via {llm_provider}: {e}")
        api_error = e # Store error

    # --- Process Final Result ---
    validated_data = {"entities": [], "risks": [], "relationships": []} # Default empty
    if extracted_json is not None:
        # Validate structure
        validated_data = {
            "entities": extracted_json.get("entities", []) if isinstance(extracted_json.get("entities"), list) else [],
            "risks": extracted_json.get("risks", []) if isinstance(extracted_json.get("risks"), list) else [],
            "relationships": extracted_json.get("relationships", []) if isinstance(extracted_json.get("relationships"), list) else []
        }
        # Check if something was actually extracted
        if any(validated_data.values()):
             print("\nSuccessfully Parsed Extracted Data.")
        else:
             print("\nWarning: LLM returned valid JSON structure, but all data arrays were empty.")

    elif api_error: # If API call failed or JSON parsing failed
         print(f"\nERROR: LLM extraction failed. No valid JSON obtained. Error: {api_error}")
    else:
         print("\nERROR: Unknown state in LLM extraction. No JSON obtained.")


    print(f"\nFinal Parsed Extracted Data:\n{json.dumps(validated_data, indent=2)}")
    return validated_data


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

   print("\n--- Local Tests Complete ---")