# nlp_processor.py

import json
from typing import List, Dict, Any, Optional
import time

# Import specific config variables needed
from config import (
    GOOGLE_AI_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL, OPENROUTER_HEADERS
)

# Attempt to import LLM libraries
try: import openai
except ImportError: openai = None

try: import google.generativeai as genai
except ImportError: genai = None

# --- Helper to dynamically get client AND key ---
def _get_llm_client_and_model(provider: str, model_name: str):
    """
    Initializes and returns the appropriate LLM client/library and model name,
    fetching the API key from config based on the provider.
    """
    api_key = None
    client_or_lib = None
    client_type = None # 'openai_compatible' or 'google_ai'

    if provider == "openai":
        api_key = OPENAI_API_KEY
        if not openai: raise ImportError("OpenAI library not installed.")
        if not api_key: raise ValueError("OpenAI API Key is missing in environment.")
        try:
            client_or_lib = openai.OpenAI(api_key=api_key)
            client_type = "openai_compatible"
            print(f"Initialized OpenAI client for provider: {provider}")
        except Exception as e: print(f"ERROR init OpenAI client: {e}"); raise e

    elif provider == "openrouter":
        api_key = OPENROUTER_API_KEY
        if not openai: raise ImportError("OpenAI library not installed (required for OpenRouter).")
        if not api_key: raise ValueError("OpenRouter API Key is missing in environment.")
        try:
            client_or_lib = openai.OpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
                default_headers=OPENROUTER_HEADERS
            )
            client_type = "openai_compatible"
            print(f"Initialized OpenAI client for provider: {provider} (Model: {model_name})")
        except Exception as e: print(f"ERROR init OpenRouter client: {e}"); raise e

    elif provider == "google_ai":
        api_key = GOOGLE_AI_API_KEY
        if not genai: raise ImportError("Google Generative AI library not installed.")
        if not api_key: raise ValueError("Google AI API Key is missing in environment.")
        try:
            genai.configure(api_key=api_key)
            # Create the specific model instance here
            client_or_lib = genai.GenerativeModel(model_name=model_name)
            client_type = "google_ai"
            print(f"Initialized Google AI client for provider: {provider} (Model: {model_name})")
        except Exception as e: print(f"ERROR init Google AI client: {e}"); raise e

    else:
        raise ValueError(f"Unsupported LLM provider specified: {provider}")

    return client_or_lib, client_type, model_name # Return client/lib and model name

# --- Keyword Generation with Dynamic Client ---
def translate_keywords_for_context(original_query: str, target_context: str,
                                   llm_provider: str, llm_model: str) -> List[str]: # Removed api_key arg
    """
    Generates ENGLISH keywords using the dynamically specified LLM (fetches key from config).
    """
    print(f"\n--- Attempting Keyword Generation via {llm_provider} (Model: {llm_model}) ---")
    if not llm_provider or not llm_model:
        print("Skipping keyword generation: Missing LLM provider or model name.")
        return [original_query]

    prompt = f"""You are an expert market analyst... (rest of prompt same)""" # Keep prompt content

    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)

        if client_type == "openai_compatible":
            response = client_or_lib.chat.completions.create(
                model=model_name_used,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5, max_tokens=150
            )
            raw_content = response.choices[0].message.content.strip()
        elif client_type == "google_ai":
            response = client_or_lib.generate_content(prompt) # client_or_lib is the model instance now
            if not response.candidates: raise ValueError(f"Google AI response blocked. Feedback: {response.prompt_feedback}")
            raw_content = response.text.strip()
        else:
            raise ValueError("Internal error: Unknown client type.") # Should not happen

        print(f"\nRaw {llm_provider} Response (Keywords):\n{raw_content}")
        keywords = [kw.strip() for kw in raw_content.split(',') if kw.strip()]
        print(f"Parsed Keywords: {keywords}")
        return keywords if keywords else [original_query]

    except Exception as e:
        print(f"ERROR during keyword generation via {llm_provider}: {e}")
        return [original_query] # Fallback

# --- Entity / Risk Extraction with Dynamic Client ---
# Schema definition remains the same
extraction_schema = {
   # ...(schema definition remains the same)...
    "type": "object", "properties": { "entities": { "type": "array", "description": "List of identified entities (companies, people, locations, organizations). Provide names in English.", "items": { "type": "object", "properties": { "name": {"type": "string"}, "type": {"type": "string", "enum": ["COMPANY", "PERSON", "LOCATION", "ORGANIZATION", "REGULATION", "OTHER"]}, "mentions": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "type"]} }, "risks": { "type": "array", "description": "List of potential risks identified (supply chain, compliance, financial, regulatory, etc.). Describe risks in English.", "items": { "type": "object", "properties": { "description": {"type": "string"}, "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]}, "related_entities": {"type": "array", "items": {"type": "string"}}, "source_urls": {"type": "array", "items": {"type": "string"}}}, "required": ["description"]} }, "relationships": { "type": "array", "description": "Identified relationships between entities (e.g., supplier, regulator). Use English names and relationship types.", "items": { "type": "object", "properties": { "entity1": {"type": "string"}, "relationship_type": {"type": "string"}, "entity2": {"type": "string"}, "context_urls": {"type": "array", "items": {"type": "string"}}}, "required": ["entity1", "relationship_type", "entity2"]} } }, "required": ["entities", "risks", "relationships"]
}

def extract_data_from_results(search_results: List[Dict[str, Any]], extraction_context: str,
                              llm_provider: str, llm_model: str) -> Dict[str, List]: # Removed api_key arg
    """
    Extracts entities, risks, relationships using the dynamically specified LLM (fetches key from config).
    """
    print(f"\n--- Attempting Data Extraction via {llm_provider} (Model: {llm_model}) ---")
    if not llm_provider or not llm_model:
        print("Skipping data extraction: Missing LLM provider or model name.")
        return {"entities": [], "risks": [], "relationships": []}
    if not search_results:
        print("Skipping data extraction: No search results provided.")
        return {"entities": [], "risks": [], "relationships": []}

    # Prepare context (same logic)
    context_text = f"Extraction Context: {extraction_context}\n\nRelevant Information (from search results):\n"
    max_chars = 8000; char_count = len(context_text); added_snippets = 0
    print(f"\nPreparing context for extraction (Max Chars: {max_chars})...")
    for result in search_results:
        snippet = result.get('snippet', ''); title = result.get('title', ''); url = result.get('url', '')
        if snippet and not snippet.isspace():
            entry = f"\n---\nTitle: {title}\nURL: {url}\nSnippet: {snippet}\n"
            if char_count + len(entry) <= max_chars: context_text += entry; char_count += len(entry); added_snippets += 1
            else: print(f"Stopping context prep early due to char limit ({char_count} chars)."); break
        else: print(f"Skipping result with empty snippet: {title} ({url})")
    if added_snippets == 0: print("No valid snippets for context."); return {"entities": [], "risks": [], "relationships": []}
    print(f"Prepared context using {added_snippets} snippets.")

    # Prepare prompt with schema (same logic)
    schema_string_for_prompt = json.dumps(extraction_schema, indent=2)
    prompt = f"""Based ONLY on the provided text snippets below... (rest of prompt is the same)... Begin analysis of the following text:\n{context_text}"""

    raw_content = None
    extracted_json = None

    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)

        if client_type == "openai_compatible":
            # Attempt JSON mode with OpenAI/OpenRouter compatible client
            request_params = {"model": model_name_used, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "response_format": {"type": "json_object"}}
            try:
                response = client_or_lib.chat.completions.create(**request_params)
                raw_content = response.choices[0].message.content
            except Exception as e_json_mode: # Catch potential error if JSON mode unsupported
                 print(f"WARNING: {llm_provider} possibly failed JSON mode ({e_json_mode}). Retrying without forcing JSON format...")
                 del request_params["response_format"]
                 response = client_or_lib.chat.completions.create(**request_params)
                 raw_content = response.choices[0].message.content

        elif client_type == "google_ai":
            # generation_config = genai.types.GenerationConfig(response_mime_type="application/json", temperature=0.1)
            generation_config = genai.types.GenerationConfig(temperature=0.1)
            response = client_or_lib.generate_content(prompt, generation_config=generation_config) # client_or_lib is the model instance
            if not response.candidates: raise ValueError(f"Google AI response blocked. Feedback: {response.prompt_feedback}")
            raw_content = response.text
        else:
             raise ValueError("Internal error: Unknown client type.")

        # --- Attempt to parse the response ---
        print(f"\nRaw {llm_provider} Response (Extraction):\n{raw_content}")
        try:
            # Clean potential markdown wrappers before parsing
            cleaned_content = raw_content.strip()
            if cleaned_content.startswith("```json"): cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"): cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
            extracted_json = json.loads(cleaned_content)
        except json.JSONDecodeError as json_e:
            print(f"ERROR: Failed to decode JSON from {llm_provider} response: {json_e}")
            print(f"Raw content was: {raw_content}")
            return {"entities": [], "risks": [], "relationships": []}

    except Exception as e:
        print(f"ERROR during data extraction API call via {llm_provider}: {e}")
        return {"entities": [], "risks": [], "relationships": []}

    # --- Process successfully parsed JSON ---
    if extracted_json is None:
         print("ERROR: Failed to obtain valid JSON data after API calls.")
         return {"entities": [], "risks": [], "relationships": []}

    validated_data = {
        "entities": extracted_json.get("entities", []) if isinstance(extracted_json.get("entities"), list) else [],
        "risks": extracted_json.get("risks", []) if isinstance(extracted_json.get("risks"), list) else [],
        "relationships": extracted_json.get("relationships", []) if isinstance(extracted_json.get("relationships"), list) else []
    }
    print(f"\nParsed Extracted Data:\n{json.dumps(validated_data, indent=2)}")
    return validated_data


# --- Local Testing Block (Needs modification to work standalone now) ---
if __name__ == "__main__":
   print("\n--- Running Local NLP Processor Tests ---")
   print("NOTE: Local testing requires LLM API keys in .env and will use the configured keys.")

   # Example: Test Keyword Translation (will use keys from .env)
   print("\nTesting Keyword Translation...")
   if config.GOOGLE_AI_API_KEY: # Test Google if key exists
       print("--> Using Google AI")
       kw_test = translate_keywords_for_context("supply chain compliance 2023", "Baidu search in China", "google_ai", config.GOOGLE_AI_MODEL_NAME)
       print(f"Google AI Keywords Result: {kw_test}")
   elif config.OPENROUTER_API_KEY: # Test OpenRouter if key exists
       print("--> Using OpenRouter")
       kw_test = translate_keywords_for_context("supply chain compliance 2023", "Baidu search in China", "openrouter", config.DEFAULT_OPENROUTER_MODEL)
       print(f"OpenRouter Keywords Result: {kw_test}")
   elif config.OPENAI_API_KEY: # Test OpenAI if key exists
        print("--> Using OpenAI")
        kw_test = translate_keywords_for_context("supply chain compliance 2023", "Baidu search in China", "openai", config.DEFAULT_OPENAI_MODEL)
        print(f"OpenAI Keywords Result: {kw_test}")
   else:
       print("No LLM API key configured in .env for testing.")


   # Example: Test Data Extraction (will use keys from .env)
   print("\nTesting Data Extraction...")
   sample_results_test = [ {"title": "Corp A Fined", "url": "...", "snippet": "Regulator Z fined Corporation A for compliance violations in Hub B."} ] # Minimal sample
   sample_extraction_context_test = "compliance risks for Corporation A"

   provider_to_test = None
   model_to_test = None
   if config.GOOGLE_AI_API_KEY:
       provider_to_test = "google_ai"; model_to_test = config.GOOGLE_AI_MODEL_NAME; print("--> Using Google AI")
   elif config.OPENROUTER_API_KEY:
       provider_to_test = "openrouter"; model_to_test = config.DEFAULT_OPENROUTER_MODEL; print("--> Using OpenRouter")
   elif config.OPENAI_API_KEY:
       provider_to_test = "openai"; model_to_test = config.DEFAULT_OPENAI_MODEL; print("--> Using OpenAI")

   if provider_to_test:
       extracted_data_test = extract_data_from_results(sample_results_test, sample_extraction_context_test, provider_to_test, model_to_test)
       print("\nFinal Extracted Data Result:")
       print(json.dumps(extracted_data_test, indent=2))
   else:
        print("No LLM API key configured in .env for testing.")


   print("\n--- Local Tests Complete ---")