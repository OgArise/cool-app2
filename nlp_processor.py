# nlp_processor.py

import openai
import json
import config # Import updated config
from typing import List, Dict, Any

# --- Initialize LLM Client based on config ---
client = None
llm_model_name = None
llm_provider_name = "None" # For logging

if config.LLM_PROVIDER == "openrouter":
    try:
        # CORRECTED INITIALIZATION: Pass default_headers directly
        client = openai.OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            default_headers=config.OPENROUTER_HEADERS # Pass headers here
        )

        llm_model_name = config.OPENROUTER_MODEL_NAME
        llm_provider_name = "OpenRouter"
        print(f"OpenAI client configured for OpenRouter (Model: {llm_model_name}).")
    except Exception as e:
        client = None # Ensure client is None if init fails
        print(f"ERROR: Failed to initialize OpenAI client for OpenRouter: {e}")
elif config.LLM_PROVIDER == "openai":
    try:
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        llm_model_name = config.OPENAI_MODEL_NAME
        llm_provider_name = "OpenAI"
        print(f"OpenAI client configured directly (Model: {llm_model_name}).")
    except Exception as e:
        client = None # Ensure client is None if init fails
        print(f"ERROR: Failed to initialize direct OpenAI client: {e}")
else:
    print("Warning: LLM client not initialized due to missing API key (OpenRouter or OpenAI).")


# --- Keyword Generation ---
def translate_keywords_for_context(original_query: str, target_context: str) -> List[str]:
    """
    Generates relevant ENGLISH keywords for a given context, based on an initial query,
    using the configured LLM provider and model.
    """
    if not client or not llm_model_name:
        print("Skipping keyword generation: LLM client not available.")
        return [original_query]

    prompt = f"""You are an expert market analyst generating ENGLISH search keywords.
Given an initial query and a target search context (like a specific region or search engine), generate a concise list of 3-5 relevant ENGLISH search keywords suitable for finding information within that specific context.

Initial Query: {original_query}
Target Search Context: {target_context}

Output ONLY the comma-separated list of generated ENGLISH keywords, with no other explanation or preamble."""

    print(f"\n--- Sending Request to {llm_provider_name} for Keyword Generation (Model: {llm_model_name}) ---")
    print(f"Prompt:\n{prompt}")

    try:
        response = client.chat.completions.create(
            model=llm_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )
        raw_content = response.choices[0].message.content.strip()
        print(f"\nRaw {llm_provider_name} Response (Keywords):\n{raw_content}")

        keywords = [kw.strip() for kw in raw_content.split(',') if kw.strip()]
        print(f"Parsed Keywords: {keywords}")
        return keywords if keywords else [original_query]
    except Exception as e:
        print(f"ERROR during keyword generation via {llm_provider_name}: {e}")
        return [original_query]

# --- Entity / Risk Extraction ---
extraction_schema = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array", "description": "List of identified entities (companies, people, locations, organizations). Provide names in English.",
            "items": { "type": "object", "properties": { "name": {"type": "string"}, "type": {"type": "string", "enum": ["COMPANY", "PERSON", "LOCATION", "ORGANIZATION", "REGULATION", "OTHER"]}, "mentions": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "type"]}
        },
        "risks": {
            "type": "array", "description": "List of potential risks identified (supply chain, compliance, financial, regulatory, etc.). Describe risks in English.",
            "items": { "type": "object", "properties": { "description": {"type": "string"}, "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]}, "related_entities": {"type": "array", "items": {"type": "string"}}, "source_urls": {"type": "array", "items": {"type": "string"}}}, "required": ["description"]}
        },
        "relationships": {
            "type": "array", "description": "Identified relationships between entities (e.g., supplier, regulator). Use English names and relationship types.",
            "items": { "type": "object", "properties": { "entity1": {"type": "string"}, "relationship_type": {"type": "string"}, "entity2": {"type": "string"}, "context_urls": {"type": "array", "items": {"type": "string"}}}, "required": ["entity1", "relationship_type", "entity2"]}
        }
    },
    "required": ["entities", "risks", "relationships"]
}

def extract_data_from_results(search_results: List[Dict[str, Any]], extraction_context: str) -> Dict[str, List]:
    """
    Extracts entities, risks, and relationships using the configured LLM.
    Outputs text fields in English. Uses JSON mode if supported, otherwise relies on prompt.
    """
    if not client or not llm_model_name:
        print("Skipping data extraction: LLM client not available.")
        return {"entities": [], "risks": [], "relationships": []}
    if not search_results:
        print("Skipping data extraction: No search results provided.")
        return {"entities": [], "risks": [], "relationships": []}

    # Prepare context
    context_text = f"Extraction Context: {extraction_context}\n\nRelevant Information (from search results):\n"
    max_chars = 8000
    char_count = len(context_text)
    print(f"\nPreparing context for extraction (Max Chars: {max_chars})...")
    added_snippets = 0
    for result in search_results:
        snippet = result.get('snippet', '')
        title = result.get('title', '')
        url = result.get('url', '')
        if snippet and not snippet.isspace():
            entry = f"\n---\nTitle: {title}\nURL: {url}\nSnippet: {snippet}\n"
            if char_count + len(entry) <= max_chars:
                context_text += entry
                char_count += len(entry)
                added_snippets += 1
            else:
                print(f"Stopping context preparation early due to character limit ({char_count} chars).")
                break
        else:
            print(f"Skipping result with empty snippet: {title} ({url})")

    if added_snippets == 0:
        print("No valid snippets found to prepare context for LLM extraction.")
        return {"entities": [], "risks": [], "relationships": []}
    print(f"Prepared context using {added_snippets} snippets.")

    # Prepare prompt with schema embedded
    schema_string_for_prompt = json.dumps(extraction_schema, indent=2)
    prompt = f"""Based ONLY on the provided text snippets below, extract key entities, potential risks, and relationships relevant to the extraction context.
IMPORTANT: Provide all names, types, and descriptions in ENGLISH.

Your response MUST be a single valid JSON object that strictly adheres to the following JSON schema:
```json
{schema_string_for_prompt}

If no relevant information for a category (entities, risks, relationships) is found in the text, return an empty array for that category (e.g., "entities": []), but ensure the overall response is still a valid JSON object matching the schema structure.
Begin analysis of the following text:
{context_text}"""

    print(f"\n--- Sending Request to {llm_provider_name} for Data Extraction (Model: {llm_model_name}) ---")

    # API Call - Attempt JSON mode
    request_params = {
        "model": llm_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    extracted_json = None

    try:
        response = client.chat.completions.create(**request_params)
        raw_content = response.choices[0].message.content
        print(f"\nRaw {llm_provider_name} Response (Extraction):\n{raw_content}")
        extracted_json = json.loads(raw_content)

    except openai.BadRequestError as e:
        if 'response_format' in str(e):
            print(f"WARNING: {llm_provider_name} API Error (likely JSON mode unsupported): {e}. Retrying without JSON format...")
            del request_params["response_format"]
            try:
                response = client.chat.completions.create(**request_params)
                raw_content = response.choices[0].message.content
                print(f"\nRaw {llm_provider_name} Response (Extraction - Retry):\n{raw_content}")
                try:
                    json_start = raw_content.find('{')
                    json_end = raw_content.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = raw_content[json_start:json_end]
                        extracted_json = json.loads(json_str)
                    else:
                        extracted_json = json.loads(raw_content)
                        print("Warning: Parsed raw content directly after retry")
                except json.JSONDecodeError as json_e_retry:
                    print(f"ERROR: Failed to decode JSON even after retry: {json_e_retry}")
                    return {"entities": [], "risks": [], "relationships": []}
            except Exception as retry_e:
                print(f"ERROR during data extraction API call (Retry): {retry_e}")
                return {"entities": [], "risks": [], "relationships": []}
        else:
            print(f"ERROR during data extraction API call: {e}")
            return {"entities": [], "risks": [], "relationships": []}

    except json.JSONDecodeError as json_e:
        print(f"ERROR: Failed to decode JSON from response: {json_e}")
        return {"entities": [], "risks": [], "relationships": []}
    except Exception as e:
        print(f"ERROR during data extraction API call: {e}")
        return {"entities": [], "risks": [], "relationships": []}

    # Process the successfully parsed JSON
    if extracted_json is None:
        print("ERROR: Failed to obtain valid JSON data after API calls and retries.")
        return {"entities": [], "risks": [], "relationships": []}

    validated_data = {
        "entities": extracted_json.get("entities", []) if isinstance(extracted_json.get("entities"), list) else [],
        "risks": extracted_json.get("risks", []) if isinstance(extracted_json.get("risks"), list) else [],
        "relationships": extracted_json.get("relationships", []) if isinstance(extracted_json.get("relationships"), list) else []
    }
    print(f"\nParsed Extracted Data:\n{json.dumps(validated_data, indent=2)}")
    return validated_data

# --- Local Testing Block ---
if __name__ == "__main__":
    print("\n--- Running Local NLP Processor Tests ---")
    
    # Test Keyword Translation
    print("\nTesting Keyword Translation...")
    sample_query_kw = "supply chain compliance issues 2023"
    sample_context_kw = "Baidu search in China for specific company supply chain info"
    translated_keywords_test = translate_keywords_for_context(sample_query_kw, sample_context_kw)
    print(f"\nTranslated Keywords Result: {translated_keywords_test}")
    
    # Test Data Extraction
    print("\nTesting Data Extraction...")
    sample_results_test = [
        {"title": "New Supply Chain Regulations Impact Global Trade - GovReport", 
         "url": "https://example.gov/report-2023-supply-chain", 
         "snippet": "The Regulatory Body C announced new compliance measures affecting manufacturers like Corporation A.", 
         "source": "google", "published_date": "2023-11-01"},
        {"title": "Corporation A Faces Scrutiny Over Supplier Ethics - Finance News", 
         "url": "https://example.finance/corp-a-ethics", 
         "snippet": "Concerns are rising about Corporation A's sourcing from suppliers in Manufacturing Hub B.", 
         "source": "google", "published_date": "2023-10-15"},
        {"title": "Baidu Search Results Snippet (Example)", 
         "url": "https://example.baidu.com/search-result-3", 
         "snippet": "Analysis shows shifting trade patterns in Manufacturing Hub B.", 
         "source": "baidu", "published_date": "2023-12-01"}
    ]
    sample_extraction_context_test = "supply chain risks and regulations involving Corporation A"
    extracted_data_test = extract_data_from_results(sample_results_test, sample_extraction_context_test)
    print("\nFinal Extracted Data Result:")
    print(json.dumps(extracted_data_test, indent=2))
    
    print("\n--- Local Tests Complete ---")