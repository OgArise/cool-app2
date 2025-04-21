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


# ===> NEW FUNCTION for Ownership Extraction <===
def extract_ownership_relationships(parent_entity_name: str,
                                    related_entity_name: str,
                                    text_snippets: List[Dict[str, Any]], # Snippets potentially mentioning ownership
                                    llm_provider: str, llm_model: str) -> Dict | None:
    """
    Uses LLM to analyze text snippets for ownership percentage/control between two entities.

    Args:
        parent_entity_name: The name of the entity assumed to be the potential owner.
        related_entity_name: The name of the entity assumed to be potentially owned.
        text_snippets: A list of search result dicts (containing 'snippet', 'url')
                       from targeted searches for ownership documents.
        llm_provider: The LLM provider ('google_ai', 'openai', 'openrouter').
        llm_model: The specific model name.

    Returns:
        A dictionary like {"relationship_type": "SUBSIDIARY|AFFILIATE|UNRELATED/OTHER",
                          "ownership_percentage": "% string or Not Stated",
                          "source_url": "URL or Not Stated"}
        Returns None if the LLM call fails or parsing fails.
    """
    print(f"\n--- Analyzing ownership: '{parent_entity_name}' owning '{related_entity_name}'? ---")
    if not llm_provider or not llm_model: return None # Need LLM config
    if not text_snippets: print("No text snippets provided for ownership check."); return None

    # Prepare context specifically for ownership check
    context = f"Analyze the potential ownership relationship where '{parent_entity_name}' is the owner/investor and '{related_entity_name}' is the owned/investee entity. Base your analysis ONLY on the following text snippets extracted from potential financial reports or official documents.\n\nFocus *only* on stated ownership percentages or clear descriptions of control/significant influence (e.g., 'wholly owned subsidiary', 'minority stake', 'equity method investment', 'consolidated entity', 'joint venture', '% stake'). Ignore mentions of simple supplier, customer, or partnership relationships unless ownership is explicitly stated.\n\nText Snippets:\n"
    max_chars = 7000 # Slightly smaller context for this focused task
    char_count = len(context)
    added_snippets = 0
    for snip in text_snippets:
        snippet_text = snip.get('snippet', '')
        if snippet_text and not snippet_text.isspace():
            entry = f"---\nSource URL: {snip.get('url', 'N/A')}\nSnippet: {snippet_text}\n"
            if char_count + len(entry) <= max_chars:
                context += entry; char_count += len(entry); added_snippets += 1
            else: break
    if added_snippets == 0: print("No valid snippets for ownership check."); return None
    print(f"Prepared ownership context using {added_snippets} snippets.")


    # Define desired JSON output structure for this task
    ownership_schema_desc = """
    {
      "relationship_type": "SUBSIDIARY | AFFILIATE | UNRELATED/OTHER", // Based on >50% ownership for SUBSIDIARY, 5-50% for AFFILIATE, or lack of evidence/other relationship
      "ownership_percentage": "Specific % found (e.g., '75%', 'approx 30%') OR 'Control Mentioned' OR 'Influence Mentioned' OR 'Not Stated'", // Report % if found, otherwise describe evidence
      "source_url": "The single most relevant Source URL from the snippets supporting the conclusion OR 'Multiple Sources' OR 'Not Stated'" // URL providing the evidence
    }"""

    prompt = f"""{context}

    Based ONLY on the text snippets provided above, determine the ownership relationship where '{parent_entity_name}' owns/invests in '{related_entity_name}'.

    Use these definitions:
    - SUBSIDIARY: Evidence of >50% ownership or explicit control statement.
    - AFFILIATE: Evidence of 5% to 50% ownership or explicit significant influence statement.
    - UNRELATED/OTHER: Evidence of <5% ownership, a different relationship, or no clear ownership evidence found in the snippets.

    Your response MUST be ONLY a single valid JSON object matching the structure described below. Do not include explanations outside the JSON.
    ```json
    {ownership_schema_desc}
    ```"""

    raw_content = None
    parsed_json = None
    api_error = None

    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        print(f"Sending ownership request to {llm_provider} ({model_name_used})...")

        if client_type == "openai_compatible":
             request_params = {"model": model_name_used, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "response_format": {"type": "json_object"}}
             try:
                 response = client_or_lib.chat.completions.create(**request_params)
                 raw_content = response.choices[0].message.content
             except Exception as e_json_mode:
                 print(f"WARNING: {llm_provider} ownership check failed JSON mode ({e_json_mode}). Retrying...")
                 del request_params["response_format"]
                 response = client_or_lib.chat.completions.create(**request_params)
                 raw_content = response.choices[0].message.content
        elif client_type == "google_ai":
            generation_config = genai.types.GenerationConfig(temperature=0.1)
            response = client_or_lib.generate_content(prompt, generation_config=generation_config)
            if not response.candidates: raise ValueError(f"Google AI ownership response blocked. Feedback: {response.prompt_feedback}")
            raw_content = response.text
        else: raise ValueError("Unknown client type")

        print(f"Raw {llm_provider} ownership response:\n>>>\n{raw_content}\n<<<")
        cleaned_content = raw_content.strip().replace("```json", "").replace("```", "").strip()
        parsed_json = json.loads(cleaned_content)

    except json.JSONDecodeError as json_e: print(f"ERROR: Failed to decode JSON from {llm_provider} ownership response: {json_e}\nRaw content: {raw_content}"); api_error = json_e
    except Exception as e: print(f"ERROR during ownership extraction call via {llm_provider}: {e}"); api_error = e

    # Validate and return
    if parsed_json and isinstance(parsed_json, dict) and parsed_json.get("relationship_type") in ["SUBSIDIARY", "AFFILIATE", "UNRELATED/OTHER"]:
        print("Successfully parsed ownership info.")
        # Ensure default values if keys are missing
        parsed_json.setdefault("ownership_percentage", "Not Stated")
        parsed_json.setdefault("source_url", "Not Stated")
        return parsed_json
    else:
        print(f"Failed to get valid ownership structure. Error: {api_error}")
        return None # Indicate failure or invalid structure

# ===> END NEW FUNCTION <===


# ===> NEW FUNCTION for Analysis Summary <===
def generate_analysis_summary(extracted_data: Dict, query: str, exposures_count: int,
                              llm_provider: str, llm_model: str) -> str:
    """Generates a concise, easy-to-understand summary of the analysis findings."""
    print(f"\n--- Attempting Analysis Summary via {llm_provider} (Model: {llm_model}) ---")
    if not llm_provider or not llm_model: return "Summary generation skipped: Missing LLM config."

    # Prepare context for the summary prompt
    summary_context = f"Original Query: {query}\n\nAnalysis Findings:\n"
    entities = extracted_data.get("entities", [])
    risks = extracted_data.get("risks", [])
    relationships = extracted_data.get("relationships", [])

    # Include key findings in the context for the LLM
    if entities: summary_context += f"- Key Entities Found ({len(entities)}): " + ", ".join([f"{e.get('name')} ({e.get('type')})" for e in entities[:5]]) + ("..." if len(entities)>5 else "") + "\n"
    if risks:
        high_risks = [r for r in risks if r.get('severity') == 'HIGH']
        med_risks = [r for r in risks if r.get('severity') == 'MEDIUM']
        summary_context += f"- Potential Risks ({len(risks)}): "
        if high_risks: summary_context += f"{len(high_risks)} High Severity (e.g., '{high_risks[0].get('description')[:80]}...'); "
        if med_risks: summary_context += f"{len(med_risks)} Medium Severity (e.g., '{med_risks[0].get('description')[:80]}...'); "
        summary_context += "\n"
    if relationships: summary_context += f"- Key Relationships Found ({len(relationships)}): " + "; ".join([f"{rel.get('entity1')} {rel.get('relationship_type')} {rel.get('entity2')}" for rel in relationships[:3]]) + ("..." if len(relationships)>3 else "") + "\n"
    if exposures_count > 0: summary_context += f"- Supply Chain Exposures Identified: {exposures_count}\n"

    if not any([entities, risks, relationships]):
        return "No significant entities, risks, or relationships were extracted from the search results to summarize."

    # Define the summary prompt for a general audience
    prompt = f"""You are a professional geopolitical and financial analyst summarizing automated research findings for a general public audience.
    Based ONLY on the summarized findings provided below, write a concise (2-4 sentences) summary highlighting the most important takeaways. Focus on the key entities involved, the nature and severity of major risks, any significant connections found, and if any supply chain exposures were noted. Make it clear, direct, and easy to understand, avoiding jargon.

    {summary_context}

    Generate ONLY the summary paragraph."""

    try:
        # Use the dynamic client logic (already defined in _get_llm_client_and_model)
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        raw_content = ""

        print(f"Sending summary request to {llm_provider} ({model_name_used})...")
        if client_type == "openai_compatible":
            response = client_or_lib.chat.completions.create( model=model_name_used, messages=[{"role": "user", "content": prompt}], temperature=0.6, max_tokens=300 ) # Allow more tokens
            raw_content = response.choices[0].message.content.strip()
        elif client_type == "google_ai":
            # Ensure safety settings allow potentially summarizing sensitive topics if applicable
            safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in [ "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT", ] ]
            generation_config = genai.types.GenerationConfig(temperature=0.6, max_output_tokens=300)
            response = client_or_lib.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
            if not response.candidates: raise ValueError(f"Google AI summary response blocked. Feedback: {response.prompt_feedback}")
            raw_content = response.text.strip()
        else: raise ValueError("Unknown client type")

        print(f"\nRaw {llm_provider} Response (Summary):\n>>>\n{raw_content}\n<<<")
        # Basic cleaning
        cleaned_summary = raw_content
        prefixes_to_remove = ["Okay, here's a summary:", "Here is a summary:", "Summary:", "Based on the findings:", "Based on the analysis:", "Here's a summary:"]
        for prefix in prefixes_to_remove:
             if cleaned_summary.lower().startswith(prefix.lower()): cleaned_summary = cleaned_summary[len(prefix):].strip()
        return cleaned_summary if cleaned_summary else "LLM returned an empty summary."

    except Exception as e:
        print(f"ERROR during summary generation via {llm_provider}: {e}")
        # If rate limited during summary, just give a simpler message
        if "429" in str(e):
             return "Could not generate summary due to API rate limits during final step."
        return f"Could not generate summary due to error: {type(e).__name__}"
# ===> END NEW FUNCTION <===


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