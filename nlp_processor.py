# nlp_processor.py

import json
from typing import List, Dict, Any, Optional
import time
import re
import traceback # Import traceback for logging

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
    print(f"Attempting to initialize LLM client for provider: {provider}...")

    if provider == "openai":
        api_key = OPENAI_API_KEY
        if not openai: raise ImportError("OpenAI library not installed.")
        if not api_key: raise ValueError("OpenAI API Key is missing in environment.")
        try:
            client_or_lib = openai.OpenAI(api_key=api_key)
            client_type = "openai_compatible"
            print(f"Initialized OpenAI client.")
        except Exception as e: print(f"ERROR init OpenAI: {e}"); raise e

    elif provider == "openrouter":
        api_key = OPENROUTER_API_KEY
        if not openai: raise ImportError("OpenAI library not installed (for OpenRouter).")
        if not api_key: raise ValueError("OpenRouter API Key is missing.")
        try:
            client_or_lib = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL, default_headers=OPENROUTER_HEADERS)
            client_type = "openai_compatible"
            print(f"Initialized OpenRouter client (Model: {model_name})")
        except Exception as e: print(f"ERROR init OpenRouter: {e}"); raise e

    elif provider == "google_ai":
        api_key = GOOGLE_AI_API_KEY
        if not genai: raise ImportError("Google Generative AI library not installed.")
        if not api_key: raise ValueError("Google AI API Key is missing.")
        try:
            genai.configure(api_key=api_key)
            # Ensure model name validity if possible, or let API call handle it
            client_or_lib = genai.GenerativeModel(model_name=model_name)
            client_type = "google_ai"
            print(f"Initialized Google AI client (Model: {model_name})")
        except Exception as e: print(f"ERROR init Google AI: {e}"); raise e
    else:
        raise ValueError(f"Unsupported LLM provider specified: {provider}")

    if client_or_lib is None:
        raise ConnectionError(f"Failed to create LLM client for provider {provider}")

    return client_or_lib, client_type, effective_model_name


# --- Keyword Generation ---
def translate_keywords_for_context(original_query: str, target_context: str,
                                   llm_provider: str, llm_model: str) -> List[str]:
    """Generates ENGLISH keywords using the specified LLM."""
    print(f"\n--- Attempting Keyword Generation via {llm_provider} (Model: {llm_model}) ---")
    if not llm_provider or not llm_model: return [original_query]

    # Stronger Prompt Instruction
    prompt = f"""You are an expert keyword generator. Given the query and context below, provide a list of 3-5 relevant ENGLISH search keywords suitable for the context.

    Initial Query: {original_query}
    Target Search Context: {target_context}

    IMPORTANT: Your entire response must contain ONLY the comma-separated list of keywords. Do NOT include any other text, explanation, or formatting."""

    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        raw_content = ""

        if client_type == "openai_compatible":
            response = client_or_lib.chat.completions.create( model=model_name_used, messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=150 )
            raw_content = response.choices[0].message.content.strip()
        elif client_type == "google_ai":
            response = client_or_lib.generate_content(prompt)
            # Ensure response.text exists and is safe to access
            raw_content = response.text.strip() if hasattr(response, 'text') and response.text else ""
            if not raw_content and response.candidates:
                 # Fallback if .text is empty but candidates exist (might be blocked)
                 print(f"Warning: Google AI response.text was empty. Prompt Feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
                 # Attempt to get content from the first candidate if safe
                 if response.candidates[0].content and response.candidates[0].content.parts:
                     raw_content = response.candidates[0].content.parts[0].text.strip()
                 else:
                     raw_content = "" # Still empty
        else:
            raise ValueError("Unknown client type.")

        print(f"\nRaw {llm_provider} Response (Keywords):\n>>>\n{raw_content}\n<<<")

        # Clean potential conversational filler
        prefixes_to_remove = ["Sure, here are the keywords:", "Here are the keywords:", "Okay, here is the list:", "Here is the list:", "Of course! Here are the keywords:"]
        cleaned_content = raw_content
        for prefix in prefixes_to_remove:
            if cleaned_content.lower().startswith(prefix.lower()): # Case-insensitive check
                cleaned_content = cleaned_content[len(prefix):].strip()
                print(f"Cleaned prefix '{prefix}'")
                break

        # Check if ONLY keywords remain
        if '\n' in cleaned_content:
             print("Warning: Detected multiple lines in keyword response after cleaning. Attempting split on first line.")
             cleaned_content = cleaned_content.split('\n')[0]

        keywords = [kw.strip() for kw in cleaned_content.split(',') if kw.strip()]

        # Check if the result looks like keywords or still conversational
        if not keywords or any(word in cleaned_content.lower() for word in ["sure", "here", "keywords", "list", "okay", "provide", "prompt", "apologies"]):
            print("Warning: Keyword response still looks conversational or empty. Returning original query.")
            return [original_query]

        print(f"Parsed Keywords: {keywords}")
        return keywords

    except Exception as e:
        print(f"ERROR during keyword generation via {llm_provider}: {e}")
        traceback.print_exc() # Print traceback for keyword errors too
        return [original_query] # Fallback

# --- Entity / Risk Extraction ---
def extract_data_from_results(search_results: List[Dict[str, Any]], extraction_context: str,
                              llm_provider: str, llm_model: str,
                              focus_on_china: bool = False) -> Dict[str, List]:
    """Extracts structured data using the specified LLM, with enhanced prompting."""
    print(f"\n--- Attempting Data Extraction via {llm_provider} (Model: {llm_model}) ---")
    if not llm_provider or not llm_model: return {"entities": [], "risks": [], "relationships": []}
    if not search_results: return {"entities": [], "risks": [], "relationships": []}

    # Prepare context - Add URL to snippet text for easier reference by LLM
    context_text = "" # Start empty, add context description later in prompt
    max_chars = 8000; char_count = 0; added_snippets = 0
    print(f"\nPreparing context for extraction (Max Chars: {max_chars})...")
    snippets_with_urls = [] # Store tuples of (url, snippet_text)
    for result in search_results:
        snippet = result.get('snippet', ''); title = result.get('title', ''); url = result.get('url', 'N/A')
        if snippet and not snippet.isspace() and url != 'N/A': # Require a valid URL now
            entry = f"\n---\nSourceURL: {url}\nTitle: {title}\nSnippet: {snippet}\n" # Label URL clearly
            if char_count + len(entry) <= max_chars:
                context_text += entry; char_count += len(entry); added_snippets += 1
                snippets_with_urls.append({"url": url, "snippet": snippet}) # Keep track
            else: print(f"Stopping context prep early ({char_count} chars)."); break
        else: print(f"Skipping empty snippet or missing URL: {title} ({url})")
    if added_snippets == 0: print("No valid snippets with URLs found for context."); return {"entities": [], "risks": [], "relationships": []}
    print(f"Prepared context using {added_snippets} snippets.")


    # --- ENHANCED EXTRACTION PROMPT ---
    # More detailed schema description within the prompt itself
    schema_description = """
    - "entities": Array of objects. Each object MUST have:
        - "name": (string, English name of the entity). Standardize (e.g., "Corp A" not "Corp. A.").
        - "type": (string enum: COMPANY, PERSON, LOCATION, ORGANIZATION, REGULATION, OTHER).
        - "mentions": (array of strings) Accurately list the specific 'SourceURL' values (provided above each snippet in the input) where this *exact entity name* was mentioned. MUST NOT be empty if entity is listed.
    - "risks": Array of objects. Each object MUST have:
        - "description": (string, Clear English description of the risk).
        - "severity": (string enum: LOW, MEDIUM, HIGH) Estimate severity based on keywords like 'significant risk', 'potential penalties', 'violations', 'failure', 'scrutiny', 'warning', 'investigation', 'major impact'. Default to MEDIUM if risk is mentioned but severity is ambiguous. Default to LOW for minor concerns.
        - "related_entities": (array of strings) List the English names of entities *specifically mentioned within the context of this risk description*. Can be empty `[]` if no specific entities are linked in the text.
        - "source_urls": (array of strings) Accurately list the specific 'SourceURL' values (provided above each snippet) where this *exact risk description* was mentioned or strongly implied. MUST NOT be empty if risk is listed.
    - "relationships": Array of objects. Each object MUST have:
        - "entity1": (string name, English). Standardize.
        - "relationship_type": (string, e.g., SUPPLIER_OF, REGULATES, PARTNERS_WITH, INVESTED_IN, ACQUIRED, PARENT_COMPANY_OF). Be specific if possible based on the text.
        - "entity2": (string name, English). Standardize.
        - "context_urls": (array of strings) Accurately list the specific 'SourceURL' values (provided above each snippet) where this *exact relationship* between entity1 and entity2 was mentioned or described. MUST NOT be empty if relationship is listed.
    """

    prompt_base = f"""Analyze the following text snippets based on the extraction context: "{extraction_context}".
Extract key entities, potential risks, and specific relationships found ONLY in the provided text snippets.

<<< MOST IMPORTANT INSTRUCTIONS >>>
1.  Your entire response MUST be ONLY a single valid JSON object. Do NOT include any text, explanation, notes, apologies, or markdown formatting like ```json before or after the JSON object.
2.  The JSON object MUST contain exactly three keys: "entities", "risks", and "relationships".
3.  The value for each key MUST be an array of objects (use an empty array `[]` if nothing is found for that category).
4.  Strictly adhere to the structure and requirements detailed below for objects within the arrays:
{schema_description}
5.  **Crucially**: Populate the "mentions", "related_entities", "source_urls", and "context_urls" arrays accurately using the exact 'SourceURL' values provided *directly above* each relevant text snippet in the input. If an entity/risk/relationship is mentioned in multiple snippets, include all corresponding SourceURLs in the respective array. These URL arrays MUST NOT be empty for any listed item.
6.  Provide all names, types, descriptions, and relationship types in ENGLISH. Standardize company names (e.g., "Example Corp" instead of "Example Corp.").
7.  If no relevant information is found for a category (entities, risks, relationships), return an empty array `[]` for that key. Do not omit the key itself.
<<< END MOST IMPORTANT INSTRUCTIONS >>>
"""
    # Add focus instruction if needed
    if focus_on_china:
        prompt_focus = """
SPECIAL INSTRUCTION: Focus extraction ONLY on entities/risks/relationships directly related to China (Chinese companies, locations in China, regulations affecting China, international companies' Chinese operations/risks)."""
        prompt = prompt_base + prompt_focus + f"\n\nBegin analysis of the text snippets provided below:\n{context_text}"
    else:
        prompt = prompt_base + f"\n\nBegin analysis of the text snippets provided below:\n{context_text}"
    # --- END ENHANCED PROMPT ---


    raw_content = None; extracted_json = None; api_error = None
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        print(f"\n--- Sending Request to {llm_provider} for Data Extraction (Model: {model_name_used}) ---")
        # print(f"Prompt Snippet:\n{prompt[:500]}...\n...\n...{prompt[-500:]}") # Log snippet for debugging

        if client_type == "openai_compatible":
            # Ensure messages format is correct
            request_params = {"model": model_name_used, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
            # Add response_format conditionally ONLY if provider isn't OpenRouter (often breaks it)
            if provider != "openrouter":
                 request_params["response_format"] = {"type": "json_object"}

            try:
                print(f"Attempting extraction with params: {request_params.keys()}")
                response = client_or_lib.chat.completions.create(**request_params)
                raw_content = response.choices[0].message.content
            except Exception as e_json_mode:
                 # Only retry if JSON mode was attempted and failed
                 if "response_format" in request_params:
                     print(f"WARNING: {llm_provider} possibly failed JSON mode ({e_json_mode}). Retrying without forcing JSON format...")
                     del request_params["response_format"] # Remove format for retry
                     response = client_or_lib.chat.completions.create(**request_params)
                     raw_content = response.choices[0].message.content
                 else:
                     # If it failed even without JSON mode initially (e.g., OpenRouter), raise the error
                     raise e_json_mode

        elif client_type == "google_ai":
            # Configure safety settings to be less restrictive if needed, but start standard
            safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in [ "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT", ] ]
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                # Add response_mime_type for Gemini models that support it (like 1.5 Flash/Pro)
                # Check model name before applying
                response_mime_type="application/json" if "gemini-1.5" in model_name_used else None
            )
            print(f"Attempting extraction with Google AI (JSON mode: {'application/json' if generation_config.response_mime_type else 'No'})...")
            try:
                response = client_or_lib.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                if not response.candidates:
                    raise ValueError(f"Google AI response blocked or empty. Feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
                raw_content = response.text # Gemini JSON mode puts JSON directly in .text
            except Exception as e_google:
                 # If JSON mode failed, try without it? Could be complex. Log error for now.
                 print(f"ERROR during Google AI extraction attempt: {e_google}")
                 print(f"Prompt Feedback (if available): {getattr(response, 'prompt_feedback', 'N/A')}")
                 traceback.print_exc()
                 api_error = e_google # Set API error and proceed to parsing check
                 raw_content = None # Ensure raw_content is None if API call failed


        else:
             raise ValueError("Unknown client type")

        if raw_content:
            print(f"\nRaw {llm_provider} Response (Extraction):\n>>>\n{raw_content}\n<<<")
            try:
                # Basic cleaning: remove potential markdown fences
                cleaned_content = raw_content.strip()
                if cleaned_content.startswith("```json"): cleaned_content = cleaned_content[7:]
                if cleaned_content.endswith("```"): cleaned_content = cleaned_content[:-3]
                cleaned_content = cleaned_content.strip()

                # Find the outermost JSON object {}
                json_start = cleaned_content.find('{')
                json_end = cleaned_content.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = cleaned_content[json_start:json_end]
                    print("Attempting to parse extracted JSON block...")
                    extracted_json = json.loads(json_str)
                else:
                    json_str = cleaned_content # Fallback if {} not found
                    if not json_str: raise ValueError("Cleaned content is empty.")
                    print("Warning: Could not find explicit '{...}' structure. Attempting to parse cleaned content directly.")
                    extracted_json = json.loads(json_str) # Try parsing directly

            except json.JSONDecodeError as json_e:
                print(f"ERROR: Failed to decode JSON after cleaning: {json_e}")
                print(f"Cleaned content was:\n>>>\n{cleaned_content}\n<<<")
                api_error = json_e # Treat JSON parsing error as an API error for reporting
                extracted_json = None # Ensure no partial JSON proceeds
            except Exception as parse_e:
                 print(f"ERROR: Unexpected error during JSON parsing: {parse_e}")
                 traceback.print_exc()
                 api_error = parse_e
                 extracted_json = None

    except Exception as e:
        print(f"ERROR during LLM extraction API call or client setup: {e}")
        traceback.print_exc()
        api_error = e
        extracted_json = None # Ensure reset


    # Process Final Result - More Robust Validation
    validated_data = {"entities": [], "risks": [], "relationships": []}
    if extracted_json is not None and isinstance(extracted_json, dict):
        print("\nValidating parsed JSON structure...")
        # Validate top-level keys and that values are lists
        entities_raw = extracted_json.get("entities", [])
        risks_raw = extracted_json.get("risks", [])
        relationships_raw = extracted_json.get("relationships", [])

        if isinstance(entities_raw, list):
            # Validate individual entities
            for entity in entities_raw:
                if isinstance(entity, dict) and \
                   entity.get("name") and isinstance(entity.get("name"), str) and \
                   entity.get("type") and isinstance(entity.get("type"), str) and \
                   entity.get("mentions") and isinstance(entity.get("mentions"), list) and \
                   all(isinstance(m, str) for m in entity["mentions"]): # Check mentions is list of strings
                    validated_data["entities"].append(entity)
                else:
                    print(f"Warning: Skipping invalid entity item: {entity}")
        else: print("Warning: 'entities' key found but value is not a list.")


        if isinstance(risks_raw, list):
            # Validate individual risks
            for risk in risks_raw:
                if isinstance(risk, dict) and \
                   risk.get("description") and isinstance(risk.get("description"), str) and \
                   risk.get("source_urls") and isinstance(risk.get("source_urls"), list) and \
                   all(isinstance(u, str) for u in risk["source_urls"]): # Check source_urls is list of strings
                   # Optional fields validation (severity, related_entities)
                   if "severity" not in risk or not isinstance(risk.get("severity"), str): risk["severity"] = "MEDIUM" # Default if missing/invalid
                   if "related_entities" not in risk or not isinstance(risk.get("related_entities"), list): risk["related_entities"] = [] # Default if missing/invalid
                   elif not all(isinstance(re, str) for re in risk["related_entities"]): risk["related_entities"] = [] # Ensure list contains strings

                   validated_data["risks"].append(risk)
                else:
                   print(f"Warning: Skipping invalid risk item (missing desc/urls or invalid types): {risk}")
        else: print("Warning: 'risks' key found but value is not a list.")

        if isinstance(relationships_raw, list):
            # Validate individual relationships
            for rel in relationships_raw:
                if isinstance(rel, dict) and \
                   rel.get("entity1") and isinstance(rel.get("entity1"), str) and \
                   rel.get("relationship_type") and isinstance(rel.get("relationship_type"), str) and \
                   rel.get("entity2") and isinstance(rel.get("entity2"), str) and \
                   rel.get("context_urls") and isinstance(rel.get("context_urls"), list) and \
                   all(isinstance(u, str) for u in rel["context_urls"]): # Check context_urls is list of strings
                    validated_data["relationships"].append(rel)
                else:
                   print(f"Warning: Skipping invalid relationship item: {rel}")
        else: print("Warning: 'relationships' key found but value is not a list.")

        if any(validated_data.values()): print("\nSuccessfully Parsed and Validated Extracted Data.")
        elif not api_error: print("\nWarning: LLM returned valid JSON structure, but all data arrays were empty or invalid.")

    # Final logging based on outcome
    if api_error:
        print(f"\nERROR: LLM extraction failed or produced invalid/unparseable JSON. Error: {type(api_error).__name__}")
        # Reset validated_data to ensure no partial/bad data is returned on error
        validated_data = {"entities": [], "risks": [], "relationships": []}
    elif not any(validated_data.values()):
        print("\nResult: No valid structured data extracted.")
    else:
         print(f"\nResult: Extracted {len(validated_data['entities'])} entities, {len(validated_data['risks'])} risks, {len(validated_data['relationships'])} relationships.")

    # print(f"\nFinal Validated Extracted Data:\n{json.dumps(validated_data, indent=2)}") # Optional: Log final structure
    return validated_data


# --- Ownership Extraction ---
def extract_ownership_relationships(parent_entity_name: str, related_entity_name: str,
                                    text_snippets: List[Dict[str, Any]],
                                    llm_provider: str, llm_model: str) -> Dict | None:
    """Uses LLM to analyze text snippets for evidence of ownership/control."""
    # Keep this function as defined in response #62 - should be robust enough for now
    print(f"\n--- Analyzing ownership: '{parent_entity_name}' owning '{related_entity_name}'? ---");
    if not llm_provider or not llm_model: print("Ownership check skipped: Missing LLM config."); return None;
    if not text_snippets: print("No snippets provided for ownership check."); return None

    # Prepare context ONLY from snippets mentioning BOTH entities
    context = f"Analyze ONLY the following text snippets for an explicit ownership or control relationship where '{parent_entity_name}' owns, controls, or has invested significantly (>5%) in '{related_entity_name}'. Focus ONLY on evidence present in these snippets.\n\nDefinitions:\n- SUBSIDIARY: Clear control stated or >50% ownership mentioned.\n- AFFILIATE: Significant investment mentioned (e.g., 5-50%) or terms like 'affiliate', 'joint venture', 'equity investment' used.\n- UNRELATED: No evidence of ownership/control found in snippets, or <5% stake mentioned.\n\nText Snippets:\n";
    max_chars = 7000; char_count = len(context); added_snippets = 0; relevant_snippets_for_prompt = []
    for snip in text_snippets:
        snippet_text = snip.get('snippet', ''); url = snip.get('url', 'N/A');
        # Check if BOTH names are present (case-insensitive) and snippet has text
        if snippet_text and url != 'N/A' and \
           re.search(r'\b' + re.escape(parent_entity_name) + r'\b', snippet_text, re.IGNORECASE) and \
           re.search(r'\b' + re.escape(related_entity_name) + r'\b', snippet_text, re.IGNORECASE):
             entry = f"---\nSource URL: {url}\nSnippet: {snippet_text}\n";
             if char_count + len(entry) <= max_chars:
                 context += entry;
                 relevant_snippets_for_prompt.append({"url": url, "snippet": snippet_text});
                 char_count += len(entry);
                 added_snippets += 1
             else:
                 print("Stopping ownership context prep early due to max chars.")
                 break
    if added_snippets == 0: print(f"No snippets found mentioning both '{parent_entity_name}' and '{related_entity_name}'. Returning UNRELATED."); return {"relationship_type": "UNRELATED", "evidence_snippet": "No relevant snippets found mentioning both entities.", "source_url": "N/A"}
    print(f"Prepared ownership context using {added_snippets} relevant snippets.")

    # Define desired JSON output structure for the LLM
    ownership_schema_desc = """{"relationship_type": "SUBSIDIARY | AFFILIATE | UNRELATED", "evidence_snippet": "Quote the specific short phrase (max 150 chars) from a snippet that provides the strongest evidence for the relationship_type, or state 'No specific evidence found in snippets'.", "source_url": "Provide the Source URL corresponding to the evidence_snippet, or 'N/A' if no specific evidence was found."}""";

    # Construct the final prompt
    prompt = f"""{context}\nBased ONLY on the text snippets provided above, determine the ownership relationship according to the definitions. Your response MUST be ONLY a single valid JSON object matching this exact structure (no extra text or markdown):\n```json\n{ownership_schema_desc}\n```"""

    raw_content = None; parsed_json = None; api_error = None
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model);
        print(f"Sending ownership request to {llm_provider} ({model_name_used})...")
        # print(f"Ownership Prompt Snippet:\n{prompt[:500]}...\n...{prompt[-500:]}") # Optional: Log prompt

        if client_type == "openai_compatible":
            request_params = {"model": model_name_used, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
            if provider != "openrouter": request_params["response_format"] = {"type": "json_object"} # Add JSON mode if supported
            try:
                response = client_or_lib.chat.completions.create(**request_params);
                raw_content = response.choices[0].message.content
            except Exception as e_json_mode:
                 if "response_format" in request_params:
                     print(f"WARNING: Ownership check failed JSON mode ({e_json_mode}). Retrying...");
                     del request_params["response_format"];
                     response = client_or_lib.chat.completions.create(**request_params);
                     raw_content = response.choices[0].message.content
                 else: raise e_json_mode # Reraise if failed without JSON mode

        elif client_type == "google_ai":
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json" if "gemini-1.5" in model_name_used else None
            )
            print(f"Attempting ownership with Google AI (JSON mode: {'application/json' if generation_config.response_mime_type else 'No'})...")
            try:
                response = client_or_lib.generate_content(prompt, generation_config=generation_config)
                if not response.candidates: raise ValueError(f"Google AI ownership response blocked: {getattr(response, 'prompt_feedback', 'N/A')}")
                raw_content = response.text
            except Exception as e_google_own:
                 print(f"ERROR during Google AI ownership attempt: {e_google_own}")
                 api_error = e_google_own
                 raw_content = None

        else: raise ValueError("Unknown client type")

        if raw_content:
            print(f"Raw {llm_provider} ownership response:\n>>>\n{raw_content}\n<<<");
            cleaned_content = raw_content.strip().replace("```json", "").replace("```", "").strip();
            json_start = cleaned_content.find('{'); json_end = cleaned_content.rfind('}') + 1;
            if json_start != -1 and json_end != -1:
                 json_str = cleaned_content[json_start:json_end]
                 parsed_json = json.loads(json_str)
            else:
                 print(f"Warning: Could not find '{{...}}' in ownership response. Content:\n{cleaned_content}")
                 # Attempt direct parse only if content looks like JSON
                 if cleaned_content.startswith("{") and cleaned_content.endswith("}"):
                     parsed_json = json.loads(cleaned_content)
                 else:
                     raise json.JSONDecodeError("Response does not appear to be JSON", cleaned_content, 0)

    except json.JSONDecodeError as json_e:
        print(f"ERROR decoding JSON ownership response: {json_e}\nRaw Content was:\n>>>\n{raw_content}\n<<<");
        api_error = json_e
    except Exception as e:
        print(f"ERROR during ownership extraction API call or parsing: {e}");
        traceback.print_exc()
        api_error = e

    # Validate and return
    if parsed_json and isinstance(parsed_json, dict) and \
       parsed_json.get("relationship_type") in ["SUBSIDIARY", "AFFILIATE", "UNRELATED"] and \
       isinstance(parsed_json.get("evidence_snippet"), str) and \
       isinstance(parsed_json.get("source_url"), str):
        print(f"Parsed ownership info: {parsed_json['relationship_type']}")
        # Truncate evidence snippet just in case
        parsed_json["evidence_snippet"] = parsed_json["evidence_snippet"][:150]
        return parsed_json
    else:
        error_msg_detail = f"LLM Error: {type(api_error).__name__}" if api_error else "Invalid JSON structure received"
        print(f"Failed to get valid ownership structure. {error_msg_detail}");
        # Return UNRELATED with error context if possible
        return {
            "relationship_type": "UNRELATED",
            "evidence_snippet": f"Analysis failed: {error_msg_detail}",
            "source_url": "N/A"
        }


# --- Analysis Summary Generation ---
def generate_analysis_summary(extracted_data: Dict, query: str, exposures_count: int,
                              llm_provider: str, llm_model: str) -> str:
    """Generates a concise summary of the analysis findings."""
    # Keep this function exactly as defined in response #71
    print(f"\n--- Attempting Analysis Summary via {llm_provider} (Model: {llm_model}) ---");
    if not llm_provider or not llm_model: return "Summary generation skipped: Missing LLM config."
    summary_context = f"Original Query: {query}\n\nAnalysis Findings:\n";
    entities = extracted_data.get("entities", []); risks = extracted_data.get("risks", []); relationships = extracted_data.get("relationships", [])
    if not any([entities, risks, relationships, exposures_count > 0]):
         return "No significant data extracted or exposures identified to generate a summary."

    # Build context string more carefully
    summary_parts = []
    if entities: summary_parts.append(f"- Identified {len(entities)} entities, including: " + ", ".join([f"{e.get('name','?')}({e.get('type','?')})" for e in entities[:5]]) + ("..." if len(entities)>5 else "."))
    if risks:
        high_risks = [r for r in risks if isinstance(r, dict) and r.get('severity') == 'HIGH']
        med_risks = [r for r in risks if isinstance(r, dict) and r.get('severity') == 'MEDIUM']
        low_risks = [r for r in risks if isinstance(r, dict) and r.get('severity') == 'LOW']
        risk_summary = f"- Found {len(risks)} potential risks: {len(high_risks)} High, {len(med_risks)} Medium, {len(low_risks)} Low."
        # Add examples if available
        example_risks = []
        if high_risks: example_risks.append(f"High risk example: '{high_risks[0].get('description', '')[:80]}...'")
        elif med_risks: example_risks.append(f"Medium risk example: '{med_risks[0].get('description', '')[:80]}...'")
        if example_risks: risk_summary += f" ({'; '.join(example_risks)})"
        summary_parts.append(risk_summary)
    if relationships: summary_parts.append(f"- Mapped {len(relationships)} relationships, such as: " + "; ".join([f"{rel.get('entity1','?')} {rel.get('relationship_type','?')} {rel.get('entity2','?')}" for rel in relationships[:3]]) + ("..." if len(relationships)>3 else "."))
    if exposures_count > 0: summary_parts.append(f"- Identified {exposures_count} potential supply chain exposures requiring further investigation.")

    summary_context += "\n".join(summary_parts)

    # Refined Summary Prompt
    prompt = f"""You are a professional analyst summarizing research findings for an executive audience based *only* on the provided 'Analysis Findings'.
Your task is to write a concise, objective, and informative summary paragraph (3-5 sentences).
Focus on the most important findings: key entities involved, the overall risk profile (mentioning counts of high/medium risks if significant), and any identified supply chain exposures. Avoid jargon. Do not add opinions or information not present in the findings.

Analysis Findings:
{summary_context}

Generate ONLY the summary paragraph based *strictly* on the findings above. Do not include a salutation or any text before or after the paragraph."""

    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model); raw_content = ""
        print(f"Sending summary request to {llm_provider} ({model_name_used})...")

        if client_type == "openai_compatible":
            response = client_or_lib.chat.completions.create( model=model_name_used, messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=350 ); # Slightly higher temp, more tokens
            raw_content = response.choices[0].message.content.strip()
        elif client_type == "google_ai":
             # Standard safety, slightly higher temp, more tokens
            safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in [ "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT", ] ];
            generation_config = genai.types.GenerationConfig(temperature=0.5, max_output_tokens=350);
            response = client_or_lib.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings);
            # Check response validity
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 raw_content = response.text.strip()
            else:
                 print(f"Warning: Google AI summary generation blocked or empty. Feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
                 raw_content = "Summary generation failed due to API response issue." # Fallback message
        else: raise ValueError("Unknown client type")

        print(f"\nRaw {llm_provider} Response (Summary):\n>>>\n{raw_content}\n<<<")

        # Clean up summary (remove potential conversational prefixes/suffixes)
        cleaned_summary = raw_content;
        prefixes_to_remove = ["Okay, here's a summary:", "Here is a summary:", "Summary:", "Based on the findings:", "Based on the analysis:", "Here's a summary paragraph:", "Here's the summary:"]
        suffixes_to_remove = ["Let me know if you need more details."]
        for prefix in prefixes_to_remove:
             if cleaned_summary.lower().startswith(prefix.lower()): cleaned_summary = cleaned_summary[len(prefix):].strip()
        for suffix in suffixes_to_remove:
              if cleaned_summary.lower().endswith(suffix.lower()): cleaned_summary = cleaned_summary[:-len(suffix)].strip()

        # Basic quality check
        if not cleaned_summary or len(cleaned_summary) < 20 or "sorry" in cleaned_summary.lower() or "cannot provide" in cleaned_summary.lower():
             print("Warning: Summary seems too short, empty, or apologetic. Returning standard failure message.")
             return f"Could not generate a meaningful summary based on the extracted data."

        return cleaned_summary

    except Exception as e:
        print(f"ERROR during summary generation via {llm_provider}: {e}");
        traceback.print_exc()
        return f"Could not generate summary due to error: {type(e).__name__}"


# --- Local Testing Block ---
if __name__ == "__main__":
    print("\n--- Running Local NLP Processor Tests ---")
    print("NOTE: Local testing requires LLM API keys in .env and will use the configured keys.")
    import config # Import config here for the test block

    # Determine which provider to test based on available keys in config
    provider_to_test = None
    model_to_test = None
    if config.GOOGLE_AI_API_KEY:
        provider_to_test = "google_ai"
        model_to_test = config.DEFAULT_GOOGLE_AI_MODEL
        print("--> Testing with Google AI")
    elif config.OPENAI_API_KEY:
        provider_to_test = "openai"
        model_to_test = config.DEFAULT_OPENAI_MODEL
        print("--> Testing with OpenAI")
    elif config.OPENROUTER_API_KEY:
        provider_to_test = "openrouter"
        model_to_test = config.DEFAULT_OPENROUTER_MODEL
        print("--> Testing with OpenRouter")

    if not provider_to_test:
        print("No LLM API keys configured in .env for testing. Exiting.")
    else:
        # --- Test Keyword Translation ---
        print("\nTesting Keyword Translation...")
        sample_query_kw = "supply chain compliance issues 2023"
        sample_context_kw = "Baidu search in China for specific company supply chain info"
        translated_keywords_test = translate_keywords_for_context(sample_query_kw, sample_context_kw, provider_to_test, model_to_test)
        print(f"\nTranslated Keywords Result: {translated_keywords_test}")
        time.sleep(1) # Small delay

        # --- Test Data Extraction ---
        print("\nTesting Data Extraction...")
        sample_results_test = [
            {"title": "New Supply Chain Regulations Impact Global Trade - GovReport", "url": "https://example.gov/report-2023-supply-chain", "snippet": "The Regulatory Body C announced new compliance measures affecting manufacturers like Corporation A, especially those operating in Manufacturing Hub B. Financial penalties may apply from 2024.", "source": "google", "published_date": "2023-11-01"},
            {"title": "Corporation A Faces Scrutiny Over Supplier Ethics - Finance News", "url": "https://example.finance/corp-a-ethics", "snippet": "Concerns are rising about Corporation A's sourcing from suppliers in Manufacturing Hub B, potentially violating international labor standards. This poses a significant reputational risk.", "source": "google", "published_date": "2023-10-15"},
            {"title": "Baidu Search Results Snippet (Example)", "url": "https://example.baidu.com/search-result-3", "snippet": "Analysis shows shifting trade patterns in Manufacturing Hub B. Regulatory Body C is monitoring closely. Investors are watching Corporation A.", "source": "baidu", "published_date": "2023-12-01"},
            {"title": "Empty Snippet Example", "url": "https://example.com/empty", "snippet": " ", "source": "google", "published_date": "2023-09-01"},
            {"title": "Missing URL Example", "url": "N/A", "snippet": "This snippet has text but no URL.", "source": "google", "published_date": "2023-09-02"}

        ]
        sample_extraction_context_test = "supply chain risks and regulations involving Corporation A and Manufacturing Hub B"
        extracted_data_test = extract_data_from_results(sample_results_test, sample_extraction_context_test, provider_to_test, model_to_test)
        print("\nFinal Extracted Data Result:")
        print(json.dumps(extracted_data_test, indent=2))
        time.sleep(1) # Small delay

        # ===> ADD Local Test for Ownership Extraction <===
        print("\nTesting Ownership Extraction...")
        # Reuse provider/model from above if already determined
        if provider_to_test:
            sample_ownership_snippets = [
                {'url': 'http://example.com/report1', 'snippet': 'In 2022, ParentCorp acquired a 60% controlling stake in SubCo Inc.'},
                {'url': 'http://example.com/news1', 'snippet': 'ParentCorp holds a significant minority investment (approx 30%) in AffiliateCorp.'},
                {'url': 'http://example.com/report2', 'snippet': 'ParentCorp and Unrelated Inc entered into a strategic partnership.'},
                {'url': 'http://example.com/sec', 'snippet': 'Exhibit 21 lists SubCo Inc as a subsidiary of ParentCorp.'}, # Added mention of parent
                {'url': 'http://example.com/other', 'snippet': 'SubCo Inc is a company based in Delaware.'} # Snippet mentions one but not both
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
            time.sleep(1) # Small delay

            ownership_result_2 = extract_ownership_relationships(
                parent_entity_name="ParentCorp",
                related_entity_name="AffiliateCorp",
                text_snippets=sample_ownership_snippets,
                llm_provider=provider_to_test,
                llm_model=model_to_test
            )
            print("\nOwnership Result (ParentCorp owning AffiliateCorp):")
            print(json.dumps(ownership_result_2, indent=2))
            time.sleep(1) # Small delay

            ownership_result_3 = extract_ownership_relationships(
                parent_entity_name="ParentCorp",
                related_entity_name="Unrelated Inc",
                text_snippets=sample_ownership_snippets,
                llm_provider=provider_to_test,
                llm_model=model_to_test
            )
            print("\nOwnership Result (ParentCorp owning Unrelated Inc):")
            print(json.dumps(ownership_result_3, indent=2))
            time.sleep(1) # Small delay

            # Test case where no snippets mention both
            ownership_result_4 = extract_ownership_relationships(
                parent_entity_name="ParentCorp",
                related_entity_name="NonExistent Inc",
                text_snippets=sample_ownership_snippets,
                llm_provider=provider_to_test,
                llm_model=model_to_test
            )
            print("\nOwnership Result (ParentCorp owning NonExistent Inc):")
            print(json.dumps(ownership_result_4, indent=2))
            time.sleep(1) # Small delay

        else:
            print("Skipping ownership tests - No LLM configured.")


        # ===> ADD Local Test for Summary Generation <===
        print("\nTesting Analysis Summary Generation...")
        if provider_to_test:
             # Use the data extracted earlier
             summary_test = generate_analysis_summary(
                 extracted_data=extracted_data_test,
                 query=sample_extraction_context_test, # Use extraction context as query for test
                 exposures_count=2, # Simulate finding 2 exposures
                 llm_provider=provider_to_test,
                 llm_model=model_to_test
             )
             print("\nGenerated Summary:")
             print(summary_test)

             # Test summary with empty data
             summary_empty_test = generate_analysis_summary(
                 extracted_data={"entities": [], "risks": [], "relationships": []},
                 query="Empty data test",
                 exposures_count=0,
                 llm_provider=provider_to_test,
                 llm_model=model_to_test
             )
             print("\nGenerated Summary (Empty Data):")
             print(summary_empty_test)
        else:
             print("Skipping summary tests - No LLM configured.")


    print("\n--- Local Tests Complete ---")