# nlp_processor.py

import json
from typing import List, Dict, Any, Optional, Mapping, Tuple, Union
import time
import re
import traceback
import sys

# Import config
import config

# Import LLM libraries directly, check availability later
try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Import pycountry at module level but check availability in functions
try:
    import pycountry
    pycountry_available_nlp = True # Use a different name to avoid conflict if orchestrator also imports
except ImportError:
    pycountry = None
    pycountry_available_nlp = False
    print("Warning: 'pycountry' not installed. Language name resolution limited in NLP.")


# Define LLM call delay for rate limiting
LLM_RELATIONSHIP_CHECK_DELAY = 1.5
LLM_TRANSLATION_DELAY = 0.3


def _get_llm_client_and_model(provider: str, model_name: str):
    """Initializes and returns the LLM client and effective model name. Checks for library availability."""
    api_key = None
    client_or_lib = None
    client_type = None
    effective_model_name = model_name

    # Check for essential config attributes
    if not hasattr(config, 'OPENAI_API_KEY') or not hasattr(config, 'OPENROUTER_API_KEY') or not hasattr(config, 'GOOGLE_AI_API_KEY') or \
       not hasattr(config, 'OPENROUTER_BASE_URL') or not hasattr(config, 'OPENROUTER_HEADERS') or \
       not hasattr(config, 'DEFAULT_GOOGLE_AI_MODEL') or not hasattr(config, 'DEFAULT_OPENAI_MODEL') or not hasattr(config, 'DEFAULT_OPENROUTER_MODEL'):
         print("Warning: Essential config attributes missing in _get_llm_client_and_model.")
         # Return gracefully if config is incomplete
         return None, None, None


    if provider == "openai":
        # Check for library availability
        if openai is None:
             raise ImportError("OpenAI library is required but not installed.")

        api_key = config.OPENAI_API_KEY
        if not api_key: raise ValueError("OpenAI API Key missing.")
        try:
            is_likely_openai_model = any(m in model_name.lower() for m in ["gpt-4", "gpt-3.5", "dall-e", "whisper"])
            if not is_likely_openai_model and config.OPENROUTER_API_KEY and config.OPENROUTER_BASE_URL:
                 client_or_lib = openai.OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL, default_headers=config.OPENROUTER_HEADERS)
                 client_type = "openai_compatible"
            else:
                client_or_lib = openai.OpenAI(api_key=api_key)
                client_type = "openai_compatible"

        except Exception as e:
            print(f"ERROR init OpenAI: {e}")
            raise e

    elif provider == "openrouter":
        # Check for library availability
        if openai is None:
             raise ImportError("OpenAI library (used by OpenRouter client) is required but not installed.")

        api_key = config.OPENROUTER_API_KEY
        if not api_key: raise ValueError("OpenRouter API Key missing.")
        if not config.OPENROUTER_BASE_URL: raise ValueError("OpenRouter Base URL missing.")
        try:
            client_or_lib = openai.OpenAI(api_key=api_key, base_url=config.OPENROUTER_BASE_URL, default_headers=config.OPENROUTER_HEADERS)
            client_type = "openai_compatible"
        except Exception as e:
            print(f"ERROR init OpenRouter: {e}")
            raise e

    elif provider == "google_ai":
        # Check for library availability
        if genai is None:
            raise ImportError("Google Generative AI library is required but not installed.")

        api_key = config.GOOGLE_AI_API_KEY
        if not api_key: raise ValueError("Google AI API Key missing.")
        try:
            genai.configure(api_key=api_key)
            try:
                list_models_response = genai.list_models()
                available_models = [m.name for m in list_models_response if m.name.startswith('models/')]
                if model_name not in available_models:
                     prefixed_model_name = f"models/{model_name}" if not model_name.startswith("models/") else model_name
                     if prefixed_model_name in available_models:
                          effective_model_name = prefixed_model_name
                     else:
                         print(f"Warning: Google AI model '{model_name}' (or '{prefixed_model_name}') not found. Using default: {config.DEFAULT_GOOGLE_AI_MODEL}")
                         effective_model_name = config.DEFAULT_GOOGLE_AI_MODEL
                         if effective_model_name not in available_models:
                              fallback_model = "models/gemini-1.5-flash-latest"
                              if fallback_model in available_models:
                                   effective_model_name = fallback_model
                              else:
                                   raise ValueError(f"Specified Google AI model '{model_name}', default '{config.DEFAULT_GOOGLE_AI_MODEL}', and fallback '{fallback_model}' not available.")

            except Exception as list_models_e:
                 print(f"Warning: Could not list Google AI models to verify '{model_name}': {list_models_e}. Proceeding with specified model name.")
                 pass

            client_or_lib = genai.GenerativeModel(model_name=effective_model_name)
            client_type = "google_ai"
        except Exception as e:
            print(f"ERROR init Google AI: {e}")
            raise e
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    if client_or_lib is None:
        raise ConnectionError(f"Failed LLM client init: {provider} for model {model_name}")

    return client_or_lib, client_type, effective_model_name

# --- Keyword Generation ---
def translate_keywords_for_context(original_query: str, target_context: str,
                                   llm_provider: str, llm_model: str) -> List[str]:
    """Generates search keywords for a given context. Checks for LLM availability."""
    print(f"\n--- Attempting Keyword Generation via {llm_provider} (Model: {llm_model}) ---")
    if not llm_provider or not llm_model:
        return [original_query]

    # Check for LLM availability before making the call
    if _get_llm_client_and_model(llm_provider, llm_model)[0] is None:
         print(f"Warning: LLM client not available for keyword generation using {llm_provider}. Skipping keyword generation.")
         return [original_query]


    prompt = f"""Expert keyword generator: Given the query and context below, provide a list of 3-5 relevant ENGLISH search keywords suitable for the context.
Initial Query: {original_query}
Target Search Context: {target_context}
IMPORTANT: Your entire response must contain ONLY the comma-separated list of keywords. Do NOT include any other text, explanation, or formatting."""

    raw_content = ""
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)

        if client_type == "openai_compatible":
            request_params = {"model": model_name_used, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 150}
            response = client_or_lib.chat.completions.create(**request_params)
            raw_content = response.choices[0].message.content.strip()

        elif client_type == "google_ai":
            # Check for library availability
            if genai is None:
                 print("Warning: Google Generative AI library not available for keyword generation.")
                 return [original_query]

            safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in genai.types.HarmCategory if c != genai.types.HarmCategory.HARM_CATEGORY_UNSPECIFIED]
            generation_config = genai.types.GenerationConfig(temperature=0.2, max_output_tokens=150)
            response = client_or_lib.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
            if not response.candidates:
                feedback = getattr(response, 'prompt_feedback', None)
                block_reason = getattr(feedback, 'block_reason', 'Unknown')
                print(f"--- ERROR: Google AI BLOCKED. Reason: {block_reason}. ---")
                raw_content = ""
            else:
                raw_content = response.text.strip()

        else:
            raise ValueError("Unknown client type.")

        prefixes_to_remove = ["Sure, here are the keywords:", "Here are the keywords:", "Okay, here is the list:", "Here is the list:", "Of course! Here are the keywords:"]
        cleaned_content = raw_content
        for prefix in prefixes_to_remove:
            if cleaned_content.lower().startswith(prefix.lower()):
                cleaned_content = cleaned_content[len(prefix):].strip()
                break

        if '\n' in cleaned_content:
             cleaned_content = cleaned_content.split('\n')[0]

        cleaned_content = cleaned_content.replace('"', '').replace("'", '').strip()

        keywords = [kw.strip() for kw in cleaned_content.split(',') if kw.strip()]

        conversational_fillers = ["sorry", "apologize", "cannot", "unable", "provide", "based", "above", "snippets", "context", "hello", "hi", "greetings"]
        if not keywords or any(re.search(r'\b' + word + r'\b', cleaned_content.lower()) for word in conversational_fillers):
             print("Warning: Keyword response conversational/empty/unhelpful. Returning original.")
             return [original_query]

        print(f"Parsed Keywords: {keywords}")
        return keywords

    except Exception as e:
        print(f"ERROR during keyword generation: {e}")
        traceback.print_exc()
        return [original_query]

# --- Translation Functions ---
def translate_text(text: str, target_language: str, llm_provider: str, llm_model: str) -> Optional[str]:
    """Translates a single piece of text using the LLM. Checks for LLM availability."""
    if not text or not isinstance(text, str) or not llm_provider or not llm_model:
         return None

    # Check for LLM availability before making the call
    if _get_llm_client_and_model(llm_provider, llm_model)[0] is None:
         print(f"Warning: LLM client not available for translation using {llm_provider}. Skipping translation.")
         return None

    target_lang_name = target_language
    # Check if pycountry is available before using it
    if pycountry_available_nlp and pycountry is not None:
        try:
            lang = pycountry.languages.get(alpha_2=target_language.lower())
            if lang:
                 target_lang_name = lang.name
            elif target_language.lower() == 'zh':
                 target_lang_name = "Chinese"
            elif target_language.lower() == 'en':
                 target_lang_name = "English"
        except Exception: # Catch any exception from pycountry lookup
            pass


    prompt = f"""Translate the following text into {target_lang_name}.
IMPORTANT: Provide ONLY the translated text. Do NOT include any introductory phrases, explanations, or markdown.

Text to translate:
{text}"""

    raw_content = None
    api_error = None
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)

        if client_type == "openai_compatible":
            # Check for library availability
            if openai is None:
                 print("Warning: OpenAI library not available for translation.")
                 return None

            request_params = {"model": model_name_used, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": min(len(text) * 3, 2000)}
            response = client_or_lib.chat.completions.create(**request_params)
            raw_content = response.choices[0].message.content

        elif client_type == "google_ai":
            # Check for library availability
            if genai is None:
                 print("Warning: Google Generative AI library not available for translation.")
                 return None

            safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in genai.types.HarmCategory if c != genai.types.HarmCategory.HARM_CATEGORY_UNSPECIFIED]
            generation_config = genai.types.GenerationConfig(temperature=0.1, max_output_tokens=min(len(text) * 3, 2000));
            response_obj = client_or_lib.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings);

            if response_obj and response_obj.candidates and response_obj.candidates[0].content and response_obj.candidates[0].content.parts:
                 raw_content = response_obj.text
            else:
                 feedback = getattr(response_obj, 'prompt_feedback', None)
                 block_reason = getattr(feedback, 'block_reason', 'Unknown')
                 print(f"Warning: Google AI Translation BLOCKED or EMPTY. Reason: {block_reason}.");
                 api_error = ValueError(f"Google AI blocked: {block_reason}")
                 raw_content = None

        else:
            raise ValueError("Unknown client type.")

        if raw_content:
            cleaned_content = raw_content.strip()
            cleaned_content = re.sub(r'^["\']|["\']$', '', cleaned_content)
            cleaned_content = re.sub(r'^```.*?\n', '', cleaned_content, flags=re.DOTALL)
            cleaned_content = re.sub(r'\n```$', '', cleaned_content)
            cleaned_content = cleaned_content.strip()

            conversational_fillers = ["sorry", "apologize", "cannot", "unable", "translate", "based on", "above", "text", "hello", "hi"]
            if any(re.search(r'\b' + word + r'\b', cleaned_content.lower()) for word in conversational_fillers):
                print(f"Warning: Translation response looks conversational: '{cleaned_content[:100]}...'")
                return None

            return cleaned_content

    except Exception as e:
        print(f"ERROR during translation: {type(e).__name__}: {e}")
        traceback.print_exc()
        api_error = e
        return None

def translate_snippets(snippets: List[Dict[str, Any]], target_language: str, llm_provider: str, llm_model: str) -> List[Dict[str, Any]]:
    """Translates a list of snippets to the target language. Checks for LLM availability."""
    if not snippets or not llm_provider or not llm_model:
         return snippets

    # Check for LLM availability before starting translation
    if _get_llm_client_and_model(llm_provider, llm_model)[0] is None:
         print(f"Warning: LLM client not available for snippet translation using {llm_provider}. Skipping translation.")
         return snippets

    print(f"--- Attempting to translate {len(snippets)} snippets to '{target_language}' ---")
    translated_snippets = []

    for snippet_data in snippets:
        # Ensure snippet_data is valid and has a snippet before attempting translation
        if not isinstance(snippet_data, dict) or not snippet_data.get('snippet') or not isinstance(snippet_data.get('snippet'), str):
             translated_snippets.append(snippet_data) # Include invalid/empty snippets as is
             continue

        original_snippet = snippet_data['snippet']
        translated_snippet_content = translate_text(original_snippet, target_language, llm_provider, llm_model)

        if translated_snippet_content is not None:
            translated_s = snippet_data.copy()
            translated_s['snippet'] = translated_snippet_content
            # Store original language and add 'translated_from' tag
            translated_s['original_language'] = snippet_data.get('original_language', 'unknown')
            translated_s['translated_from'] = translated_s['original_language'] # Record what it was translated from
            # Keep the original snippet if needed for reference (optional)
            # translated_s['original_snippet'] = original_snippet

            translated_snippets.append(translated_s)
        else:
            # If translation failed or returned None, keep the original snippet
            print(f"Warning: Failed to translate snippet (URL: {snippet_data.get('url', 'N/A')}). Including original.")
            translated_snippets.append(snippet_data)

        time.sleep(LLM_TRANSLATION_DELAY)

    print(f"--- Finished snippet translation. Translated/processed {len(translated_snippets)} snippets. ---")
    return translated_snippets

def _prepare_context_text(search_results: List[Dict[str, Any]]) -> Tuple[str, int]:
    """Prepares the formatted context string from search results."""
    context_text = ""
    max_chars = 8000
    char_count = 0
    added_snippets = 0
    for result in search_results:
        # Ensure result is a dict and has basic required fields before processing
        if not isinstance(result, dict) or not result.get('url'):
             # print(f"Skipping invalid result in _prepare_context_text: {result}") # Optional logging
             continue

        snippet = result.get('snippet', '')
        title = result.get('title', '')
        url = result.get('url', 'N/A')
        original_lang = result.get('original_language') # Get original_language (can be None)
        translated_from = result.get('translated_from') # Get translated_from tag


        translation_info = ""
        if translated_from:
             # Indicate if it was translated and from which language
             translation_info = f" (Translated from {translated_from})"
        # Check if original_lang is a string AND not English AND it hasn't been translated
        elif isinstance(original_lang, str) and original_lang.lower() not in ['en', 'english'] and not translated_from:
             # Indicate original language if known and not English (and not translated)
             translation_info = f" (Original Language: {original_lang})"


        # Only include snippets that are not empty or just whitespace and have a valid URL
        if snippet and not snippet.isspace() and url != 'N/A':
            # Optional: enforce a minimum snippet length to filter very short ones
            if isinstance(snippet, str) and len(snippet) < 50: continue # Skip snippets shorter than 50 chars (check if snippet is string)

            entry = f"\n---\nSourceURL: {url}{translation_info}\nTitle: {title}\nSnippet: {snippet}\n"
            if char_count + len(entry) <= max_chars:
                context_text += entry
                char_count += len(entry)
                added_snippets += 1
            else:
                # Stop adding snippets if max_chars is reached
                break
    return context_text, added_snippets

def _call_llm_and_parse_json(prompt: str, llm_provider: str, llm_model: str,
                             function_name: str, attempt_json_mode: bool = True) -> Optional[Dict]:
    """Helper function to call LLM expecting a JSON response and parse it. Checks for LLM availability."""
    raw_content = None
    api_error = None
    parsed_json = None
    try:
        # Check for LLM availability before making the call
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        if client_or_lib is None:
             print(f"--- [{function_name}] LLM client not available for {llm_provider}. Skipping LLM call. ---")
             return None

        start_time = time.time()
        response_obj = None

        if client_type == "openai_compatible":
            # Check for library availability
            if openai is None:
                 print(f"--- [{function_name}] OpenAI library not available. Skipping LLM call. ---")
                 return None

            request_params = {"model": model_name_used, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
            # Attempt JSON mode only if the provider is OpenAI AND the model supports it
            # OpenRouter often supports JSON mode even with Google models, but it's safer to tie it to OpenAI provider + specific models
            if attempt_json_mode and llm_provider == "openai" and any(m in model_name_used.lower() for m in ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]):
                 request_params["response_format"] = {"type": "json_object"}

            try:
                 response = client_or_lib.chat.completions.create(**request_params)
                 raw_content = response.choices[0].message.content
                 response_obj = response
            except Exception as e_call:
                 # Retry without JSON mode if the error suggests response_format issue
                 if attempt_json_mode and llm_provider == "openai" and "response_format" in request_params and \
                    any(err_txt in str(e_call).lower() for err_txt in ["response_format", "json_object", "messages must contain the word 'json'", "invalid response format"]):
                     print(f"--- [{function_name}] WARNING: OpenAI JSON mode likely failed ({type(e_call).__name__}). Retrying WITHOUT JSON mode... ---")
                     del request_params["response_format"]
                     try:
                         response = client_or_lib.chat.completions.create(**request_params)
                         raw_content = response.choices[0].message.content
                         response_obj = response
                     except Exception as e_retry:
                         print(f"--- [{function_name}] ERROR on retry without JSON mode: {type(e_retry).__name__}: {e_retry} ---")
                         api_error = e_retry
                         raw_content = None
                 else:
                      # Log other OpenAI API call errors
                      print(f"--- [{llm_provider}] API call error: {type(e_call).__name__}: {e_call} ---")
                      traceback.print_exc() # Print traceback for unexpected API errors
                      api_error = e_call
                      raw_content = None

        elif client_type == "google_ai":
            # Check for library availability
            if genai is None:
                 print(f"--- [{function_name}] Google Generative AI library not available. Skipping LLM call. ---")
                 return None

            safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in genai.types.HarmCategory if c != genai.types.HarmCategory.HARM_CATEGORY_UNSPECIFIED]
            # Attempt JSON mime type only if the model name suggests it supports it (e.g., Gemini 1.5)
            try_json_mime = attempt_json_mode and "gemini-1.5" in model_name_used.lower()
            generation_config = genai.types.GenerationConfig(temperature=0.1, response_mime_type="application/json" if try_json_mime else None )

            try:
                response = client_or_lib.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings);
                response_obj = response
                if not response.candidates:
                    feedback = getattr(response, 'prompt_feedback', None)
                    block_reason = getattr(feedback, 'block_reason', 'Unknown')
                    print(f"--- ERROR: Google AI BLOCKED. Reason: {block_reason}. ---")
                    api_error = ValueError(f"Google AI blocked: {block_reason}")
                    raw_content = None
                elif hasattr(response_obj, 'text'):
                     raw_content = response_obj.text
                else:
                    # Handle cases where response_obj.text is missing but candidates exist
                    print(f"--- WARNING: Google AI response has candidates but no text attribute. Response object: {response_obj} ---")
                    raw_content = None # Treat as empty response

            except Exception as e_google:
                print(f"--- [{function_name}] ERROR Google AI call: {type(e_google).__name__}: {e_google} ---")
                traceback.print_exc()
                api_error = e_google
                raw_content = None
        else:
            raise ValueError("Unknown client type.")

        duration = time.time() - start_time
        # print(f"--- [{function_name}] LLM call took {duration:.2f} seconds. ---") # Optional timing log


        if raw_content:
            # Clean and parse JSON from the raw content
            try:
                cleaned_content = raw_content.strip()
                # Remove markdown code block syntax if present
                if cleaned_content.startswith("```json"): cleaned_content = cleaned_content[7:]
                if cleaned_content.endswith("```"): cleaned_content = cleaned_content[:-3]
                cleaned_content = cleaned_content.strip()

                # Find the first '{' and last '}' to isolate the JSON object
                json_start = cleaned_content.find('{')
                json_end = cleaned_content.rfind('}') + 1

                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_str = cleaned_content[json_start:json_end]
                    # Basic check that it looks like a JSON object
                    if json_str.startswith('{') and json_str.endswith('}'):
                         parsed_json = json.loads(json_str)
                    else:
                         # If braces were found but don't form a valid JSON object string,
                         # try parsing the whole cleaned content as a fallback.
                         print(f"--- [{function_name}] WARNING: Found braces, but string doesn't look like a valid JSON object. Attempting full cleaned content parse.")
                         try:
                              parsed_json = json.loads(cleaned_content)
                         except Exception as e_fallback_parse:
                              print(f"--- [{function_name}] ERROR fallback parsing cleaned content: {type(e_fallback_parse).__name__}: {e_fallback_parse} ---")
                              api_error = e_fallback_parse
                              parsed_json = None
                else:
                    # If no braces were found, assume the whole cleaned content should be JSON
                    print(f"--- [{function_name}] WARNING: Could not find JSON braces. Attempting full cleaned content parse.")
                    try:
                         parsed_json = json.loads(cleaned_content)
                    except Exception as e_full_parse:
                         print(f"--- [{function_name}] ERROR parsing cleaned content: {type(e_full_parse).__name__}: {e_full_parse} ---")
                         api_error = e_full_parse
                         parsed_json = None

            except json.JSONDecodeError as json_e:
                print(f"--- [{function_name}] ERROR decoding JSON: {json_e} ---")
                # Print the content that failed to decode (potentially large)
                print(f"--- [{function_name}] Content that failed to decode (first 500 chars): {cleaned_content[:500]} ---")
                api_error = json_e
                parsed_json = None
            except Exception as parse_e:
                print(f"--- [{function_name}] ERROR during JSON parsing: {type(parse_e).__name__}: {parse_e} ---")
                traceback.print_exc()
                api_error = parse_e
                parsed_json = None
        else:
             print(f"--- [{function_name}] ERROR: Raw content from LLM was None or empty. Cannot parse. ---")
             # Set a generic error if no API error occurred but content was empty
             if api_error is None:
                 api_error = ValueError("LLM returned None or empty content")

        # Final check: ensure the parsed result is a dictionary as expected
        if parsed_json is not None and isinstance(parsed_json, dict):
             print(f"--- [{function_name}] Successfully Parsed JSON response. ---")
             return parsed_json
        elif api_error is not None:
             # If an API error occurred, report it
             print(f"--- [{function_name}] ERROR: LLM call/parsing failed. Error: {type(api_error).__name__} ---")
             return None # Return None on error
        else:
             # If parsed_json is not a dict (e.g., list, string, None after fallbacks) or some other issue
             print(f"--- [{function_name}] ERROR: JSON obtained but not a dict, or unknown parsing issue. Parsed type: {type(parsed_json).__name__} ---")
             # Optionally print the parsed content if not a dict for debugging
             # print(f"--- [{function_name}] Parsed content: {parsed_json} ---")
             return None # Return None if the final result isn't the expected format

    except Exception as e:
        print(f"--- [{function_name}] ERROR during LLM call/setup (outer catch): {type(e).__name__}: {e} ---")
        traceback.print_exc()
        return None

def extract_entities_only(search_results: List[Dict[str, Any]], extraction_context: str,
                          llm_provider: str, llm_model: str) -> List[Dict]:
    """Extracts entities (COMPANY, ORGANIZATION, REGULATORY_AGENCY, SANCTION) based on schema. Checks for LLM availability."""
    print("\n--- [NLP Entities] Attempting Entity Extraction ---")
    function_name = "NLP Entities"
    if not search_results: print(f"--- [{function_name}] No search results, skipping. ---"); return []
    # Check for LLM availability
    if _get_llm_client_and_model(llm_provider, llm_model)[0] is None:
         print(f"--- [{function_name}] LLM client not available for {llm_provider}. Skipping entity extraction. ---")
         return []

    context_text, added_snippets = _prepare_context_text(search_results)
    if added_snippets == 0: print(f"--- [{function_name}] No valid snippets. ---"); return []
    print(f"--- [{function_name}] Prepared context using {added_snippets} snippets. ---")

    # Modified prompt to clarify REGULATORY_AGENCY definition
    entity_schema = """- "entities": Array of objects. Each MUST have: "name" (string, English), "type" (enum: COMPANY, ORGANIZATION, REGULATORY_AGENCY, SANCTION), "mentions" (array of SourceURLs where name appeared, MUST NOT be empty)."""
    prompt = f"""Analyze text snippets based on context: "{extraction_context}". Extract ONLY entities of type COMPANY, ORGANIZATION, REGULATORY_AGENCY, or SANCTION matching the schema below, found ONLY in the snippets.

Schema:
{entity_schema}

Definitions:
- COMPANY: A business entity.
- ORGANIZATION: A group or body (can include government bodies, non-profits, international organizations like OECD, although prioritize regulatory bodies for REGULATORY_AGENCY type).
- REGULATORY_AGENCY: A specific governmental **body, department, or authority** responsible for overseeing or enforcing regulations or sanctions. **Do NOT list country names (like 'United States', 'China', 'UK') as REGULATORY_AGENCY.** Examples: "Securities and Exchange Commission", "Office of Foreign Assets Control", "State Administration for Market Regulation (SAMR)", "European Commission", "China Securities Regulatory Commission (CSRC)".
- SANCTION: A specific measure, list, or action taken against an entity, often named (e.g., "Magnitsky Act sanctions", "Entity List designation", "OFAC's SDN List").

Response MUST be ONLY a single valid JSON object like {{"entities": [...]}}. Use empty array [] if no entities found. Do not include entities of other types (PERSON, LOCATION, REGULATION, OTHER, etc., unless explicitly listed in enum). No explanations or markdown.

Begin analysis of text snippets:
{context_text}
"""
    parsed_json = _call_llm_and_parse_json(prompt, llm_provider, llm_model, function_name, attempt_json_mode=True)

    validated_entities = []
    allowed_entity_types = ["COMPANY", "ORGANIZATION", "REGULATORY_AGENCY", "SANCTION"]
    # Add a simple check to filter common country names if they are tagged as REGULATORY_AGENCY by mistake
    common_country_names_lower = {"united states", "us", "china", "cn", "united kingdom", "uk", "germany", "de", "india", "in", "france", "fr", "japan", "jp", "canada", "ca", "australia", "au"} # Add more as needed
    # Add known broad non-Chinese organizations that might be misidentified as companies or regulators
    common_non_chinese_orgs_lower = {"oecd", "nato", "un", "world bank", "imf", "european union", "eu"}

    if parsed_json is not None and isinstance(parsed_json.get("entities"), list):
        for entity in parsed_json["entities"]:
             # Ensure the item is a dictionary before trying to access keys
             if not isinstance(entity, dict):
                  print(f"--- [{function_name}] Skipping invalid entity item (not a dictionary): {entity} ---")
                  continue

             entity_name = entity.get("name")
             entity_type = entity.get("type")
             entity_mentions = entity.get("mentions")

             if entity_name and isinstance(entity_name, str) and entity_type in allowed_entity_types and \
                isinstance(entity_mentions, list) and entity_mentions and all(isinstance(m, str) for m in entity_mentions):

                 entity_name_lower = entity_name.lower()

                 # Post-processing check: If type is REGULATORY_AGENCY and name is a common country name, filter it out
                 if entity_type == "REGULATORY_AGENCY" and entity_name_lower in common_country_names_lower:
                      print(f"--- [{function_name}] Filtering out likely incorrect REGULATORY_AGENCY '{entity_name}' (common country name). ---")
                      continue # Skip adding this entity

                 # Post-processing check: If type is COMPANY or ORGANIZATION and name is a common non-Chinese org, filter it out
                 if entity_type in ["COMPANY", "ORGANIZATION"] and entity_name_lower in common_non_chinese_orgs_lower:
                      print(f"--- [{function_name}] Filtering out likely irrelevant COMPANY/ORGANIZATION '{entity_name}' (common non-Chinese org). ---")
                      continue # Skip adding this entity


                 validated_entities.append(entity) # Add the entity if it passes checks
             else: print(f"--- [{function_name}] Skipping invalid or incomplete entity item (missing name/type/mentions or invalid types): {entity} ---")
    else:
         # Log if the expected 'entities' list was not found or was not a list
         print(f"--- [{function_name}] Failed to parse valid 'entities' list from LLM response. Parsed content type: {type(parsed_json).__name__}. Content: {parsed_json} ---")

    print(f"--- [{function_name}] Returning {len(validated_entities)} validated entities ({', '.join(allowed_entity_types)} only). ---")
    return validated_entities

def extract_risks_only(search_results: List[Dict[str, Any]], extraction_context: str,
                       llm_provider: str, llm_model: str) -> List[Dict]:
    """Extracts only risks (desc, severity, source_urls) based on the schema. Checks for LLM availability."""
    print("\n--- [NLP Risks] Attempting Risk Extraction (without related entities) ---")
    function_name = "NLP Risks"
    if not search_results: print(f"--- [{function_name}] No search results, skipping. ---"); return []
    # Check for LLM availability
    if _get_llm_client_and_model(llm_provider, llm_model)[0] is None:
         print(f"--- [{function_name}] LLM client not available for {llm_provider}. Skipping risk extraction. ---")
         return []

    context_text, added_snippets = _prepare_context_text(search_results)
    if added_snippets == 0: print(f"--- [{function_name}] No valid snippets. ---"); return []
    print(f"--- [{function_name}] Prepared context using {added_snippets} snippets. ---")

    # Updated risk schema to potentially include risk_category
    risk_schema = """- "risks": Array of objects. Each MUST have: "description" (string, English, summary of the risk, keep concise, about 4-10 words), "severity" (enum: LOW, MEDIUM, HIGH, SEVERE), "risk_category" (string, e.g., 'Compliance', 'Financial', 'Environmental', 'Labor', 'Governance', 'Supply Chain'), "source_urls" (array of SourceURLs where risk mentioned, MUST NOT be empty)."""
    prompt = f"""Analyze text snippets based on context: "{extraction_context}". Extract ONLY potential risks matching the schema below, found ONLY in the snippets.

Schema:
{risk_schema}

Infer SEVERE for major violations/penalties. HIGH for significant issues. MEDIUM default.
Keep the "description" field concise, a summary of the risk in about 4-10 words.
Assign a single, relevant "risk_category" from the examples provided or infer a new one.
Response MUST be ONLY a single valid JSON object like {{"risks": [...]}}. Use empty array [] if no risks found. No explanations or markdown.

Begin analysis of text snippets:
{context_text}
"""
    parsed_json = _call_llm_and_parse_json(prompt, llm_provider, llm_model, function_name, attempt_json_mode=True)

    validated_risks = []
    if parsed_json is not None and isinstance(parsed_json.get("risks"), list):
        for risk in parsed_json["risks"]:
            # Ensure the item is a dictionary before checking keys
            if not isinstance(risk, dict):
                 print(f"--- [{function_name}] Skipping invalid risk item (not a dictionary): {risk} ---")
                 continue

            risk_desc = risk.get("description")
            risk_urls = risk.get("source_urls")
            risk_severity = risk.get("severity")
            risk_category = risk.get("risk_category") # Extract category

            if risk_desc and isinstance(risk_desc, str) and isinstance(risk_urls, list) and risk_urls and all(isinstance(u, str) for u in risk_urls):
                if risk_severity not in ["LOW", "MEDIUM", "HIGH", "SEVERE"]: risk["severity"] = "MEDIUM"
                # Ensure risk_category is a string, default if missing or invalid
                if not risk_category or not isinstance(risk_category, str): risk["risk_category"] = "UNKNOWN"
                else: risk["risk_category"] = risk_category.strip() # Clean up category string

                risk["related_entities"] = [] # Initialize related_entities list
                risk['_source_type'] = 'snippet' # Add internal source type
                validated_risks.append(risk)
            else: print(f"--- [{function_name}] Skipping invalid risk item (missing desc/urls or invalid types): {risk} ---")
    else:
         # Log if the expected 'risks' list was not found or was not a list
         print(f"--- [{function_name}] Failed to parse valid 'risks' list from LLM response. Parsed content type: {type(parsed_json).__name__}. Content: {parsed_json} ---")


    print(f"--- [{function_name}] Returning {len(validated_risks)} validated risks (initially without related_entities). ---")
    return validated_risks

def link_entities_to_risk(risks: List[Dict],
                          list_of_entity_names: List[str],
                          all_snippets_map: Mapping[str, Dict[str, Any]],
                          llm_provider: str, llm_model: str) -> List[Dict]:
    """
    Takes extracted risks and entity names (from snippets), and uses LLM to populate the 'related_entities' field for each risk
    based on the original snippets associated with the risk's source URLs. Returns updated risks list. Checks for LLM availability.
    """
    print(f"\n--- [NLP Linker] Attempting to Link Entities (from snippets) to Risks (from snippets) ---")
    function_name = "NLP Linker"
    # Filter for risks that came from snippets and need linking
    snippet_risks_to_link = [r for r in risks if isinstance(r, dict) and r.get('_source_type', 'snippet') == 'snippet' and "related_entities" not in r]
    # Keep risks that already have related entities (e.g., from structured data) or are not from snippets
    other_risks = [r for r in risks if not isinstance(r, dict) or r not in snippet_risks_to_link]


    if not snippet_risks_to_link: print(f"--- [{function_name}] No snippet risks provided that need linking, skipping. ---"); return risks
    if not list_of_entity_names:
        print(f"--- [{function_name}] No entity names (from snippets) provided, cannot link snippet risks. Returning risks as is. ---")
        # Ensure snippet risks that needed linking get an empty related_entities list
        for risk in snippet_risks_to_link:
             if isinstance(risk, dict):
                  risk["related_entities"] = []
                  # Re-add to other_risks list to ensure all originals are returned
                  other_risks.append(risk)
        return other_risks


    # Check for LLM availability before starting linking
    if _get_llm_client_and_model(llm_provider, llm_model)[0] is None:
         print(f"--- [{function_name}] LLM client not available for {llm_provider}. Skipping entity linking. ---")
         # Ensure snippet risks that needed linking get an empty related_entities list
         for risk in snippet_risks_to_link:
              if isinstance(risk, dict):
                   risk["related_entities"] = []
                   other_risks.append(risk)
         return other_risks


    updated_risks_from_linking = [] # This will hold the risks after attempting linking

    print(f"--- [{function_name}] Will check {len(snippet_risks_to_link)} snippet risks against {len(list_of_entity_names)} filtered entities...")

    for risk_index, risk in enumerate(snippet_risks_to_link):
        # Re-check validity before processing
        if not isinstance(risk, dict) or not risk.get('description') or not risk.get('source_urls'):
            print(f"--- [{function_name}] Skipping invalid snippet risk object at index {risk_index}. ---")
            # Add an empty related_entities list even if invalid and append to the return list
            if isinstance(risk, dict) and "related_entities" not in risk: risk["related_entities"] = []
            updated_risks_from_linking.append(risk)
            continue # Move to the next risk

        risk_desc = risk['description']; risk_urls = risk['source_urls']
        # Prepare context with snippets associated with this specific risk
        risk_context_text = ""; added_snippets = 0; max_chars = 7000
        context_intro = f"The following risk was identified: \"{risk_desc}\". It was found in text snippets from these sources: {', '.join(risk_urls)}. Identify which of the following entities are directly mentioned *within the context of this specific risk* in the snippets.\n\nList of Potential Entities: [{', '.join([f"'{name}'" for name in list_of_entity_names])}]\n\nRelevant Text Snippets:\n"
        char_count = len(context_intro) # Start char count with intro length

        snippets_for_this_risk = [all_snippets_map.get(url) for url in risk_urls if url in all_snippets_map]
        # Filter out None results and invalid snippet formats
        snippets_for_this_risk = [s for s in snippets_for_this_risk if isinstance(s, dict) and s.get('snippet') and isinstance(s.get('snippet'), str)]


        for snippet_data in snippets_for_this_risk:
            snippet_text = snippet_data['snippet']
            url = snippet_data.get('url', 'N/A')
            title = snippet_data.get('title', '') # Get title
            original_lang = snippet_data.get('original_language', 'unknown')
            translated_from = snippet_data.get('translated_from')
            translation_info = ""
            if translated_from: translation_info = f" (Translated from {translated_from})"
            elif original_lang != 'unknown':
                 if original_lang.lower() not in ['en', 'english']: translation_info = f" (Original Language: {original_lang})"


            if len(snippet_text) < 50: continue # Skip very short snippets

            entry = f"---\nSourceURL: {url}{translation_info}\nTitle: {title}\nSnippet: {snippet_text}\n"
            if char_count + len(entry) <= max_chars:
                risk_context_text += entry
                char_count += len(entry)
                added_snippets+=1
            else: break # Stop adding snippets if max length is reached

        if added_snippets == 0:
            print(f"--- [{function_name}] No valid snippets found for risk: '{risk_desc[:50]}...'. Setting empty related_entities. ---")
            risk["related_entities"] = []
            risk['_source_type'] = 'snippet' # Ensure source type is kept
            updated_risks_from_linking.append(risk)
            continue # Move to the next risk

        prompt = f"""{context_intro}{risk_context_text}
Based *strictly* on the provided Relevant Text Snippets, identify ONLY the entities from the "List of Potential Entities" that are mentioned *directly together* with the specific risk described above ("{risk_desc}").

Your response MUST be ONLY a single valid JSON object with a single key "related_entities" containing a JSON array of strings (the names of the directly involved entities). Example: {{"related_entities": ["Entity A", "Entity C"]}} or {{"related_entities": []}} if none are directly involved in this specific risk context. Do not include any other text, explanation, or formatting."""

        print(f"--- [{function_name}] Linking risk #{risk_index+1}/{len(snippet_risks_to_link)}: '{risk_desc[:50]}...' using {added_snippets} snippets. ---")

        related_entity_names = []
        parsed_json = _call_llm_and_parse_json(prompt, llm_provider, llm_model, function_name, attempt_json_mode=True)

        if parsed_json is not None and isinstance(parsed_json.get("related_entities"), list):
             entity_list_from_llm = parsed_json["related_entities"]
             # Validate that all items in the list are strings
             if all(isinstance(item, str) for item in entity_list_from_llm):
                 # Filter the list from LLM to only include names that were in the input list of potential entities
                 input_names_lower = {name.lower() for name in list_of_entity_names}
                 related_entity_names = [name.strip() for name in entity_list_from_llm if isinstance(name, str) and name.strip().lower() in input_names_lower]
                 print(f"--- [{function_name}] Parsed and filtered related entities: {related_entity_names} ---")
             else: print(f"--- [{function_name}] WARNING: 'related_entities' array contained non-string items: {entity_list_from_llm} ---")
        elif parsed_json is not None:
             print(f"--- [{function_name}] WARNING: LLM response JSON did not contain a valid 'related_entities' list (got type {type(parsed_json.get('related_entities')).__name__}). Full response: {parsed_json} ---")
        else: print(f"--- [{function_name}] WARNING: LLM call or parsing failed for risk linking. ---")

        risk["related_entities"] = related_entity_names # Update the risk dictionary with the linked entities
        risk['_source_type'] = 'snippet' # Ensure source type is kept
        updated_risks_from_linking.append(risk) # Add the updated risk to the list

        # Add a delay between LLM calls for linking
        time.sleep(LLM_RELATIONSHIP_CHECK_DELAY)

    # Return the combined list of risks that were updated and those that were not snippet risks
    return updated_risks_from_linking + other_risks

def extract_relationships_only(search_results: List[Dict[str, Any]], extraction_context: str,
                               entities: List[Dict],
                               llm_provider: str, llm_model: str) -> List[Dict]:
    """Extracts only specific ownership relationships (SUBSIDIARY_OF, PARENT_COMPANY_OF, AFFILIATE_OF, JOINT_VENTURE_PARTNER) based on the schema from snippets. Checks for LLM availability."""
    print("\n--- [NLP Relationships (Ownership Only)] Attempting Relationship Extraction (Ownership Only) ---")
    function_name = "NLP Relationships (Ownership Only)"
    # Filter search results for those with valid snippets and URLs
    relevant_search_results = [s for s in search_results if isinstance(s, dict) and s.get('snippet') and isinstance(s.get('snippet'), str) and s.get('url') and isinstance(s.get('url'), str)]

    if not relevant_search_results or not entities: print(f"--- [{function_name}] No relevant search results or entities, skipping. ---"); return []
    # Check for LLM availability
    if _get_llm_client_and_model(llm_provider, llm_model)[0] is None:
         print(f"--- [{function_name}] LLM client not available for {llm_provider}. Skipping relationship extraction. ---")
         return []

    context_text, added_snippets = _prepare_context_text(relevant_search_results)
    if added_snippets == 0: print(f"--- [{function_name}] No valid snippets. ---"); return []
    print(f"--- [{function_name}] Prepared context using {added_snippets} snippets. ---")

    # Filter entities to only include Companies and Organizations with valid names
    entity_names = [e['name'] for e in entities if isinstance(e, dict) and e.get('name') and e.get('type') in ["COMPANY", "ORGANIZATION"]]
    if not entity_names: print(f"--- [{function_name}] No COMPANY/ORGANIZATION entity names available from filtered entities. ---"); return []
    input_names_lower = {name.lower() for name in entity_names}

    # Updated relationship schema for ownership types, including JOINT_VENTURE_PARTNER and removing ACQUIRED/RELATED_COMPANY
    rel_schema = """- "relationships": Array of objects. Each MUST have: "entity1" (string name, English - from the provided entity list).
        - "relationship_type" (string, ONLY choose from: "SUBSIDIARY_OF", "PARENT_COMPANY_OF", "AFFILIATE_OF", "JOINT_VENTURE_PARTNER").
        - "entity2" (string name, English - from the provided entity list).
        - "context_urls" (array of SourceURLs where relationship mentioned, MUST NOT be empty)."""

    prompt = f"""Analyze text snippets based on context: "{extraction_context}". Given the entities previously identified from the search results: {', '.join([f"'{name}'" for name in entity_names])}.

Extract ONLY explicitly stated **ownership, affiliate, or joint venture relationships** between these specific entities, matching the schema below.

Schema:
{rel_schema}

Focus ONLY on relationships where BOTH entity1 and entity2 are from the provided list of identified COMPANY or ORGANIZATION entities.
The relationship_type MUST be one of: SUBSIDIARY_OF, PARENT_COMPANY_OF, AFFILIATE_OF, or JOINT_VENTURE_PARTNER.
Response MUST be ONLY a single valid JSON object like {{"relationships": [...]}}. Use empty array [] if no relationships found. Do not include any other text, explanation, or formatting."""


    parsed_json = _call_llm_and_parse_json(prompt, llm_provider, llm_model, function_name, attempt_json_mode=True)

    validated_relationships = []
    # Updated allowed relationship types for validation
    allowed_types_lower = {"subsidiary_of", "parent_company_of", "affiliate_of", "joint_venture_partner"}

    if parsed_json is not None and isinstance(parsed_json.get("relationships"), list):
        for rel in parsed_json["relationships"]:
             # Ensure the item is a dictionary before checking keys
             if not isinstance(rel, dict):
                  print(f"--- [{function_name}] Skipping invalid relationship item (not a dictionary): {rel} ---")
                  continue

             entity1_name = rel.get("entity1"); entity2_name = rel.get("entity2"); rel_type_raw = rel.get("relationship_type")
             context_urls = rel.get("context_urls")

             if entity1_name and isinstance(entity1_name, str) and entity2_name and isinstance(entity2_name, str) and \
                rel_type_raw and isinstance(rel_type_raw, str) and rel_type_raw.lower() in allowed_types_lower and \
                isinstance(context_urls, list) and context_urls and all(isinstance(u, str) for u in context_urls):

                 e1_name_lower = entity1_name.strip().lower()
                 e2_name_lower = entity2_name.strip().lower()

                 # Double check that both entities are in the original list of COMPANY/ORGANIZATION names provided
                 if e1_name_lower in input_names_lower and e2_name_lower in input_names_lower:
                      rel['_source_type'] = 'snippet' # Add internal source type
                      validated_relationships.append(rel)
                 else:
                      # This shouldn't happen if the LLM followed instructions, but safety check
                      print(f"--- [{function_name}] Skipping relationship item because one or both entities ('{entity1_name}', '{entity2_name}') are not in the input list of COMPANY/ORGANIZATION entities: {rel} ---")
             else: print(f"--- [{function_name}] Skipping invalid or incomplete relationship item (doesn't match schema, not allowed type, or missing mandatory fields): {rel} ---")
    else:
         # Log if the expected 'relationships' list was not found or was not a list
         print(f"--- [{function_name}] Failed to parse valid 'relationships' list from LLM response. Parsed content type: {type(parsed_json).__name__}. Content: {parsed_json} ---")


    print(f"--- [{function_name}] Returning {len(validated_relationships)} validated relationships (Ownership/Affiliate/JV only). ---")
    return validated_relationships

def extract_ownership_involving_entity(text_snippets: List[Dict[str, Any]], target_entity_name: str,
                                       llm_provider: str, llm_model: str) -> List[Dict]:
    """
    Analyzes text snippets to find *any* ownership or affiliate relationships
    where the target_entity_name is involved (as parent, subsidiary, affiliate, etc.).
    Returns a list of relationship dictionaries. Checks for LLM availability.
    This function is used in search_engines.search_for_ownership_docs, not the main orchestrator flow.
    It's kept here but not directly used in the steps above.
    """
    print(f"\n--- [NLP Targeted Relationships] Attempting to find relationships involving '{target_entity_name}' ---")
    function_name = "NLP Targeted Relationships"
    if not llm_provider or not llm_model: print(f"--- [{function_name}] Skipping: Missing LLM config."); return []
    # Filter text snippets for those with valid snippets and URLs
    relevant_text_snippets = [s for s in text_snippets if isinstance(s, dict) and s.get('snippet') and isinstance(s.get('snippet'), str) and s.get('url') and isinstance(s.get('url'), str)]

    if not relevant_text_snippets: print(f"--- [{function_name}] No relevant snippets provided."); return []
    # Check for LLM availability
    if _get_llm_client_and_model(llm_provider, llm_model)[0] is None:
         print(f"--- [{function_name}] LLM client not available for {llm_provider}. Skipping relationship extraction. ---")
         return []

    context = f"Analyze ONLY the following text snippets for explicit evidence of ownership, subsidiary, affiliate, joint venture, acquired, or significantly related company relationships where '{target_entity_name}' is involved.\n\nDefinitions:\n- PARENT_COMPANY_OF: Entity A is described as owning, controlling, or being the parent of Entity B (e.g., 'Entity A acquired Entity B', 'Entity B is a subsidiary of Entity A', 'Entity A has controlling stake in Entity B').\n- SUBSIDIARY_OF: Entity A is described as being owned or controlled by Entity B (e.g., 'Entity A is a subsidiary of Entity B').\n- AFFILIATE_OF: Entity A is described as an affiliate, associate, or joint venture partner of Entity B, usually implying significant but not majority ownership (e.g., 'Entity A is an affiliate of Entity B', 'Entity A and Entity B are joint venture partners').\n- ACQUIRED: Entity A acquired Entity B.\n- JOINT_VENTURE_PARTNER: Entity A and Entity B are described as partners in a joint venture.\n- RELATED_COMPANY: Entities mentioned together in a context that suggests a significant link (e.g. mentioned in the same corporate filing, linked through investment portfolio) but the exact nature of relationship (parent/subsidiary/affiliate) isn't explicitly clear from the text snippets.\n\nRelevant Text Snippets:\n";
    max_chars = 7000; char_count = len(context); added_snippets = 0; snippets_for_llm = []
    for snip in relevant_text_snippets:
        snippet_text = snip['snippet']; url = snip['url']; title = snip.get('title', '')
        original_lang = snip.get('original_language', 'unknown')
        translated_from = snip.get('translated_from')
        translation_info = ""
        if translated_from:
             translation_info = f" (Translated from {translated_from})"
        elif original_lang != 'unknown':
             if original_lang.lower() not in ['en', 'english']: translation_info = f" (Original Language: {original_lang})"


        if len(snippet_text) >= 50: # Skip very short snippets
             entry = f"---\nSourceURL: {url}{translation_info}\nTitle: {title}\nSnippet: {snippet_text}\n"
             if char_count + len(entry) <= max_chars:
                 context += entry
                 char_count += len(entry)
                 added_snippets += 1
                 snippets_for_llm.append(snip)
             else:
                 break # Stop adding snippets if max length is reached

    if added_snippets == 0: print(f"--- [{function_name}] No valid snippets provided."); return []
    print(f"--- [{function_name}] Prepared context: {added_snippets} snippets. ---")

    # Updated relationship schema for targeted search (includes more types)
    rel_schema = """- "relationships": Array of objects. Each MUST have: "entity1" (string name, English - one of the entities in the relationship).
        - "relationship_type" (string, ONLY choose from: "PARENT_COMPANY_OF", "SUBSIDIARY_OF", "AFFILIATE_OF", "ACQUIRED", "JOINT_VENTURE_PARTNER", "RELATED_COMPANY").
        - "entity2" (string name, English - the other entity in the relationship).
        - "context_urls" (array of SourceURLs where relationship mentioned, MUST NOT be empty)."""

    prompt = f"""{context}
Based ONLY on the provided snippets and definitions, extract ALL ownership, subsidiary, affiliate, joint venture, acquired, or significantly related company relationships where '{target_entity_name}' is one of the entities involved. Use the relationship_type that best fits the description in the snippet.

Schema:
{rel_schema}

Ensure BOTH entity1 and entity2 are COMPANY or ORGANIZATION names. If the exact nature isn't clear but they are mentioned together in a corporate/financial context suggesting a link, use "RELATED_COMPANY".
Response MUST be ONLY a single valid JSON object like {{"relationships": [...]}}. Use empty array [] if no relationships found. Do not include any other text, explanation, or formatting."""


    parsed_json = _call_llm_and_parse_json(prompt, llm_provider, llm_model, function_name, attempt_json_mode=True)

    validated_relationships = []
    # Define allowed types for this specific function's output
    allowed_types_lower = {"parent_company_of", "subsidiary_of", "affiliate_of", "acquired", "joint_venture_partner", "related_company"}

    if parsed_json is not None and isinstance(parsed_json.get("relationships"), list):
        for rel in parsed_json["relationships"]:
             # Ensure the item is a dictionary before checking keys
             if not isinstance(rel, dict):
                  print(f"--- [{function_name}] Skipping invalid relationship item (not a dictionary): {rel} ---")
                  continue

             entity1_name = rel.get("entity1"); entity2_name = rel.get("entity2"); rel_type_raw = rel.get("relationship_type")
             context_urls = rel.get("context_urls")

             if entity1_name and isinstance(entity1_name, str) and entity2_name and isinstance(entity2_name, str) and \
                rel_type_raw and isinstance(rel_type_raw, str) and rel_type_raw.lower() in allowed_types_lower and \
                isinstance(context_urls, list) and context_urls and all(isinstance(u, str) for u in context_urls) and \
                (entity1_name.strip().lower() == target_entity_name.lower() or entity2_name.strip().lower() == target_entity_name.lower()):

                 rel['_source_type'] = 'targeted_snippet_llm' # Add internal source type
                 validated_relationships.append(rel)
             else: print(f"--- [{function_name}] Skipping invalid or incomplete relationship item (doesn't match schema, not allowed type, or doesn't involve target entity): {rel} ---")
    else:
         # Log if the expected 'relationships' list was not found or was not a list
         print(f"--- [{function_name}] Failed to parse valid 'relationships' list from LLM response. Parsed content type: {type(parsed_json).__name__}. Content: {parsed_json} ---")


    print(f"--- [{function_name}] Returning {len(validated_relationships)} validated relationships involving '{target_entity_name}'. ---")
    return validated_relationships


def extract_regulatory_sanction_relationships(search_results: List[Dict[str, Any]], extraction_context: str,
                                              entities: List[Dict],
                                              llm_provider: str, llm_model: str) -> List[Dict]:
    """
    Analyzes text snippets to find explicit relationships between identified Entities
    (Companies/Organizations/Agencies/Sanctions) related to regulations and sanctions.
    Returns a list of relationship dictionaries. Checks for LLM availability.
    """
    print("\n--- [NLP Regulatory/Sanction Relationships] Attempting Extraction ---")
    function_name = "NLP Regulatory/Sanction Relationships"
    # Filter search results for those with valid snippets and URLs
    relevant_search_results = [s for s in search_results if isinstance(s, dict) and s.get('snippet') and isinstance(s.get('snippet'), str) and s.get('url') and isinstance(s.get('url'), str)]

    if not relevant_search_results or not entities: print(f"--- [{function_name}] No relevant search results or entities, skipping. ---"); return []
    # Check for LLM availability
    if _get_llm_client_and_model(llm_provider, llm_model)[0] is None:
         print(f"--- [{function_name}] LLM client not available for {llm_provider}. Skipping relationship extraction. ---")
         return []

    context_text, added_snippets = _prepare_context_text(relevant_search_results)
    if added_snippets == 0: print(f"--- [{function_name}] No valid snippets. ---"); return []
    print(f"--- [{function_name}] Prepared context using {added_snippets} snippets. ---")

    entity_names = [e['name'] for e in entities if isinstance(e, dict) and e.get('name')]
    if not entity_names: print(f"--- [{function_name}] No entity names available. ---"); return []
    # Create sets of entity names by type for easier lookup (use original casing for lookup)
    company_org_names = {e.get('name','') for e in entities if isinstance(e, dict) and e.get('type') in ['COMPANY', 'ORGANIZATION'] and e.get('name')}
    reg_agency_names = {e.get('name','') for e in entities if isinstance(e, dict) and e.get('type') == 'REGULATORY_AGENCY' and e.get('name')}
    sanction_names = {e.get('name','') for e in entities if isinstance(e, dict) and e.get('type') == 'SANCTION' and e.get('name')}
    # Create lowercased sets for case-insensitive matching
    company_org_names_lower = {name.lower() for name in company_org_names}
    reg_agency_names_lower = {name.lower() for name in reg_agency_names}
    sanction_names_lower = {name.lower() for name in sanction_names}
    all_entity_names_lower = company_org_names_lower.union(reg_agency_names_lower, sanction_names_lower)


    rel_schema = """- "relationships": Array of objects. Each MUST have: "entity1" (string name, English - from the provided entity list).
        - "relationship_type" (string, ONLY choose from: "REGULATED_BY", "ISSUED_BY", "SUBJECT_TO", "MENTIONED_WITH").
        - "entity2" (string name, English - from the provided entity list).
        - "context_urls" (array of SourceURLs where relationship mentioned, MUST NOT be empty)."""

    prompt = f"""Analyze text snippets based on context: "{extraction_context}". Given the entities previously identified from the search results: {', '.join([f"'{name}'" for name in entity_names])}.

Extract ONLY explicit relationships between these specific entities related to regulatory actions or sanctions, matching the schema below.

Schema:
{rel_schema}

Relationship Types:
- REGULATED_BY: Entity1 (a COMPANY or ORGANIZATION from the list) is described as being regulated by Entity2 (a REGULATORY_AGENCY from the list).
- ISSUED_BY: Entity1 (a SANCTION from the list) was issued by Entity2 (a REGULATORY_AGENCY from the list).
- SUBJECT_TO: Entity1 (a COMPANY or ORGANIZATION from the list) is explicitly mentioned as being subject to, affected by, or receiving Entity2 (a SANCTION or REGULATORY_AGENCY from the list).
- MENTIONED_WITH: Entity1 and Entity2 (any type from the list: COMPANY, ORGANIZATION, REGULATORY_AGENCY, SANCTION) are mentioned together in a context that suggests a regulatory or sanction-related connection, but the exact nature of REGULATED_BY, ISSUED_BY, or SUBJECT_TO is not explicit.

Focus ONLY on relationships where BOTH entity1 and entity2 are from the provided list of identified entities.
The relationship_type MUST be one of: REGULATED_BY, ISSUED_BY, SUBJECT_TO, or MENTIONED_WITH.
Response MUST be ONLY a single valid JSON object like {{"relationships": [...]}}. Use empty array [] if no relationships found. Do not include any other text, explanation, or formatting."""


    parsed_json = _call_llm_and_parse_json(prompt, llm_provider, llm_model, function_name, attempt_json_mode=True)

    validated_relationships = []
    allowed_types_lower = {"regulated_by", "issued_by", "subject_to", "mentioned_with"}

    if parsed_json is not None and isinstance(parsed_json.get("relationships"), list):
        for rel in parsed_json["relationships"]:
             # Ensure the item is a dictionary before checking keys
             if not isinstance(rel, dict):
                  print(f"--- [{function_name}] Skipping invalid relationship item (not a dictionary): {rel} ---")
                  continue

             entity1_name = rel.get("entity1"); entity2_name = rel.get("entity2"); rel_type_raw = rel.get("relationship_type")
             context_urls = rel.get("context_urls")

             if entity1_name and isinstance(entity1_name, str) and entity2_name and isinstance(entity2_name, str) and \
                rel_type_raw and isinstance(rel_type_raw, str) and rel_type_raw.lower() in allowed_types_lower and \
                isinstance(context_urls, list) and context_urls and all(isinstance(u, str) for u in context_urls):

                 e1_name_lower = entity1_name.strip().lower()
                 e2_name_lower = entity2_name.strip().lower()

                 # Double check that both entities are in the original list of identified entities
                 if e1_name_lower in all_entity_names_lower and e2_name_lower in all_entity_names_lower:

                      # Perform type validity check based on the specific relationship type
                      is_valid_rel_type = True
                      # Find the *original* entity objects to get their types
                      e1_obj = next((e for e in entities if isinstance(e, dict) and e.get('name','') and e['name'].lower() == e1_name_lower), None)
                      e2_obj = next((e for e in entities if isinstance(e, dict) and e.get('name','') and e['name'].lower() == e2_name_lower), None)

                      if e1_obj is None or e2_obj is None:
                           # This shouldn't happen if names were in all_entity_names_lower, but safety check
                           print(f"--- [{function_name}] Safety check failed: Could not find entity object for linked name ('{entity1_name}' or '{entity2_name}'). Skipping relationship {rel}")
                           continue # Skip if we can't find the entity objects to check types

                      e1_type = e1_obj.get('type')
                      e2_type = e2_obj.get('type')

                      if r_type_upper == "REGULATED_BY":
                           if e1_type not in ["COMPANY", "ORGANIZATION"] or e2_type != "REGULATORY_AGENCY": is_valid_rel_type = False
                      elif r_type_upper == "ISSUED_BY":
                           if e1_type != "SANCTION" or e2_type != "REGULATORY_AGENCY": is_valid_rel_type = False
                      elif r_type_upper == "SUBJECT_TO":
                           if e1_type not in ["COMPANY", "ORGANIZATION"] or e2_type not in ["SANCTION", "REGULATORY_AGENCY"]: is_valid_rel_type = False
                      elif r_type_upper == "MENTIONED_WITH":
                          # MENTIONED_WITH can be between any of the allowed entity types
                          if e1_type not in allowed_entity_types or e2_type not in allowed_entity_types: is_valid_rel_type = False


                      if is_valid_rel_type:
                           rel['_source_type'] = 'snippet' # Add internal source type
                           validated_relationships.append(rel)
                      else:
                           print(f"--- [{function_name}] Skipping relationship item due to type mismatch for relation type '{rel_type_raw}': {rel} (Entity1 Type: {e1_type}, Entity2 Type: {e2_type}) ---")

                 else:
                      # This shouldn't happen if the LLM followed instructions, but safety check
                      print(f"--- [{function_name}] Skipping relationship item because one or both entities ('{entity1_name}', '{entity2_name}') are not in the input list of entities: {rel} ---")
             else: print(f"--- [{function_name}] Skipping invalid or incomplete relationship item (doesn't match schema, not allowed type, or missing mandatory fields): {rel} ---")
    else:
         # Log if the expected 'relationships' list was not found or was not a list
         print(f"--- [{function_name}] Failed to parse valid 'relationships' list from LLM response. Parsed content type: {type(parsed_json).__name__}. Content: {parsed_json} ---")


    print(f"--- [{function_name}] Returning {len(validated_relationships)} validated relationships (Regulatory/Sanction). ---")
    return validated_relationships


def _map_linkup_relation_type(linkup_type: str) -> str:
     """Maps Linkup's structured relationship types to internal types for storage."""
     if not isinstance(linkup_type, str): return "RELATED_COMPANY"
     type_lower = linkup_type.lower()
     if type_lower == 'subsidiary': return 'PARENT_COMPANY_OF'
     elif type_lower == 'affiliate': return 'AFFILIATE_OF'
     elif type_lower == 'parent': return 'SUBSIDIARY_OF'
     elif type_lower == 'joint venture': return 'JOINT_VENTURE_PARTNER'
     elif type_lower == 'investee': return 'INVESTEE'
     elif type_lower == 'acquired': return 'ACQUIRED'
     elif type_lower == 'private company': return "RELATED_COMPANY"
     else: return "RELATED_COMPANY"

def process_linkup_structured_data(linkup_structured_results_list: List[Dict[str, Any]], original_query: str) -> Dict[str, List]:
    """
    Processes a list of structured data results received from Linkup API calls
    and converts them into the internal entities, risks, and relationships format.
    This function correctly extracts data from the 'data' dictionary key within each result item.
    Handles potential regulatory/sanction data if the schemas were updated and structure matches.
    Checks for LLM availability (though not strictly needed for this process, kept for consistency).
    """
    print("\n--- [NLP Structured Processor] Processing Linkup structured data list ---")
    processed_entities = []
    processed_risks = []
    processed_relationships = []

    if not linkup_structured_results_list or not isinstance(linkup_structured_results_list, list):
        print("--- [NLP Structured Processor] No structured data list or invalid format provided. ---")
        return {"entities": [], "risks": [], "relationships": []}

    for result_item in linkup_structured_results_list:
        # Expecting items like {"entity": "Entity Name", "schema": "schema_name", "data": {...}}
        if not isinstance(result_item, dict):
             print(f"Warning: Skipping invalid structured result item format (not a dict): {result_item}")
             continue

        entity_name_context = result_item.get("entity") # The entity name the search was targeted for
        schema_name = result_item.get("schema")       # The schema name (e.g., "ownership", "key_risks")
        structured_data_content = result_item.get("data") # The actual structured data dictionary

        if not entity_name_context or not isinstance(entity_name_context, str) or not schema_name or not isinstance(schema_name, str) or not structured_data_content or not isinstance(structured_data_content, dict):
            print(f"Warning: Skipping invalid structured result item format (missing keys or data not dict): {result_item}")
            continue

        print(f"--- [NLP Structured Processor] Processing structured data for query entity '{entity_name_context}' using schema '{schema_name}' ---")

        # --- Process data based on the schema name ---

        # --- Process Ownership Schema Results ---
        if schema_name == "ownership":
            # The main company name in the structured data, fallback to the entity name context
            main_company_in_data = structured_data_content.get("company_name", entity_name_context)

            linkup_relations_list = structured_data_content.get("ownership_relationships")
            if isinstance(linkup_relations_list, list):
                for rel_item in linkup_relations_list:
                     if isinstance(rel_item, dict):
                          parent_name = rel_item.get("parent_company")
                          sub_name = rel_item.get("subsidiary_affiliate")
                          relation_type_raw = rel_item.get("relation_type")
                          source_url = rel_item.get("source_url") # Source URL for this specific relationship item

                          mapped_rel_type = _map_linkup_relation_type(relation_type_raw)

                          entity1_name_mapped = parent_name
                          entity2_name_mapped = sub_name

                          if entity1_name_mapped and entity2_name_mapped and isinstance(entity1_name_mapped, str) and isinstance(entity2_name_mapped, str) and mapped_rel_type:
                               internal_relation = {
                                   "entity1": entity1_name_mapped.strip(),
                                   "relationship_type": mapped_rel_type,
                                   "entity2": entity2_name_mapped.strip(),
                                   "context_urls": [source_url] if source_url and isinstance(source_url, str) else [], # Use source URL from the item
                                   "_source_type": "linkup_structured"
                               }
                               processed_relationships.append(internal_relation)

                               # Add related entities found in the structured data as COMPANY type
                               # These entities are inferred from the relationship data itself
                               for name, entity_type in [(entity1_name_mapped, "COMPANY"), (entity2_name_mapped, "COMPANY")]:
                                   if name and isinstance(name, str):
                                        is_already_added = any(e.get("name","").lower() == name.strip().lower() for e in processed_entities)
                                        if not is_already_added:
                                             processed_entities.append({
                                                 "name": name.strip(),
                                                 "type": entity_type,
                                                 "mentions": [source_url] if source_url and isinstance(source_url, str) else [], # Link mentions to source URL
                                                 "_source_type": "linkup_structured"
                                             })

                          else:
                               print(f"Warning: Skipping structured ownership relationship due to missing data or unrecognised type: {rel_item}")
            elif linkup_relations_list is not None:
                 print(f"Warning: Expected 'ownership_relationships' to be a list in schema '{schema_name}', but got {type(linkup_relations_list)} for entity '{entity_name_context}'")


        # --- Process Key Risks Schema Results ---
        elif schema_name == "key_risks":
            # The main company name in the structured data, fallback to the entity name context
            main_company_in_data = structured_data_content.get("company_name", entity_name_context)

            linkup_risks_list = structured_data_content.get("key_risks_identified")
            if isinstance(linkup_risks_list, list):
                for risk_item in linkup_risks_list:
                    if isinstance(risk_item, dict) and risk_item.get("risk_description"):
                        source_url = risk_item.get("source_url") # Source URL for this specific risk item
                        # Extract risk_category if available
                        risk_category = risk_item.get("risk_category")
                        if not risk_category or not isinstance(risk_category, str): risk_category = "UNKNOWN"

                        internal_risk = {
                            "description": risk_item.get("risk_description").strip() if isinstance(risk_item.get("risk_description"), str) else "N/A",
                            "severity": risk_item.get("reported_severity", "MEDIUM").upper() if isinstance(risk_item.get("reported_severity"), str) else "MEDIUM",
                            "risk_category": risk_category.strip(), # Add extracted/default category
                            "source_urls": [source_url] if source_url and isinstance(source_url, str) else [], # Use source URL from the item
                            # Related entities for structured risks are the main company the search was for
                            "related_entities": [main_company_in_data.strip()] if main_company_in_data and isinstance(main_company_in_data, str) else [],
                            "_source_type": "linkup_structured"
                        }
                        # Basic validation for risk description
                        if internal_risk["description"] != "N/A":
                            processed_risks.append(internal_risk)
                            # Add the main company entity if not already added
                            if main_company_in_data and isinstance(main_company_in_data, str) and not any(e.get("name","").lower() == main_company_in_data.strip().lower() for e in processed_entities):
                                 processed_entities.append({"name": main_company_in_data.strip(), "type": "COMPANY", "mentions": [source_url] if source_url and isinstance(source_url, str) else [], "_source_type": "linkup_structured"})
                        else:
                            print(f"Warning: Skipping structured risk due to missing description: {risk_item}")


            elif linkup_risks_list is not None:
                 print(f"Warning: Expected 'key_risks_identified' to be a list in schema '{schema_name}', but got {type(linkup_risks_list)} for entity '{entity_name_context}'")

        # --- Process Regulatory/Sanction Schema Results (Hypothetical - depends on actual schemas) ---
        # This is placeholder logic assuming you define schemas like LINKUP_SCHEMA_REGULATIONS
        # and LINKUP_SCHEMA_SANCTIONS and Linkup returns lists under specific keys.
        # elif schema_name == "regulations": # Hypothetical schema name
        #     main_entity_in_data = structured_data_content.get("entity_name", entity_name_context) # Hypothetical key name
        #     regulations_list = structured_data_content.get("regulations_found") # Hypothetical list key name
        #     if isinstance(regulations_list, list):
        #         for reg_item in regulations_list:
        #             if isinstance(reg_item, dict) and reg_item.get("regulation_name"): # Hypothetical key names
        #                  reg_name = reg_item["regulation_name"]
        #                  agency_name = reg_item.get("issuing_agency")
        #                  affected_entities = reg_item.get("affected_companies", []) # Hypothetical key name (list of affected company names)
        #                  source_url = reg_item.get("source_url")

        #                  # Add Sanction/Regulation entity (Type 'SANCTION' for Regulations found this way)
        #                  if reg_name and isinstance(reg_name, str):
        #                       if not any(e.get("name","").lower() == reg_name.strip().lower() for e in processed_entities):
        #                            processed_entities.append({"name": reg_name.strip(), "type": "SANCTION", "mentions": [source_url] if source_url and isinstance(source_url, str) else [], "_source_type": "linkup_structured"})
        #                  # Add Regulatory Agency entity
        #                  if agency_name and isinstance(agency_name, str):
        #                       if not any(e.get("name","").lower() == agency_name.strip().lower() for e in processed_entities):
        #                            processed_entities.append({"name": agency_name.strip(), "type": "REGULATORY_AGENCY", "mentions": [source_url] if source_url and isinstance(source_url, str) else [], "_source_type": "linkup_structured"})

        #                  # Add relationships
        #                  if reg_name and agency_name and isinstance(reg_name, str) and isinstance(agency_name, str):
        #                       # Relationship: Sanction ISSUED_BY Agency
        #                       processed_relationships.append({"entity1": reg_name.strip(), "relationship_type": "ISSUED_BY", "entity2": agency_name.strip(), "context_urls": [source_url] if source_url and isinstance(source_url, str) else [], "_source_type": "linkup_structured"})
        #                  if reg_name and affected_entities and isinstance(reg_name, str) and isinstance(affected_entities, list):
        #                       for affected_entity_name in affected_entities:
        #                            if affected_entity_name and isinstance(affected_entity_name, str):
        #                                 # Relationship: Company SUBJECT_TO Sanction
        #                                 processed_relationships.append({"entity1": affected_entity_name.strip(), "relationship_type": "SUBJECT_TO", "entity2": reg_name.strip(), "context_urls": [source_url] if source_url and isinstance(source_url, str) else [], "_source_type": "linkup_structured"})
        #                                 # Can also add a risk related to this entity and sanction
        #                                 # internal_risk = {
        #                                 #      "description": f"Subject to sanction: {sanc_name.strip()}",
        #                                 #      "severity": severity if severity and isinstance(severity, str) else "SEVERE", # Use reported severity or default
        #                                 #      "source_urls": [source_url] if source_url and isinstance(source_url, str) else [],
        #                                 #      "related_entities": [target_entity_name.strip()],
        #                                 #      "_source_type": "linkup_structured_derived_risk" # Mark as derived risk
        #                                 # }
        #                                 # processed_risks.append(internal_risk)


        #     elif regulations_list is not None:
        #          print(f"Warning: Expected 'regulations_found' to be a list in schema '{schema_name}', but got {type(regulations_list)} for entity '{entity_name_context}'")

        # elif schema_name == "sanctions": # Hypothetical schema name
        #     main_entity_in_data = structured_data_content.get("entity_name", entity_name_context) # Hypothetical key name
        #     sanctions_list = structured_data_content.get("sanctions_found") # Hypothetical list key name
        #     if isinstance(sanctions_list, list):
        #         for sanc_item in sanctions_list:
        #             if isinstance(sanc_item, dict) and sanc_item.get("sanction_name"): # Hypothetical key names
        #                  sanc_name = sanc_item["sanction_name"]
        #                  issuing_body_name = sanc_item.get("issuing_body")
        #                  target_entities = sanc_item.get("target_entities", []) # Hypothetical key name (list of target company names)
        #                  source_url = sanc_item.get("source_url")
        #                  severity = sanc_item.get("severity")

        #                  # Add Sanction entity
        #                  if sanc_name and isinstance(sanc_name, str):
        #                       if not any(e.get("name","").lower() == sanc_name.strip().lower() for e in processed_entities):
        #                            processed_entities.append({"name": sanc_name.strip(), "type": "SANCTION", "mentions": [source_url] if source_url and isinstance(source_url, str) else [], "_source_type": "linkup_structured"})
        #                  # Add Issuing Body entity (assuming Issuing Body could be a REGULATORY_AGENCY or ORGANIZATION/COUNTRY)
        #                  if issuing_body_name and isinstance(issuing_body_name, str):
        #                       # Need logic to guess type or check if already identified
        #                       # For simplicity, let's add as ORGANIZATION if not already AGENCY
        #                       existing_entity = next((e for e in processed_entities if e.get("name","").lower() == issuing_body_name.strip().lower()), None)
        #                       if not existing_entity:
        #                             processed_entities.append({"name": issuing_body_name.strip(), "type": "ORGANIZATION", "mentions": [source_url] if source_url and isinstance(source_url, str) else [], "_source_type": "linkup_structured"})
        #                       # If it exists and isn't REGULATORY_AGENCY, update its type? Maybe not needed for basic KG.

        #                  # Add relationships
        #                  if sanc_name and issuing_body_name and isinstance(sanc_name, str) and isinstance(issuing_body_name, str):
        #                       # Relationship: Sanction ISSUED_BY IssuingBody (Agency or Org)
        #                       processed_relationships.append({"entity1": sanc_name.strip(), "relationship_type": "ISSUED_BY", "entity2": issuing_body_name.strip(), "context_urls": [source_url] if source_url and isinstance(source_url, str) else [], "_source_type": "linkup_structured"})
        #                  if sanc_name and target_entities and isinstance(sanc_name, str) and isinstance(target_entities, list):
        #                       for target_entity_name in target_entities:
        #                            if target_entity_name and isinstance(target_entity_name, str):
        #                                 # Relationship: Company/Org SUBJECT_TO Sanction
        #                                 processed_relationships.append({"entity1": target_entity_name.strip(), "relationship_type": "SUBJECT_TO", "entity2": sanc_name.strip(), "context_urls": [source_url] if source_url and isinstance(source_url, str) else [], "_source_type": "linkup_structured"})
        #                                 # Can also add a risk related to this entity and sanction
        #                                 # internal_risk = {
        #                                 #      "description": f"Subject to sanction: {sanc_name.strip()}",
        #                                 #      "severity": severity if severity and isinstance(severity, str) else "SEVERE", # Use reported severity or default
        #                                 #      "source_urls": [source_url] if source_url and isinstance(source_url, str) else [],
        #                                 #      "related_entities": [target_entity_name.strip()],
        #                                 #      "_source_type": "linkup_structured_derived_risk" # Mark as derived risk
        #                                 # }
        #                                 # processed_risks.append(internal_risk)


        #     elif sanctions_list is not None:
        #          print(f"Warning: Expected 'sanctions_found' to be a list in schema '{schema_name}', but got {type(sanctions_list)} for entity '{entity_name_context}'")


    print(f"--- [NLP Structured Processor] Finished processing structured data. Extracted: E:{len(processed_entities)}, R:{len(processed_risks)}, Rel:{len(processed_relationships)} ---")

    return {
        "entities": processed_entities,
        "risks": processed_risks,
        "relationships": processed_relationships
    }


def generate_analysis_summary(results: Dict[str, Any], query: str, exposures_count: int,
                              llm_provider: str, llm_model: str) -> str:
    """Generates a concise analysis summary using the LLM. Accepts the full results dict."""
    print(f"\n--- Attempting Analysis Summary via {llm_provider} (Model: {llm_model}) ---");
    function_name = "NLP Summary"
    if not llm_provider or not llm_model: return "Summary generation skipped: Missing LLM config."
    # Check for LLM availability
    if _get_llm_client_and_model(llm_provider, llm_model)[0] is None:
         print(f"--- [{function_name}] LLM client not available for {llm_provider}. Skipping summary generation. ---")
         return "Summary generation skipped: LLM client not available."

    # Use the full results dict passed in
    entities = results.get("final_extracted_data", {}).get("entities", [])
    risks = results.get("final_extracted_data", {}).get("risks", [])
    relationships = results.get("final_extracted_data", {}).get("relationships", [])
    structured_raw_data_list = results.get("linkup_structured_data", [])
    exposures_list = results.get("high_risk_exposures", []) # Use the list of generated exposures


    if not any([entities, risks, relationships, exposures_count > 0, structured_raw_data_list]): return "No significant data extracted or exposures identified to generate a summary."
    summary_parts = []
    if entities:
        # Include new entity types in summary count/list
        company_count = len([e for e in entities if isinstance(e, dict) and e.get('type') == 'COMPANY'])
        org_count = len([e for e in entities if isinstance(e, dict) and e.get('type') == 'ORGANIZATION'])
        reg_agency_count = len([e for e in entities if isinstance(e, dict) and e.get('type') == 'REGULATORY_AGENCY'])
        sanction_count = len([e for e in entities if isinstance(e, dict) and e.get('type') == 'SANCTION'])
        entity_types_summary = []
        if company_count: entity_types_summary.append(f"{company_count} Companies")
        if org_count: entity_types_summary.append(f"{org_count} Organizations")
        if reg_agency_count: entity_types_summary.append(f"{reg_agency_count} Regulatory Agencies")
        if sanction_count: entity_types_summary.append(f"{sanction_count} Sanctions")

        # Include a sample of entity names, prioritizing Companies/Orgs involved in exposures if possible
        sample_entity_names = []
        exposed_entity_names = {exp.get('Entity','') for exp in exposures_list if isinstance(exp, dict) and exp.get('Entity')}
        # Add exposed entities first
        for e in entities:
             if isinstance(e, dict) and e.get('name') and e.get('name') in exposed_entity_names:
                  sample_entity_names.append(f"{e['name']}({e.get('type','?')})")
        # Add other entities if needed to fill the sample
        for e in entities:
             if isinstance(e, dict) and e.get('name') and e.get('name') not in exposed_entity_names:
                  if len(sample_entity_names) < 7: # Limit sample size
                       sample_entity_names.append(f"{e['name']}({e.get('type','?')})")
                  else: break

        entity_list_str = ", ".join(sample_entity_names)
        if len(entities) > 7: entity_list_str += "..."
        if entity_types_summary: summary_parts.append(f"- Entities ({len(entities)} total: {', '.join(entity_types_summary)}): " + entity_list_str + ".")
        elif entity_list_str: summary_parts.append(f"- Entities ({len(entities)}): " + entity_list_str + ".")


    if risks:
        high_risks = [r for r in risks if isinstance(r, dict) and r.get('severity') in ['HIGH', 'SEVERE']]
        med_risks = [r for r in risks if isinstance(r, dict) and r.get('severity') == 'MEDIUM']
        low_risks = [r for r in risks if isinstance(r, dict) and r.get('severity') == 'LOW']
        risk_summary = f"- Risks ({len(risks)} total): {len(high_risks)} High/Severe, {len(med_risks)} Medium, {len(low_risks)} Low."

        # Include sample risk descriptions, prioritizing High/Severe risks linked to exposed entities
        sample_risk_descriptions = []
        exposed_entity_names_lower = {name.lower() for name in exposed_entity_names}
        # Find High/Severe risks linked to exposed entities
        relevant_high_severe_risks = [r for r in high_risks if any(isinstance(e_name, str) and e_name.lower() in exposed_entity_names_lower for e_name in r.get('related_entities', []))]
        for r in relevant_high_severe_risks:
             if isinstance(r.get('description'), str):
                  if len(sample_risk_descriptions) < 3:
                       # Include category if available, otherwise just description and severity
                       desc_part = r['description'].strip()[:80] + '...' if len(r['description'].strip()) > 80 else r['description'].strip()
                       category_part = f" ({r.get('risk_category', 'UNKNOWN')})" if isinstance(r.get('risk_category'), str) and r.get('risk_category') != 'UNKNOWN' else ''
                       severity_part = f" ({r.get('severity','?')})"
                       sample_risk_descriptions.append(f"'{desc_part}'{category_part}{severity_part}")
                  else: break

        # Add other High/Severe risks if sample isn't full
        if len(sample_risk_descriptions) < 3:
             for r in high_risks:
                  if isinstance(r.get('description'), str) and r not in relevant_high_severe_risks: # Avoid duplicates
                       if len(sample_risk_descriptions) < 3:
                            desc_part = r['description'].strip()[:80] + '...' if len(r['description'].strip()) > 80 else r['description'].strip()
                            category_part = f" ({r.get('risk_category', 'UNKNOWN')})" if isinstance(r.get('risk_category'), str) and r.get('risk_category') != 'UNKNOWN' else ''
                            severity_part = f" ({r.get('severity','?')})"
                            sample_risk_descriptions.append(f"'{desc_part}'{category_part}{severity_part}")
                       else: break


        if sample_risk_descriptions: risk_summary += f" (Examples: {'; '.join(sample_risk_descriptions)})"
        summary_parts.append(risk_summary)


    if relationships:
        # Count relationship types relevant to sheet/KG (excluding those explicitly filtered)
        # Use the allowed types defined in orchestrator for sheet saving
        allowed_sheet_rel_types = ["PARENT_COMPANY_OF", "SUBSIDIARY_OF", "AFFILIATE_OF", "JOINT_VENTURE_PARTNER", "REGULATED_BY", "ISSUED_BY", "SUBJECT_TO", "MENTIONED_WITH"]

        relevant_rels = [r for r in relationships if isinstance(r, dict) and r.get('relationship_type') in allowed_sheet_rel_types]

        ownership_types = ["PARENT_COMPANY_OF", "SUBSIDIARY_OF", "AFFILIATE_OF", "JOINT_VENTURE_PARTNER"]
        reg_sanc_types = ["REGULATED_BY", "ISSUED_BY", "SUBJECT_TO"]
        mentioned_type = ["MENTIONED_WITH"]

        ownership_rels_count = len([r for r in relevant_rels if r.get('relationship_type') in ownership_types])
        reg_sanc_rels_count = len([r for r in relevant_rels if r.get('relationship_type') in reg_sanc_types])
        mentioned_rels_count = len([r for r in relevant_rels if r.get('relationship_type') in mentioned_type])

        rel_types_summary = []
        if ownership_rels_count: rel_types_summary.append(f"{ownership_rels_count} Ownership/Affiliate/JV")
        if reg_sanc_rels_count: rel_types_summary.append(f"{reg_sanc_rels_count} Regulatory/Sanction Action")
        if mentioned_rels_count: rel_types_summary.append(f"{mentioned_rels_count} Mentioned With")


        # Include sample relationships, prioritizing those involving exposed entities
        sample_rels = []
        # Find relationships involving exposed entities
        exposed_entity_names_lower = {name.lower() for name in exposed_entity_names}
        relevant_rels_for_exposed = [rel for rel in relevant_rels if any(isinstance(e_name, str) and e_name.lower() in exposed_entity_names_lower for e_name in [rel.get('entity1',''), rel.get('entity2','')] )]
        for rel in relevant_rels_for_exposed:
             if isinstance(rel.get('entity1'), str) and isinstance(rel.get('entity2'), str) and isinstance(rel.get('relationship_type'), str):
                  if len(sample_rels) < 3: # Limit sample size
                       sample_rels.append(f"{rel['entity1'].strip()} {rel['relationship_type'].replace('_', ' ').title()} {rel['entity2'].strip()}")
                  else: break
        # Add other relevant relationships if sample isn't full
        if len(sample_rels) < 3:
             for rel in relevant_rels:
                  if isinstance(rel.get('entity1'), str) and isinstance(rel.get('entity2'), str) and isinstance(rel.get('relationship_type'), str) and rel not in relevant_rels_for_exposed:
                       if len(sample_rels) < 3:
                             sample_rels.append(f"{rel['entity1'].strip()} {rel['relationship_type'].replace('_', ' ').title()} {rel['entity2'].strip()}")
                       else: break


        rel_list_str = "; ".join(sample_rels)
        if len(relevant_rels) > 3: rel_list_str += "..."
        if rel_types_summary: summary_parts.append(f"- Relationships ({len(relevant_rels)} total: {', '.join(rel_types_summary)}): " + rel_list_str + ".")
        elif rel_list_str: summary_parts.append(f"- Relationships ({len(relevant_rels)}): " + rel_list_str + ".")


    if exposures_count > 0:
         # Include sample exposures in the summary
         sample_exposures = []
         for exp in exposures_list[:2]: # Sample first 2 exposures
              if isinstance(exp, dict) and exp.get('Entity'):
                   exp_summary = f"'{exp['Entity'].strip()}' ({exp.get('Risk_Type','?').replace(' Risk','')}, {exp.get('Risk_Severity','?')})" # Use cleaned Risk_Type label
                   sample_exposures.append(exp_summary)
         exposure_summary = f"- Identified {exposures_count} potential High/Severe Risk Exposures linked to Chinese Companies/Orgs via Ownership/Affiliate/JV relationships."
         if sample_exposures: exposure_summary += f" (Examples: {'; '.join(sample_exposures)})"
         summary_parts.append(exposure_summary)


    if structured_raw_data_list:
         schema_counts = {}
         for item in structured_raw_data_list:
              schema_name = item.get("schema", "unknown")
              schema_counts[schema_name] = schema_counts.get(schema_name, 0) + 1
         structured_summary = f"- Linkup Structured Data Found: {len(structured_raw_data_list)} results across {len(schema_counts)} schemas."
         if schema_counts:
              structured_summary += " Schemas: " + ", ".join([f"{name} ({count})" for name, count in schema_counts.items()])
         summary_parts.append(structured_summary)

    # Check KG update status for summary
    # Access kg_update_status directly from the passed results dictionary
    kg_status = results.get('kg_update_status', 'not run')
    kg_summary = f"- Knowledge Graph Update Status: {kg_status}."
    if kg_status and 'skipped' in kg_status:
         kg_summary += " (KG update skipped or encountered an issue)."
    summary_parts.append(kg_summary)


    summary_context = "Original Query: " + query + "\n\nAnalysis Findings:\n" + "\n".join(summary_parts)

    prompt = f"""Based ONLY on the 'Analysis Findings' below, write a concise, objective summary paragraph (3-5 sentences) suitable for executives. Highlight the total number of entities found and their types, the profile of identified risks focusing on High/Severe, the types and count of relationships found (especially ownership/affiliate/JV and regulatory/sanction related), the number of High Risk Exposures identified, and the status of the Knowledge Graph update. Mention if Linkup structured data contributed to the findings.

Analysis Findings:
{summary_context}

Output ONLY the summary paragraph. Do not include headings, bullet points, or conversational text."""

    try:
        llm_to_use = llm_model; provider_to_use = llm_provider
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(provider_to_use, llm_to_use); raw_content = ""
        print(f"Sending summary request to {provider_to_use} ({model_name_used})...")

        messages = [{"role": "user", "content": prompt}]
        temperature = 0.5
        max_tokens = 350

        if client_type == "openai_compatible":
             if openai is None:
                 print("Warning: OpenAI library not available for summary generation.")
                 return "Summary generation skipped: OpenAI library missing."

             response = client_or_lib.chat.completions.create(
                 model=model_name_used,
                 messages=messages,
                 temperature=temperature,
                 max_tokens=max_tokens
             )
             raw_content = response.choices[0].message.content.strip()

        elif client_type == "google_ai":
            if genai is None:
                 print("Warning: Google Generative AI library not available for summary generation.")
                 return "Summary generation skipped: Google Generative AI library missing."

            safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in genai.types.HarmCategory if c != genai.types.HarmCategory.HARM_CATEGORY_UNSPECIFIED];
            generation_config = genai.types.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens);
            response_obj = client_or_lib.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings);

            if response_obj and response_obj.candidates and response_obj.candidates[0].content and response_obj.candidates[0].content.parts:
                 raw_content = response_obj.text.strip()
            else:
                 feedback = getattr(response_obj, 'prompt_feedback', None)
                 block_reason = getattr(feedback, 'block_reason', 'Unknown')
                 print(f"Warning: Google AI summary blocked/empty. Reason: {block_reason}.");
                 raw_content = "Summary generation skipped: Google AI response error."
        else:
            raise ValueError("Unknown client type")

        cleaned_summary = ""
        if raw_content:
            cleaned_content = raw_content.strip()
            cleaned_content = re.sub(r'^["\']|["\']$', '', cleaned_content)
            cleaned_content = re.sub(r'^```.*?\n', '', cleaned_content, flags=re.DOTALL)
            cleaned_content = re.sub(r'\n```$', '', cleaned_content)
            cleaned_summary = cleaned_content.strip()

            conversational_fillers = ["sorry", "apologize", "cannot", "unable", "provide", "based", "above", "text", "hello", "hi"]
            if not cleaned_summary or len(cleaned_summary) < 50 or any(re.search(r'\b' + word + r'\b', cleaned_summary.lower()) for word in conversational_fillers):
                print("Warning: Summary short/empty/apologetic.");
                return f"Could not generate a meaningful summary based on the extracted data."
        else:
             return f"Could not generate a meaningful summary based on the extracted data."


        return cleaned_summary

    except Exception as e:
        print(f"ERROR summary generation: {e}")
        traceback.print_exc()
        return f"Could not generate summary due to error: {type(e).__name__}"

if __name__ == "__main__":
    print("\n--- Running Local NLP Processor Tests ---")
    print("NOTE: Local testing requires LLM API keys in .env and and search API keys.")
    provider_to_test = None; model_to_test = None
    import config
    # Need to import search_engines for the local test to run fully
    try:
        import search_engines
        search_engines_available_nlp_test = True
    except ImportError:
        search_engines = None
        search_engines_available_nlp_test = False
        print("Warning: search_engines.py not available. Limited NLP tests.")


    # Determine provider based on configured keys, preferring OpenRouter if available
    if config and config.OPENROUTER_API_KEY: provider_to_test = "openrouter"; model_to_test = config.DEFAULT_OPENROUTER_MODEL; print("--> Testing with OpenRouter")
    elif config and config.OPENAI_API_KEY: provider_to_test = "openai"; model_to_test = config.DEFAULT_OPENAI_MODEL; print("--> Testing with OpenAI")
    elif config and config.GOOGLE_AI_API_KEY: provider_to_test = "google_ai"; model_to_test = config.DEFAULT_GOOGLE_AI_MODEL; print("--> Testing with Google AI")

    if not provider_to_test: print("No LLM API keys configured. Exiting tests.")
    else:
        # Initialize LLM client once for testing and check availability
        try:
             test_client, test_client_type, test_model_used = _get_llm_client_and_model(provider_to_test, model_to_test)
             if test_client is None:
                  print("Failed to initialize LLM client for testing. Exiting tests.")
                  provider_to_test = None
             else:
                print(f"Successfully initialized LLM: {test_client_type} / {test_model_used}")
        except Exception as e:
             print(f"Failed to initialize LLM client for testing: {e}. Exiting tests.")
             provider_to_test = None


    # Only run tests if provider is available and search_engines module is available for relevant functions
    if provider_to_test and search_engines_available_nlp_test and \
       hasattr(search_engines, 'standardize_result') and callable(search_engines.standardize_result):


        print("\nTesting Keyword Translation...")
        # Pass provider/model to the function call
        kws = translate_keywords_for_context("supply chain compliance issues 2023", "Baidu search in China", provider_to_test, model_to_test); print(f"Keywords: {kws}"); time.sleep(1)

        print("\nTesting Text Translation (English to Chinese)...")
        test_english_text = "Hello, world! This is a test snippet."
        # Pass provider/model to the function call
        translated_chinese = translate_text(test_english_text, 'zh', provider_to_test, model_to_test)
        print(f"Original: {test_english_text}\nTranslated (Chinese): {translated_chinese}"); time.sleep(2)

        print("\nTesting Text Translation (Chinese to English)...")
        test_chinese_text = "你好，世界！这是一个测试片段。"
        # Pass provider/model to the function call
        translated_english = translate_text(test_chinese_text, 'en', provider_to_test, model_to_test)
        print(f"Original: {test_chinese_text}\nTranslated (English): {translated_english}"); time.sleep(2)

        print("\nTesting Snippet Translation (Chinese to English)...")
        test_snippets_zh = [
             {"title": "中文新闻标题", "url": "https://example.com/zh1", "snippet": "这是一段中文测试文本，关于公司合规。", "source": "serpapi_baidu", "original_language": "zh"},
             {"title": "另一篇中文文章", "url": "https://example.com/zh2", "snippet": "中国公司受到新环境法规的影响。", "source": "linkup_snippet_step1", "original_language": "zh"},
             # Add an English one to test it's not translated
             {"title": "English Article", "url": "https://example.com/en1", "snippet": "This is an English test snippet.", "source": "google_cse", "original_language": "en"}
        ]
        # Pass provider/model to the function call
        translated_snippets_list = translate_snippets(test_snippets_zh, 'en', provider_to_test, model_to_test)
        print("\nTranslated Snippets:")
        print(json.dumps(translated_snippets_list, indent=2)); time.sleep(1)


        print("\nTesting Multi-Call Data Extraction & Linking (using sample snippets including translated)...")
        # Combine original and translated snippets for extraction test
        sample_results_combined = translated_snippets_list + [s for s in test_snippets_zh if isinstance(s, dict) and s.get('original_language') == 'en'] # Ensure English ones are included

        # Need some sample English snippets as well for varied content
        sample_results_en = [
             {"title": "Report A: Company X fined $5M for forced labor in its supply chain in Xinjiang", "url": "https://example.com/a", "snippet": "A new report alleges Company X uses forced labor in Xinjiang. Partner Co, a key distributor for Company X, is also mentioned. This is a severe risk.", "source": "news", "published_date": "2023-07-01", "original_language": "en"},
             {"title": "Report B: Investigation finds Company Y subsidiary, Sub Z, violated severe environmental laws in China", "url": "https://example.com/b", "snippet": "Sub Z, a wholly-owned subsidiary of Company Y, based in China, violated severe environmental laws. This incident poses a high risk to Company Y.", "source": "news", "published_date": "2023-06-15", "original_language": "en"},
             {"title": "SEC Fines Company Z for Compliance Issues", "url": "https://example.com/sec_fine", "snippet": "The Securities and Exchange Commission (SEC) fined Company Z $10 million for failing to comply with reporting regulations.", "source": "news", "published_date": "2024-01-10", "original_language": "en"},
             {"title": "OFAC Sanctions Against Entity A", "url": "https://example.com/ofac_sanction", "snippet": "The Office of Foreign Assets Control (OFAC) announced new sanctions targeting Entity A due to its activities.", "source": "news", "published_date": "2024-02-01", "original_language": "en"},
             {"title": "Company P and Company Q Announce Joint Venture in China", "url": "https://example.com/jv", "snippet": "Company P and Company Q are forming a joint venture to expand operations in the Chinese market.", "source": "news", "published_date": "2024-03-01", "original_language": "en"},
             # Example that might trigger Reg/Sanc relationship
             {"title": "SAMR Investigation into Company S", "url": "https://example.com/samr_investigation", "snippet": "China's State Administration for Market Regulation (SAMR) has launched an investigation into anti-competitive practices by Company S.", "source": "news", "published_date": "2024-04-01", "original_language": "en"}
        ]

        # Combine all snippets for the extraction test
        all_snippets_for_extraction_test = sample_results_combined + sample_results_en
        all_test_snippets_map_combined = {r['url']: r for r in all_snippets_for_extraction_test if isinstance(r, dict) and r.get('url')}


        test_context = "financial sector compliance, regulatory actions, and sanctions"

        print("\nExtracting Entities (COMPANY, ORGANIZATION, REGULATORY_AGENCY, SANCTION) from Combined/Translated Snippets...")
        # Pass provider/model to the function call
        test_entities = extract_entities_only(all_snippets_for_extraction_test, test_context, provider_to_test, model_to_test);
        print("\nExtracted Entities:")
        print(json.dumps(test_entities, indent=2)); time.sleep(1)

        print("\nExtracting Risks (Initial) from Combined/Translated Snippets...")
        # Pass provider/model to the function call
        test_risks_initial = extract_risks_only(all_snippets_for_extraction_test, test_context, provider_to_test, model_to_test);
        print("\nExtracted Risks (Initial):")
        print(json.dumps(test_risks_initial, indent=2)); time.sleep(1)

        print("\nLinking Entities to Risks (using filtered entity names and the combined map)...")
        # Use only entities from the validated list with names for linking
        test_entity_names = [e['name'] for e in test_entities if isinstance(e, dict) and e.get('name')]
        test_risks_linked = [];
        if test_risks_initial and test_entity_names:
            # Pass provider/model to the function call and the combined map
            test_risks_linked = link_entities_to_risk(test_risks_initial, test_entity_names, all_test_snippets_map_combined, provider_to_test, model_to_test);
            print("\nRisks after Linking:")
            print(json.dumps(test_risks_linked, indent=2))
        else:
            print("\nSkipping entity linking (no initial risks or entities with names).")
            test_risks_linked = test_risks_initial # Return initial risks if linking skipped
        time.sleep(1)

        print("\nExtracting Relationships (Ownership only) from Combined/Translated Snippets...")
        # Use only Company and Organization entities from the validated list for ownership relationships
        test_entities_company_org = [e for e in test_entities if isinstance(e, dict) and e.get('type') in ["COMPANY", "ORGANIZATION"]]
        test_relationships_ownership = [];
        if test_entities_company_org:
            # Pass provider/model to the function call and the combined list of snippets
            test_relationships_ownership = extract_relationships_only(all_snippets_for_extraction_test, test_context, test_entities_company_org, provider_to_test, model_to_test);
            print("\nExtracted Relationships (Ownership/Affiliate/JV only):")
            print(json.dumps(test_relationships_ownership, indent=2))
        else:
            print("\nSkipping ownership relationship extraction - no COMPANY/ORGANIZATION entities found.")
        time.sleep(1)

        print("\nExtracting Regulatory/Sanction Relationships from Combined/Translated Snippets...")
        # Use all validated entities for Regulatory/Sanction relationships
        test_relationships_reg_sanc = [];
        if test_entities:
            # Pass provider/model to the function call and the combined list of snippets
            test_relationships_reg_sanc = extract_regulatory_sanction_relationships(all_snippets_for_extraction_test, test_context, test_entities, provider_to_test, model_to_test);
            print("\nExtracted Regulatory/Sanction Relationships:")
            print(json.dumps(test_relationships_reg_sanc, indent=2))
        else:
            print("\nSkipping regulatory/sanction relationship extraction - no entities found.")
        time.sleep(1)

        # Combine all relationships for the test output
        test_relationships_combined = test_relationships_ownership + test_relationships_reg_sanc


        print("\nTesting Processing of Linkup Structured Data (using hypothetical schema and data)...")
        # This hypothetical data should match the structure expected by process_linkup_structured_data
        hypothetical_structured_data_list = [
            {
                "entity": "Acme Corp", # Entity the search was for
                "schema": "key_risks", # Schema used
                "data": { # The actual data returned by Linkup for this schema
                    "company_name": "Acme Corp",
                    "key_risks_identified": [
                        {
                            "risk_description": "Reported environmental violations in China",
                            "risk_category": "environmental",
                            "reported_severity": "high",
                            "source_date": "2023-05-10",
                            "source_description": "News Article",
                            "source_url": "https://linkup.so/source/env_report_123"
                        },
                         { # Another risk for the same company
                            "risk_description": "Compliance issues with import regulations",
                            "risk_category": "compliance",
                            "reported_severity": "medium",
                            "source_date": "2024-01-20",
                            "source_description": "Govt. Notice",
                            "source_url": "https://linkup.so/source/govt_notice_456"
                        }
                    ]
                }
            },
             {
             "entity": "Beta Industries", # Entity the search was for
             "schema": "ownership", # Schema used
             "data": { # The actual data returned by Linkup for this schema
                  "company_name": "Beta Industries",
                  "ownership_relationships": [
                      {
                          "parent_company": "Global Holdings",
                          "subsidiary_affiliate": "Beta Industries",
                          "relation_type": "subsidiary",
                          "stake_percentage": 100,
                          "source_date": "2022-11-01",
                          "source_description": "SEC Filing",
                          "source_url": "https://linkup.so/source/filing_789"
                      },
                       { # Another relationship for Beta Industries
                           "parent_company": "Beta Industries",
                           "subsidiary_affiliate": "SubCo Alpha",
                           "relation_type": "parent", # Note: Linkup uses 'parent', we map to SUBSIDIARY_OF
                           "stake_percentage": 51,
                           "source_date": "2023-03-15",
                           "source_description": "Annual Report",
                           "source_url": "https://linkup.so/source/annual_report_111"
                       }
                  ]
             }
        },
         { # Hypothetical structured data for a Regulator/Sanction
             "entity": "SAMR", # Entity the search was for
             "schema": "regulatory_actions", # Hypothetical schema name
             "data": { # Hypothetical data structure
                  "regulator_name": "State Administration for Market Regulation (SAMR)",
                  "actions_found": [
                      {
                          "action_type": "Investigation",
                          "description": "Investigation into anti-competitive practices",
                          "affected_entities": ["Company S", "Company T"], # List of affected companies
                          "source_date": "2024-04-01",
                          "source_url": "https://linkup.so/source/samr_action_1"
                      }
                  ]
             }
         }
        ]
        # Process the hypothetical structured data
        processed_structured = process_linkup_structured_data(hypothetical_structured_data_list, "Test Structured Data Query");
        print("\nProcessed Structured Data (NLP Output):");
        print(json.dumps(processed_structured, indent=2)); time.sleep(1)

        # Example of combining processed structured data with snippet data for a final view
        # Note: Orchestrator handles actual merging, this is just to show the combined format
        test_final_entities = test_entities + processed_structured.get("entities", [])
        test_final_risks = test_risks_linked + processed_structured.get("risks", [])
        test_final_relationships = test_relationships_combined + processed_structured.get("relationships", [])

        print("\n--- Combined Data from Snippets and Hypothetical Structured Data (Example) ---")
        print(f"Entities: {len(test_final_entities)}")
        print(f"Risks: {len(test_final_risks)}")
        print(f"Relationships: {len(test_final_relationships)}")

        # Example of generating a summary using the combined data
        print("\n--- Generating Summary with Combined Data ---")
        # For the summary test here, we'll create a dummy results dictionary
        # that resembles what the orchestrator passes, including the KG status.
        dummy_results_for_summary_test = {
             "query": "Combined Test Data Analysis",
             "final_extracted_data": {
                  "entities": test_final_entities,
                  "risks": test_final_risks,
                  "relationships": test_final_relationships,
             },
             "linkup_structured_data": hypothetical_structured_data_list,
             "high_risk_exposures": [], # Assuming no exposures generated by this test data alone
             "kg_update_status": "success" # Dummy KG status for summary test
        }
        dummy_exposure_count = 0 # Assuming no exposures generated by this test data alone


        test_summary = generate_analysis_summary(
            dummy_results_for_summary_test, # Pass the dummy results dict
            "Combined Test Data Analysis",
            dummy_exposure_count,
            provider_to_test,
            model_to_test
        )
        print("\nGenerated Summary (Test):")
        print(test_summary)


    else:
        print("\nSkipping Linkup structured data processing test: Linkup search is not enabled or functions unavailable.")


    print("\n--- Local Tests Complete ---")
