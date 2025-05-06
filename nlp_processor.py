# nlp_processor.py

import json
from typing import List, Dict, Any, Optional, Mapping, Tuple, Union
import time
import re
import traceback
import sys
import asyncio # Import asyncio for async operations

# Import config
import config

# Import LLM libraries directly, check availability later
# Assuming these libraries (openai, google.generativeai) have async capabilities
try:
    import openai
    openai_available = True
except ImportError:
    openai = None
    openai_available = False
    print("Warning: OpenAI library not installed. OpenAI/OpenRouter functionality disabled.")


try:
    import google.generativeai as genai
    google_genai_available = True
except ImportError:
    genai = None
    google_genai_available = False
    print("Warning: Google Generative AI library not installed. Google AI functionality disabled.")


# Import pycountry at module level but check availability in functions
try:
    import pycountry
    pycountry_available_nlp = True # Use a different name to avoid conflict if orchestrator also imports
except ImportError:
    pycountry = None
    pycountry_available_nlp = False
    print("Warning: 'pycountry' not installed. Language name resolution limited in NLP.")


# Define LLM concurrency limit using a Semaphore
# Adjust this value based on the rate limits of your LLM provider(s) and available resources.
# A higher value allows more concurrent calls, potentially speeding up processing,
# but risks hitting rate limits or overwhelming resources.
# Example: GPT-4o-mini might have a higher limit than GPT-4. OpenRouter depends on the specific model.
# Start with a conservative value and increase if monitoring shows capacity is available.
DEFAULT_LLM_CONCURRENCY_LIMIT = 5 # Limit concurrent LLM calls

# Initialize the semaphore
LLM_SEMAPHORE = asyncio.Semaphore(DEFAULT_LLM_CONCURRENCY_LIMIT)

# FIX: Define allowed_entity_types globally so it's accessible to multiple functions
ALLOWED_ENTITY_TYPES = ["COMPANY", "ORGANIZATION", "REGULATORY_AGENCY", "SANCTION"]


def _get_llm_client_and_model(provider: str, model_name: str):
    """Initializes and returns the LLM client and effective model name. Checks for library availability."""
    # This function remains synchronous as it's purely initialization and config check
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
        if not openai_available:
             raise ImportError("OpenAI library is required but not installed.")

        api_key = config.OPENAI_API_KEY
        if not api_key: raise ValueError("OpenAI API Key missing.")
        try:
            # Use openai.AsyncOpenAI for async operations
            is_likely_openai_model = any(m in model_name.lower() for m in ["gpt-4", "gpt-3.5", "dall-e", "whisper"])
            if not is_likely_openai_model and config.OPENROUTER_API_KEY and config.OPENROUTER_BASE_URL:
                 # Use OpenRouter base URL with AsyncOpenAI if API key is present
                 client_or_lib = openai.AsyncOpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL, default_headers=config.OPENROUTER_HEADERS)
                 client_type = "openai_compatible"
            else:
                client_or_lib = openai.AsyncOpenAI(api_key=api_key)
                client_type = "openai_compatible"

        except Exception as e:
            print(f"ERROR init OpenAI: {e}")
            raise e

    elif provider == "openrouter":
        # Check for library availability
        if not openai_available: # OpenRouter client uses OpenAI library
             raise ImportError("OpenAI library (used by OpenRouter client) is required but not installed.")

        api_key = config.OPENROUTER_API_KEY
        if not api_key: raise ValueError("OpenRouter API Key missing.")
        if not config.OPENROUTER_BASE_URL: raise ValueError("OpenRouter Base URL missing.")
        try:
            # Use openai.AsyncOpenAI for async operations with OpenRouter base URL
            client_or_lib = openai.AsyncOpenAI(api_key=api_key, base_url=config.OPENROUTER_BASE_URL, default_headers=config.OPENROUTER_HEADERS)
            client_type = "openai_compatible"
        except Exception as e:
            print(f"ERROR init OpenRouter: {e}")
            raise e

    elif provider == "google_ai":
        # Check for library availability
        if not google_genai_available:
            raise ImportError("Google Generative AI library is required but not installed.")

        api_key = config.GOOGLE_AI_API_KEY
        if not api_key: raise ValueError("Google AI API Key missing.")
        try:
            genai.configure(api_key=api_key)
            try:
                # Check models asynchronously if the SDK allows, or rely on sync method here.
                # The current genai.list_models is sync.
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

            # For Google AI, the model itself acts as the client instance
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

# --- Async LLM Call Helper for Text Responses ---
async def _call_llm_and_get_text(prompt: str, llm_provider: str, llm_model: str, function_name: str, max_tokens: int = 2000) -> Optional[str]:
    """Helper function to call LLM expecting a text response (async). Acquires semaphore."""
    raw_content = None
    api_error = None
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        if client_or_lib is None:
            print(f"--- [{function_name}] LLM client not available for {llm_provider}. Skipping LLM call. ---")
            return None

        # start_time = time.time() # Optional timing
        async with LLM_SEMAPHORE: # Acquire the semaphore - limits concurrent LLM calls
            if client_type == "openai_compatible":
                if not openai_available: raise ImportError("OpenAI library not available")
                request_params = {"model": model_name_used, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": max_tokens}
                try:
                    response = await client_or_lib.chat.completions.create(**request_params)
                    raw_content = response.choices[0].message.content
                except Exception as e_call:
                     print(f"--- [{function_name}] ERROR during async OpenAI call: {type(e_call).__name__}: {e_call} ---")
                     api_error = e_call

            elif client_type == "google_ai":
                if not google_genai_available: raise ImportError("Google GenAI not available")
                safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in genai.types.HarmCategory if c != genai.types.HarmCategory.HARM_CATEGORY_UNSPECIFIED]
                generation_config = genai.types.GenerationConfig(temperature=0.1, max_output_tokens=max_tokens);
                try:
                    response_obj = await client_or_lib.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings);
                    if response_obj and not response_obj.candidates: # Check if response_obj is valid and has candidates
                        feedback = getattr(response_obj, 'prompt_feedback', None)
                        block_reason = getattr(feedback, 'block_reason', 'Unknown')
                        print(f"--- ERROR: Google AI BLOCKED for text response. Reason: {block_reason}. ---")
                        api_error = ValueError(f"Google AI blocked: {block_reason}")
                        raw_content = None # Ensure raw_content is None if blocked

                    elif response_obj and hasattr(response_obj, 'text'): # Check if response_obj is valid and has text attribute
                         raw_content = response_obj.text
                    else:
                        # Handle cases where response_obj is not valid or text is missing but no block
                        print(f"--- WARNING: Google AI response invalid or missing text attribute. Response object: {response_obj} ---")
                        raw_content = None # Treat as empty response

                except Exception as e_google:
                    print(f"--- [{function_name}] ERROR during async Google AI call: {type(e_google).__name__}: {e_google} ---")
                    api_error = e_google
            else:
                raise ValueError("Unknown client type.")

        # duration = time.time() - start_time # Optional timing
        # if raw_content: print(f"--- [{function_name}] LLM text call took {duration:.2f} seconds. ---") # Optional timing log


        if raw_content:
            cleaned_content = raw_content.strip()
            # Remove markdown code block syntax if present (more robust)
            cleaned_content = re.sub(r'^```.*?(\n|$)', '', cleaned_content, flags=re.IGNORECASE | re.DOTALL) # Use ignorecase and dotall
            cleaned_content = re.sub(r'```$', '', cleaned_content)
            # Remove leading/trailing quotes that some models add
            cleaned_content = re.sub(r'^["\']|["\']$', '', cleaned_content)
            cleaned_content = cleaned_content.strip()
            return cleaned_content

        else:
             print(f"--- [{function_name}] ERROR: Raw content from LLM was None or empty. Cannot return text. ---")
             # Set a generic error if no API error occurred but content was empty
             if api_error is None:
                 api_error = ValueError("LLM returned None or empty content")
             return None

    except Exception as e:
        print(f"--- [{function_name}] ERROR during async LLM text call setup/outer catch: {type(e).__name__}: {e} ---")
        traceback.print_exc()
        return None


# --- Async LLM Call Helper for JSON Responses ---
async def _call_llm_and_parse_json(prompt: str, llm_provider: str, llm_model: str,
                             function_name: str, attempt_json_mode: bool = True, max_tokens: int = 4000) -> Optional[Dict]:
    """Helper function to call LLM expecting a JSON response and parse it (async). Acquires semaphore."""
    raw_content = None
    api_error = None
    parsed_json = None
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        if client_or_lib is None:
             print(f"--- [{function_name}] LLM client not available for {llm_provider}. Skipping LLM call. ---")
             return None

        # start_time = time.time() # Optional timing

        async with LLM_SEMAPHORE: # Acquire the semaphore - limits concurrent LLM calls
            if client_type == "openai_compatible":
                if not openai_available: raise ImportError("OpenAI library not available")

                request_params = {"model": model_name_used, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": max_tokens}
                # Attempt JSON mode only if the provider is OpenAI OR OpenRouter AND the model supports it
                if attempt_json_mode and (llm_provider == "openai" or llm_provider == "openrouter") and any(m in model_name_used.lower() for m in ["gpt-4", "gpt-3.5-turbo", "gpt-4o", "gemini", "qwen", "command"]): # Added more model name indicators
                     request_params["response_format"] = {"type": "json_object"}
                     # Add a hint to the prompt for models that might need it (e.g. some OpenAI models require "json" keyword)
                     # Although the response_format parameter is primary, this can help.
                     if "json" not in prompt.lower() and "json object" not in prompt.lower() and "json array" not in prompt.lower():
                          prompt = "Provide the response as a JSON object.\n\n" + prompt
                          request_params["messages"] = [{"role": "user", "content": prompt}] # Update messages with new prompt


                try:
                     response = await client_or_lib.chat.completions.create(**request_params)
                     raw_content = response.choices[0].message.content
                     # response_obj = response # Optional: store response object for debugging
                except Exception as e_call:
                     # Retry without JSON mode if the error suggests response_format issue
                     if attempt_json_mode and "response_format" in request_params and \
                        any(err_txt in str(e_call).lower() for err_txt in ["response_format", "json_object", "messages must contain the word 'json", "invalid response format", "model is not available with this response_format type"]): # Added more error indicators
                         print(f"--- [{function_name}] WARNING: LLM JSON mode likely failed ({type(e_call).__name__}). Retrying WITHOUT JSON mode... ---")
                         del request_params["response_format"]
                         # Restore original prompt for retry if we modified it
                         if "Provide the response as a JSON object." in prompt:
                             request_params["messages"] = [{"role": "user", "content": prompt.replace("Provide the response as a JSON object.\n\n", "", 1)}]
                             print("--- [{function_name}] Restored original prompt for retry.")

                         try:
                             response = await client_or_lib.chat.completions.create(**request_params)
                             raw_content = response.choices[0].message.content
                             # response_obj = response
                         except Exception as e_retry:
                             print(f"--- [{function_name}] ERROR on retry without JSON mode: {type(e_retry).__name__}: {e_retry} ---")
                             # traceback.print_exc() # Suppress verbose retry traceback
                             api_error = e_retry
                             raw_content = None
                     else:
                          # Log other OpenAI API call errors
                          print(f"--- [{llm_provider}] API call error: {type(e_call).__name__}: {e_call} ---")
                          traceback.print_exc() # Print traceback for unexpected API errors
                          api_error = e_call
                          raw_content = None

            elif client_type == "google_ai":
                if not google_genai_available: raise ImportError("Google GenAI not available")
                safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in genai.types.HarmCategory if c != genai.types.HarmCategory.HARM_CATEGORY_UNSPECIFIED]
                # Attempt JSON mime type only if the model name suggests it supports it (e.g., Gemini 1.5)
                try_json_mime = attempt_json_mode and "gemini-1.5" in model_name_used.lower()
                generation_config = genai.types.GenerationConfig(temperature=0.1, max_output_tokens=max_tokens, response_mime_type="application/json" if try_json_mime else None )

                try:
                    response_obj = await client_or_lib.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings);
                    if response_obj and not response_obj.candidates: # Check if response_obj is valid and has candidates
                        feedback = getattr(response_obj, 'prompt_feedback', None)
                        block_reason = getattr(feedback, 'block_reason', 'Unknown')
                        print(f"--- ERROR: Google AI BLOCKED for JSON response. Reason: {block_reason}. ---")
                        api_error = ValueError(f"Google AI blocked: {block_reason}")
                        raw_content = None # Ensure raw_content is None if blocked

                    elif response_obj and hasattr(response_obj, 'text'): # Check if response_obj is valid and has text attribute
                         raw_content = response_obj.text
                    else:
                        # Handle cases where response_obj is not valid or text is missing but no block
                        print(f"--- WARNING: Google AI response invalid or missing text attribute. Response object: {response_obj} ---")
                        raw_content = None # Treat as empty response

                except Exception as e_google:
                    print(f"--- [{function_name}] ERROR during async Google AI call: {type(e_google).__name__}: {e_google} ---")
                    traceback.print_exc()
                    api_error = e_google
                    raw_content = None
            else:
                raise ValueError("Unknown client type.")

        # duration = time.time() - start_time # Optional timing
        # if raw_content: print(f"--- [{function_name}] LLM JSON call took {duration:.2f} seconds. ---") # Optional timing log


        if raw_content:
            # Clean and parse JSON from the raw content
            try:
                cleaned_content = raw_content.strip()
                # Remove markdown code block syntax if present (more robust)
                cleaned_content = re.sub(r'^```json\s*', '', cleaned_content, flags=re.IGNORECASE | re.DOTALL) # Use ignorecase and dotall
                cleaned_content = re.sub(r'```$', '', cleaned_content)
                # Remove leading/trailing quotes that some models add
                cleaned_content = re.sub(r'^["\']|["\']$', '', cleaned_content)
                cleaned_content = cleaned_content.strip()

                # Find the first '{' or '[' and last '}' or ']' to isolate the JSON
                json_start = min(cleaned_content.find('{') if '{' in cleaned_content else float('inf'),
                                 cleaned_content.find('[') if '[' in cleaned_content else float('inf'))
                json_end_obj = cleaned_content.rfind('}') + 1
                json_end_array = cleaned_content.rfind(']') + 1
                json_end = max(json_end_obj if json_end_obj > json_start else -1,
                               json_end_array if json_end_array > json_start else -1)


                if json_start != float('inf') and json_end != -1 and json_end > json_start:
                    json_str = cleaned_content[json_start:json_end]
                    # Basic check that it looks like a JSON object or array
                    if (json_str.startswith('{') and json_str.endswith('}')) or (json_str.startswith('[') and json_str.endswith(']')):
                         parsed_json = json.loads(json_str)
                    else:
                         # If braces/brackets were found but don't form a valid JSON object string,
                         # try parsing the whole cleaned content as a fallback.
                         print(f"--- [{function_name}] WARNING: Found JSON delimiters, but string doesn't look like valid JSON. Attempting full cleaned content parse.")
                         try:
                              parsed_json = json.loads(cleaned_content)
                         except Exception as e_fallback_parse:
                              print(f"--- [{function_name}] ERROR fallback parsing cleaned content: {type(e_fallback_parse).__name__}: {e_fallback_parse} ---")
                              api_error = e_fallback_parse
                              parsed_json = None
                else:
                    # If no JSON delimiters were found, assume the whole cleaned content should be JSON
                    print(f"--- [{function_name}] WARNING: Could not find JSON delimiters. Attempting full cleaned content parse.")
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
             return None

        # Final check: ensure the parsed result is a dictionary as expected by most extraction functions
        if parsed_json is not None and isinstance(parsed_json, dict):
             # print(f"--- [{function_name}] Successfully Parsed JSON response. ---") # Suppress frequent log
             return parsed_json
        # Allow list for certain expected JSON outputs (like relationship lists if schema changes)
        # Wrap list in dict for consistent return type expectation (most functions expect dict)
        elif parsed_json is not None and isinstance(parsed_json, list):
             # print(f"--- [{function_name}] Successfully Parsed JSON response as list. ---") # Suppress frequent log
             return {"items": parsed_json} # Returning a dict with 'items' key
        elif api_error is not None:
             # If an API error occurred, report it
             print(f"--- [{function_name}] ERROR: LLM call/parsing failed. Error: {type(api_error).__name__} ---")
             return None # Return None on error
        else:
             # If parsed_json is not a dict/list (e.g., string, None after fallbacks) or some other issue
             print(f"--- [{function_name}] ERROR: JSON obtained but not dict/list, or unknown parsing issue. Parsed type: {type(parsed_json).__name__}. Content: {parsed_json} ---")
             # Optionally print the parsed content if not a dict for debugging
             # print(f"--- [{function_name}] Parsed content: {parsed_json} ---")
             return None # Return None if the final result isn't the expected format

    except Exception as e:
        print(f"--- [{function_name}] ERROR during async LLM call/setup (outer catch): {type(e).__name__}: {e} ---")
        traceback.print_exc()
        return None


# --- Keyword Generation ---
# This function now uses the async _call_llm_and_get_text helper
async def translate_keywords_for_context(original_query: str, target_context: str,
                                   llm_provider: str, llm_model: str) -> List[str]:
    """Generates search keywords for a given context (async). Checks for LLM availability."""
    print(f"\n--- Attempting async Keyword Generation via {llm_provider} (Model: {llm_model}) ---")
    if not llm_provider or not llm_model:
        print("Warning: Missing LLM config for keyword generation. Skipping.")
        return [original_query]

    # FIX: Updated prompt to strongly emphasize relevance to the *initial query*
    prompt = f"""Expert keyword generator: Given the INITIAL QUERY and TARGET SEARCH CONTEXT below, provide a comma-separated list of 3-5 highly relevant ENGLISH search keywords. Prioritize keywords directly related to the INITIAL QUERY. If the INITIAL QUERY or TARGET SEARCH CONTEXT includes non-English terms (like Chinese), you may include relevant non-English keywords suitable for search engines targeting that region.
INITIAL QUERY: {original_query}
TARGET SEARCH CONTEXT: {target_context}
IMPORTANT: Your entire response must contain ONLY the comma-separated list of keywords. Do NOT include any other text, explanation, or formatting."""


    # Use the async text helper
    raw_content = await _call_llm_and_get_text(
         prompt, llm_provider, llm_model, "Keyword Generation", max_tokens=150
    )

    if not raw_content:
         print("Warning: Keyword generation returned no content. Returning original query.")
         return [original_query]


    cleaned_content = raw_content.strip()
    # Remove markdown code block syntax if present (more robust)
    cleaned_content = re.sub(r'^```.*?(\n|$)', '', cleaned_content, flags=re.IGNORECASE | re.DOTALL) # Use ignorecase and dotall
    cleaned_content = re.sub(r'```$', '', cleaned_content)
    # Remove leading/trailing quotes that some models add
    cleaned_content = re.sub(r'^["\']|["\']$', '', cleaned_content)
    cleaned_content = cleaned_content.strip()

    # FIX: Refined conversational filler check to be more specific and less prone to false positives
    # Only filter out responses that *only* contain conversational text or look like empty/failure states
    # Allow keywords that might contain common words if they are part of a valid phrase
    conversational_only_phrases = ["sorry", "apologize", "cannot", "unable", "provide", "based on the text", "hello", "hi", "greetings", "summary"] # Added summary
    # Check if the entire cleaned response (case-insensitive, stripped) is one of the filler phrases
    is_pure_filler = cleaned_content.lower() in conversational_only_phrases
    # Check if the response contains any words *besides* potential fillers. If not, it's likely conversational/empty.
    words_in_response = [word.strip() for word in cleaned_content.split() if word.strip()]
    is_only_fillers = all(re.search(r'\b' + re.escape(word) + r'\b', cleaned_content.lower()) for word in conversational_only_phrases) and len(words_in_response) < 5 # Very few words and all look like fillers

    # Also check for obvious signs of failure if the response is too short to contain meaningful keywords
    is_too_short_for_keywords = len(cleaned_content) < 10 # Keywords should be longer than ~10 chars


    if not cleaned_content or is_pure_filler or is_only_fillers or is_too_short_for_keywords:
         print(f"Warning: Keyword response conversational/empty/unhelpful ('{cleaned_content[:100]}...'). Returning original.")
         return [original_query]


    keywords = [kw.strip() for kw in cleaned_content.split(',') if kw.strip()]

    # Filter out keywords that are just punctuation or very short non-meaningful strings
    keywords = [kw for kw in keywords if len(kw) > 1 and not all(c in '.,!?"\'' for c in kw)]

    # Final check: If after cleaning and splitting, the keywords list is empty, return original query
    if not keywords:
        print(f"Warning: Keyword generation resulted in an empty list after cleaning ('{cleaned_content[:100]}...'). Returning original.")
        return [original_query]


    print(f"Parsed Keywords: {keywords}")
    return keywords

# --- Translation Functions ---
# This function now uses the async _call_llm_and_get_text helper
async def translate_text(text: str, target_language: str, llm_provider: str, llm_model: str) -> Optional[str]:
    """Translates a single piece of text using the LLM (async). Checks for LLM availability."""
    if not text or not isinstance(text, str) or not llm_provider or not llm_model:
         return None

    # Check for LLM availability before making the call
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        if client_or_lib is None:
            print(f"Warning: LLM client not available for translation using {llm_provider}. Skipping translation.")
            return None
    except Exception as e:
         print(f"Error getting LLM client for translation: {e}. Skipping translation.")
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
IMPORTANT: Provide ONLY the translated text. Do NOT include any introductory phrases, explanations, or markdown."""

    # Put text to translate separately to avoid issues with f-string and user text
    full_prompt = f"{prompt}\n\nText to translate:\n{text}"

    # Use the async text helper
    raw_content = await _call_llm_and_get_text(
        full_prompt, llm_provider, llm_model, "Translation", max_tokens=min(len(text) * 3, 2000) # Adjust max tokens based on input size
    )

    if not raw_content:
         print(f"Warning: Translation returned no content for text: '{text[:50]}...'.")
         return None


    cleaned_content = raw_content.strip()
    # Remove markdown code block syntax if present (more robust)
    cleaned_content = re.sub(r'^```.*?(\n|$)', '', cleaned_content, flags=re.IGNORECASE | re.DOTALL) # Use ignorecase and dotall
    cleaned_content = re.sub(r'```$', '', cleaned_content)
    # Remove leading/trailing quotes that some models add
    cleaned_content = re.sub(r'^["\']|["\']$', '', cleaned_content)
    cleaned_content = cleaned_content.strip()


    # FIX: Use the refined conversational and short checks for translation specifically
    # Check if the entire response looks like only conversational fillers or is too short/empty
    conversational_only_phrases = ["sorry", "apologize", "cannot", "unable", "translate", "based on the text", "hello", "hi", "greetings"]
    # Check if the entire cleaned response (case-insensitive, stripped) is one of the filler phrases
    is_pure_filler = cleaned_content.lower() in conversational_only_phrases
    # Check if it's too short compared to the original text, allowing for very short valid translations
    # This check is complex. Let's simplify: just check if the response is excessively short AND contains few words,
    # or if it seems to contain only filler words.
    words_in_response = [word.strip() for word in cleaned_content.split() if word.strip()]
    is_suspiciously_short = len(cleaned_content) < max(len(text) * 0.1, 5) # Too short (less than 10% of original, or less than 5 chars)
    is_only_fillers = len(words_in_response) < 5 and any(re.search(r'\b' + re.escape(word) + r'\b', cleaned_content.lower()) for word in conversational_only_phrases) # Very few words AND contain fillers


    if not cleaned_content or is_pure_filler or is_suspiciously_short or is_only_fillers:
        print(f"Warning: Translation response short/empty/conversational: '{cleaned_content[:100]}...' for text '{text[:50]}...'.")
        return None

    return cleaned_content

async def translate_snippets(snippets: List[Dict[str, Any]], target_language: str, llm_provider: str, llm_model: str) -> List[Dict[str, Any]]:
    """Translates a list of snippets to the target language (async). Checks for LLM availability."""
    if not snippets or not llm_provider or not llm_model:
         return snippets

    # Check for LLM availability before starting translation
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        if client_or_lib is None:
            print(f"Warning: LLM client not available for snippet translation using {llm_provider}. Skipping translation.")
            return snippets
    except Exception as e:
         print(f"Error getting LLM client for snippet translation: {e}. Skipping translation.")
         return snippets


    print(f"--- Attempting to translate {len(snippets)} snippets to '{target_language}' asynchronously ---")
    translated_snippets = []

    async def translate_single_snippet(snippet_data):
         # Ensure snippet_data is valid and has a snippet before attempting translation
         if not isinstance(snippet_data, dict) or not snippet_data.get('snippet') or not isinstance(snippet_data.get('snippet'), str):
              return snippet_data # Return invalid/empty snippets as is

         original_snippet = snippet_data['snippet']
         # Await the async translate_text call
         translated_snippet_content = await translate_text(original_snippet, target_language, llm_provider, llm_model)

         if translated_snippet_content is not None:
             translated_s = snippet_data.copy()
             translated_s['snippet'] = translated_snippet_content
             # Store original language and add 'translated_from' tag
             translated_s['original_language'] = snippet_data.get('original_language', 'unknown')
             translated_s['translated_from'] = translated_s['original_language'] # Record what it was translated from
             # Keep the original snippet if needed for reference (optional)
             # translated_s['original_snippet'] = original_snippet

             return translated_s
         else:
             # If translation failed or returned None, keep the original snippet
             print(f"Warning: Failed to translate snippet (URL: {snippet_data.get('url', 'N/A')}). Including original.")
             return snippet_data


    # Create a list of tasks for each snippet translation
    translation_tasks = [translate_single_snippet(s) for s in snippets]

    # Run translation tasks concurrently
    # asyncio.gather automatically respects the semaphore used within translate_single_snippet's call to _call_llm_and_get_text
    translated_snippets = await asyncio.gather(*translation_tasks)


    print(f"--- Finished async snippet translation. Translated/processed {len(translated_snippets)} snippets. ---")
    return translated_snippets


def _prepare_context_text(search_results: List[Dict[str, Any]]) -> Tuple[str, int]:
    """Performs synchronous preparation of formatted context string from search results."""
    # This function remains synchronous as it does not involve external I/O
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
             translation_info = f" (Translated from {translated_from})"
        # Check if original_lang is a string AND not English AND it hasn't been translated
        elif isinstance(original_lang, str) and original_lang.lower() not in ['en', 'english'] and not translated_from:
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

# _call_llm_and_parse_json is already an async helper now


# Extraction functions using _call_llm_and_parse_json now need to be async
async def extract_entities_only(search_results: List[Dict[str, Any]], extraction_context: str,
                          llm_provider: str, llm_model: str) -> List[Dict]:
    """Extracts entities (COMPANY, ORGANIZATION, REGULATORY_AGENCY, SANCTION) based on schema (async). Checks for LLM availability."""
    print("\n--- [NLP Entities] Attempting async Entity Extraction ---")
    function_name = "NLP Entities"
    if not search_results: print(f"--- [{function_name}] No search results, skipping. ---"); return []
    # Check for LLM availability
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        if client_or_lib is None:
             print(f"Warning: LLM client not available for {llm_provider}. Skipping entity extraction. ---")
             return []
    except Exception as e:
         print(f"Error getting LLM client for entity extraction: {e}. Skipping entity extraction.")
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
    # No chunking needed here as we're sending the whole context at once to the LLM
    # for initial entity extraction.
    parsed_json = await _call_llm_and_parse_json(prompt, llm_provider, llm_model, function_name, attempt_json_mode=True)

    validated_entities = []
    # Use the globally defined ALLOWED_ENTITY_TYPES
    # allowed_entity_types = ["COMPANY", "ORGANIZATION", "REGULATORY_AGENCY", "SANCTION"] # Removed local definition
    # Add a simple check to filter common country names if they are tagged as REGULATORY_AGENCY by mistake
    common_country_names_lower = {"united states", "us", "china", "cn", "united kingdom", "uk", "germany", "de", "india", "in", "france", "fr", "japan", "jp", "canada", "ca", "australia", "au"} # Add more as needed
    # Add known broad non-Chinese organizations that might be misidentified as companies or regulators
    common_non_chinese_orgs_lower = {"oecd", "nato", "un", "world bank", "imf", "european union", "eu"}

    # Ensure parsed_json is a dictionary and contains the 'entities' key which is a list
    entities_list_from_llm = parsed_json.get("entities", []) if isinstance(parsed_json, dict) else []


    if isinstance(entities_list_from_llm, list):
        for entity in entities_list_from_llm:
             # Ensure the item is a dictionary before trying to access keys
             if not isinstance(entity, dict):
                  print(f"--- [{function_name}] Skipping invalid entity item (not a dictionary): {entity} ---")
                  continue

             entity_name = entity.get("name")
             entity_type = entity.get("type")
             entity_mentions = entity.get("mentions")

             # Use the global ALLOWED_ENTITY_TYPES here
             if entity_name and isinstance(entity_name, str) and entity_type in ALLOWED_ENTITY_TYPES and \
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
         print(f"--- [{function_name}] Failed to parse valid 'entities' list from LLM response. Parsed content type: {type(parsed_json.get('entities')).__name__ if isinstance(parsed_json, dict) else type(parsed_json).__name__}. Content: {parsed_json} ---")


    print(f"--- [{function_name}] Returning {len(validated_entities)} validated entities ({', '.join(ALLOWED_ENTITY_TYPES)} only). ---")
    return validated_entities

# Extraction functions using _call_llm_and_parse_json now need to be async
async def extract_risks_only(search_results: List[Dict[str, Any]], extraction_context: str,
                       llm_provider: str, llm_model: str) -> List[Dict]:
    """Extracts only risks (desc, severity, source_urls) based on the schema (async). Checks for LLM availability."""
    print("\n--- [NLP Risks] Attempting async Risk Extraction (without related entities) ---")
    function_name = "NLP Risks"
    if not search_results: print(f"--- [{function_name}] No search results, skipping. ---"); return []
    # Check for LLM availability
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        if client_or_lib is None:
             print(f"Warning: LLM client not available for {llm_provider}. Skipping risk extraction. ---")
             return []
    except Exception as e:
         print(f"Error getting LLM client for risk extraction: {e}. Skipping risk extraction.")
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
    # Await the async LLM call helper
    parsed_json = await _call_llm_and_parse_json(prompt, llm_provider, llm_model, function_name, attempt_json_mode=True)

    validated_risks = []
    # Ensure parsed_json is a dictionary and contains the 'risks' key which is a list
    risks_list_from_llm = parsed_json.get("risks", []) if isinstance(parsed_json, dict) else []

    if isinstance(risks_list_from_llm, list):
        for risk in risks_list_from_llm:
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
         print(f"--- [{function_name}] Failed to parse valid 'risks' list from LLM response. Parsed content type: {type(parsed_json.get('risks')).__name__ if isinstance(parsed_json, dict) else type(parsed_json).__name__}. Content: {parsed_json} ---")


    print(f"--- [{function_name}] Returning {len(validated_risks)} validated risks (initially without related_entities). ---")
    return validated_risks

# Risk linking function uses _call_llm_and_parse_json and can process risks concurrently
async def link_entities_to_risk(risks: List[Dict],
                          list_of_entity_names: List[str],
                          all_snippets_map: Mapping[str, Dict[str, Any]],
                          llm_provider: str, llm_model: str) -> List[Dict]:
    """
    Takes extracted risks and entity names (from snippets), and uses LLM to populate the 'related_entities' field for each risk
    based on the original snippets associated with the risk's source URLs (async). Returns updated risks list. Checks for LLM availability.
    """
    print(f"\n--- [NLP Linker] Attempting to Link Entities (from snippets) to Risks (from snippets) asynchronously ---")
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
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        if client_or_lib is None:
             print(f"Warning: LLM client not available for {llm_provider}. Skipping entity linking. ---")
             # Ensure snippet risks that needed linking get an empty related_entities list
             for risk in snippet_risks_to_link:
                  if isinstance(risk, dict):
                       risk["related_entities"] = []
                       other_risks.append(risk)
             return risks
    except Exception as e:
         print(f"Error getting LLM client for entity linking: {e}. Skipping entity linking.")
         # Ensure snippet risks that needed linking get an empty related_entities list
         for risk in snippet_risks_to_link:
              if isinstance(risk, dict):
                   risk["related_entities"] = []
                   other_risks.append(risk)
         return risks


    updated_risks_from_linking = [] # This will hold the risks after attempting linking

    print(f"--- [{function_name}] Will check {len(snippet_risks_to_link)} snippet risks against {len(list_of_entity_names)} filtered entities asynchronously...")

    async def link_single_risk(risk_index, risk):
        # Re-check validity before processing
        if not isinstance(risk, dict) or not risk.get('description') or not risk.get('source_urls'):
            print(f"--- [{function_name}] Skipping invalid snippet risk object at index {risk_index}. ---")
            # Add an empty related_entities list even if invalid and append to the return list
            if isinstance(risk, dict) and "related_entities" not in risk: risk["related_entities"] = []
            return risk # Return the potentially modified invalid risk

        risk_desc = risk['description']; risk_urls = risk['source_urls']
        # Prepare context with snippets associated with this specific risk
        risk_context_text = ""; added_snippets = 0; max_chars = 7000
        # Construct the list of potential entities string separately
        potential_entities_list_str = ', '.join([f"'{name}'" for name in list_of_entity_names])
        # Construct the context_intro string, embedding the pre-formatted list string
        context_intro = f"The following risk was identified: \"{risk_desc}\". It was found in text snippets from these sources: {', '.join(risk_urls)}. Identify which of the following entities are directly mentioned *within the context of this specific risk* in the snippets.\n\nList of Potential Entities: [ {potential_entities_list_str} ]\n\nRelevant Text Snippets:\n"


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
            return risk # Return the risk with empty related entities


        prompt = f"""{context_intro}{risk_context_text}
Based *strictly* on the provided Relevant Text Snippets, identify ONLY the entities from the "List of Potential Entities" that are mentioned *directly together* with the specific risk described above ("{risk_desc}").

Your response MUST be ONLY a single valid JSON object with a single key "related_entities" containing a JSON array of strings (the names of the directly involved entities). Example: {{"related_entities": ["Entity A", "Entity C"]}} or {{"related_entities": []}} if none are directly involved in this specific risk context. Do not include any other text, explanation, or formatting."""

        print(f"--- [{function_name}] Linking risk #{risk_index+1}/{len(snippet_risks_to_link)}: '{risk_desc[:50]}...' using {added_snippets} snippets. ---")

        related_entity_names = []
        # Use async LLM call helper for this risk
        # Max tokens for this call might be less than overall JSON calls, as it's just a list of names
        parsed_json = await _call_llm_and_parse_json(prompt, llm_provider, llm_model, function_name, attempt_json_mode=True, max_tokens=500) # Limited tokens

        # Ensure parsed_json is a dictionary and contains the 'related_entities' key which is a list
        entity_list_from_llm = parsed_json.get("related_entities", []) if isinstance(parsed_json, dict) else []

        if isinstance(entity_list_from_llm, list):
             # Validate that all items in the list are strings
             if all(isinstance(item, str) for item in entity_list_from_llm):
                 # Filter the list from LLM to only include names that were in the input list of potential entities
                 input_names_lower = {name.lower() for name in list_of_entity_names}
                 related_entity_names = [name.strip() for name in entity_list_from_llm if isinstance(name, str) and name.strip().lower() in input_names_lower]
                 # print(f"--- [{function_name}] Parsed and filtered related entities: {related_entity_names} ---") # Suppress frequent log
             else: print(f"--- [{function_name}] WARNING: 'related_entities' array contained non-string items: {entity_list_from_llm} ---")
        else: # Log if the expected 'related_entities' list was not found or was not a list
             print(f"--- [{function_name}] WARNING: LLM response JSON did not contain a valid 'related_entities' list (got type {type(parsed_json.get('related_entities')).__name__ if isinstance(parsed_json, dict) else type(parsed_json).__name__}). Full response: {parsed_json} ---")


        risk_copy = risk.copy() # Work on a copy to avoid modifying the original list during concurrent processing
        risk_copy["related_entities"] = related_entity_names # Update the risk dictionary with the linked entities
        risk_copy['_source_type'] = 'snippet' # Ensure source type is kept

        return risk_copy # Return the updated risk


    # Create a list of tasks for each risk linking operation
    linking_tasks = [link_single_risk(i, risk) for i, risk in enumerate(snippet_risks_to_link)]

    # Run linking tasks concurrently
    # asyncio.gather automatically respects the semaphore used within link_single_risk's call to _call_llm_and_parse_json
    updated_risks_from_linking = await asyncio.gather(*linking_tasks)

    print(f"--- [{function_name}] Finished async entity linking. ---")

    # Return the combined list of risks that were updated and those that were not snippet risks
    return updated_risks_from_linking + other_risks

# Extraction functions using _call_llm_and_parse_json now need to be async
async def extract_relationships_only(search_results: List[Dict[str, Any]], extraction_context: str,
                               entities: List[Dict],
                               llm_provider: str, llm_model: str) -> List[Dict]:
    """Extracts only specific ownership relationships (SUBSIDIARY_OF, PARENT_COMPANY_OF, AFFILIATE_OF, JOINT_VENTURE_PARTNER) based on the schema from snippets (async). Checks for LLM availability."""
    print("\n--- [NLP Relationships (Ownership Only)] Attempting async Relationship Extraction (Ownership Only) ---")
    function_name = "NLP Relationships (Ownership Only)"
    # Filter search results for those with valid snippets and URLs
    relevant_search_results = [s for s in search_results if isinstance(s, dict) and s.get('snippet') and isinstance(s.get('snippet'), str) and s.get('url') and isinstance(s.get('url'), str)]

    if not relevant_search_results or not entities: print(f"--- [{function_name}] No relevant search results or entities, skipping. ---"); return []
    # Check for LLM availability
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        if client_or_lib is None:
             print(f"Warning: LLM client not available for {llm_provider}. Skipping relationship extraction. ---")
             return []
    except Exception as e:
         print(f"Error getting LLM client for relationship extraction: {e}. Skipping.")
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
        - "entity2" (string name, English - the other entity in the relationship).
        - "context_urls" (array of SourceURLs where relationship mentioned, MUST NOT be empty)."""

    prompt = f"""Analyze text snippets based on context: "{extraction_context}". Given the entities previously identified from the search results: {', '.join([f"'{name}'" for name in entity_names])}.

Extract ONLY explicitly stated **ownership, affiliate, or joint venture relationships** between these specific entities, matching the schema below.

Schema:
{rel_schema}

Focus ONLY on relationships where BOTH entity1 and entity2 are from the provided list of identified COMPANY or ORGANIZATION entities.
The relationship_type MUST be one of: SUBSIDIARY_OF, PARENT_COMPANY_OF, AFFILIATE_OF, or JOINT_VENTURE_PARTNER.
Response MUST be ONLY a single valid JSON object like {{"relationships": [...]}}. Use empty array [] if no relationships found. No explanations or markdown.

Begin analysis of text snippets:
{context_text}
"""
    # Await the async LLM call helper
    parsed_json = await _call_llm_and_parse_json(prompt, llm_provider, llm_model, function_name, attempt_json_mode=True)

    validated_relationships = []
    # Updated allowed relationship types for validation
    allowed_types_lower = {"subsidiary_of", "parent_company_of", "affiliate_of", "joint_venture_partner"}

    # Ensure parsed_json is a dictionary and contains the 'relationships' key which is a list
    relationships_list_from_llm = parsed_json.get("relationships", []) if isinstance(parsed_json, dict) else []

    if isinstance(relationships_list_from_llm, list):
        for rel in relationships_list_from_llm:
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
         print(f"--- [{function_name}] Failed to parse valid 'relationships' list from LLM response. Parsed content type: {type(parsed_json.get('relationships')).__name__ if isinstance(parsed_json, dict) else type(parsed_json).__name__}. Content: {parsed_json} ---")


    print(f"--- [{function_name}] Returning {len(validated_relationships)} validated relationships (Ownership/Affiliate/JV only). ---")
    return validated_relationships

# Extraction functions using _call_llm_and_parse_json now need to be async
async def extract_ownership_involving_entity(text_snippets: List[Dict[str, Any]], target_entity_name: str,
                                       llm_provider: str, llm_model: str) -> List[Dict]:
    """
    Analyzes text snippets to find *any* ownership or affiliate relationships
    where the target_entity_name is involved (as parent, subsidiary, affiliate, etc.) (async).
    Returns a list of relationship dictionaries. Checks for LLM availability.
    This function is used in search_engines.search_for_ownership_docs, not the main orchestrator flow.
    """
    print(f"\n--- [NLP Targeted Relationships] Attempting async relationships involving '{target_entity_name}' ---")
    function_name = "NLP Targeted Relationships"
    if not llm_provider or not llm_model: print(f"--- [{function_name}] Skipping: Missing LLM config."); return []
    # Filter text snippets for those with valid snippets and URLs
    relevant_text_snippets = [s for s in text_snippets if isinstance(s, dict) and s.get('snippet') and isinstance(s.get('snippet'), str) and s.get('url') and isinstance(s.get('url'), str)]

    if not relevant_text_snippets: print(f"--- [{function_name}] No relevant snippets provided."); return []
    # Check for LLM availability
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        if client_or_lib is None:
             print(f"Warning: LLM client not available for {llm_provider}. Skipping relationship extraction. ---")
             return []
    except Exception as e:
         print(f"Error getting LLM client for targeted relationship extraction: {e}. Skipping.")
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

    # Await the async LLM call helper
    parsed_json = await _call_llm_and_parse_json(prompt, llm_provider, llm_model, function_name, attempt_json_mode=True)

    validated_relationships = []
    # Define allowed types for this specific function's output
    allowed_types_lower = {"parent_company_of", "subsidiary_of", "affiliate_of", "acquired", "joint_venture_partner", "related_company"}

    # Ensure parsed_json is a dictionary and contains the 'relationships' key which is a list
    relationships_list_from_llm = parsed_json.get("relationships", []) if isinstance(parsed_json, dict) else []


    if isinstance(relationships_list_from_llm, list):
        for rel in relationships_list_from_llm:
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

                 # Double check that AT LEAST ONE entity is the target entity (case-insensitive)
                 if e1_name_lower == target_entity_name.lower() or e2_name_lower == target_entity_name.lower():
                      rel['_source_type'] = 'targeted_snippet_llm' # Add internal source type
                      validated_relationships.append(rel)
                 else:
                      # This shouldn't happen if the LLM followed instructions, but safety check
                      print(f"--- [{function_name}] Skipping relationship item because neither entity ('{entity1_name}', '{entity2_name}') is the target entity '{target_entity_name}': {rel} ---")
             else: print(f"--- [{function_name}] Skipping invalid or incomplete relationship item (doesn't match schema, not allowed type, or missing mandatory fields): {rel} ---")
    else:
         # Log if the expected 'relationships' list was not found or was not a list
         print(f"--- [{function_name}] Failed to parse valid 'relationships' list from LLM response. Parsed content type: {type(parsed_json.get('relationships')).__name__ if isinstance(parsed_json, dict) else type(parsed_json).__name__}. Content: {parsed_json} ---")


    print(f"--- [{function_name}] Returning {len(validated_relationships)} validated relationships involving '{target_entity_name}'. ---")
    return validated_relationships


# Extraction functions using _call_llm_and_parse_json now need to be async
async def extract_regulatory_sanction_relationships(search_results: List[Dict[str, Any]], extraction_context: str,
                                              entities: List[Dict],
                                              llm_provider: str, llm_model: str) -> List[Dict]:
    """
    Analyzes text snippets to find explicit relationships between identified Entities
    (Companies/Organizations/Agencies/Sanctions) related to regulations and sanctions (async).
    Returns a list of relationship dictionaries. Checks for LLM availability.
    """
    print("\n--- [NLP Regulatory/Sanction Relationships] Attempting async Extraction ---")
    function_name = "NLP Regulatory/Sanction Relationships"
    # Filter search results for those with valid snippets and URLs
    relevant_search_results = [s for s in search_results if isinstance(s, dict) and s.get('snippet') and isinstance(s.get('snippet'), str) and s.get('url') and isinstance(s.get('url'), str)]

    if not relevant_search_results or not entities: print(f"--- [{function_name}] No relevant search results or entities, skipping. ---"); return []
    # Check for LLM availability
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        if client_or_lib is None:
             print(f"Warning: LLM client not available for {llm_provider}. Skipping relationship extraction. ---")
             return []
    except Exception as e:
         print(f"Error getting LLM client for reg/sanc relationship extraction: {e}. Skipping.")
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
        - "entity2" (string name, English - the other entity in the relationship).
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

    # Await the async LLM call helper
    parsed_json = await _call_llm_and_parse_json(prompt, llm_provider, llm_model, function_name, attempt_json_mode=True)

    validated_relationships = []
    allowed_types_lower = {"regulated_by", "issued_by", "subject_to", "mentioned_with"}

    # Ensure parsed_json is a dictionary and contains the 'relationships' key which is a list
    relationships_list_from_llm = parsed_json.get("relationships", []) if isinstance(parsed_json, dict) else []

    if isinstance(relationships_list_from_llm, list):
        for rel in relationships_list_from_llm:
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
                      r_type_upper = rel_type_raw.upper() # Use the raw type for checking against upper case allowed types

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
                          if e1_type not in ALLOWED_ENTITY_TYPES or e2_type not in ALLOWED_ENTITY_TYPES: is_valid_rel_type = False # Use global list


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
         print(f"--- [{function_name}] Failed to parse valid 'relationships' list from LLM response. Parsed content type: {type(parsed_json.get('relationships')).__name__ if isinstance(parsed_json, dict) else type(parsed_json).__name__}. Content: {parsed_json} ---")


    print(f"--- [{function_name}] Returning {len(validated_relationships)} validated relationships (Regulatory/Sanction). ---")
    return validated_relationships


def _map_linkup_relation_type(linkup_type: str) -> str:
     """Maps Linkup's structured relationship types to internal types for storage."""
     # This function remains synchronous as it's pure data transformation
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
    This function remains synchronous as it primarily involves data parsing and transformation.
    """
    print("\n--- [NLP Structured Processor] Processing Linkup structured data list (synchronous) ---")
    processed_entities = []
    processed_risks = []
    processed_relationships = []

    if not linkup_structured_results_list or not isinstance(linkup_structured_results_list, list):
        print("--- [NLP Structured Processor] No structured data list or invalid format provided. ---")
        return {"entities": [], "risks": [], "relationships": []}

    for result_item in linkup_structured_results_list:
        # Expecting items like {"entity": "Entity Name", "schema": "schema_name", "data": {...}}
        # Or the raw structured content dict itself from LinkupStructuredSearch
        if not isinstance(result_item, dict):
             print(f"Warning: Skipping invalid structured result item format (not a dict): {result_item}")
             continue

        entity_name_context = result_item.get("entity") # The entity name the search was targeted for (if using our wrapper format)
        schema_name = result_item.get("schema")       # The schema name (e.g., "ownership", "key_risks") (if using our wrapper format)
        structured_data_content = result_item.get("data") # The actual structured data dictionary (if using our wrapper format)

        # Determine the actual data content and schema name based on the format received
        if structured_data_content is not None and isinstance(structured_data_content, dict) and schema_name is not None and isinstance(schema_name, str):
             # This looks like our wrapper format {"entity": ..., "schema": ..., "data": {...}}
             pass # Use the extracted variables as they are
        elif result_item.get("ownership_relationships") is not None or result_item.get("key_risks_identified") is not None or result_item.get("actions_found") is not None: # Added check for hypothetical types
             # This looks like the raw structured content dict itself
             structured_data_content = result_item # The data is the item itself
             # Try to infer schema name from keys
             if structured_data_content.get("ownership_relationships") is not None: schema_name = "ownership"
             elif structured_data_content.get("key_risks_identified") is not None: schema_name = "key_risks"
             elif structured_data_content.get("actions_found") is not None: schema_name = "regulatory_actions" # Assuming this key for a hypothetical schema
             else: schema_name = "unknown_structured" # Fallback
             entity_name_context = structured_data_content.get("company_name", structured_data_content.get("regulator_name", "Unknown Entity")) # Try getting entity name from content

        else:
             print(f"Warning: Skipping invalid structured result item format (doesn't match known structured formats): {result_item}")
             continue # Skip this item


        if not entity_name_context or not isinstance(entity_name_context, str) or not schema_name or not isinstance(schema_name, str) or not structured_data_content or not isinstance(structured_data_content, dict):
             print(f"Warning: Skipping structured result item after format check (missing crucial data: entity_name_context='{entity_name_context}', schema_name='{schema_name}', structured_data_content is dict? {isinstance(structured_data_content, dict)}): {result_item}")
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
        #                                 # Relationship: Company SUBJECT_TO Sanction
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


# Summary generation needs to be async
async def generate_analysis_summary(results: Dict[str, Any], query: str, exposures_count: int,
                              llm_provider: str, llm_model: str) -> str:
    """Generates a concise analysis summary using the LLM (async). Accepts the full results dict."""
    print(f"\n--- Attempting async Analysis Summary via {llm_provider} (Model: {llm_model}) ---");
    function_name = "NLP Summary"
    if not llm_provider or not llm_model: return "Summary generation skipped: Missing LLM config."
    # Check for LLM availability
    try:
        client_or_lib, client_type, model_name_used = _get_llm_client_and_model(llm_provider, llm_model)
        if client_or_lib is None:
             print(f"Warning: LLM client not available for {llm_provider}. Skipping summary generation. ---")
             return "Summary generation skipped: LLM client not available."
    except Exception as e:
         print(f"Error getting LLM client for summary generation: {e}. Skipping.")
         return f"Summary generation skipped due to LLM client error: {type(e).__name__}"


    # Use the FILTERED data from the results dict passed in
    entities = results.get("final_extracted_data", {}).get("entities", [])
    risks = results.get("final_extracted_data", {}).get("risks", [])
    relationships = results.get("final_extracted_data", {}).get("relationships", [])
    # Exposures list is also taken from the results dict, and is the filtered list
    exposures_list = results.get("high_risk_exposures", [])
    exposures_count_filtered = len(exposures_list) # Use the count of the filtered list

    # Structured raw data is still used for summary context, even if it didn't contribute to final filtered data
    structured_raw_data_list = results.get("linkup_structured_data", [])
    structured_data_present = bool(structured_raw_data_list)

    # Use the FILTERED counts for the summary log
    print(f"[Step 5.5 Summary] Data for summary (FILTERED): E:{len(entities)}, R:{len(risks)}, Rel:{len(relationships)}, Exp:{exposures_count_filtered}, Structured:{structured_data_present}.")


    if not any([entities, risks, relationships, exposures_count_filtered > 0, structured_data_present]): return "No significant data extracted or exposures identified across steps to generate a summary."
    summary_parts = []
    if entities:
        # Include new entity types in summary count/list (based on FILTERED entities)
        company_count = len([e for e in entities if isinstance(e, dict) and e.get('type') == 'COMPANY'])
        org_count = len([e for e in entities if isinstance(e, dict) and e.get('type') == 'ORGANIZATION'])
        reg_agency_count = len([e for e in entities if isinstance(e, dict) and e.get('type') == 'REGULATORY_AGENCY'])
        sanction_count = len([e for e in entities if isinstance(e, dict) and e.get('type') == 'SANCTION'])
        entity_types_summary = []
        if company_count: entity_types_summary.append(f"{company_count} Companies")
        if org_count: entity_types_summary.append(f"{org_count} Organizations")
        if reg_agency_count: entity_types_summary.append(f"{reg_agency_count} Regulatory Agencies")
        if sanction_count: entity_types_summary.append(f"{sanction_count} Sanctions")

        # Include a sample of entity names (from FILTERED entities)
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
        # Risk counts based on FILTERED risks
        high_risks = [r for r in risks if isinstance(r, dict) and r.get('severity') in ['HIGH', 'SEVERE']]
        med_risks = [r for r in risks if isinstance(r, dict) and r.get('severity') == 'MEDIUM']
        low_risks = [r for r in risks if isinstance(r, dict) and r.get('severity') == 'LOW']
        risk_summary = f"- Risks ({len(risks)} total): {len(high_risks)} High/Severe, {len(med_risks)} Medium, {len(low_risks)} Low."

        # Include sample risk descriptions (from FILTERED risks)
        sample_risk_descriptions = []
        # Find High/Severe risks linked to exposed entities (using names from FILTERED exposures)
        exposed_entity_names_lower = {name.lower() for name in exposed_entity_names}
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
        # Count relationship types relevant to sheet/KG (based on FILTERED relationships)
        # Use the allowed types defined in orchestrator for sheet saving
        allowed_sheet_rel_types = ["PARENT_COMPANY_OF", "SUBSIDIARY_OF", "AFFILIATE_OF", "JOINT_VENTURE_PARTNER", "REGULATED_BY", "ISSUED_BY", "SUBJECT_TO", "MENTIONED_WITH"]

        # relevant_rels = [r for r in relationships if isinstance(r, dict) and r.get('relationship_type') in allowed_sheet_rel_types] # Relationships are already filtered

        ownership_types = ["PARENT_COMPANY_OF", "SUBSIDIARY_OF", "AFFILIATE_OF", "JOINT_VENTURE_PARTNER"]
        reg_sanc_types = ["REGULATED_BY", "ISSUED_BY", "SUBJECT_TO"]
        mentioned_type = ["MENTIONED_WITH"]

        ownership_rels_count = len([r for r in relationships if r.get('relationship_type') in ownership_types]) # Count from filtered list
        reg_sanc_rels_count = len([r for r in relationships if r.get('relationship_type') in reg_sanc_types]) # Count from filtered list
        mentioned_rels_count = len([r for r in relationships if r.get('relationship_type') in mentioned_type]) # Count from filtered list

        rel_types_summary = []
        if ownership_rels_count: rel_types_summary.append(f"{ownership_rels_count} Ownership/Affiliate/JV")
        if reg_sanc_rels_count: rel_types_summary.append(f"{reg_sanc_rels_count} Regulatory/Sanction Action")
        if mentioned_rels_count: rel_types_summary.append(f"{mentioned_rels_count} Mentioned With")


        # Include sample relationships (from FILTERED relationships)
        sample_rels = []
        # Find relationships involving exposed entities (using names from FILTERED exposures)
        exposed_entity_names_lower = {name.lower() for name in exposed_entity_names}
        relevant_rels_for_exposed = [rel for rel in relationships if any(isinstance(e_name, str) and e_name.lower() in exposed_entity_names_lower for e_name in [rel.get('entity1',''), rel.get('entity2','')] )]
        for rel in relevant_rels_for_exposed:
             if isinstance(rel.get('entity1'), str) and isinstance(rel.get('entity2'), str) and isinstance(rel.get('relationship_type'), str):
                  if len(sample_rels) < 3: # Limit sample size
                       sample_rels.append(f"{rel['entity1'].strip()} {rel['relationship_type'].replace('_', ' ').title()} {rel['entity2'].strip()}")
                  else: break
        # Add other relevant relationships if sample isn't full
        if len(sample_rels) < 3:
             for rel in relationships:
                  if isinstance(rel.get('entity1'), str) and isinstance(rel.get('entity2'), str) and isinstance(rel.get('relationship_type'), str) and rel not in relevant_rels_for_exposed:
                       if len(sample_rels) < 3:
                             sample_rels.append(f"{rel['entity1'].strip()} {rel['relationship_type'].replace('_', ' ').title()} {rel['entity2'].strip()}")
                       else: break


        rel_list_str = "; ".join(sample_rels)
        if len(relationships) > 3: rel_list_str += "..."
        if rel_types_summary: summary_parts.append(f"- Relationships ({len(relationships)} total: {', '.join(rel_types_summary)}): " + rel_list_str + ".")
        elif rel_list_str: summary_parts.append(f"- Relationships ({len(relationships)}): " + rel_list_str + ".")


    if exposures_count_filtered > 0: # Use filtered count
         # Include sample exposures in the summary (from FILTERED list)
         sample_exposures = []
         for exp in exposures_list[:2]: # Sample first 2 exposures from the filtered list
              if isinstance(exp, dict) and exp.get('Entity'):
                   exp_summary = f"'{exp['Entity'].strip()}' ({exp.get('Risk_Type','?').replace(' Risk','')}, {exp.get('Risk_Severity','?')})" # Use cleaned Risk_Type label
                   sample_exposures.append(exp_summary)
         exposure_summary = f"- Identified {exposures_count_filtered} potential High/Severe Risk Exposures linked to Chinese Companies/Orgs via Ownership/Affiliate/JV relationships." # Use filtered count in text
         if sample_exposures: exposure_summary += f" (Examples: {'; '.join(sample_exposures)})"
         summary_parts.append(exposure_summary)


    if structured_raw_data_list:
         schema_counts = {}
         for item in structured_raw_data_list:
              # Use the 'schema' key if present, fallback to inferring from 'data' content keys if item is raw content
              schema_name_for_count = item.get("schema", "unknown")
              if schema_name_for_count == "unknown" and isinstance(item.get("data"), dict): # If wrapper format and schema is missing
                   data_content = item["data"]
                   if data_content.get("ownership_relationships") is not None: schema_name_for_count = "ownership"
                   elif data_content.get("key_risks_identified") is not None: schema_name_for_count = "key_risks"
                   elif data_content.get("actions_found") is not None: schema_name_for_count = "regulatory_actions" # Assuming this key
                   else: schema_name_for_count = "unknown_content"

              elif schema_name_for_count == "unknown" and isinstance(item, dict): # If item is raw content dict
                   if item.get("ownership_relationships") is not None: schema_name_for_count = "ownership"
                   elif item.get("key_risks_identified") is not None: schema_name_for_count = "key_risks"
                   elif item.get("actions_found") is not None: schema_name_for_count = "regulatory_actions"
                   else: schema_name_for_count = "unknown_raw"


              schema_counts[schema_name_for_count] = schema_counts.get(schema_name_for_count, 0) + 1
         structured_summary = f"- Linkup Structured Data Found: {len(structured_raw_data_list)} items across {len(schema_counts)} schemas."
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

        # Use the async text helper for summary generation
        raw_content = await _call_llm_and_get_text(
             prompt, llm_provider, llm_model, function_name, max_tokens=350 # Limit tokens for summary
        )

        cleaned_summary = ""
        if raw_content:
            cleaned_content = raw_content.strip()
            cleaned_content = re.sub(r'^["\']|["\']$', '', cleaned_content)
            cleaned_content = re.sub(r'^```.*?\n', '', cleaned_content, flags=re.DOTALL)
            cleaned_content = re.sub(r'\n```$', '', cleaned_content)
            cleaned_summary = cleaned_content.strip()

            # FIX: Use the refined conversational and short checks from translate_text
            conversational_only_phrases = ["sorry", "apologize", "cannot", "unable", "provide", "based on the text", "hello", "hi", "greetings", "summary"] # Added summary
            is_pure_filler = cleaned_content.lower() in conversational_only_phrases
            # Check if the response contains any words *besides* potential fillers. If not, it's likely conversational/empty.
            words_in_response = [word.strip() for word in cleaned_content.split() if word.strip()]
            is_only_fillers = len(words_in_response) < 5 and any(re.search(r'\b' + re.escape(word) + r'\b', cleaned_content.lower()) for word in conversational_only_phrases) # Very few words AND contain fillers

            is_too_short = len(cleaned_summary) < 50 # Check minimum length for a summary


            if not cleaned_summary or is_pure_filler or is_too_short or is_only_fillers:
                print(f"Warning: Summary short/empty/apologetic: '{cleaned_summary[:100]}...'.");
                return f"Could not generate a meaningful summary based on the extracted data."
        else:
             return f"Could not generate a meaningful summary based on the extracted data."


        return cleaned_summary

    except Exception as e:
        print(f"ERROR async summary generation: {e}")
        traceback.print_exc()
        return f"Could not generate summary due to error: {type(e).__name__}"

if __name__ == "__main__":
    # Main execution block needs to run the async orchestrator
    async def main_test_run():
        print("\n--- Running Local Orchestrator Tests ---")
        print("NOTE: Requires LLM API keys and search API keys in .env.")
        print("Ensure Neo4j is running if KG update is enabled.")
        print("Ensure Google Sheets is configured if saving is enabled.")

        test_query = "Corporate tax evasion cases in China 2023"
        test_country = "cn"

        test_llm_provider = "openai"
        # Ensure config is loaded for default model
        if config:
            test_llm_model = config.DEFAULT_OPENAI_MODEL if hasattr(config, 'DEFAULT_OPENAI_MODEL') else "gpt-4o-mini"
        else:
            test_llm_model = "gpt-4o-mini"


        print(f"\nRunning analysis for query: '{test_query}' in country: '{test_country}'")

        try:
            # Await the async run_analysis function
            test_run_results = await run_analysis(
                initial_query=test_query,
                llm_provider=test_llm_provider,
                llm_model=test_llm_model,
                specific_country_code=test_country,
                max_global_results=20,
                max_specific_results=20
            )

            print("\n--- Test Run Results ---")
            # Print results nicely, but avoid printing huge lists directly
            printable_results = test_run_results.copy()
            # FIX: These counts should now match the filtered data in final_extracted_data
            # Use the counts already present in the results dict from Step 5Prep/main_api
            # if 'linkup_structured_data' in printable_results: # Removed raw list from UI response
            #      printable_results['linkup_structured_data_count'] = len(printable_results['linkup_structured_data'])
            #      del printable_results['linkup_structured_data']
            if 'wayback_results' in printable_results:
                 # The UI receives the limited list, so len is correct
                 printable_results['wayback_results_count'] = len(printable_results['wayback_results'])
                 # Optionally truncate or remove details if the list is very long
                 # printable_results['wayback_results_sample'] = printable_results['wayback_results'][:3]
                 # del printable_results['wayback_results'] # Kept limited list in UI response
            # FIX: final_extracted_data now contains filtered data in results dict sent to UI
            if 'final_extracted_data' in printable_results:
                 # Summarize counts from the filtered lists
                 printable_results['final_extracted_data_counts'] = {
                     k: len(v) for k, v in printable_results['final_extracted_data'].items()
                 }
                 # Optionally remove the full lists if they are large (UI now expects counts + exposures list)
                 del printable_results['final_extracted_data']
            # FIX: high_risk_exposures now contains the filtered list in results dict sent to UI
            if 'high_risk_exposures' in printable_results:
                 # UI receives the list, so len is correct
                 printable_results['high_risk_exposures_count'] = len(printable_results['high_risk_exposures'])
                 # The UI needs the list for the table, so don't delete the original list
                 # del printable_results['high_risk_exposures'] # Removed deletion


            # Clean up steps data to show counts rather than full extracted_data lists
            if 'steps' in printable_results:
                 for step in printable_results['steps']:
                      if isinstance(step, dict) and 'extracted_data' in step:
                           # These counts should already be in 'extracted_data_counts' added in Step 5Prep
                           if isinstance(step.get('extracted_data'), dict): # Check if it's a dict before accessing keys
                                step['extracted_data_counts'] = {k: len(v) for k,v in step['extracted_data'].items()}
                           del step['extracted_data']


            print(json.dumps(printable_results, indent=2))

        except Exception as e:
            print(f"\n--- Test Run Exception ---")
            print(f"An exception occurred during the test run: {type(e).__name__}: {e}")
            traceback.print_exc()

        print("\n--- Local Orchestrator Tests Complete ---")

    # Run the async main test function
    try:
        asyncio.run(main_test_run())
    except KeyboardInterrupt:
        print("\nOrchestrator test run interrupted.")
    except Exception as e:
        print(f"\n--- Critical Orchestrator Test Failure ---")
        print(f"An unhandled exception occurred during the main async test run: {type(e).__name__}: {e}")
        traceback.print_exc()