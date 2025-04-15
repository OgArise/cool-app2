# nlp_processor.py

import openai
import json
import config  # To get API key
from typing import List, Dict, Any

# Configure OpenAI client (ensure v1.0+ syntax)
# Needs OPENAI_API_KEY in your .env file
if config.OPENAI_API_KEY:
    try:
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        print("OpenAI client initialized successfully.")
    except Exception as e:
        client = None
        print(f"ERROR: Failed to initialize OpenAI client: {e}")
else:
    client = None
    print("Warning: OpenAI client not initialized due to missing API key in .env")


# --- Keyword Generation (Focus on English Keywords for Target Context) ---
def translate_keywords_for_context(original_query: str, target_context: str) -> List[str]:
    """
    Generates relevant ENGLISH keywords for a given context, based on an initial query.
    """
    if not client:
        print("Skipping keyword generation: OpenAI client not available.")
        return [original_query]  # Fallback if no API key

    prompt = f"""You are an expert market analyst generating ENGLISH search keywords.
Given an initial query and a target search context (like a specific region or search engine), generate a concise list of 3-5 relevant ENGLISH search keywords suitable for finding information within that specific context.

Initial Query: {original_query}
Target Search Context: {target_context}

Output ONLY the comma-separated list of generated ENGLISH keywords, with no other explanation or preamble."""

    print("\n--- Sending Request to OpenAI for Keyword Generation ---")
    print(f"Prompt:\n{prompt}")  # Log the prompt being sent

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Or gpt-4-turbo, gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150  # Increased slightly for keyword lists
        )
        raw_content = response.choices[0].message.content.strip()
        print(f"\nRaw OpenAI Response (Keywords):\n{raw_content}")  # Log the raw response

        # Basic split, might need more robust parsing
        keywords = [kw.strip() for kw in raw_content.split(',') if kw.strip()]
        print(f"Parsed Keywords: {keywords}")
        return keywords if keywords else [original_query]  # Fallback if parsing fails or no keywords generated
    except Exception as e:
        print(f"ERROR during keyword generation: {e}")
        return [original_query]  # Fallback on error


# --- Entity / Risk Extraction (Output in English) ---
# Define the desired JSON structure for the LLM
extraction_schema = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "description": "List of identified entities (companies, people, locations, organizations). Provide names in English.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the entity in English."},
                    "type": {
                        "type": "string",
                        "enum": ["COMPANY", "PERSON", "LOCATION", "ORGANIZATION", "REGULATION", "OTHER"],
                        "description": "Type of the entity."
                    },
                    "mentions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs where this entity was significantly mentioned."
                    }
                },
                "required": ["name", "type"]
            }
        },
        "risks": {
            "type": "array",
            "description": "List of potential risks identified (supply chain, compliance, financial, regulatory, etc.). Describe risks in English.",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Description of the potential risk in English."},
                    "severity": {
                        "type": "string",
                        "enum": ["LOW", "MEDIUM", "HIGH"],
                        "description": "Estimated severity."
                    },
                    "related_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names (in English) of entities primarily associated with this risk."
                    },
                    "source_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs providing evidence for this risk."
                    }
                },
                "required": ["description"]
            }
        },
        "relationships": {
            "type": "array",
            "description": "Identified relationships between entities (e.g., supplier, regulator). Use English names and relationship types.",
            "items": {
                "type": "object",
                "properties": {
                    "entity1": {"type": "string", "description": "Name (in English) of the first entity."},
                    "relationship_type": {
                        "type": "string",
                        "description": "Type of relationship (e.g., SUPPLIER_OF, REGULATED_BY, PARTNER_WITH, LOCATED_IN)."
                    },
                    "entity2": {"type": "string", "description": "Name (in English) of the second entity."},
                    "context_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs providing context for this relationship."
                    }
                },
                "required": ["entity1", "relationship_type", "entity2"]
            }
        }
    },
    "required": ["entities", "risks", "relationships"]
}


def extract_data_from_results(search_results: List[Dict[str, Any]], extraction_context: str) -> Dict[str, List]:
    """
    Extracts entities, risks, and relationships from search result snippets using an LLM.
    Outputs all text fields (names, descriptions) in English.
    Uses JSON mode and instructs the LLM to follow the schema via the prompt.
    """
    if not client:
        print("Skipping data extraction: OpenAI client not available.")
        return {"entities": [], "risks": [], "relationships": []}  # Empty if no key
    if not search_results:
        print("Skipping data extraction: No search results provided.")
        return {"entities": [], "risks": [], "relationships": []}

    # Prepare context from snippets (limit size)
    context_text = f"Extraction Context: {extraction_context}\n\nRelevant Information (from search results):\n"
    max_chars = 8000  # Adjust based on model context window and cost tolerance
    char_count = len(context_text)

    print(f"\nPreparing context for extraction (Max Chars: {max_chars})...")
    added_snippets = 0
    for result in search_results:
        snippet = result.get('snippet', '')
        title = result.get('title', '')
        url = result.get('url', '')
        # Basic check to ensure snippet is not empty or just whitespace
        if snippet and not snippet.isspace():
            entry = f"\n---\nTitle: {title}\nURL: {url}\nSnippet: {snippet}\n"
            if char_count + len(entry) <= max_chars:
                context_text += entry
                char_count += len(entry)
                added_snippets += 1
            else:
                print(f"Stopping context preparation early due to character limit ({char_count} chars).")
                break  # Stop adding snippets
        else:
            print(f"Skipping result with empty snippet: {title} ({url})")

    # If no snippets were added (e.g., all results had empty snippets), don't call LLM
    if added_snippets == 0:
        print("No valid snippets found to prepare context for LLM extraction.")
        return {"entities": [], "risks": [], "relationships": []}

    print(f"Prepared context using {added_snippets} snippets.")

    # Include the schema description within the main prompt
    # Convert the schema dictionary to a JSON string to embed it
    schema_string_for_prompt = json.dumps(extraction_schema, indent=2)

    # Define the prompt for the LLM, explicitly requesting English
    prompt = f"""Based ONLY on the provided text snippets below, extract key entities, potential risks, and relationships relevant to the extraction context. IMPORTANT: Provide all names, types, and descriptions in ENGLISH. Your response MUST be a single valid JSON object that strictly adheres to the following JSON schema:
```json
{schema_string_for_prompt}
If no relevant information for a category (entities, risks, relationships) is found in the text, return an empty array for that category (e.g., "entities": []), but ensure the overall response is still a valid JSON object matching the schema structure.
Begin analysis of the following text:
{context_text}"""

    print("\n--- Sending Request to OpenAI for Data Extraction ---")
    # Avoid printing potentially very long context to logs unless necessary for deep debug
    print(f"Context Text Sent (first 500 chars):\n{context_text[:500]}...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using a capable model is important for schema adherence
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},  # Use JSON mode
            temperature=0.1  # Lower temperature for better schema adherence
        )

        raw_content = response.choices[0].message.content
        print(f"\nRaw OpenAI Response (Extraction):\n{raw_content}")  # Log the raw response

        # Attempt to parse the JSON string from the response
        extracted_json = json.loads(raw_content)

        # Basic validation - ensure keys exist, default to empty lists if not
        # Also validate that the values are lists as expected by the schema
        validated_data = {
            "entities": extracted_json.get("entities", []) if isinstance(extracted_json.get("entities"), list) else [],
            "risks": extracted_json.get("risks", []) if isinstance(extracted_json.get("risks"), list) else [],
            "relationships": extracted_json.get("relationships", []) if isinstance(extracted_json.get("relationships"), list) else []
        }

        # Optional: Add more robust validation here if needed (e.g., check item structure within arrays)
        print(f"\nParsed Extracted Data:\n{json.dumps(validated_data, indent=2)}")
        return validated_data
    except json.JSONDecodeError as json_e:
        print(f"ERROR: Failed to decode JSON from OpenAI response: {json_e}")
        print(f"Raw content was: {raw_content}")
        return {"entities": [], "risks": [], "relationships": []}  # Return empty on JSON error
    except Exception as e:
        print(f"ERROR during data extraction API call: {e}")

        # Check if it's an OpenAI API error specifically
        if isinstance(e, openai.APIError):
            print(f"OpenAI API Error Details: Status Code={e.status_code}, Type={e.type}, Message={e.message}")
        return {"entities": [], "risks": [], "relationships": []}  # Return empty on other errors


# --- Local Testing Block ---
# This code only runs if you execute this file directly (e.g., python nlp_processor.py)
if __name__ == "__main__":
    print("\n--- Running Local NLP Processor Tests ---")

    # --- Test Keyword Translation ---
    print("\nTesting Keyword Translation...")
    sample_query_kw = "supply chain compliance issues 2023"
    sample_context_kw = "Baidu search in China for specific company supply chain info"
    translated_keywords_test = translate_keywords_for_context(sample_query_kw, sample_context_kw)
    print(f"\nTranslated Keywords Result: {translated_keywords_test}")

    # --- Test Data Extraction ---
    print("\nTesting Data Extraction...")

    # Create sample search results (use English snippets for easier debugging)
    sample_results_test = [
        {
            "title": "New Supply Chain Regulations Impact Global Trade - GovReport",
            "url": "https://example.gov/report-2023-supply-chain",
            "snippet": "The Regulatory Body C announced new compliance measures affecting manufacturers like Corporation A, especially those operating in Manufacturing Hub B. Financial penalties may apply from 2024.",
            "source": "google",
            "published_date": "2023-11-01"
        },
        {
            "title": "Corporation A Faces Scrutiny Over Supplier Ethics - Finance News",
            "url": "https://example.finance/corp-a-ethics",
            "snippet": "Concerns are rising about Corporation A's sourcing from suppliers in Manufacturing Hub B, potentially violating international labor standards. This poses a significant reputational risk.",
            "source": "google",
            "published_date": "2023-10-15"
        },
        {
            "title": "Baidu Search Results Snippet (Example)",
            "url": "https://example.baidu.com/search-result-3",
            "snippet": "Analysis shows shifting trade patterns in Manufacturing Hub B. Regulatory Body C is monitoring closely. Investors are watching Corporation A.",  # Simulating a different result
            "source": "baidu",
            "published_date": "2023-12-01"
        },
        {
            "title": "Empty Snippet Example",  # Test case for empty snippet handling
            "url": "https://example.com/empty",
            "snippet": " ",
            "source": "google",
            "published_date": "2023-09-01"
        }
    ]
    sample_extraction_context_test = "supply chain risks and regulations involving Corporation A and Manufacturing Hub B"
    extracted_data_test = extract_data_from_results(sample_results_test, sample_extraction_context_test)
    print("\nFinal Extracted Data Result:")
    print(json.dumps(extracted_data_test, indent=2))

    print("\n--- Local Tests Complete ---")