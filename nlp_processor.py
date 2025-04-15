import openai
import json
import config
from typing import List, Dict, Any

# Configure OpenAI client (ensure v1.0+ syntax)
if config.OPENAI_API_KEY:
    client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
else:
    client = None
    print("Warning: OpenAI client not initialized due to missing API key.")

# --- Keyword Translation ---
def translate_keywords_for_context(original_query: str, target_context: str) -> List[str]:
    if not client: return [original_query] # Fallback if no API key

    prompt = f"""You are an expert market analyst specializing in keyword generation for different search engines and regions. Your task is to take an input query and a target context, and generate a concise list of 3-5 highly relevant search keywords suitable for that specific context.

Input Query: {original_query}
Target Context: {target_context}

Output ONLY the comma-separated list of generated keywords, with no other explanation or preamble."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Or gpt-4-turbo, gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=100
        )
        keywords_str = response.choices[0].message.content.strip()
        # Basic split, might need more robust parsing
        keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
        return keywords if keywords else [original_query] # Fallback if parsing fails
    except Exception as e:
        print(f"Error translating keywords: {e}")
        return [original_query] # Fallback on error

# --- Entity / Risk Extraction ---
# Define the desired JSON structure for the LLM
extraction_schema = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array", "description": "List of identified entities (companies, people, locations, organizations).",
            "items": {
                "type": "object",
                "properties": { "name": {"type": "string"}, "type": {"type": "string", "enum": ["COMPANY", "PERSON", "LOCATION", "ORGANIZATION", "REGULATION", "OTHER"]}, "mentions": {"type": "array", "items": {"type": "string"}}},
                "required": ["name", "type"]
            }
        },
        "risks": {
            "type": "array", "description": "List of potential risks identified (supply chain, compliance, financial, regulatory, etc.).",
            "items": {
                "type": "object",
                "properties": { "description": {"type": "string"}, "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]}, "related_entities": {"type": "array", "items": {"type": "string"}}, "source_urls": {"type": "array", "items": {"type": "string"}}},
                "required": ["description"]
            }
        },
        "relationships": {
            "type": "array", "description": "Identified relationships between entities (e.g., supplier, regulator).",
            "items": {
                "type": "object",
                "properties": { "entity1": {"type": "string"}, "relationship_type": {"type": "string"}, "entity2": {"type": "string"}, "context_urls": {"type": "array", "items": {"type": "string"}}},
                "required": ["entity1", "relationship_type", "entity2"]
            }
        }
    },
    "required": ["entities", "risks", "relationships"]
}

def extract_data_from_results(search_results: List[Dict[str, Any]], extraction_context: str) -> Dict[str, List]:
    if not client: return {"entities": [], "risks": [], "relationships": []} # Empty if no key
    if not search_results: return {"entities": [], "risks": [], "relationships": []}

    # Prepare context from snippets (limit size)
    context_text = f"Extraction Context: {extraction_context}\n\nRelevant Information:\n"
    max_chars = 8000 # Adjust based on model context window
    char_count = len(context_text)

    for result in search_results:
        snippet = result.get('snippet', '')
        title = result.get('title', '')
        url = result.get('url', '')
        entry = f"\n---\nTitle: {title}\nURL: {url}\nSnippet: {snippet}\n"
        if char_count + len(entry) <= max_chars:
            context_text += entry
            char_count += len(entry)
        else:
            break # Stop adding snippets

    prompt = f"""Based ONLY on the provided text snippets below, extract key entities, potential risks, and relationships relevant to the extraction context. Adhere strictly to the provided JSON schema for your output. If no relevant information for a category (entities, risks, relationships) is found in the text, return an empty array for that category.

{context_text}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Or gpt-4-turbo with response_format
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object", "schema": extraction_schema}, # Use JSON mode
            temperature=0.2
        )
        # Newer OpenAI libraries might put JSON directly in content or a specific attribute
        # Check the structure of 'response.choices[0].message'
        content = response.choices[0].message.content
        # content = response.choices[0].message.function_call.arguments # If using older function calling
        extracted_json = json.loads(content) # Parse the JSON string from the response

        # Basic validation - ensure keys exist
        return {
            "entities": extracted_json.get("entities", []),
            "risks": extracted_json.get("risks", []),
            "relationships": extracted_json.get("relationships", [])
        }

    except Exception as e:
        print(f"Error extracting data: {e}")
        return {"entities": [], "risks": [], "relationships": []} # Return empty on error