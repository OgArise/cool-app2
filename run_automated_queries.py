# run_automated_queries.py

from typing import List, Dict, Any, Optional
import time
import json
import requests
from datetime import datetime
import os
import traceback
import sys

# Google Sheets Imports
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    import os
    google_sheets_library_available = True
except ImportError:
    print("ERROR: Google API libraries not installed (`google-api-python-client google-auth-httplib2 google-auth-oauthlib`). Automation script cannot run.")
    google_sheets_library_available = False

# Import config ONLY for variables DEFINED within it (like keys, GSheet ID, etc.)
try:
    import config
except ImportError:
     print("ERROR: config.py not found. Make sure it exists and defines necessary variables.")
     config = None

# --- Google Sheets Setup ---
# Initialize global gsheet_service variable *before* its usage in functions
gsheet_service = None

QUERIES_SHEET_NAME = "Queries"
STATUS_COLUMN = "Status"
QUERY_COLUMN = "Query"
LLM_PROVIDER_COLUMN = "LLM Provider"
LLM_MODEL_COLUMN = "LLM Model"
RESULT_SUMMARY_COLUMN = "Result Summary"

# --- Configuration ---
BACKEND_API_URL = os.getenv(
    "BACKEND_API_URL_SCHEDULER",
    os.getenv(
        "BACKEND_API_URL",
        "http://localhost:8000"
    )
)
ANALYZE_ENDPOINT = f"{BACKEND_API_URL}/analyze"
print(f"Automation targeting Backend API: {BACKEND_API_URL}")

# Increase timeout for the API request to accommodate longer analysis runs
# Increased to 2000 seconds as requested.
API_REQUEST_TIMEOUT = 2000

# Initialize Google Sheets related variables to None or False
SHEET_ID = None
GCP_SERVICE_ACCOUNT_JSON_STR = None
SERVICE_ACCOUNT_INFO = None
google_sheets_configured = False
google_sheets_available = False

# Process config if it was imported successfully
if config:
    SHEET_ID = config.GOOGLE_SHEET_ID
    GCP_SERVICE_ACCOUNT_JSON_STR = config.GCP_SERVICE_ACCOUNT_JSON_STR

    # Check if essential config values are present
    if SHEET_ID and isinstance(SHEET_ID, str) and GCP_SERVICE_ACCOUNT_JSON_STR and isinstance(GCP_SERVICE_ACCOUNT_JSON_STR, str):
         try:
             SERVICE_ACCOUNT_INFO = json.loads(GCP_SERVICE_ACCOUNT_JSON_STR)
             if isinstance(SERVICE_ACCOUNT_INFO, dict):
                 print("Google Sheets configuration seems complete.")
                 google_sheets_configured = True
             else:
                  print("Warning: GCP_SERVICE_ACCOUNT_JSON_STR does not contain a valid JSON object for service account info.")
                  SERVICE_ACCOUNT_INFO = None
         except Exception as e:
              print(f"ERROR parsing GCP JSON from config: {e}")
              SERVICE_ACCOUNT_INFO = None
    elif config.GOOGLE_SHEET_ID:
         print("Warning: GCP_SERVICE_ACCOUNT_JSON_STR environment variable not set or empty, but GOOGLE_SHEET_ID is. Google Sheets saving will be disabled.")

    DEFAULT_PROVIDER = "openai"
    DEFAULT_MODEL = config.DEFAULT_OPENAI_MODEL if config and hasattr(config, 'DEFAULT_OPENAI_MODEL') else "gpt-4o-mini"
    FALLBACK_LLM_CONFIG = (DEFAULT_PROVIDER, DEFAULT_MODEL)

    if config and (getattr(config, 'OPENAI_API_KEY', None) or getattr(config, 'OPENROUTER_API_KEY', None) or getattr(config, 'GOOGLE_AI_API_KEY', None)):
         if DEFAULT_PROVIDER == "openai" and not getattr(config, 'OPENAI_API_KEY', None):
              if getattr(config, 'OPENROUTER_API_KEY', None) and hasattr(config, 'DEFAULT_OPENROUTER_MODEL'): FALLBACK_LLM_CONFIG = ("openrouter", config.DEFAULT_OPENROUTER_MODEL)
              elif getattr(config, 'GOOGLE_AI_API_KEY', None) and hasattr(config, 'DEFAULT_GOOGLE_AI_MODEL'): FALLBACK_LLM_CONFIG = ("google_ai", config.DEFAULT_GOOGLE_AI_MODEL)

    if FALLBACK_LLM_CONFIG == (None, None):
         print("ERROR: No valid default LLM configuration found. Automation script will not be able to run queries if LLM columns in sheet are empty.")

# The overall availability depends on library available AND configured credentials are valid
google_sheets_available = google_sheets_library_available and google_sheets_configured

if not google_sheets_available:
    print("Warning: Google Sheets saving will be disabled due to missing libraries or configuration.")

DELAY_BETWEEN_QUERIES = 30 # Delay after processing a single query
DELAY_AFTER_BATCH = 60   # Delay after processing a batch of 10 queries
DELAY_IF_NO_QUERIES = 600 # Delay when no pending queries are found

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

def _get_gsheet_service_automation():
    """Authenticates for the automation script. Checks for library availability."""
    global gsheet_service
    if not google_sheets_library_available: return None
    if not google_sheets_configured: return None

    if gsheet_service is None:
        if SERVICE_ACCOUNT_INFO is None: return None
        try:
            creds = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
            gsheet_service = build('sheets', 'v4', credentials=creds, cache_discovery=False)
            print("Google Sheets service authenticated successfully for automation.")
        except Exception as e:
            print(f"ERROR: Failed to authenticate Google Sheets service: {type(e).__name__}: {e}")
            traceback.print_exc()
            gsheet_service = None
    return gsheet_service

def _get_pending_queries(service) -> List[Dict]:
    """Reads the Queries sheet and returns rows marked 'Pending'. Checks for service."""
    pending_queries = []
    if service is None: return pending_queries

    try:
        header_range = f"{QUERIES_SHEET_NAME}!1:1"
        header_result = service.spreadsheets().values().get(
             spreadsheetId=SHEET_ID, range=header_range
        ).execute()
        header_row = header_result.get('values', [[]])[0]

        if not header_row:
            print("Queries sheet appears empty or has no header row.")
            return pending_queries

        # Ensure header row values are treated as strings before stripping and mapping
        header_map = {str(h).strip(): idx for idx, h in enumerate(header_row) if str(h).strip()}
        # print(f"Debug Header Map: {header_map}") # Debug print


        try:
            query_col_index = header_map[QUERY_COLUMN]
            status_col_index = header_map[STATUS_COLUMN]
            # Use .get() with a default of None for optional columns
            provider_col_index = header_map.get(LLM_PROVIDER_COLUMN)
            model_col_index = header_map.get(LLM_MODEL_COLUMN)
            summary_col_index = header_map.get(RESULT_SUMMARY_COLUMN)

            # Determine the last column index to read data efficiently
            max_col_index = max( [idx for idx in [query_col_index, status_col_index, provider_col_index, model_col_index, summary_col_index] if idx is not None] )
            data_range_letter = chr(ord('A') + max_col_index)
            data_range = f"{QUERIES_SHEET_NAME}!A:{data_range_letter}"
            # print(f"Debug Reading data range: {data_range}") # Debug print

        except KeyError as e:
             print(f"ERROR: Missing required column in header ('{e.args[0]}' not found). Required: '{QUERY_COLUMN}', '{STATUS_COLUMN}'. Headers found: {header_row}")
             return pending_queries
        except Exception as e:
             print(f"ERROR determining data range based on headers: {type(e).__name__}: {e}. Headers: {header_row}")
             traceback.print_exc()
             return pending_queries


        data_result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID, range=data_range).execute()
        rows = data_result.get('values', [])

        if len(rows) < 2:
            print("No data rows found in Queries sheet beyond header.")
            return pending_queries

        # Process data rows (skip header row)
        for i, row in enumerate(rows[1:], start=2): # i is the 1-based row index in the sheet
            # Ensure row values are treated as strings and handle rows shorter than max_col_index
            get_col_value = lambda index: str(row[index]).strip() if index is not None and index < len(row) else None

            query = get_col_value(query_col_index)
            status = get_col_value(status_col_index)
            llm_provider_sheet = get_col_value(provider_col_index)
            llm_model_sheet = get_col_value(model_col_index)

            # Check for pending status and a valid query
            if status and status.lower() == 'pending' and query and query.strip():
                 query_info = {
                     'row_index': i, # Store 1-based sheet row index
                     'query': query.strip(), # Store stripped query
                     'status_col_letter': chr(ord('A') + status_col_index),
                     'summary_col_letter': chr(ord('A') + summary_col_index) if summary_col_index is not None else None,
                     'llm_provider_sheet': llm_provider_sheet,
                     'llm_model_sheet': llm_model_sheet,
                 }
                 pending_queries.append(query_info)
                 print(f"Found pending query '{query_info['query'][:50]}...' at row {i}")
            # Optional: Log rows with invalid status or missing query if needed for debugging sheet input
            # elif query and query.strip():
            #      print(f"Skipping row {i}: Status is '{status}', not 'Pending'. Query: '{query.strip()[:50]}...'")
            # elif status and status.lower() == 'pending':
            #      print(f"Skipping row {i}: Status is 'Pending' but query is empty or missing.")


    except Exception as e:
        print(f"ERROR reading pending queries from Google Sheet: {type(e).__name__}: {e}")
        traceback.print_exc()

    print(f"Finished checking sheet. Found {len(pending_queries)} pending queries.")
    return pending_queries

def _update_query_status(service, row_index: int, status_col_letter: str, new_status: str, summary_col_letter: Optional[str], result_summary: Optional[str] = None):
    """Updates the status (and optionally adds a result summary) for a specific row. Checks for service."""
    if service is None:
        print(f"GSheet service unavailable for status update. Cannot update row {row_index} with status '{new_status}'.")
        return
    try:
        range_to_update_status = f"{QUERIES_SHEET_NAME}!{status_col_letter}{row_index}"
        # print(f"Debug Updating status cell: {range_to_update_status} with '{new_status}'") # Debug print

        update_cells = [{ 'range': range_to_update_status, 'values': [[new_status]] }]

        if summary_col_letter and result_summary is not None:
             range_to_update_summary = f"{QUERIES_SHEET_NAME}!{summary_col_letter}{row_index}"
             # print(f"Debug Updating summary cell: {range_to_update_summary} with '{result_summary[:50]}...'") # Debug print
             # Ensure summary doesn't exceed cell limits (Google Sheets cell max is 50,000 chars, but keeping shorter is good)
             summary_to_save = str(result_summary)[:49999] # Using a high limit just in case
             update_cells.append({ 'range': range_to_update_summary, 'values': [[summary_to_save]] })
        elif summary_col_letter and result_summary is None:
             # If summary_col_letter exists but summary is None, clear the summary cell
             range_to_update_summary = f"{QUERIES_SHEET_NAME}!{summary_col_letter}{row_index}"
             update_cells.append({ 'range': range_to_update_summary, 'values': [['']] })


        body = {'value_input_option': 'USER_ENTERED', 'data': update_cells}

        request = service.spreadsheets().values().batchUpdate(
            spreadsheetId=SHEET_ID,
            body=body
        )
        response = request.execute()
        # print(f"Debug Batch update response: {response}") # Debug print

    except Exception as e:
        print(f"ERROR updating status/summary for row {row_index} in Google Sheet: {type(e).__name__}: {e}")
        traceback.print_exc()

def run_automation():
    print("Starting AI Analyst Automation...")

    # Check essential prerequisites early
    if not google_sheets_available:
        print("Exiting: Google Sheets is not available or configured.")
        return

    if FALLBACK_LLM_CONFIG == (None, None):
        print("Exiting: No valid default LLM configuration available from config.py.")
        return

    service = _get_gsheet_service_automation()
    if service is None:
        print("Exiting: Cannot obtain Google Sheets service.")
        return

    run_counter = 0 # Counter for queries processed in the current loop cycle

    while True:
        start_loop_time = time.time()
        print(f"\n[{datetime.now().isoformat()}] Checking for pending queries...")
        pending_queries = _get_pending_queries(service)

        if not pending_queries:
            print(f"No pending queries found. Sleeping for {DELAY_IF_NO_QUERIES} seconds...")
            time.sleep(DELAY_IF_NO_QUERIES)
            run_counter = 0 # Reset counter when no jobs are found
            continue

        print(f"Found {len(pending_queries)} pending queries.")
        run_counter = 0 # Reset counter for the new batch

        # Process pending queries up to a batch limit if desired (e.g., process first 10 found)
        # pending_queries_batch = pending_queries[:10] # Process first 10

        for job in pending_queries: # Iterate through the selected batch
            run_counter += 1 # Increment counter for each job processed in this batch
            print(f"\n[{datetime.now().isoformat()}] Processing Query (Row {job['row_index']}, Job {run_counter}/{len(pending_queries)}): '{job['query'][:50]}...'")

            llm_provider_sheet = job.get('llm_provider_sheet')
            llm_model_sheet = job.get('llm_model_sheet')

            # Basic validation for sheet LLM config strings
            is_provider_valid = isinstance(llm_provider_sheet, str) and llm_provider_sheet.strip() and "|" not in llm_provider_sheet
            is_model_valid = isinstance(llm_model_sheet, str) and llm_model_sheet.strip() and "|" not in llm_model_sheet

            llm_provider_to_send = None
            llm_model_to_send = None

            if is_provider_valid and is_model_valid:
                 # Use sheet values if both are provided and look reasonably valid
                 llm_provider_to_send = llm_provider_sheet.strip()
                 llm_model_to_send = llm_model_sheet.strip()
                 print(f"  Using LLM from sheet: Provider='{llm_provider_to_send}', Model='{llm_model_to_send}'")
            else:
                 # Use fallback if sheet values are missing, empty, or invalid
                 llm_provider_to_send, llm_model_to_send = FALLBACK_LLM_CONFIG
                 # Check if fallback itself is valid
                 if llm_provider_to_send is None or llm_model_to_send is None:
                       print(f"ERROR: Fallback LLM config is also invalid ({FALLBACK_LLM_CONFIG}). Cannot proceed with job.")
                       _update_query_status(service, job['row_index'], job['status_col_letter'], "Error: No Valid LLM Config", job['summary_col_letter'], "Automation failed to get valid LLM config.")
                       time.sleep(1) # Small delay before next job
                       continue # Skip this job and move to the next

                 print(f"  Sheet LLM config invalid/missing/corrupted ('{llm_provider_sheet}', '{llm_model_sheet}'). Using fallback default: Provider='{llm_provider_to_send}', Model='{llm_model_to_send}'")


            # Set status to Processing before API call
            _update_query_status(service, job['row_index'], job['status_col_letter'], "Processing...", job['summary_col_letter'])

            payload = {
                "query": job['query'],
                "global_context": "global financial news and legal filings", # Fixed context for now
                "specific_context": "search for specific company examples and details", # Fixed context for now
                "specific_country": 'cn', # Fixed country for now
                "max_global": 20, # Fixed limits for now
                "max_specific": 20, # Fixed limits for now
                "llm_provider": llm_provider_to_send,
                "llm_model": llm_model_to_send
            }

            # Initialize result summary and status for this job
            result_summary = f"LLM: {llm_provider_to_send} ({llm_model_to_send})"
            new_status = "Error: Unknown" # Default status in case of unexpected failure

            try:
                print(f"Calling backend API: {ANALYZE_ENDPOINT} with Payload LLM: {payload['llm_provider']} / {payload['llm_model']}")
                # Use the increased timeout value
                response = requests.post(ANALYZE_ENDPOINT, json=payload, timeout=API_REQUEST_TIMEOUT)

                if response.status_code == 200:
                    results = response.json()
                    print(f"Backend analysis completed successfully (HTTP 200 OK). Duration: {results.get('run_duration_seconds', 'N/A')}s")

                    # Extract relevant info from results for the summary
                    entities_count = len(results.get("final_extracted_data", {}).get("entities", []))
                    risks_count = len(results.get("final_extracted_data", {}).get("risks", []))
                    relationships_count = len(results.get("final_extracted_data", {}).get("relationships", [])) # Add relationships count
                    exposures_count = len(results.get("high_risk_exposures", []))
                    kg_status = results.get('kg_update_status', '?')
                    duration = results.get('run_duration_seconds', 'N/A')
                    backend_llm_used = results.get('llm_used', 'N/A')
                    backend_error = results.get('error') # Get backend error message

                    # Build the result summary string
                    result_summary = f"LLM: {backend_llm_used} | E:{entities_count}, R:{risks_count}, Rel:{relationships_count}, Exp:{exposures_count}, KG:{kg_status}, Time:{duration}s"

                    if backend_error and backend_error != "None" and backend_error != "": # Check if backend reported an error
                        new_status = f"Error: {backend_error}"
                        # Append backend error message to summary if it exists
                        error_detail_str = backend_error[:100] + '...' if len(backend_error) > 100 else backend_error
                        result_summary += f" | Backend Error: {error_detail_str}"
                    else:
                        # Status is Complete only if backend reported no error
                        new_status = "Complete"

                    # Optionally include the analysis summary text in the sheet summary column
                    analysis_summary_text = results.get("analysis_summary")
                    if analysis_summary_text and isinstance(analysis_summary_text, str):
                         # Append analysis summary to the sheet cell summary
                         # Use a separator like " --- " or "\n" depending on desired sheet formatting
                         # Max sheet cell length is large, but let's take a reasonable chunk
                         summary_text_for_cell = analysis_summary_text[:500] + '...' if len(analysis_summary_text) > 500 else analysis_summary_text
                         result_summary = f"{result_summary} --- Summary: {summary_text_for_cell}"

                else:
                     # Handle non-200 status codes from the API
                     print(f"ERROR: Backend API returned status {response.status_code}.")
                     new_status = f"Error: Backend HTTP {response.status_code}"
                     error_detail = f"HTTP {response.status_code}"
                     try:
                          # Try to get error detail from JSON response
                          error_json = response.json()
                          if 'detail' in error_json:
                               error_detail = json.dumps(error_json['detail'])[:200] + '...' if len(json.dumps(error_json['detail'])) > 200 else json.dumps(error_json['detail'])
                          else:
                               error_detail = response.text[:200] + '...'
                     except json.JSONDecodeError:
                          # If response is not JSON, use raw text
                          error_detail = response.text[:200] + '...'
                     result_summary += f" | Detail: {error_detail}"
                     print(f"Response Text: {response.text}") # Log the full response text for debugging


            except requests.exceptions.Timeout:
                 print(f"ERROR: Timeout calling backend API after {API_REQUEST_TIMEOUT} seconds.")
                 new_status = "Error: Backend Timeout";
                 result_summary += f" | Timeout ({API_REQUEST_TIMEOUT}s)"
            except requests.exceptions.ConnectionError as e:
                 print(f"ERROR: Connection Error calling backend: {e}. Is backend running at {ANALYZE_ENDPOINT}?")
                 new_status = "Error: Backend Connect";
                 result_summary += f" | Connect Error: {type(e).__name__}"
            except requests.exceptions.RequestException as e:
                 print(f"ERROR: Request Exception calling backend: {type(e).__name__}: {e}")
                 new_status = f"Error: Backend Request ({type(e).__name__})";
                 result_summary += f" | Request Failed: {type(e).__name__}"
            except Exception as e:
                 print(f"ERROR: Unexpected error during job processing: {type(e).__name__}: {e}")
                 traceback.print_exc()
                 new_status = f"Error: Script Error ({type(e).__name__})";
                 result_summary += f" | Script Error: {type(e).__name__}"

            # Update the status and summary in the Google Sheet for the current job
            _update_query_status(service, job['row_index'], job['status_col_letter'], new_status, job['summary_col_letter'], result_summary)

            # Implement batch processing delay
            if run_counter % 10 == 0: # Apply batch delay after every 10 queries processed in this batch
                 print(f"Completed {run_counter} queries in this batch. Sleeping for {DELAY_AFTER_BATCH} seconds before next query...")
                 time.sleep(DELAY_AFTER_BATCH)
            else:
                 # Apply delay between individual queries
                 print(f"Sleeping for {DELAY_BETWEEN_QUERIES} seconds before next query...")
                 time.sleep(DELAY_BETWEEN_QUERIES)

        # This point is reached after processing all pending queries found in the current sheet check
        print("Finished processing current batch of pending queries.")
        # The loop will continue, checking the sheet again after the delay_if_no_queries if no new ones are found.

if __name__ == "__main__":
    print("\n--- Running Automated Queries Script ---")
    print("NOTE: This script continuously checks a Google Sheet for 'Pending' queries and runs them via the backend API.")
    print("Ensure API keys are configured in .env and the backend API is running.")

    # Optional: Add a small initial delay before the first sheet check
    # time.sleep(5)

    run_automation()
    print("\n--- Automated Queries Script Finished ---")