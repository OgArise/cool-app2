# streamlit_ui.py

import streamlit as st
import json
from datetime import datetime
import os
# Replaced requests with httpx as needed for async function used with asyncio.run
# import requests
import httpx # Import httpx
import asyncio # Import asyncio
import pandas as pd
import traceback
from typing import Optional, Dict, Any, List # Import List for type hinting

# Import config
try:
    import config
except ImportError:
    config = None
    st.warning("config.py not found. Using default values.")

# Import pycountry
try:
    import pycountry
    pycountry_available = True
except ImportError:
    print("Warning: 'pycountry' not installed. Using basic country list.")
    pycountry_available = False


DEFAULT_BACKEND_URL = "http://localhost:8000"
BACKEND_API_URL = os.getenv("BACKEND_API_URL", DEFAULT_BACKEND_URL)
# Ensure BACKEND_API_URL does not have a trailing slash before appending the endpoint path
CLEANED_BACKEND_API_URL = BACKEND_API_URL.rstrip('/')
ANALYZE_ENDPOINT = f"{CLEANED_BACKEND_API_URL}/analyze"

GOOGLE_SHEET_ID_FROM_CONFIG = getattr(config, 'GOOGLE_SHEET_ID', None) if config else None

# Define the specific GID for the Exposures tab if known
# You can find this GID in the URL when you are viewing the specific tab in Google Sheets:
# https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit#gid=YOUR_EXPOSURES_SHEET_GID
EXPOSURES_SHEET_GID = "1468712289" # Replace with the actual GID of your Exposures tab if different

GOOGLE_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID_FROM_CONFIG}/edit" if GOOGLE_SHEET_ID_FROM_CONFIG else None
GOOGLE_EXPOSURES_SHEET_URL = f"{GOOGLE_SHEET_URL}#gid={EXPOSURES_SHEET_GID}" if GOOGLE_SHEET_URL and EXPOSURES_SHEET_GID else None


LLM_PROVIDERS = { "Google AI": "google_ai", "OpenAI": "openai", "OpenRouter": "openrouter" }
DEFAULT_MODELS = {
    "google_ai": getattr(config, 'DEFAULT_GOOGLE_AI_MODEL', "models/gemini-1.5-flash-latest") if config else "models/gemini-1.5-flash-latest",
    "openai": getattr(config, 'DEFAULT_OPENAI_MODEL', "gpt-4o-mini") if config else "gpt-4o-mini",
    "openrouter": getattr(config, 'DEFAULT_OPENROUTER_MODEL', "qwen/qwen3-235b-a22b:free") if config else "qwen/qwen3-235b-a22b:free"
}

# Determine the default provider key based on the name 'OpenAI'
# Find the index of 'OpenAI' in the list of provider names
try:
    default_provider_index = list(LLM_PROVIDERS.keys()).index("OpenAI")
except ValueError:
    default_provider_index = 0 # Fallback to the first provider if OpenAI is not in the list


def get_country_options():
    options = {"Global": "global"}
    if pycountry_available:
        try:
            countries = sorted([(country.name, country.alpha_2.lower()) for country in pycountry.countries])
            preferred_order = {"Global": "global", "China": "cn", "United States": "us"}
            final_options = {}
            # Add preferred countries first in specified order
            for name, code in preferred_order.items():
                 if name == "Global" or any(c_name == name or c_code == code for c_name, c_code in countries):
                      final_options[name] = code
            # Add all other countries alphabetically
            for name, code in countries:
                 if name not in final_options and code not in final_options.values():
                      final_options[name] = code
            options = final_options
        except Exception as e:
             print(f"Error loading countries from pycountry: {e}. Using basic list.")
             options.update({"China": "cn", "United States": "us", "United Kingdom": "uk", "India": "in", "Germany": "de"})
    else:
         options.update({"China": "cn", "United States": "us", "United Kingdom": "uk", "India": "in", "Germany": "de"})
    return options

COUNTRY_OPTIONS = get_country_options()
COUNTRY_DISPLAY_NAMES = list(COUNTRY_OPTIONS.keys())

def update_contexts():
    """Updates the default context fields based on the initial query input."""
    query = st.session_state.get("initial_query_input", "")
    # Only update if the query has changed AND the contexts haven't been manually edited
    # Check if contexts are still the *original* default ones before overriding
    current_global = st.session_state.get("global_context_input", "")
    current_specific = st.session_state.get("specific_context_input", "")
    original_default_global = "Global financial news and legal filings for compliance issues"
    original_default_specific = "Search for specific company examples and regulatory actions"

    if query and query != st.session_state.get("_last_updated_query", ""):
         # Check if contexts are still the *last auto-generated* ones based on the previous query OR the original defaults
         last_auto_global = f"Global financial news and legal filings for '{st.session_state.get('_last_updated_query', '')}'"
         last_auto_specific = f"Search for specific company examples and regulatory actions related to '{st.session_state.get('_last_updated_query', '')}'"

         if current_global in [original_default_global, last_auto_global] and \
            current_specific in [original_default_specific, last_auto_specific]:
            st.session_state.global_context_input = f"Global financial news and legal filings for '{query}'"
            st.session_state.specific_context_input = f"Search for specific company examples and regulatory actions related to '{query}'"
            print(f"Updated contexts based on query: '{query}'")

         st.session_state._last_updated_query = query # Always update the last processed query
    elif not query:
        # Reset to default generic contexts if query is cleared
        st.session_state.global_context_input = original_default_global
        st.session_state.specific_context_input = original_default_specific
        st.session_state._last_updated_query = ""
        print("Query cleared, reset contexts to default.")


# Initialize session state variables if they don't exist
if "global_context_input" not in st.session_state: st.session_state.global_context_input = "Global financial news and legal filings for compliance issues"
if "specific_context_input" not in st.session_state: st.session_state.specific_context_input = "Search for specific company examples and regulatory actions"
if 'analysis_status' not in st.session_state: st.session_state.analysis_status = "IDLE"
if 'analysis_payload' not in st.session_state: st.session_state.analysis_payload = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'error_message' not in st.session_state: st.session_state.error_message = None
if '_last_updated_query' not in st.session_state: st.session_state._last_updated_query = ""
# Set initial query default to blank
if "initial_query_input" not in st.session_state: st.session_state.initial_query_input = ""

# Initialize LLM model text inputs for each provider
for provider_key in LLM_PROVIDERS.values():
     session_key_model = f"{provider_key}_model_input"
     if session_key_model not in st.session_state:
          st.session_state[session_key_model] = DEFAULT_MODELS.get(provider_key, "")

# Remove the analysis_task initialization as it's no longer used for polling
# if 'analysis_task' not in st.session_state:
#     st.session_state.analysis_task = None


# Set the page title for the browser tab and the main title on the page
st.set_page_config(page_title="The China Analyst AI Agent", layout="wide") # FIX: Updated page_title
st.title("🕵️ The China Analyst AI Agent")


st.markdown("Enter query, select LLM/Country. API Keys configured on backend.")

# Display backend URL status
if BACKEND_API_URL.startswith("YOUR_"): st.error("Backend API URL needs config. Please set the `BACKEND_API_URL` environment variable.")
elif BACKEND_API_URL == DEFAULT_BACKEND_URL: st.info(f"Targeting local backend API ({BACKEND_API_URL})..")
# Use the cleaned URL for display
else: st.info(f"Targeting Backend API: {CLEANED_BACKEND_API_URL}")

st.sidebar.title("LLM Selection")
# Use LLM_PROVIDERS keys for display, values for internal logic
# Set default index based on finding 'OpenAI'
selected_provider_name = st.sidebar.selectbox("Select LLM Provider", options=list(LLM_PROVIDERS.keys()), index=default_provider_index, key='sidebar_llm_provider_name_select' )
selected_provider_key = LLM_PROVIDERS[selected_provider_name]

# Get the current model value for the selected provider from session state
session_key_model = f"{selected_provider_key}_model_input"
llm_model_value_for_input = st.session_state.get(session_key_model)

# If the value is empty (e.g., first time selecting this provider), populate with default
if not llm_model_value_for_input:
     llm_model_value_for_input = DEFAULT_MODELS.get(selected_provider_key, "")
     st.session_state[session_key_model] = llm_model_value_for_input # Update session state with default

# Text input for the model name, linked to the session state key
llm_model = st.sidebar.text_input(
    f"Model Name for {selected_provider_name}",
    value=llm_model_value_for_input,
    key=session_key_model,
    help=f"e.g., {DEFAULT_MODELS.get(selected_provider_key, 'default')}"
)

st.sidebar.caption("✨ Tip: OpenRouter is free. OpenAI & Google requires paid credits.")

st.subheader("1. Enter Your Initial Query")
# Set initial query default to blank
initial_query_value = st.text_input(
    "Initial Search Query:",
    value=st.session_state.get("initial_query_input", ""), # Default to blank
    key="initial_query_input",
    on_change=update_contexts, # Trigger context update when query changes
    help="Enter the primary query to initiate the analysis."
)
# Ensure contexts are updated on initial load if the default wasn't blank and hasn't been processed
if st.session_state.initial_query_input and not st.session_state.get("_last_updated_query"):
    update_contexts()


st.subheader("2. Configure Search & Run Analysis")
with st.form("analysis_form"):
    st.markdown("**Search Contexts (Auto-updated based on query)**")
    # Link text areas directly to session state keys managed by update_contexts
    global_context = st.text_area( "Global Search Context", value=st.session_state.global_context_input, key="global_context_input", height=100, help="Describes the focus area for broad searches." )
    specific_context = st.text_area( "Specific Search Context", value=st.session_state.specific_context_input, key="specific_context_input", height=100, help="Describes the focus area for country-specific searches." )

    st.markdown("**Other Parameters**")
    col1, col2 = st.columns(2)
    with col1:
        # Find the index for the default country ('cn') for the selectbox
        try:
            default_country_index_select = COUNTRY_DISPLAY_NAMES.index("China") if "China" in COUNTRY_DISPLAY_NAMES else 0
        except ValueError:
            default_country_index_select = 0 # Fallback if China is not in the list for some reason


        selected_country_name_widget_value = st.selectbox(
            "Specific Country Search Target",
            options=COUNTRY_DISPLAY_NAMES,
            index=default_country_index_select,
            key='country_select',
            help="Select 'Global' or a specific country for the targeted search."
        )
    with col2:
        max_global_results = st.number_input("Max Global Results Per Search Engine", min_value=1, max_value=50, value=20, key='max_global_input', help="Maximum number of search results requested from *each* enabled global search engine.")
        max_specific_results = st.number_input("Max Specific Results Per Search Engine", min_value=1, max_value=50, value=20, key='max_specific_input', help="Maximum number of search results requested from *each* enabled country-specific search engine.")

    submitted = st.form_submit_button("Run Analysis")

# --- Asynchronous API Call Function ---
# This function will be run in the background by asyncio
async def run_analysis_async(payload):
    """Makes the asynchronous API call to the backend."""
    try:
        # Use httpx.AsyncClient for asynchronous requests
        # Use a timeout that is reasonable for the API to respond
        # It should be less than Streamlit's overall script timeout if possible
        api_timeout_seconds = 1800 # 30 minutes (should match or be less than backend processing timeout)
        async with httpx.AsyncClient(timeout=api_timeout_seconds) as client:
            print(f"Streamlit making async API call to {ANALYZE_ENDPOINT}")
            response = await client.post(ANALYZE_ENDPOINT, json=payload)

        # Check for specific HTTP status codes returned by the backend
        if response.status_code == 500:
             # If backend returns 500, it might contain error details in the body
             try:
                 error_details = response.json()
                 error_msg = error_details.get("detail", "Backend returned 500 Internal Server Error")
                 # Check if the backend included partial/summary results in the error body
                 results_summary_data = error_details.get("results_summary")
                 if results_summary_data:
                      # Return status COMPLETE_WITH_ERROR and include the summary results
                      return {"status": "COMPLETE_WITH_ERROR", "results": results_summary_data, "error_message": error_msg}
                 else:
                      # No summary results included, just return the error status
                      return {"status": "ERROR", "results": None, "error_message": error_msg}
             except json.JSONDecodeError:
                 # If the 500 response body is not JSON
                 error_msg = f"Backend returned 500 Internal Server Error. Response body not JSON: {response.text[:200]}..."
                 print(error_msg)
                 return {"status": "ERROR", "results": None, "error_message": error_msg}
        else:
             # For any other non-200 status codes, raise the exception
             response.raise_for_status() # Raise an exception for bad status codes (4xx or other 5xx)

        # If status is 200 OK, parse the JSON response (which is now the smaller subset)
        results_data = response.json()
        print(f"Streamlit received backend analysis response (subset). Duration: {results_data.get('run_duration_seconds', 'N/A')}s")

        # The backend now returns a boolean/string error field, not raising HTTPException for backend logic errors.
        # The status is 'COMPLETE' if we reach here without an HTTPStatusError.
        # Check the `backend_error` field in the returned data.
        if results_data.get("backend_error") and results_data["backend_error"] != "None" and results_data["backend_error"] != "":
            print(f"Streamlit received results with backend_error field: {results_data['backend_error']}")
            return {"status": "COMPLETE_WITH_ERROR", "results": results_data, "error_message": f"Analysis completed with backend error: {results_data['backend_error']}"}
        else:
            return {"status": "COMPLETE", "results": results_data, "error_message": None}


    except httpx.TimeoutException:
        error_msg = f"Request to backend API timed out after {api_timeout_seconds} seconds. The analysis might still be running on the backend. Check Google Sheet for results."
        print(error_msg)
        return {"status": "ERROR", "results": None, "error_message": error_msg}
    except httpx.RequestError as e:
        error_msg = f"An HTTP request error occurred calling backend API at {ANALYZE_ENDPOINT}: {e}"
        print(error_msg)
        traceback.print_exc()
        return {"status": "ERROR", "results": None, "error_message": error_msg}
    except Exception as e:
        error_msg = f"An unexpected error occurred during the API request: {type(e).__name__}: {e}"
        print(f"--- UNEXPECTED ERROR IN STREAMLIT ASYNC CALL ---")
        traceback.print_exc()
        return {"status": "ERROR", "results": None, "error_message": error_msg}


# --- Synchronous Wrapper to Run Async Code ---
def run_analysis_sync_wrapper(payload):
    """Synchronously runs the async API call and returns its result."""
    try:
        # Use asyncio.run to execute the async function
        return asyncio.run(run_analysis_async(payload))
    except Exception as e:
        # Catch any exceptions from asyncio.run or the async function itself
        error_msg = f"Error during synchronous async execution: {type(e).__name__}: {e}"
        print(f"--- ERROR IN SYNC ASYNC WRAPPER ---")
        traceback.print_exc()
        return {"status": "ERROR", "results": None, "error_message": error_msg}


# --- Analysis Trigger and Execution ---
# This block runs when the form is submitted OR st.rerun() is called while status is RUNNING
if submitted and st.session_state.analysis_status != "RUNNING":
    # Capture form values into session state payload variables upon submission
    st.session_state.payload_query = st.session_state.initial_query_input
    st.session_state.payload_global_context = st.session_state.global_context_input
    st.session_state.payload_specific_context = st.session_state.specific_context_input
    st.session_state.payload_specific_country_name = st.session_state.country_select # Store name, convert to code for API
    st.session_state.payload_max_global = st.session_state.max_global_input
    st.session_state.payload_max_specific = st.session_state.max_specific_input

    # Capture selected LLM provider and model
    st.session_state.payload_llm_provider = selected_provider_key
    st.session_state.payload_llm_model = st.session_state.get(f"{selected_provider_key}_model_input")

    # --- Input Validation ---
    validation_errors = []
    if not st.session_state.get('payload_query') or not st.session_state.get('payload_query').strip():
        validation_errors.append("Please enter an initial search query.")
    if not st.session_state.get('payload_llm_provider') or not st.session_state.get('payload_llm_provider').strip() or \
       not st.session_state.get('payload_llm_model') or not st.session_state.get('payload_llm_model').strip() or \
       st.session_state.get('payload_llm_model') == "unknown":
        validation_errors.append("Please select a valid LLM Provider and Model in the sidebar.")
    if BACKEND_API_URL.startswith("YOUR_"):
         validation_errors.append("Backend API URL needs configuration. Please set the `BACKEND_API_URL` environment variable.")
    if st.session_state.get('payload_specific_country_name') == "Global" and (st.session_state.get('payload_specific_context') == "Search for specific company examples and regulatory actions" or ('related to' in st.session_state.get('specific_context_input','').lower() and st.session_state.get('_last_updated_query','').lower() not in st.session_state.get('specific_context_input','').lower())):
         # Warn if country is Global but specific context is still country-focused based on default text
         st.warning("You selected 'Global' for the country target, but the 'Specific Search Context' still seems focused on specific companies/actions related to a country. Consider adjusting the specific context for a global search.")
         # Decided not to block, just warn


    if validation_errors:
        # Display errors and reset status
        for err in validation_errors: st.error(err)
        st.session_state.analysis_status = "IDLE"
        st.session_state.analysis_payload = None # Clear payload on validation failure
        st.session_state.error_message = "Validation failed. Check inputs."
        st.session_state.analysis_results = None # Clear previous results
        print(f"Form submission failed validation: {validation_errors}")
    else:
        # Inputs are valid, prepare payload for API call
        specific_country_code_to_send = COUNTRY_OPTIONS.get(st.session_state.payload_specific_country_name, "us") # Convert name to code

        payload = {
            "query": st.session_state.payload_query,
            "global_context": st.session_state.payload_global_context,
            "specific_context": st.session_state.payload_specific_context,
            "specific_country": specific_country_code_to_send,
            "max_global": st.session_state.payload_max_global,
            "max_specific": st.session_state.payload_max_specific,
            "llm_provider": st.session_state.payload_llm_provider,
            "llm_model": st.session_state.payload_llm_model
        }
        st.session_state.analysis_payload = payload # Store payload in state
        st.session_state.analysis_results = None # Clear previous results
        st.session_state.error_message = None # Clear previous error
        st.session_state.analysis_status = "RUNNING" # Set status to RUNNING
        print(f"Form submitted successfully. Payload set in state. Triggering rerun for API call.")
        # print(f"Captured Payload: {payload}") # Avoid logging sensitive details like full contexts
        st.rerun() # Rerun the script to execute the API call block

# --- API Call Execution Block ---
# This block runs if the status is "RUNNING"
elif st.session_state.analysis_status == "RUNNING":
     st.info(f"Analysis in progress... Sending request to backend API ({ANALYZE_ENDPOINT}).")
     payload_to_send = st.session_state.get('analysis_payload')

     if not payload_to_send or not isinstance(payload_to_send, dict):
         # Should not happen if validation worked, but safety check
         error_msg = "Internal error: Analysis payload not found or is invalid in session state."
         st.error(error_msg)
         st.session_state.analysis_status = "ERROR"
         st.session_state.error_message = error_msg
         st.session_state.analysis_payload = None # Clear invalid payload
         print(f"--- STREAMLIT STATE ERROR: {error_msg} ---")
         st.rerun() # Rerun to show error state
     else:
         # Display LLM being used based on the payload
         llm_display = f"{payload_to_send.get('llm_provider','?')}: {payload_to_send.get('llm_model','?')}"
         st.write(f"Using LLM: {llm_display}")

         with st.spinner("Analysis in progress... This may take several minutes."):
            # --- Call the synchronous wrapper which runs the async API call ---
            task_result = run_analysis_sync_wrapper(payload_to_send)

            # Update session state based on the result received from the wrapper
            st.session_state.analysis_status = task_result.get("status", "ERROR") # Use status from task result
            st.session_state.analysis_results = task_result.get("results")
            st.session_state.error_message = task_result.get("error_message")

         # --- IMPORTANT: Trigger a rerun to move to the display block ---
         # This ensures the UI updates after the blocking spinner and state change
         st.rerun()


# --- Analysis Complete/Error Display Block ---
# This block runs if the status is COMPLETE, COMPLETE_WITH_ERROR, or ERROR
elif st.session_state.analysis_status in ["COMPLETE", "COMPLETE_WITH_ERROR"]:
    results = st.session_state.analysis_results
    # Ensure results is a dictionary before trying to access keys
    if not isinstance(results, dict):
        st.error("Analysis completed, but received invalid results data from backend.")
        st.session_state.analysis_status = "ERROR" # Set status to error for safety
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
        else:
             st.error("Backend returned data in an unexpected format.")
        st.json(results) # Display whatever was received for debugging
    else:
        # Proceed with displaying results if they are valid
        if st.session_state.analysis_status == "COMPLETE":
            st.success("Analysis complete!")
        elif st.session_state.analysis_status == "COMPLETE_WITH_ERROR":
            # Display the error message received from the backend
            st.warning(st.session_state.error_message)


        st.subheader("Analysis Summary")
        st.markdown("**Key Takeaways:**")
        # Display the analysis summary text
        summary_text = results.get("analysis_summary", "Summary could not be generated or analysis failed early.")
        st.info(summary_text)

        # Display key metrics in columns
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        with col_metrics1: st.metric("LLM Used", results.get("llm_used", "N/A"))
        with col_metrics2: st.metric("Total Duration (s)", results.get("run_duration_seconds", "N/A"))
        with col_metrics3: st.metric("KG Update Status", results.get("kg_update_status", "N/A"))


        # Display counts of extracted data types
        st.markdown("**Extracted Data Counts (before filtering for sheet/KG):**")
        col_counts1, col_counts2, col_counts3, col_counts4, col_counts5 = st.columns(5)
        extracted_counts = results.get("extracted_data_counts", {})
        with col_counts1: st.metric("Entities", extracted_counts.get("entities", "N/A"))
        with col_counts2: st.metric("Risks", extracted_counts.get("risks", "N/A"))
        with col_counts3: st.metric("Relationships", extracted_counts.get("relationships", "N/A"))
        with col_counts4: st.metric("Linkup Structured Results", results.get("linkup_structured_data_count", "N/A"))
        with col_counts5: st.metric("URLs Checked (Wayback)", results.get("wayback_results_count", "N/A"))


        # Get exposures count from the results dictionary
        exposures_list_from_results = results.get("high_risk_exposures", [])
        exposures_count = len(exposures_list_from_results)
        st.metric("High Risk Exposures Found (Saved to Sheet)", exposures_count)


        # Add link to Google Sheet Exposures tab - This block is already here
        if GOOGLE_EXPOSURES_SHEET_URL:
             st.markdown(f"[View All Results & High Risk Exposures in Google Sheet]({GOOGLE_EXPOSURES_SHEET_URL})")
             if EXPOSURES_SHEET_GID == "YOUR_EXPOSURES_SHEET_GID":
                 st.caption("Note: Update `EXPOSURES_SHEET_GID` in `streamlit_ui.py` with your actual GID for a direct link.")
        elif GOOGLE_SHEET_URL:
             st.markdown(f"[View All Results in Google Sheet]({GOOGLE_SHEET_URL})")
             st.caption("Note: Google Sheet Exposures tab link not configured.")
        else:
             st.caption("Google Sheet link not configured (GOOGLE_SHEET_ID missing or invalid).")


        # --- Display High Risk Exposures Table ---
        st.subheader("Identified High Risk Exposures (Current Run)")
        if exposures_list_from_results:
            try:
                # Create a Pandas DataFrame from the exposures list
                exposures_df = pd.DataFrame(exposures_list_from_results)
                # Select and order columns for display
                display_cols = ["Entity", "Subsidiary/Affiliate", "Parent Company", "Risk_Severity", "Risk_Type", "Explanation", "Main_Sources"]
                # Ensure all display_cols exist in the DataFrame before selecting
                existing_cols = [col for col in display_cols if col in exposures_df.columns]
                st.dataframe(exposures_df[existing_cols], use_container_width=True)
            except Exception as exp_df_e:
                st.warning(f"Error displaying exposures table: {exp_df_e}")
                # Fallback to displaying JSON if table creation fails
                st.json(exposures_list_from_results)
        else:
            st.write("No high risk exposures identified in this run.")


        # --- Optional Expander for Step Details ---
        st.markdown("---") # Separator

        with st.expander("Run Steps & Details", expanded=False):
            st.subheader("Run Steps & Durations");
            if results.get("steps"):
                try:
                    steps_data = []
                    for step in results["steps"]:
                         if isinstance(step, dict):
                               # Use extracted_data_counts if available, otherwise fallback to original keys if they exist
                              entity_count_step = step.get("extracted_data_counts", {}).get("entities", step.get("extracted_data",{}).get("entities","N/A")) # Fallback check includes original structure
                              risk_count_step = step.get("extracted_data_counts", {}).get("risks", step.get("extracted_data",{}).get("risks","N/A"))
                              rel_count_step = step.get("extracted_data_counts", {}).get("relationships", step.get("extracted_data",{}).get("relationships","N/A"))

                              # If fallback returned a list, get its length
                              if isinstance(entity_count_step, list): entity_count_step = len(entity_count_step)
                              if isinstance(risk_count_step, list): risk_count_step = len(risk_count_step)
                              if isinstance(rel_count_step, list): rel_count_step = len(rel_count_step)


                              steps_data.append({
                                  "Name": step.get("name", "N/A"),
                                  "Duration (s)": step.get("duration", "N/A"),
                                  "Status": step.get("status", "N/A"),
                                  "Search Results": step.get("search_results_count", "N/A"),
                                  "Structured Results": step.get("structured_results_count", "N/A"), # Count from Step 1.5
                                  "Exposures Found": step.get("exposures_found", "N/A"), # Count from Step 3.5
                                  "URLs Checked": step.get("urls_checked", "N/A"), # Count from Step 4
                                  "KG Status": step.get("kg_update_status", "N/A"), # Status from Step 5.1
                                  # Display counts from extracted_data_counts if available
                                  "Entities Extracted (Step)": entity_count_step,
                                  "Risks Extracted (Step)": risk_count_step,
                                  "Relationships Extracted (Step)": rel_count_step,
                                  "Error Message": step.get("error_message", "") # Error message from any step
                              })
                         else:
                              steps_data.append({"Name": "Invalid Step Data", "Status": "Error", "Error Message": "Step data is not a dictionary."})
                    steps_df = pd.DataFrame(steps_data)
                    # Define column order and drop columns where all values are N/A, "", or None
                    col_order = [
                        "Name", "Status", "Duration (s)", "Error Message",
                        "Search Results", "Structured Results",
                        "Entities Extracted (Step)", "Risks Extracted (Step)", "Relationships Extracted (Step)",
                        "Exposures Found", "URLs Checked", "KG Status",
                    ]
                    # Filter for columns that exist in the DataFrame and are not all empty/N/A
                    cols_to_display = [col for col in col_order if col in steps_df.columns and not steps_df[col].isnull().all() and not (steps_df[col] == '').all()]

                    steps_df = steps_df.reindex(columns=cols_to_display)
                    st.dataframe(steps_df, use_container_width=True)
                except Exception as df_e: # FIX: This except block must directly follow the try block with correct indentation
                    st.warning(f"Error displaying steps table: {df_e}")
                    st.json(results.get("steps", "No steps data."))
            else: st.write("No step details available.")

    # Remove expanders for full raw data lists to reduce Streamlit state size
    # The data is saved to Google Sheets, so the link above serves as access point
    # with st.expander("Final Extracted Data (Combined Raw)", expanded=False):
    #     st.subheader("Final Extracted Data (Before Filtering for Sheet/KG)")
    #     final_data = results.get("final_extracted_data", {})
    #     if final_data:
    #          st.json(final_data)
    #     else:
    #          st.write("No final extracted data available.")

    # with st.expander("Linkup Raw Structured Data", expanded=False):
    #     st.subheader("Raw Structured Data Collected from Linkup")
    #     structured_data_list = results.get("linkup_structured_data", [])
    #     if structured_data_list:
    #          st.json(structured_data_list)
    #     else:
    #          st.write("No raw structured data collected from Linkup.")

    # with st.expander("Wayback Machine Results", expanded=False):
    #     st.subheader("Wayback Machine Check Results")
    #     wayback_results_list = results.get("wayback_results", [])
    #     if wayback_results_list:
    #         st.json(wayback_results_list)
    #     else:
    #          st.write("No wayback machine results available.")

    with st.expander("Full Raw API Response JSON (Limited Data)", expanded=False):
        st.subheader("Complete Raw JSON Output (Subset for UI)")
        # Display the exact data structure received by the UI
        st.json(results)

# --- Error Display Block ---
# This block runs if the status is ERROR (API request failed, timeout, etc.)
elif st.session_state.analysis_status == "ERROR":
    st.error("Analysis Failed!")
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
    else:
        st.error("An unknown error occurred during processing.")

    # Optionally display any partial results or steps if available, even in error state
    # The error message might contain partial results if the backend returned a 500 with body
    # Or st.session_state.analysis_results might hold partial results if the API call itself failed after some steps ran
    partial_results = st.session_state.get('analysis_results')
    if partial_results and isinstance(partial_results, dict):
         st.subheader("Partial Results (if any)")
         # Display partial metrics
         col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
         with col_metrics1: st.metric("LLM Used", partial_results.get("llm_used", "N/A"))
         with col_metrics2: st.metric("Partial Duration (s)", partial_results.get("run_duration_seconds", "N/A"))
         with col_metrics3: st.metric("KG Update Status (Partial)", partial_results.get("kg_update_status", "N/A"))

         # Display counts of extracted data types if available in partial results
         st.markdown("**Partial Extracted Data Counts:**")
         col_counts1, col_counts2, col_counts3, col_counts4, col_counts5 = st.columns(5)
         extracted_counts = partial_results.get("extracted_data_counts", {})
         with col_counts1: st.metric("Entities", extracted_counts.get("entities", "N/A"))
         with col_counts2: st.metric("Risks", extracted_counts.get("risks", "N/A"))
         with col_counts3: st.metric("Relationships", extracted_counts.get("relationships", "N/A"))
         with col_counts4: st.metric("Linkup Structured Results", partial_results.get("linkup_structured_data_count", "N/A"))
         with col_counts5: st.metric("URLs Checked (Wayback)", partial_results.get("wayback_results_count", "N/A"))

         # Display partial steps
         with st.expander("Completed Steps (Partial Run)", expanded=True):
            st.subheader("Run Steps & Durations");
            if partial_results.get("steps"):
                try: # FIX: Ensure this try block is correctly structured
                    steps_data = []
                    for step in partial_results["steps"]:
                         if isinstance(step, dict):
                               # Use extracted_data_counts if available, otherwise fallback to original keys if they exist
                              entity_count_step = step.get("extracted_data_counts", {}).get("entities", step.get("extracted_data",{}).get("entities","N/A")) # Fallback check includes original structure
                              risk_count_step = step.get("extracted_data_counts", {}).get("risks", step.get("extracted_data",{}).get("risks","N/A"))
                              rel_count_step = step.get("extracted_data_counts", {}).get("relationships", step.get("extracted_data",{}).get("relationships","N/A"))

                              # If fallback returned a list, get its length
                              if isinstance(entity_count_step, list): entity_count_step = len(entity_count_step)
                              if isinstance(risk_count_step, list): risk_count_step = len(risk_count_step)
                              if isinstance(rel_count_step, list): rel_count_step = len(rel_count_step)


                              steps_data.append({
                                  "Name": step.get("name", "N/A"),
                                  "Duration (s)": step.get("duration", "N/A"),
                                  "Status": step.get("status", "N/A"),
                                  "Search Results": step.get("search_results_count", "N/A"),
                                  "Structured Results": step.get("structured_results_count", "N/A"), # Count from Step 1.5
                                  "Exposures Found": step.get("exposures_found", "N/A"), # Count from Step 3.5
                                  "URLs Checked": step.get("urls_checked", "N/A"), # Count from Step 4
                                  "KG Status": step.get("kg_update_status", "N/A"), # Status from Step 5.1
                                  # Display counts from extracted_data_counts if available
                                  "Entities Extracted (Step)": entity_count_step,
                                  "Risks Extracted (Step)": risk_count_step,
                                  "Relationships Extracted (Step)": rel_count_step,
                                  "Error Message": step.get("error_message", "") # Error message from any step
                              })
                         else: steps_data.append({"Name": "Invalid Step Data", "Status": "Error"})
                    steps_df = pd.DataFrame(steps_data)
                    # Define column order and drop columns where all values are N/A, "", or None
                    col_order = [
                        "Name", "Status", "Duration (s)", "Error Message",
                        "Search Results", "Structured Results",
                        "Entities Extracted (Step)", "Risks Extracted (Step)", "Relationships Extracted (Step)",
                        "Exposures Found", "URLs Checked", "KG Status",
                    ]
                    # Filter for columns that exist in the DataFrame and are not all empty/N/A
                    cols_to_display = [col for col in col_order if col in steps_df.columns and not steps_df[col].isnull().all() and not (steps_df[col] == '').all()]

                    steps_df = steps_df.reindex(columns=cols_to_display)
                    st.dataframe(steps_df, use_container_width=True)
                except Exception as df_e: # FIX: This except block must directly follow the try block with correct indentation
                    st.warning(f"Error displaying partial steps table: {df_e}")
                    st.json(partial_results.get("steps", "No steps data."))
            else: st.write("No step details available.")

         with st.expander("Partial Raw API Response JSON (if any)", expanded=False):
              st.json(partial_results)


# --- Initial/Idle State Display ---
else:
    st.info("Enter query and click 'Run Analysis' to begin.")