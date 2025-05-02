# streamlit_ui.py

import streamlit as st
import json
from datetime import datetime
import os
# import requests # Switched to httpx for async
import httpx # Import httpx
import asyncio # Import asyncio
import pandas as pd
import traceback

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
    "openrouter": getattr(config, 'DEFAULT_OPENROUTER_MODEL', "google/gemini-flash-1.5") if config else "google/gemini-flash-1.5"
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


st.set_page_config(page_title="AI Analyst Agent", layout="wide")
st.title("🕵️ AI Analyst Agent Interface")
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

st.sidebar.caption("✨ Tip: Google AI & OpenRouter often have free tiers. OpenAI requires paid credits.")

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
        # Corrected the key for max_specific_input to match the assignment in the submitted block
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

        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        results_data = response.json()
        print(f"Streamlit received backend analysis response. Duration: {results_data.get('run_duration_seconds', 'N/A')}s")

        # Check if the backend reported an error in the response body
        if results_data.get("error") and results_data["error"] != "None" and results_data["error"] != "":
             return {"status": "COMPLETE_WITH_ERROR", "results": results_data, "error_message": f"Backend Orchestrator reported an error: {results_data['error']}"}
        else:
             return {"status": "COMPLETE", "results": results_data, "error_message": None}

    except httpx.TimeoutException:
        error_msg = f"Request to backend API timed out after {api_timeout_seconds} seconds. The analysis might still be running on the backend."
        print(error_msg)
        return {"status": "ERROR", "results": None, "error_message": error_msg}
    except httpx.RequestError as e:
        error_msg = f"An HTTP request error occurred calling backend API at {ANALYZE_ENDPOINT}: {e}"
        print(error_msg)
        traceback.print_exc()
        return {"status": "ERROR", "results": None, "error_message": error_msg}
    except httpx.HTTPStatusError as e:
         error_msg = f"Backend API returned HTTP error {e.response.status_code} for {e.request.url}"
         try:
              error_detail = e.response.json().get('detail', e.response.text[:200] + '...')
              error_msg += f"\nDetail: {json.dumps(error_detail)}"
         except json.JSONDecodeError:
              error_msg += f"\nResponse: {e.response.text[:200]}..."
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


# --- Analysis Trigger and Execution Logic ---
# This block now directly calls the synchronous wrapper and updates state afterwards
if submitted: # Remove the status check here, as the wrapper will block anyway
    # Capture form values into session state payload variables upon submission
    st.session_state.payload_query = st.session_state.initial_query_input
    st.session_state.payload_global_context = st.session_state.global_context_input
    st.session_state.payload_specific_context = st.session_state.specific_context_input
    st.session_state.payload_specific_country_name = st.session_state.country_select # Store name, convert to code for API
    st.session_state.payload_max_global = st.session_state.max_global_input
    # Corrected line: Assign value from max_specific_input widget
    st.session_state.payload_max_specific = st.session_state.max_specific_input

    # Capture selected LLM provider and model
    selected_provider_name_from_state = st.session_state.sidebar_llm_provider_name_select # Use sidebar selectbox value
    st.session_state.payload_llm_provider = LLM_PROVIDERS[selected_provider_name_from_state] # Get the key from the name
    st.session_state.payload_llm_model = st.session_state.get(f"{st.session_state.payload_llm_provider}_model_input") # Use the stored model input for that key


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
    if st.session_state.get('payload_specific_country_name') == "Global" and (st.session_state.get('payload_specific_context') == "Search for specific company examples and regulatory actions" or ('related to' in st.session_state.get('specific_context_input','').lower() and "global financial news" not in st.session_state.get('specific_context_input','').lower())):
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

        print(f"Form submitted successfully. Payload set in state. Starting synchronous API task via wrapper.")

        # --- Execute the synchronous wrapper for the async API call ---
        with st.spinner("Analysis in progress... Please wait. (Calling backend API...)"):
            task_result = run_analysis_sync_wrapper(payload)

        # Update session state based on the result received from the wrapper
        st.session_state.analysis_status = task_result.get("status", "ERROR") # Use status from task result
        st.session_state.analysis_results = task_result.get("results")
        st.session_state.error_message = task_result.get("error_message")

        # No need for rerun here, as the script execution continues after the blocking call


# --- Analysis Complete/Error Display Block ---
# This block runs if the status is COMPLETE, COMPLETE_WITH_ERROR, or ERROR (or IDLE initially)
# The previous 'elif analysis_status == "RUNNING"' block is removed
if st.session_state.analysis_status in ["COMPLETE", "COMPLETE_WITH_ERROR", "ERROR"]: # Include ERROR here

    # Display error message if status is ERROR or COMPLETE_WITH_ERROR
    if st.session_state.analysis_status == "ERROR":
        st.error("Analysis Failed!")
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
        else:
            st.error("An unknown error occurred during processing.")
    elif st.session_state.analysis_status == "COMPLETE_WITH_ERROR":
        st.warning("Analysis completed with reported backend errors.")
        if st.session_state.error_message:
             st.error(st.session_state.error_message)


    # Proceed to display results if analysis_results are available, even in error state
    if st.session_state.get('analysis_results') and isinstance(st.session_state.analysis_results, dict):
        results = st.session_state.analysis_results

        st.subheader("Analysis Summary")
        st.markdown("**Key Takeaways:**")
        # Display the analysis summary text
        summary_text = results.get("analysis_summary", "Summary could not be generated or analysis failed early.")
        st.info(summary_text)

        # Display key metrics in columns
        col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
        with col_metrics1: st.metric("LLM Used", results.get("llm_used", "N/A"))
        with col_metrics2: st.metric("Total Duration (s)", results.get("run_duration_seconds", "N/A"))
        with col_metrics3: st.metric("KG Update Status", results.get("kg_update_status", "N/A"))

        # Get exposures count from the results dictionary
        exposures_list_from_results = results.get("high_risk_exposures", [])
        exposures_count = len(exposures_list_from_results)
        with col_metrics4: st.metric("High Risk Exposures Found", exposures_count)

        # Add link to Google Sheet Exposures tab
        if GOOGLE_EXPOSURES_SHEET_URL:
             st.markdown(f"[View All Results & High Risk Exposures in Google Sheet]({GOOGLE_EXPOSURES_SHEET_URL})")
             if EXPOSURES_SHEET_GID == "1468712289": # Check if the default GID is still the placeholder
                 st.caption("Note: Update `EXPOSURES_SHEET_GID` in `streamlit_ui.py` with your actual GID for a direct link to your sheet's Exposures tab.")
             elif GOOGLE_EXPOSURES_SHEET_URL != GOOGLE_SHEET_URL:
                  st.caption(f"Linking directly to Exposures tab (GID: {EXPOSURES_SHEET_GID}).")
             # else it's just the base sheet link

        elif GOOGLE_SHEET_URL:
             st.markdown(f"[View All Results in Google Sheet]({GOOGLE_SHEET_URL})")
             st.caption("Note: Google Sheet Exposures tab GID not configured for a direct link.")
        else:
             st.caption("Google Sheet link not configured (GOOGLE_SHEET_ID missing or invalid).")


        # --- Display High Risk Exposures Table ---
        st.subheader("Identified High Risk Exposures (Current Run)")
        if exposures_list_from_results:
            try:
                # Create a Pandas DataFrame from the exposures list
                exposures_df = pd.DataFrame(exposures_list_from_results)
                # Select and order columns for display
                # Ensure 'Main_Sources' is included if it exists
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


        # --- Optional Expanders for Detailed Data ---
        st.markdown("---") # Separator

        with st.expander("Run Steps & Details", expanded=False):
            st.subheader("Run Steps & Durations");
            if results.get("steps"):
                try:
                    steps_data = []
                    for step in results["steps"]:
                         if isinstance(step, dict):
                              steps_data.append({
                                  "Name": step.get("name", "N/A"),
                                  "Duration (s)": step.get("duration", "N/A"),
                                  "Status": step.get("status", "N/A"),
                                  "Search Results": step.get("search_results_count", "N/A"),
                                  "Structured Results": step.get("structured_results_count", "N/A"),
                                  "Exposures Found": step.get("exposures_found", "N/A"), # Count from Step 3.5
                                  "URLs Checked": step.get("urls_checked", "N/A"), # Count from Step 4
                                  "KG Status": step.get("kg_update_status", "N/A"), # Status from Step 5.1
                                  # Display counts from extracted_data_counts if available
                                  "Entities Extracted": step.get("extracted_data_counts", {}).get("entities", "N/A"),
                                  "Risks Extracted": step.get("extracted_data_counts", {}).get("risks", "N/A"),
                                  "Relationships Extracted": step.get("extracted_data_counts", {}).get("relationships", "N/A"),
                                  "Error Message": step.get("error_message", "") # Error message from any step
                              })
                         else:
                              steps_data.append({"Name": "Invalid Step Data", "Status": "Error", "Error Message": "Step data is not a dictionary."})
                    steps_df = pd.DataFrame(steps_data)
                    # Define column order and drop columns where all values are N/A, "", or None
                    col_order = [
                        "Name", "Status", "Duration (s)", "Error Message",
                        "Search Results", "Structured Results",
                        "Entities Extracted", "Risks Extracted", "Relationships Extracted",
                        "Exposures Found", "URLs Checked", "KG Status",
                    ]
                    # Filter for columns that exist in the DataFrame and are not all empty/N/A
                    cols_to_display = [col for col in col_order if col in steps_df.columns and not steps_df[col].isnull().all() and not (steps_df[col] == '').all()]

                    steps_df = steps_df.reindex(columns=cols_to_display)
                    st.dataframe(steps_df, use_container_width=True)
                except Exception as df_e:
                    st.warning(f"Error displaying steps table: {df_e}")
                    st.json(results.get("steps", "No steps data."))
            else: st.write("No step details available.")

        # Removed the expanders for raw Extracted Data, Structured Data, and Wayback Results
        # based on the request to simplify the display and focus only on Exposures table + Summary


        with st.expander("Full Raw Results JSON", expanded=False):
            st.subheader("Complete Raw JSON Output")
            # Ensure the original high_risk_exposures list is included in the full raw JSON
            # (It's already there as results["high_risk_exposures"])
            st.json(results)

    # --- Display error message if analysis_results are None (e.g., API call failed before returning data) ---
    elif st.session_state.analysis_status == "ERROR" and st.session_state.get('analysis_results') is None:
         # Error message already displayed above
         pass # Nothing more to display if no results were returned

# --- Initial/Idle State Display ---
else:
    st.info("Enter query and click 'Run Analysis' to begin.")