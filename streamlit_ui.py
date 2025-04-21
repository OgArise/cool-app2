# streamlit_ui.py

import streamlit as st
import json
from datetime import datetime
import os
import requests
import pandas as pd

# Try to import config - handle gracefully if it fails during local dev maybe
try:
    import config
except ImportError:
    config = None # Set to None if not found
    st.warning("config.py not found. Default values will be used. Ensure config.py is present for proper operation.")

# Attempt to import pycountry
try:
    import pycountry
    pycountry_available = True
except ImportError:
    print("Warning: 'pycountry' not installed. Using basic country list.")
    pycountry_available = False

# --- Configuration ---
# Use config object if available, otherwise use defaults/env vars directly
DEFAULT_BACKEND_URL = "http://localhost:8000"
BACKEND_API_URL = os.getenv("BACKEND_API_URL", DEFAULT_BACKEND_URL)
ANALYZE_ENDPOINT = f"{BACKEND_API_URL}/analyze"
GOOGLE_SHEET_ID_FROM_CONFIG = getattr(config, 'GOOGLE_SHEET_ID', None) # Safely get from config
GOOGLE_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID_FROM_CONFIG}/edit" if GOOGLE_SHEET_ID_FROM_CONFIG else None

# --- LLM Options ---
LLM_PROVIDERS = { "Google AI": "google_ai", "OpenAI": "openai", "OpenRouter": "openrouter" }
# Get defaults safely from config object or use hardcoded defaults
DEFAULT_MODELS = {
    "google_ai": getattr(config, 'DEFAULT_GOOGLE_AI_MODEL', "models/gemini-1.5-flash-latest"),
    "openai": getattr(config, 'DEFAULT_OPENAI_MODEL', "gpt-4o"),
    "openrouter": getattr(config, 'DEFAULT_OPENROUTER_MODEL', "google/gemini-flash-1.5")
}
PROVIDER_LINKS = { "Google AI": "...", "OpenAI": "...", "OpenRouter": "..." } # Add actual links if desired

# --- Country List Generation ---
def get_country_options():
    # ... (keep the get_country_options function as before) ...
    options = {"Global": "global"}
    if pycountry_available:
        try:
            countries = sorted([(country.name, country.alpha_2.lower()) for country in pycountry.countries])
            preferred = {"China": "cn", "United States": "us"}; final_options = {"Global": "global"}
            for name, code in preferred.items():
                 if name in dict(countries): final_options[name] = code
            for name, code in countries:
                 if name not in final_options: final_options[name] = code
            options = final_options
        except Exception as e: print(f"Error loading countries: {e}"); options.update({"China": "cn", "United States": "us"})
    else: options.update({"China": "cn", "United States": "us"})
    return options
COUNTRY_OPTIONS = get_country_options(); COUNTRY_DISPLAY_NAMES = list(COUNTRY_OPTIONS.keys())


# --- Callback Function to Update Contexts ---
# This function will now be called by the text_input *outside* the form
def update_contexts():
    query = st.session_state.get("initial_query_input", "") # Get current query safely
    if query:
        st.session_state.global_context_input = f"Global financial news and legal filings for '{query}'"
        st.session_state.specific_context_input = f"Search for specific company examples and regulatory actions related to '{query}'"
    else: # Reset to defaults if query is cleared
        st.session_state.global_context_input = "Global financial news and legal filings for compliance issues"
        st.session_state.specific_context_input = "Search for specific company examples and regulatory actions"

# --- Initialize Session State for Contexts if they don't exist ---
if "global_context_input" not in st.session_state:
    st.session_state.global_context_input = "Global financial news and legal filings for compliance issues"
if "specific_context_input" not in st.session_state:
    st.session_state.specific_context_input = "Search for specific company examples and regulatory actions"


# --- Streamlit App Layout ---
st.set_page_config(page_title="AI Analyst Agent", layout="wide")
st.title("🕵️ AI Analyst Agent Interface")
st.markdown("Enter query, select LLM/Country. API Keys configured on backend.")

if BACKEND_API_URL.startswith("YOUR_"): st.error("Backend API URL needs config.")
elif BACKEND_API_URL == DEFAULT_BACKEND_URL: st.info("Targeting local backend API.")
else: st.info(f"Targeting Backend API: {BACKEND_API_URL}")


# --- LLM Configuration UI (Sidebar) ---
# (Keep the sidebar code exactly the same as before - selecting provider/model)
st.sidebar.title("LLM Selection")
selected_provider_name = st.sidebar.selectbox( "Select LLM Provider", options=list(LLM_PROVIDERS.keys()), index=0 )
selected_provider_key = LLM_PROVIDERS[selected_provider_name]
default_model = DEFAULT_MODELS.get(selected_provider_key, "")
session_key_model = f"{selected_provider_key}_model"
if session_key_model not in st.session_state: st.session_state[session_key_model] = default_model
llm_model = st.sidebar.text_input( f"Model Name for {selected_provider_name}", key=session_key_model, help=f"e.g., {default_model}" )
st.sidebar.caption("✨ Tip: Google AI & OpenRouter often have free tiers. OpenAI requires paid credits.")


# ===> Query Input OUTSIDE the Form <===
st.subheader("1. Enter Your Initial Query")
initial_query_value = st.text_input(
    "Initial Search Query:",
    st.session_state.get("initial_query_input", "Corporate tax evasion cases 2020-2023"), # Get value from session state if exists
    key="initial_query_input", # Assign key for session state access
    on_change=update_contexts # Callback is allowed here!
)
# ===> END CHANGE <===


# --- Main Input Form (Without Query Input) ---
st.subheader("2. Configure Search & Run Analysis")
with st.form("analysis_form"):
    # Context fields are now driven by session state, updated by the input above
    st.markdown("**Search Contexts (Auto-updated based on query)**")
    global_context = st.text_area( "Global Search Context", key="global_context_input", height=100 )
    specific_context = st.text_area( "Specific Search Context", key="specific_context_input", height=100 )

    st.markdown("**Other Parameters**")
    col1, col2 = st.columns(2)
    with col1:
        default_country_index = 0 # Global
        selected_country_name = st.selectbox( "Specific Country Search Target", options=COUNTRY_DISPLAY_NAMES, index=default_country_index )
    with col2:
        max_global_results = st.number_input("Max Global Results", min_value=1, max_value=50, value=5)
        max_specific_results = st.number_input("Max Specific Results", min_value=1, max_value=50, value=5)

    # Submit button remains inside the form
    submitted = st.form_submit_button("Run Analysis")


# --- Execution and Output ---
# Initialize results in session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if submitted:
    # --- Read values needed for API call ---
    # Read query from session state as it's outside the form now
    query_to_run = st.session_state.get("initial_query_input", "")
    # Read other values directly from form variables
    country_to_run = selected_country_name
    glob_context_to_run = global_context
    spec_context_to_run = specific_context
    max_glob_to_run = max_global_results
    max_spec_to_run = max_specific_results
    # Get current LLM selection from sidebar state
    provider_to_run = selected_provider_key
    model_to_run = llm_model

    # Basic validation before making API call
    if not query_to_run:
        st.warning("Please enter an initial search query.")
        st.stop() # Stop script execution for this run
    if not provider_to_run or not model_to_run:
        st.warning("Please select Provider/Model in sidebar.")
        st.stop()
    if BACKEND_API_URL.startswith("YOUR_"):
        st.error("Backend API URL is not configured.")
        st.stop()

    # Set state to trigger processing display and clear old results
    st.session_state.analysis_results = "RUNNING"
    st.rerun()

# --- Display Area ---
if st.session_state.analysis_results == "RUNNING":
     with st.spinner("Analysis in progress... Please wait."):
        # Fetch parameters from session state (where they were stored before rerun)
        # This part seems redundant now as we read directly above, but keep pattern if needed
        payload = {
            "query": st.session_state.get("initial_query_input", ""), # Read query from state again
            "global_context": st.session_state.get("global_context_input"), # Read context from state
            "specific_context": st.session_state.get("specific_context_input"), # Read context from state
            "specific_country": COUNTRY_OPTIONS.get(st.session_state.get("run_country_name", "Global"), "us"), # Need to read country from state if needed here
            "max_global": st.session_state.get("run_max_global", 5),
            "max_specific": st.session_state.get("run_max_specific", 5),
            "llm_provider": st.session_state.get("run_llm_provider", "google_ai"),
            "llm_model": st.session_state.get("run_llm_model", "unknown")
        }
        # Re-fetch the values needed for the payload directly *before* the rerun was triggered.
        # This state management needs careful review. Let's simplify.
        # We'll use the variables captured just before the rerun instead of session state here.

        # Construct payload using variables captured *before* st.rerun() in the 'if submitted:' block
        specific_country_code_to_send = COUNTRY_OPTIONS.get(st.session_state.run_country_name, "us") # Use state saved before rerun
        payload = {
            "query": st.session_state.run_query,
            "global_context": st.session_state.run_global_context,
            "specific_context": st.session_state.run_specific_context,
            "specific_country": specific_country_code_to_send,
            "max_global": st.session_state.run_max_global,
            "max_specific": st.session_state.run_max_specific,
            "llm_provider": st.session_state.run_llm_provider,
            "llm_model": st.session_state.run_llm_model
        }

        results_data = None; error_message = None
        try:
            print(f"Sending payload to backend: {payload}") # Log payload for debugging
            response = requests.post(ANALYZE_ENDPOINT, json=payload, timeout=300)
            if response.status_code == 200: results_data = response.json()
            else:
                 error_message = f"Backend API request failed! Status: {response.status_code}"
                 try: error_detail = response.json(); error_message += f"\nDetail: {json.dumps(error_detail)}"
                 except json.JSONDecodeError: error_message += f"\nResponse: {response.text[:500]}"
        except requests.exceptions.Timeout: error_message = "Request to backend API timed out."
        except requests.exceptions.RequestException as e: error_message = f"Could not connect to backend API: {e}"
        except Exception as e: error_message = f"Unexpected error during analysis request: {e}"

        st.session_state.analysis_results = results_data if results_data else {"error_message": error_message}
        st.rerun() # Rerun again to display final results

elif isinstance(st.session_state.analysis_results, dict):
    # Display results stored in session state
    results = st.session_state.analysis_results
    st.subheader("Analysis Summary")
    if results.get("error_message"):
        st.error(results["error_message"])
    else:
        # Display Supply Chain Summary and Link
        exposures = results.get("supply_chain_exposures", [])
        col_sum1, col_sum2 = st.columns([1,3]) # Allocate space
        with col_sum1:
             st.metric("Supply Chain Exposures Found", len(exposures))
        with col_sum2:
             if GOOGLE_SHEET_URL:
                 # Construct link to the specific tab - requires knowing the GID
                 # Find GID manually: Open sheet, go to tab, URL ends in #gid=NUMBER
                 # Replace YOUR_GID_HERE with the actual GID of the 'Supply Chain Exposures' tab
                 exposures_gid = "YOUR_GID_HERE" # <-- *** REPLACE MANUALLY ***
                 if exposures_gid != "YOUR_GID_HERE":
                      sheet_link = f"{GOOGLE_SHEET_URL}#gid={exposures_gid}"
                      st.markdown(f"[View Exposure Details in Google Sheet]({sheet_link})", unsafe_allow_html=True)
                 else:
                      st.markdown(f"[View Full Results Sheet]({GOOGLE_SHEET_URL})", unsafe_allow_html=True)
                      st.caption("(Add GID for 'Supply Chain Exposures' tab in code for direct link)")
             else:
                 st.caption("Google Sheet link not configured.")

        # Display other metrics
        st.metric("LLM Used", results.get("llm_used", "N/A"))
        st.metric("Total Duration (seconds)", results.get("run_duration_seconds", "N/A"))
        st.metric("KG Update Status", results.get("kg_update_status", "N/A"))
        if results.get("error"): st.error(f"Backend Orchestration Error: {results['error']}")

        # Display Steps
        st.subheader("Run Steps & Durations");
        if results.get("steps"):
            try: steps_df = pd.DataFrame(results["steps"]); st.dataframe(steps_df)
            except Exception as df_e: st.warning(f"Could not display steps table: {df_e}"); st.json(results["steps"])
        else: st.write("No step details.")

        # Display Expanders for detailed data
        with st.expander("Final Extracted Data (Combined)", expanded=False): st.json(results.get("final_extracted_data", {}))
        with st.expander("Identified Supply Chain Exposures (Details)", expanded=False): st.json(exposures)
        with st.expander("Wayback Machine Results", expanded=False): st.json(results.get("wayback_results", []))
        with st.expander("Full Raw Results JSON", expanded=False): st.json(results)