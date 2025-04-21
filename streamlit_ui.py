# streamlit_ui.py

import streamlit as st
import json
from datetime import datetime
import os
import requests
import pandas as pd

# Try to import config - handle gracefully
try:
    import config
except ImportError:
    config = None
    st.warning("config.py not found. Using default values.")

# Attempt to import pycountry
try:
    import pycountry
    pycountry_available = True
except ImportError:
    print("Warning: 'pycountry' not installed. Using basic country list.")
    pycountry_available = False

# --- Configuration ---
DEFAULT_BACKEND_URL = "http://localhost:8000"
BACKEND_API_URL = os.getenv("BACKEND_API_URL", DEFAULT_BACKEND_URL)
ANALYZE_ENDPOINT = f"{BACKEND_API_URL}/analyze"
# Safely get Sheet ID from config or set to None
GOOGLE_SHEET_ID_FROM_CONFIG = getattr(config, 'GOOGLE_SHEET_ID', None)
GOOGLE_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID_FROM_CONFIG}/edit" if GOOGLE_SHEET_ID_FROM_CONFIG else None

# --- LLM Options ---
LLM_PROVIDERS = { "Google AI": "google_ai", "OpenAI": "openai", "OpenRouter": "openrouter" }
DEFAULT_MODELS = {
    "google_ai": getattr(config, 'DEFAULT_GOOGLE_AI_MODEL', "models/gemini-1.5-flash-latest"),
    "openai": getattr(config, 'DEFAULT_OPENAI_MODEL', "gpt-4o"),
    "openrouter": getattr(config, 'DEFAULT_OPENROUTER_MODEL', "google/gemini-flash-1.5")
}
# PROVIDER_LINKS = { ... } # Optional

# --- Country List Generation ---
def get_country_options():
    options = {"Global": "global"}; # ... (rest of function same as before) ...
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
def update_contexts():
    query = st.session_state.get("initial_query_input", "")
    if query:
        st.session_state.global_context_input = f"Global financial news and legal filings for '{query}'"
        st.session_state.specific_context_input = f"Search for specific company examples and regulatory actions related to '{query}'"
    else:
        st.session_state.global_context_input = "Global financial news and legal filings for compliance issues"
        st.session_state.specific_context_input = "Search for specific company examples and regulatory actions"

# --- Initialize Session State ---
# Initialize contexts if they don't exist
if "global_context_input" not in st.session_state: st.session_state.global_context_input = "Global financial news and legal filings for compliance issues"
if "specific_context_input" not in st.session_state: st.session_state.specific_context_input = "Search for specific company examples and regulatory actions"
# Initialize analysis state if it doesn't exist
if 'analysis_status' not in st.session_state: st.session_state.analysis_status = "IDLE" # IDLE, RUNNING, COMPLETE, ERROR
if 'analysis_payload' not in st.session_state: st.session_state.analysis_payload = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'error_message' not in st.session_state: st.session_state.error_message = None


# --- Streamlit App Layout ---
st.set_page_config(page_title="AI Analyst Agent", layout="wide")
st.title("🕵️ AI Analyst Agent Interface")
st.markdown("Enter query, select LLM/Country. API Keys configured on backend.")

if BACKEND_API_URL.startswith("YOUR_"): st.error("Backend API URL needs config.")
elif BACKEND_API_URL == DEFAULT_BACKEND_URL: st.info("Targeting local backend API.")
else: st.info(f"Targeting Backend API: {BACKEND_API_URL}")


# --- LLM Configuration UI (Sidebar) ---
st.sidebar.title("LLM Selection")
selected_provider_name = st.sidebar.selectbox("Select LLM Provider", options=list(LLM_PROVIDERS.keys()), index=0 )
selected_provider_key = LLM_PROVIDERS[selected_provider_name]
default_model = DEFAULT_MODELS.get(selected_provider_key, "")
session_key_model = f"{selected_provider_key}_model"
if session_key_model not in st.session_state: st.session_state[session_key_model] = default_model
llm_model = st.sidebar.text_input( f"Model Name for {selected_provider_name}", key=session_key_model, help=f"e.g., {default_model}" )
st.sidebar.caption("✨ Tip: Google AI & OpenRouter often have free tiers. OpenAI requires paid credits.")


# --- Query Input OUTSIDE the Form ---
st.subheader("1. Enter Your Initial Query")
initial_query_value = st.text_input( "Initial Search Query:", st.session_state.get("initial_query_input", "Corporate tax evasion cases 2020-2023"), key="initial_query_input", on_change=update_contexts )


# --- Main Input Form ---
st.subheader("2. Configure Search & Run Analysis")
with st.form("analysis_form"):
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

    submitted = st.form_submit_button("Run Analysis")

# --- Execution Logic (Triggered on Form Submit) ---
if submitted:
    # Read current values directly from widgets/state
    query_to_run = st.session_state.get("initial_query_input", "")
    provider_to_run = selected_provider_key # From sidebar selectbox
    model_to_run = llm_model # From sidebar text input
    glob_context_to_run = global_context # From form text_area
    spec_context_to_run = specific_context # From form text_area
    country_to_run = selected_country_name # From form selectbox
    max_glob_to_run = max_global_results # From form number_input
    max_spec_to_run = max_specific_results # From form number_input

    # Validation
    if not query_to_run: st.warning("Please enter an initial search query.")
    elif not provider_to_run or not model_to_run: st.warning("Please select Provider/Model in sidebar.")
    elif BACKEND_API_URL.startswith("YOUR_"): st.error("Backend API URL is not configured.")
    else:
        # Construct payload *before* rerun
        specific_country_code_to_send = COUNTRY_OPTIONS.get(country_to_run, "us")
        payload = {
            "query": query_to_run,
            "global_context": glob_context_to_run,
            "specific_context": spec_context_to_run,
            "specific_country": specific_country_code_to_send,
            "max_global": max_glob_to_run,
            "max_specific": max_spec_to_run,
            "llm_provider": provider_to_run,
            "llm_model": model_to_run
        }
        # Store payload and set status before rerun
        st.session_state.analysis_payload = payload
        st.session_state.analysis_results = None # Clear previous results
        st.session_state.error_message = None # Clear previous errors
        st.session_state.analysis_status = "RUNNING"
        print(f"Form submitted. Payload set. Rerunning. Payload: {payload}") # Debug print
        st.rerun()


# --- Display Area (Handles RUNNING, COMPLETE, ERROR states) ---
if st.session_state.analysis_status == "RUNNING":
     st.info(f"Sending request to backend ({ANALYZE_ENDPOINT})...")
     # Retrieve payload stored before rerun
     payload_to_send = st.session_state.analysis_payload
     llm_display = f"{payload_to_send.get('llm_provider','?')} ({payload_to_send.get('llm_model','?')})"
     st.write(f"Using LLM: {llm_display}")
     with st.spinner("Analysis in progress... Please wait."):
        results_data = None; error_msg = None
        try:
            print(f"Making API call with payload: {payload_to_send}") # Debug print
            response = requests.post(ANALYZE_ENDPOINT, json=payload_to_send, timeout=300)
            if response.status_code == 200:
                results_data = response.json()
                st.session_state.analysis_status = "COMPLETE"
            else:
                 error_msg = f"Backend API request failed! Status Code: {response.status_code}"
                 try: error_detail = response.json(); error_msg += f"\nDetail: {json.dumps(error_detail)}"
                 except json.JSONDecodeError: error_msg += f"\nResponse: {response.text[:500]}"
                 st.session_state.analysis_status = "ERROR"
        except requests.exceptions.Timeout: error_msg = "Request to backend API timed out."; st.session_state.analysis_status = "ERROR"
        except requests.exceptions.RequestException as e: error_msg = f"Could not connect to backend API: {e}"; st.session_state.analysis_status = "ERROR"
        except Exception as e: error_msg = f"Unexpected error during analysis request: {e}"; st.session_state.analysis_status = "ERROR"; traceback.print_exc() # Print traceback for unexpected errors

        # Store results or error message
        st.session_state.analysis_results = results_data
        st.session_state.error_message = error_msg

        # Rerun to display results/error
        st.rerun()

elif st.session_state.analysis_status == "COMPLETE" and isinstance(st.session_state.analysis_results, dict):
    # --- Display Success Results ---
    results = st.session_state.analysis_results
    st.success("Analysis complete!")
    st.subheader("Analysis Summary")
    exposures = results.get("supply_chain_exposures", [])
    col_sum1, col_sum2 = st.columns([1,3]);
    with col_sum1: st.metric("Supply Chain Exposures Found", len(exposures))
    with col_sum2:
        if GOOGLE_SHEET_URL:
             exposures_gid = "1468712289" # <-- *** REPLACE MANUALLY ***
             if exposures_gid != "1468712289": sheet_link = f"{GOOGLE_SHEET_URL}#gid={exposures_gid}"; st.markdown(f"[View Exposure Details]({sheet_link})")
             else: st.markdown(f"[View Full Results Sheet]({GOOGLE_SHEET_URL})"); st.caption("(Add GID for direct link)")
        else: st.caption("Google Sheet link not configured.")
    st.metric("LLM Used", results.get("llm_used", "N/A"))
    st.metric("Total Duration (s)", results.get("run_duration_seconds", "N/A"))
    st.metric("KG Update Status", results.get("kg_update_status", "N/A"))
    if results.get("error"): st.error(f"Backend Orchestration Error: {results['error']}")
    st.subheader("Run Steps & Durations");
    if results.get("steps"):
        try: steps_df = pd.DataFrame(results["steps"]); st.dataframe(steps_df)
        except Exception as df_e: st.warning(f"Steps table error: {df_e}"); st.json(results["steps"])
    else: st.write("No step details.")
    with st.expander("Final Extracted Data (Combined)", expanded=False): st.json(results.get("final_extracted_data", {}))
    with st.expander("Identified Supply Chain Exposures (Details)", expanded=False): st.json(exposures)
    with st.expander("Wayback Machine Results", expanded=False): st.json(results.get("wayback_results", []))
    with st.expander("Full Raw Results JSON", expanded=False): st.json(results)

elif st.session_state.analysis_status == "ERROR":
    # --- Display Error Message ---
    st.error("Analysis Failed!")
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
    else:
        st.error("An unknown error occurred during processing.")

# else: # Initial IDLE state
#    st.write("Enter query and click 'Run Analysis'.")