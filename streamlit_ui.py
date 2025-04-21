# streamlit_ui.py

import streamlit as st
import json
from datetime import datetime
import os
import requests
import pandas as pd
import config # Import config to get default models and SHEET_ID

# Attempt to import pycountry
try: import pycountry; pycountry_available = True
except ImportError: print("Warning: 'pycountry' not installed."); pycountry_available = False

# --- Configuration ---
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
ANALYZE_ENDPOINT = f"{BACKEND_API_URL}/analyze"
GOOGLE_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{config.GOOGLE_SHEET_ID}/edit" if config.GOOGLE_SHEET_ID else None

# --- LLM Options ---
LLM_PROVIDERS = { "Google AI": "google_ai", "OpenAI": "openai", "OpenRouter": "openrouter" } # Shorten OR name
DEFAULT_MODELS = { "google_ai": config.DEFAULT_GOOGLE_AI_MODEL, "openai": config.DEFAULT_OPENAI_MODEL, "openrouter": config.DEFAULT_OPENROUTER_MODEL }
PROVIDER_LINKS = { "Google AI": "...", "OpenAI": "...", "OpenRouter": "..." }

# --- Country List Generation ---
def get_country_options():
    options = {"Global": "global"}
    if pycountry_available:
        try:
            countries = sorted([(country.name, country.alpha_2.lower()) for country in pycountry.countries])
            # Move China and US towards the top after Global if they exist
            preferred = {"China": "cn", "United States": "us"}
            final_options = {"Global": "global"}
            for name, code in preferred.items():
                 if name in dict(countries): final_options[name] = code
            for name, code in countries:
                 if name not in final_options: final_options[name] = code # Add remaining sorted countries
            options = final_options
        except Exception as e:
             print(f"Error loading countries from pycountry: {e}. Using basic list."); options.update({"China": "cn", "United States": "us"})
    else: options.update({"China": "cn", "United States": "us"})
    return options
COUNTRY_OPTIONS = get_country_options(); COUNTRY_DISPLAY_NAMES = list(COUNTRY_OPTIONS.keys())


# --- Function to update context based on query ---
def update_contexts():
    query = st.session_state.initial_query_input # Get current query
    if query:
        st.session_state.global_context_input = f"Global financial news and legal filings for '{query}'"
        st.session_state.specific_context_input = f"Search for specific company examples and regulatory actions related to '{query}'"
    # If query is empty, reset to defaults (optional)
    # else:
    #    st.session_state.global_context_input = "Global financial news and legal filings for compliance issues"
    #    st.session_state.specific_context_input = "Search for specific company examples and regulatory actions"

# --- Streamlit App Layout ---
st.set_page_config(page_title="AI Analyst Agent", layout="wide")
st.title("🕵️ AI Analyst Agent Interface")
st.markdown("Enter query, select LLM/Country. API Keys configured on backend.")

if BACKEND_API_URL.startswith("YOUR_"): st.error("Backend API URL needs config.")
else: st.info(f"Targeting Backend API: {BACKEND_API_URL}")

# --- LLM Configuration UI (Sidebar) ---
st.sidebar.title("LLM Selection")
selected_provider_name = st.sidebar.selectbox("Select LLM Provider", options=list(LLM_PROVIDERS.keys()), index=0 )
selected_provider_key = LLM_PROVIDERS[selected_provider_name]
default_model = DEFAULT_MODELS.get(selected_provider_key, "")
session_key_model = f"{selected_provider_key}_model"
if session_key_model not in st.session_state: st.session_state[session_key_model] = default_model
llm_model = st.sidebar.text_input(f"Model Name for {selected_provider_name}", key=session_key_model, help=f"e.g., {default_model}")
st.sidebar.caption("✨ Tip: Google AI & OpenRouter often have free tiers. OpenAI requires paid credits.")


# --- Main Input Form ---
with st.form("analysis_form"):
    # ===> Use on_change for dynamic context update <===
    initial_query = st.text_input(
        "Initial Search Query",
        "Corporate tax evasion cases 2020-2023",
        key="initial_query_input", # Assign key for session state access
        on_change=update_contexts # Call function when input changes
    )
    # ===> END CHANGE <===

    st.subheader("Search Configuration (Optional)")
    col1, col2 = st.columns(2)
    with col1:
        # ===> Use session state keys for dynamic context text areas <===
        global_context = st.text_area(
            "Global Search Context",
            key="global_context_input", # Assign key
            height=100
        )
        specific_context = st.text_area(
            "Specific Search Context",
            key="specific_context_input", # Assign key
            height=100
        )
         # Initialize contexts on first run if not already set
        if "global_context_input" not in st.session_state:
             update_contexts() # Call manually once to set initial state
        # ===> END CHANGE <===
    with col2:
        default_country_index = 0 # Global
        selected_country_name = st.selectbox("Specific Country Search Target", options=COUNTRY_DISPLAY_NAMES, index=default_country_index )
        max_global_results = st.number_input("Max Global Results", min_value=1, max_value=50, value=5)
        max_specific_results = st.number_input("Max Specific Results", min_value=1, max_value=50, value=5)

    submitted = st.form_submit_button("Run Analysis")

# Initialize results in session state to persist across reruns if form not submitted again
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# --- Execution and Output ---
if submitted:
    # Store current selections for API call
    st.session_state.run_query = initial_query
    st.session_state.run_global_context = global_context
    st.session_state.run_specific_context = specific_context
    st.session_state.run_country_name = selected_country_name
    st.session_state.run_max_global = max_global_results
    st.session_state.run_max_specific = max_specific_results
    st.session_state.run_llm_provider = selected_provider_key
    st.session_state.run_llm_model = llm_model
    st.session_state.analysis_results = "RUNNING" # Indicate processing

    # Clear previous results display immediately
    st.rerun() # Rerun to clear old results and show spinner

# --- Display Area (Runs even if form not submitted, uses session state) ---
if st.session_state.analysis_results == "RUNNING":
     with st.spinner("Analysis in progress... Please wait."):
        # Fetch parameters from session state
        payload = {
            "query": st.session_state.run_query,
            "global_context": st.session_state.run_global_context,
            "specific_context": st.session_state.run_specific_context,
            "specific_country": COUNTRY_OPTIONS.get(st.session_state.run_country_name, "us"),
            "max_global": st.session_state.run_max_global,
            "max_specific": st.session_state.run_max_specific,
            "llm_provider": st.session_state.run_llm_provider,
            "llm_model": st.session_state.run_llm_model
        }
        results_data = None
        error_message = None
        try:
            response = requests.post(ANALYZE_ENDPOINT, json=payload, timeout=300)
            if response.status_code == 200:
                results_data = response.json()
            else:
                 error_message = f"Backend API request failed! Status Code: {response.status_code}"
                 try: error_detail = response.json(); error_message += f"\nDetail: {json.dumps(error_detail)}"
                 except json.JSONDecodeError: error_message += f"\nResponse: {response.text[:500]}" # Show snippet
        except requests.exceptions.Timeout: error_message = "Request to backend API timed out."
        except requests.exceptions.RequestException as e: error_message = f"Could not connect to backend API: {e}"
        except Exception as e: error_message = f"Unexpected error during analysis: {e}"

        st.session_state.analysis_results = results_data if results_data else {"error_message": error_message} # Store results or error
        st.rerun() # Rerun again to display results

elif isinstance(st.session_state.analysis_results, dict):
    # Display results stored in session state
    results = st.session_state.analysis_results
    st.subheader("Analysis Summary")
    if results.get("error_message"): # Display error if analysis failed
        st.error(results["error_message"])
    else:
        # ===> Display Supply Chain Exposure Summary <===
        exposures = results.get("supply_chain_exposures", [])
        st.metric("Supply Chain Exposures Found", len(exposures))
        if GOOGLE_SHEET_URL:
             # Link to the specific sheet tab (assuming sheet names match)
             sheet_link = f"{GOOGLE_SHEET_URL}#gid={SHEET_NAME_EXPOSURES}" # Simple link, might need sheet GID if names change
             st.markdown(f"[View Exposure Details in Google Sheet]({sheet_link})", unsafe_allow_html=True)
        else:
             st.caption("Google Sheet link not configured.")

        # Display other metrics
        st.metric("LLM Used", results.get("llm_used", "N/A"))
        st.metric("Total Duration (seconds)", results.get("run_duration_seconds", "N/A"))
        st.metric("KG Update Status", results.get("kg_update_status", "N/A"))
        if results.get("error"): st.error(f"Backend Orchestration Error: {results['error']}")

        # Display Steps
        st.subheader("Run Steps & Durations")
        if results.get("steps"):
            try: steps_df = pd.DataFrame(results["steps"]); st.dataframe(steps_df)
            except Exception as df_e: st.warning(f"Could not display steps table: {df_e}"); st.json(results["steps"])
        else: st.write("No step details.")

        # Display Expanders for detailed data
        with st.expander("Final Extracted Data (Combined)", expanded=False): st.json(results.get("final_extracted_data", {}))
        with st.expander("Identified Supply Chain Exposures (Details)", expanded=False): st.json(exposures) # Show details here too
        with st.expander("Wayback Machine Results", expanded=False): st.json(results.get("wayback_results", []))
        with st.expander("Full Raw Results JSON", expanded=False): st.json(results)

# else: # Initial state before first run
#    st.write("Enter query and click 'Run Analysis'.")