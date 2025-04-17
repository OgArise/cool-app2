# streamlit_ui.py

import streamlit as st
import json
from datetime import datetime
import os
import requests

# --- Configuration ---
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
ANALYZE_ENDPOINT = f"{BACKEND_API_URL}/analyze"

# Define LLM Options
LLM_PROVIDERS = {
    "Google AI": "google_ai",
    "OpenAI": "openai",
    "OpenRouter (via OpenAI SDK)": "openrouter"
}
DEFAULT_MODELS = {
    "google_ai": "models/gemini-1.5-flash-latest", # Standard API name
    "openai": "gpt-4o",
    "openrouter": "google/gemini-flash-1.5" # Example
}
PROVIDER_LINKS = { # Keep links for user reference maybe
    "Google AI": "https://aistudio.google.com/app/apikey",
    "OpenAI": "https://platform.openai.com/api-keys",
    "OpenRouter": "https://openrouter.ai/keys"
}

# --- Streamlit App Layout ---
st.set_page_config(page_title="AI Analyst Agent", layout="wide")
st.title("🕵️ AI Analyst Agent Interface")
st.markdown("Enter query/context. Select LLM in sidebar. API Keys configured on backend.") # Updated text

if BACKEND_API_URL == "http://localhost:8000": st.info("Targeting local backend API.")
else: st.info(f"Targeting Backend API: {BACKEND_API_URL}")

# --- LLM Configuration UI (Sidebar) ---
st.sidebar.title("LLM Selection")
selected_provider_name = st.sidebar.selectbox(
    "Select LLM Provider", options=list(LLM_PROVIDERS.keys()), index=0
)
selected_provider_key = LLM_PROVIDERS[selected_provider_name]

default_model = DEFAULT_MODELS.get(selected_provider_key, "")
llm_model = st.sidebar.text_input(
    f"Model Name for {selected_provider_name}",
    value=st.session_state.get(f"{selected_provider_key}_model", default_model),
    help=f"e.g., {default_model}. Ensure backend has API key for {selected_provider_name}."
)

# ===> REMOVE API KEY INPUT FIELD <===
# api_key_input = st.sidebar.text_input(...) # REMOVED
# ===> END REMOVE <===

# Update session state for model name only
st.session_state[f"{selected_provider_key}_model"] = llm_model

# --- Main Input Form ---
with st.form("analysis_form"):
    initial_query = st.text_input("Initial Search Query", "financial statements fraud china")
    st.subheader("Search Configuration (Optional)")
    col1, col2 = st.columns(2)
    with col1:
        global_search_context = st.text_area("Global Search Context", "global financial news and legal filings for compliance issues", height=100)
        specific_search_context = st.text_area("Specific Search Context", "Baidu search in China for specific company supply chain info", height=100)
    with col2:
        COUNTRY_MAP = {"China": "cn", "United States": "us", "United Kingdom": "uk", "India": "in", "Germany": "de", "France": "fr", "Japan": "jp", "Russia": "ru", "Brazil": "br", "Canada": "ca", "Australia": "au"}
        COUNTRY_NAMES = list(COUNTRY_MAP.keys()); default_country_index = COUNTRY_NAMES.index("China") if "China" in COUNTRY_NAMES else 0
        selected_country_name = st.selectbox("Specific Country Name", options=COUNTRY_NAMES, index=default_country_index)
        max_global_results = st.number_input("Max Global Results", min_value=1, max_value=50, value=5)
        max_specific_results = st.number_input("Max Specific Results", min_value=1, max_value=50, value=5)

    submitted = st.form_submit_button("Run Analysis")

# --- Execution and Output ---
if submitted:
    if not initial_query: st.warning("Please enter an initial search query.")
    elif not selected_provider_key or not llm_model: st.warning("Please select Provider/Model in sidebar.")
    elif BACKEND_API_URL.startswith("YOUR_"): st.error("Backend API URL is not configured.")
    else:
        st.info(f"Sending request to backend ({ANALYZE_ENDPOINT})...")
        st.write(f"Using LLM: {selected_provider_name} - Model: {llm_model}")
        progress_bar = st.progress(0); status_text = st.empty()
        try:
            status_text.text("Sending request to backend API..."); progress_bar.progress(10)
            specific_country_code_to_send = COUNTRY_MAP.get(selected_country_name, "us")
            # ===> REMOVE llm_api_key from Payload <===
            payload = {
                "query": initial_query,
                "global_context": global_search_context,
                "specific_context": specific_search_context,
                "specific_country": specific_country_code_to_send,
                "max_global": max_global_results,
                "max_specific": max_specific_results,
                "llm_provider": selected_provider_key,
                "llm_model": llm_model
                # "llm_api_key": current_api_key # REMOVED
            }
            # ===> END REMOVE <===
            # st.write("Sending payload:", payload) # Debug if needed

            status_text.text("Waiting for backend analysis..."); progress_bar.progress(20)
            response = requests.post(ANALYZE_ENDPOINT, json=payload, timeout=300)

            progress_bar.progress(90); status_text.text("Processing response...")

            # --- Process Backend Response (same as before) ---
            if response.status_code == 200:
                results = response.json(); st.success("Analysis complete!")
                st.subheader("Analysis Summary")
                st.metric("LLM Used", results.get("llm_used", f"{selected_provider_name} ({llm_model})"))
                st.metric("Total Duration (seconds)", results.get("run_duration_seconds", "N/A"))
                st.metric("KG Update Status", results.get("kg_update_status", "N/A"))
                if results.get("error"): st.error(f"Backend Orchestration Error: {results['error']}")
                st.subheader("Run Steps & Durations");
                if results.get("steps"):
                    import pandas as pd
                    try: steps_df = pd.DataFrame(results["steps"]); st.dataframe(steps_df)
                    except Exception as df_e: st.warning(f"Could not display steps as table: {df_e}"); st.json(results["steps"])
                else: st.write("No step details available.")
                with st.expander("Final Extracted Data (Combined)", expanded=True): st.json(results.get("final_extracted_data", {}))
                with st.expander("Wayback Machine Results", expanded=False): st.json(results.get("wayback_results", []))
                with st.expander("Full Raw Results JSON", expanded=False): st.json(results)
            else:
                st.error(f"Backend API request failed!"); st.metric("Status Code", response.status_code)
                try: error_detail = response.json(); st.json(error_detail)
                except json.JSONDecodeError: st.text("Raw error response:"); st.code(response.text)
            progress_bar.progress(100)

        except requests.exceptions.Timeout: st.error("Request to backend API timed out."); status_text.text("Request timed out.")
        except requests.exceptions.RequestException as e: st.error(f"Could not connect to backend API: {e}"); status_text.text("Connection error.")
        except Exception as e: st.error(f"Streamlit UI error: {e}"); import traceback; st.code(traceback.format_exc()); status_text.text("UI error.")
        finally:
            progress_bar.empty()
            if 'status_text' in locals(): status_text.empty()