# streamlit_ui.py

import streamlit as st
import json
from datetime import datetime
import os
import requests # To make HTTP requests to the backend

# --- Configuration ---
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "YOUR_RENDER_FASTAPI_URL_HERE") # e.g., "https://my-ai-analyst-full.onrender.com"
ANALYZE_ENDPOINT = f"{BACKEND_API_URL}/analyze"

# ===> ADD COUNTRY MAPPING <===
COUNTRY_MAP = {
    "China": "cn",
    "United States": "us",
    "United Kingdom": "uk",
    "India": "in",
    "Germany": "de",
    "France": "fr",
    "Japan": "jp",
    "Russia": "ru",
    "Brazil": "br",
    "Canada": "ca",
    "Australia": "au",
    # Add more countries as needed
}
COUNTRY_NAMES = list(COUNTRY_MAP.keys())
# ===> END OF ADDITION <===

# --- Streamlit App Layout ---
st.set_page_config(page_title="AI Analyst Agent", layout="wide")
st.title("🕵️ AI Analyst Agent Interface")
st.markdown("Enter your query and context to run the analysis pipeline.")

if BACKEND_API_URL == "YOUR_RENDER_FASTAPI_URL_HERE":
    st.warning("Warning: Backend API URL is not configured. Set the BACKEND_API_URL environment variable on Render.")
else:
    st.info(f"Targeting Backend API: {BACKEND_API_URL}")

# --- Input Fields ---
with st.form("analysis_form"):
    initial_query = st.text_input("Initial Search Query", "financial statements fraud china") # Updated default example

    st.subheader("Configuration (Optional)")
    col1, col2 = st.columns(2)
    with col1:
        global_search_context = st.text_area("Global Search Context", "global financial news and legal filings for compliance issues", height=100)
        specific_search_context = st.text_area("Specific Search Context", "Baidu search in China for specific company supply chain info", height=100)
    with col2:
        # ===> CHANGE TO SELECTBOX <===
        # Default to China if available, otherwise first in list
        default_country_index = COUNTRY_NAMES.index("China") if "China" in COUNTRY_NAMES else 0
        selected_country_name = st.selectbox(
            "Specific Country Name",
            options=COUNTRY_NAMES,
            index=default_country_index,
            help="Select the country for the specific (e.g., Baidu) search."
        )
        # ===> END OF CHANGE <===
        max_global_results = st.number_input("Max Global Results", min_value=1, max_value=50, value=5)
        max_specific_results = st.number_input("Max Specific Results", min_value=1, max_value=50, value=5)

    submitted = st.form_submit_button("Run Analysis")

# --- Execution and Output ---
if submitted:
    if not initial_query:
        st.warning("Please enter an initial search query.")
    elif BACKEND_API_URL == "YOUR_RENDER_FASTAPI_URL_HERE":
        st.error("Cannot run analysis: Backend API URL is not configured.")
    else:
        st.info(f"Sending request to backend: {ANALYZE_ENDPOINT}...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Sending request to backend API...")
            progress_bar.progress(10)

            # --- Prepare request payload ---
            # ===> USE COUNTRY MAP TO GET CODE <===
            specific_country_code_to_send = COUNTRY_MAP.get(selected_country_name, "us") # Default to 'us' if lookup fails
            # ===> END OF CHANGE <===

            payload = {
                "query": initial_query,
                "global_context": global_search_context,
                "specific_context": specific_search_context,
                "specific_country": specific_country_code_to_send, # Send the code
                "max_global": max_global_results,
                "max_specific": max_specific_results
            }
            st.write("Sending payload:", payload) # Display what's being sent

            # --- Call the Backend API ---
            status_text.text("Waiting for backend analysis (can take several minutes)...")
            response = requests.post(ANALYZE_ENDPOINT, json=payload, timeout=300)

            progress_bar.progress(90)
            status_text.text("Processing response from backend...")

            # --- Process Backend Response ---
            if response.status_code == 200:
                results = response.json()
                st.success("Analysis complete!")

                # --- Display Results ---
                st.subheader("Analysis Summary")
                st.metric("Total Duration (seconds)", results.get("run_duration_seconds", "N/A"))
                st.metric("KG Update Status", results.get("kg_update_status", "N/A"))

                if results.get("error"):
                    st.error(f"An error occurred during backend orchestration: {results['error']}")

                st.subheader("Run Steps & Durations")
                if results.get("steps"):
                    import pandas as pd # Import pandas here
                    try:
                        # Attempt to create DataFrame, handle potential errors if format isn't perfect
                        steps_df = pd.DataFrame(results["steps"])
                        st.dataframe(steps_df)
                    except Exception as df_e:
                        st.warning(f"Could not display steps as table: {df_e}")
                        st.json(results["steps"]) # Fallback to JSON display
                else:
                     st.write("No step details available.")

                with st.expander("Final Extracted Data (Combined)", expanded=True): # Expand by default now
                     st.json(results.get("final_extracted_data", {}))

                with st.expander("Wayback Machine Results", expanded=False):
                     st.json(results.get("wayback_results", []))

                with st.expander("Full Raw Results JSON", expanded=False):
                     st.json(results)

            else:
                st.error(f"Backend API request failed!")
                st.metric("Status Code", response.status_code)
                try:
                    error_detail = response.json()
                    st.json(error_detail)
                except json.JSONDecodeError:
                    st.text("Raw error response:")
                    st.code(response.text)

            progress_bar.progress(100)

        except requests.exceptions.Timeout:
             st.error("The request to the backend API timed out (took longer than 5 minutes).")
             status_text.text("Request timed out.")
        except requests.exceptions.RequestException as e:
             st.error(f"Could not connect to the backend API at {ANALYZE_ENDPOINT}.")
             st.error(f"Error details: {e}")
             status_text.text("Connection error.")
        except Exception as e:
            st.error(f"An unexpected error occurred in the Streamlit UI: {e}")
            import traceback
            st.code(traceback.format_exc())
            status_text.text("UI error.")
        finally:
            progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()