# streamlit_ui.py

import streamlit as st
import json
from datetime import datetime
import os
import requests # To make HTTP requests to the backend

# --- Configuration ---
# Get the Backend API URL from Environment Variable (best practice for Render)
# Fallback to a placeholder - **REPLACE THIS or set the ENV VAR on Render**
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "YOUR_RENDER_FASTAPI_URL_HERE")
ANALYZE_ENDPOINT = f"{BACKEND_API_URL}/analyze"

# --- Streamlit App Layout ---
st.set_page_config(page_title="AI Analyst Agent", layout="wide")
st.title("🕵️ AI Analyst Agent Interface")
st.markdown("Enter your query and context to run the analysis pipeline.")

# Display the backend URL being targeted (useful for debugging)
if BACKEND_API_URL == "YOUR_RENDER_FASTAPI_URL_HERE":
    st.warning("Warning: Backend API URL is not configured. Set the BACKEND_API_URL environment variable on Render.")
else:
    st.info(f"Targeting Backend API: {BACKEND_API_URL}")

# --- Input Fields ---
with st.form("analysis_form"):
    initial_query = st.text_input("Initial Search Query", "supply chain compliance issues 2023")

    st.subheader("Configuration (Optional)")
    col1, col2 = st.columns(2)
    with col1:
        global_search_context = st.text_area("Global Search Context", "global financial news and legal filings for compliance issues", height=100)
        specific_search_context = st.text_area("Specific Search Context", "Baidu search in China for specific company supply chain info", height=100)
    with col2:
        specific_country_code = st.text_input("Specific Country Code (e.g., cn, us)", "cn")
        max_global_results = st.number_input("Max Global Results", min_value=1, max_value=50, value=5) # Reduced default for faster testing
        max_specific_results = st.number_input("Max Specific Results", min_value=1, max_value=50, value=5) # Reduced default

    submitted = st.form_submit_button("Run Analysis")

# --- Execution and Output ---
if submitted:
    if not initial_query:
        st.warning("Please enter an initial search query.")
    elif BACKEND_API_URL == "YOUR_RENDER_FASTAPI_URL_HERE":
        st.error("Cannot run analysis: Backend API URL is not configured.")
    else:
        st.info(f"Sending request to backend: {ANALYZE_ENDPOINT}...")
        # Use placeholders for progress and status messages
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Sending request to backend API...")
            progress_bar.progress(10)

            # --- Prepare request payload for the FastAPI backend ---
            payload = {
                "query": initial_query,
                "global_context": global_search_context,
                "specific_context": specific_search_context,
                "specific_country": specific_country_code, # Matches FastAPI model field name
                "max_global": max_global_results,       # Matches FastAPI model field name
                "max_specific": max_specific_results    # Matches FastAPI model field name
            }

            # --- Call the Backend API ---
            status_text.text("Waiting for backend analysis (can take several minutes)...")
            # Set a longer timeout as the analysis can take time
            response = requests.post(ANALYZE_ENDPOINT, json=payload, timeout=300) # 300 seconds = 5 minutes

            progress_bar.progress(90)
            status_text.text("Processing response from backend...")

            # --- Process Backend Response ---
            if response.status_code == 200:
                results = response.json() # Get the JSON data returned by the backend
                st.success("Analysis complete!")

                # --- Display Results ---
                st.subheader("Analysis Summary")
                st.metric("Total Duration (seconds)", results.get("run_duration_seconds", "N/A"))
                st.metric("KG Update Status", results.get("kg_update_status", "N/A"))

                if results.get("error"):
                    st.error(f"An error occurred during backend orchestration: {results['error']}")

                st.subheader("Run Steps & Durations")
                if results.get("steps"):
                    # Use st.dataframe for better table-like display
                    import pandas as pd
                    steps_df = pd.DataFrame(results["steps"])
                    st.dataframe(steps_df)
                else:
                    st.write("No step details available.")

                # Use expanders to avoid cluttering the page
                with st.expander("Final Extracted Data (Combined)", expanded=False):
                    st.json(results.get("final_extracted_data", {}))

                with st.expander("Wayback Machine Results", expanded=False):
                    st.json(results.get("wayback_results", []))

                # Expander for the full raw result (for debugging)
                with st.expander("Full Raw Results JSON", expanded=False):
                     st.json(results)

            else:
                # Handle errors from the backend API
                st.error(f"Backend API request failed!")
                st.metric("Status Code", response.status_code)
                try:
                    # Try to display the error detail from the backend's JSON response
                    error_detail = response.json()
                    st.json(error_detail)
                except json.JSONDecodeError:
                    # If the response wasn't JSON, show the raw text
                    st.text("Raw error response:")
                    st.code(response.text)

            progress_bar.progress(100)

        except requests.exceptions.Timeout:
            st.error("The request to the backend API timed out (took longer than 5 minutes). The analysis might still be running on the backend, but the UI stopped waiting.")
            status_text.text("Request timed out.")
        except requests.exceptions.RequestException as e:
             st.error(f"Could not connect to the backend API at {ANALYZE_ENDPOINT}.")
             st.error(f"Error details: {e}")
             status_text.text("Connection error.")
        except Exception as e:
            # Catch any other unexpected errors in the Streamlit code
            st.error(f"An unexpected error occurred in the Streamlit UI: {e}")
            import traceback
            st.code(traceback.format_exc())
            status_text.text("UI error.")
        finally:
            # Ensure progress bar and status text are cleared
            progress_bar.empty()
            if 'status_text' in locals(): # Check if variable exists
                status_text.empty()