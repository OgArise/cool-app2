# main_api.py

from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import traceback
import asyncio # Import asyncio
import search_engines # Import search_engines to access check_linkup_balance

import orchestrator
import config

app = FastAPI(title="AI Analyst Agent API")

class AnalysisRequest(BaseModel):
    query: str
    global_context: Optional[str] = "global financial news and legal filings"
    specific_context: Optional[str] = "search for specific company examples and details"
    specific_country: Optional[str] = 'us'
    max_global: Optional[int] = 20
    max_specific: Optional[int] = 20
    llm_provider: str = Field(..., description="LLM provider ('openai', 'google_ai', 'openrouter')")
    llm_model: str = Field(..., description="Specific model name for the selected provider")

# Define a response model that matches the subset of data we will return to the UI
class AnalysisResponse(BaseModel):
    query: str
    llm_used: str
    run_duration_seconds: Optional[float]
    analysis_summary: str
    kg_update_status: str
    high_risk_exposures: List[Dict[str, Any]] # Keep the full list of exposures for display
    backend_error: Optional[str] = None
    steps: List[Dict[str, Any]] # Keep steps for debugging flow
    # Add counts for other data types
    extracted_data_counts: Dict[str, int]
    linkup_structured_data_count: int
    wayback_results_count: int

# Add an async startup event handler to check Linkup balance
@app.on_event("startup")
async def startup_event():
    print("\n--- Running API Startup Tasks ---")
    if search_engines.linkup_library_available and search_engines.linkup_search_enabled:
        print("Checking Linkup balance during API startup...")
        try:
            balance = await search_engines.check_linkup_balance()
            if balance is not None:
                print(f"Linkup Available Credits: {balance:.4f}")
            else:
                print("Could not retrieve Linkup credit balance.")
        except Exception as e:
            print(f"Error during Linkup balance check at startup: {e}")
            traceback.print_exc()
    else:
        print("Linkup search not enabled or library not available. Skipping balance check.")
    print("--- API Startup Tasks Complete ---\n")


# Convert the endpoint function to async and await the orchestrator call
@app.post("/analyze", response_model=AnalysisResponse)
async def run_analysis_endpoint(request: AnalysisRequest = Body(...)):
    """
    Triggers the AI Analyst analysis pipeline using selected LLM provider/model (async).
    API Key is read from backend environment variables based on provider.
    Returns a summary of the analysis results, including exposures, but not all raw extracted data.
    """
    print(f"Received analysis request for query: '{request.query}'")
    print(f"LLM Config received from UI: Provider=request.llm_provider, Model=request.llm_model")

    if not request.llm_provider or not request.llm_model:
         raise HTTPException(status_code=400, detail="Missing required LLM configuration in request (provider, model).")

    try:
        # Call the orchestrator to run the full analysis pipeline and await its completion
        full_results = await orchestrator.run_analysis(
            initial_query=request.query,
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
            global_search_context=request.global_context,
            specific_search_context=request.specific_context,
            specific_country_code=request.specific_country,
            max_global_results=request.max_global,
            max_specific_results=request.max_specific
        )

        # --- Prepare the subset of results to return to the UI ---
        # Extract necessary data from the full results dictionary
        # These counts now reflect the filtered data from orchestrator's Step 5
        extracted_data_counts = {
            "entities": len(full_results.get("final_extracted_data", {}).get("entities", [])),
            "risks": len(full_results.get("final_extracted_data", {}).get("risks", [])),
            "relationships": len(full_results.get("final_extracted_data", {}).get("relationships", [])),
        }

        # Create the response dictionary
        response_data = {
            "query": full_results.get("query", request.query),
            "llm_used": full_results.get("llm_used", f"{request.llm_provider} ({request.llm_model})"),
            "run_duration_seconds": full_results.get("run_duration_seconds"),
            "analysis_summary": full_results.get("analysis_summary", "Summary could not be generated or analysis failed early."), # Default message matches UI expectation
            "kg_update_status": full_results.get("kg_update_status", "not_run"),
            "high_risk_exposures": full_results.get("high_risk_exposures", []), # Return the list of exposures (which is now the filtered list)
            "backend_error": full_results.get("error"), # Return the backend error message if any
            "steps": full_results.get("steps", []), # Return the steps details
            "extracted_data_counts": extracted_data_counts, # Return counts (now filtered counts)
            "linkup_structured_data_count": len(full_results.get("linkup_structured_data", [])), # Return count of raw structured items
            "wayback_results_count": len(full_results.get("wayback_results", [])), # Return count of wayback results (limited list)
        }

        # If the orchestrator reported an error, return a 500 HTTP status code
        # and include the response data in the body.
        if response_data.get("backend_error") and response_data["backend_error"] != "None" and response_data["backend_error"] != "":
            print(f"Orchestrator reported error: {response_data['backend_error']}. Returning 500.")
            # Create a custom error response body that includes the results summary
            error_response_body = {
                 "detail": f"Analysis failed on backend: {response_data['backend_error']}",
                 "results_summary": response_data # Include the partial/summary results in the error body
            }
            # Note: Returning a custom body requires not using response_model directly for the HTTPException.
            # The Streamlit UI async handler is designed to parse this.
            raise HTTPException(status_code=500, detail=error_response_body)


        print("Analysis completed successfully by orchestrator. Returning subset of results.")
        # Return the subset of results as defined by the AnalysisResponse model
        return AnalysisResponse(**response_data)

    except HTTPException as http_exc:
         # Re-raise HTTPExceptions raised internally (e.g. from missing LLM config)
         raise http_exc
    except Exception as e:
        print(f"--- UNEXPECTED ERROR IN API ENDPOINT ---")
        error_type = type(e).__name__
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"Error Type: {error_type}\nError Message: {error_msg}\nTraceback:\n{error_traceback}")
        # For unexpected errors, return a generic 500 with traceback info
        # Note: This error won't contain 'results_summary' because the error happened outside the orchestrator's try/finally
        raise HTTPException(status_code=500, detail=f"Internal server error during analysis: {error_type} - {error_msg}\nTraceback: {error_traceback[:500]}...") # Limit traceback length


@app.get("/")
def read_root():
    """Basic health check endpoint."""
    print("Root endpoint accessed.")
    return {"message": "AI Analyst Agent API is running."}

if __name__ == "__main__":
    import uvicorn
    print("Running FastAPI application with uvicorn...")
    # Uvicorn automatically handles running the @app.on_event("startup") async function
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)