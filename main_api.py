# main_api.py

from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback # For logging unexpected errors

# Import the orchestrator function
import orchestrator

# Define the FastAPI app
app = FastAPI(title="AI Analyst Agent API")

# Define the request model for the /analyze endpoint
class AnalysisRequest(BaseModel):
    query: str
    global_context: Optional[str] = "global financial news and legal filings"
    specific_context: Optional[str] = "Baidu search in China for specific company supply chain info"
    specific_country: Optional[str] = 'cn' # Note: field name matches Pydantic, passed to orchestrator
    max_global: Optional[int] = 10
    max_specific: Optional[int] = 10

# Define the main analysis endpoint
@app.post("/analyze")
async def run_analysis_endpoint(request: AnalysisRequest = Body(...)):
    """
    Triggers the AI Analyst analysis pipeline via the orchestrator.
    """
    print(f"Received analysis request for query: '{request.query}'")
    try:
        # Call the orchestrator function with parameters from the request body
        results = orchestrator.run_analysis(
            initial_query=request.query,
            global_search_context=request.global_context,
            specific_search_context=request.specific_context,
            specific_country_code=request.specific_country, # Map request field to function arg name
            max_global_results=request.max_global,           # Map request field to function arg name
            max_specific_results=request.max_specific          # Map request field to function arg name
        )

        # Check if the orchestrator caught an error and included it in the results
        if "error" in results and results["error"]:
            # Log the specific error received from the orchestrator to backend logs
            print(f"Orchestrator returned error: {results['error']}")
            # Raise HTTPException using the detailed error message from the orchestrator
            # This error will be sent back to the client (e.g., Streamlit UI)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {results['error']}")

        # If no error was found in results, return the successful results
        print("Analysis completed successfully by orchestrator.")
        return results

    except HTTPException as http_exc:
         # Re-raise HTTPExceptions directly (like the one we raise above)
         raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during API call setup or within the orchestrator itself
        # if the orchestrator's try/except failed somehow (should be rare)
        print(f"--- UNEXPECTED ERROR IN API ENDPOINT ---")
        error_type = type(e).__name__
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"Error Type: {error_type}")
        print(f"Error Message: {error_msg}")
        print(f"Traceback:\n{error_traceback}")
        # Return a generic internal server error to the client
        raise HTTPException(status_code=500, detail=f"Internal server error in API endpoint: {error_type}")

# Define a simple root endpoint for health checks
@app.get("/")
def read_root():
    """Basic health check endpoint."""
    print("Root endpoint accessed.")
    return {"message": "AI Analyst Agent API is running."}

# --- How to run locally (for testing) ---
# In terminal: uvicorn main_api:app --reload --port 8000