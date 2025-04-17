# main_api.py

from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field # Import Field
from typing import Optional
import traceback

# Import the orchestrator function
import orchestrator
# Import config just to log backend config on startup if desired
import config

# Define the FastAPI app
app = FastAPI(title="AI Analyst Agent API")

# Define the request model for the /analyze endpoint
# Remove llm_api_key field
class AnalysisRequest(BaseModel):
    query: str
    global_context: Optional[str] = "global financial news and legal filings"
    specific_context: Optional[str] = "Baidu search in China for specific company supply chain info"
    specific_country: Optional[str] = 'cn'
    max_global: Optional[int] = 10
    max_specific: Optional[int] = 10
    # LLM Selection fields (key is removed)
    llm_provider: str = Field(..., description="LLM provider ('openai', 'google_ai', 'openrouter')")
    llm_model: str = Field(..., description="Specific model name for the selected provider")
    # llm_api_key: str = Field(..., description="API Key for the selected LLM provider") # REMOVED


# Define the main analysis endpoint
@app.post("/analyze")
async def run_analysis_endpoint(request: AnalysisRequest = Body(...)):
    """
    Triggers the AI Analyst analysis pipeline using selected LLM provider/model.
    API Key is read from backend environment variables based on provider.
    """
    print(f"Received analysis request for query: '{request.query}'")
    print(f"Using LLM Config: Provider={request.llm_provider}, Model={request.llm_model}")

    # Basic validation of incoming LLM params
    if not request.llm_provider or not request.llm_model:
         raise HTTPException(status_code=400, detail="Missing required LLM configuration in request (provider, model).")

    try:
        # Call the orchestrator function, passing parameters
        # Note: llm_api_key is NOT passed anymore
        results = orchestrator.run_analysis(
            initial_query=request.query,
            # Pass LLM details (provider/model only)
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
            # Pass other params
            global_search_context=request.global_context,
            specific_search_context=request.specific_context,
            specific_country_code=request.specific_country,
            max_global_results=request.max_global,
            max_specific_results=request.max_specific
        )

        # Check if the orchestrator caught an error
        if "error" in results and results["error"]:
            print(f"Orchestrator returned error: {results['error']}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {results['error']}")

        print("Analysis completed successfully by orchestrator.")
        return results

    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        # Catch unexpected errors
        print(f"--- UNEXPECTED ERROR IN API ENDPOINT ---")
        error_type = type(e).__name__; error_msg = str(e); error_traceback = traceback.format_exc()
        print(f"Error Type: {error_type}\nError Message: {error_msg}\nTraceback:\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Internal server error in API endpoint: {error_type}")

# Define a simple root endpoint for health checks
@app.get("/")
def read_root():
    """Basic health check endpoint."""
    print("Root endpoint accessed.")
    return {"message": "AI Analyst Agent API is running."}

# How to run locally: uvicorn main_api:app --reload --port 8000