# main_api.py

from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import traceback

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

@app.post("/analyze")
async def run_analysis_endpoint(request: AnalysisRequest = Body(...)):
    """
    Triggers the AI Analyst analysis pipeline using selected LLM provider/model.
    API Key is read from backend environment variables based on provider.
    """
    print(f"Received analysis request for query: '{request.query}'")
    print(f"LLM Config received from UI: Provider={request.llm_provider}, Model={request.llm_model}")

    if not request.llm_provider or not request.llm_model:
         raise HTTPException(status_code=400, detail="Missing required LLM configuration in request (provider, model).")

    try:
        results = orchestrator.run_analysis(
            initial_query=request.query,
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
            global_search_context=request.global_context,
            specific_search_context=request.specific_context,
            specific_country_code=request.specific_country,
            max_global_results=request.max_global,
            max_specific_results=request.max_specific
        )

        if "error" in results and results["error"]:
            print(f"Orchestrator returned error: {results['error']}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {results['error']}")

        print("Analysis completed successfully by orchestrator.")
        return results

    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        print(f"--- UNEXPECTED ERROR IN API ENDPOINT ---")
        error_type = type(e).__name__; error_msg = str(e); error_traceback = traceback.format_exc()
        print(f"Error Type: {error_type}\nError Message: {error_msg}\nTraceback:\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Internal server error in API endpoint: {error_type}")

@app.get("/")
def read_root():
    """Basic health check endpoint."""
    print("Root endpoint accessed.")
    return {"message": "AI Analyst Agent API is running."}

if __name__ == "__main__":
    import uvicorn
    print("Running FastAPI application with uvicorn...")
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)