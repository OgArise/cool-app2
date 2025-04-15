from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
import orchestrator
from typing import Optional

app = FastAPI(title="AI Analyst Agent API")

class AnalysisRequest(BaseModel):
    query: str
    global_context: Optional[str] = "global financial news and legal filings"
    specific_context: Optional[str] = "Baidu search in China for specific company supply chain info"
    specific_country: Optional[str] = 'cn'
    max_global: Optional[int] = 10
    max_specific: Optional[int] = 10

@app.post("/analyze")
async def run_analysis_endpoint(request: AnalysisRequest = Body(...)):
    """
    Triggers the AI Analyst analysis pipeline.
    """
    try:
        results = orchestrator.run_analysis(
            initial_query=request.query,
            global_search_context=request.global_context,
            specific_search_context=request.specific_context,
            specific_country_code=request.specific_country,
            max_global_results=request.max_global,
            max_specific_results=request.max_specific
        )
        if "error" in results:
             # Return a server error if the orchestrator caught one
             raise HTTPException(status_code=500, detail=f"Analysis failed: {results['error']}")
        return results
    except Exception as e:
        # Catch any unexpected errors during the API call itself
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Add a root endpoint for basic check
@app.get("/")
def read_root():
    return {"message": "AI Analyst Agent API is running."}

# To run (similar to the previous FastAPI service):
# uvicorn main_api:app --reload