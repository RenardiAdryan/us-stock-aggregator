from fastapi import FastAPI, HTTPException

from schema import QueryRequest, QueryResponse
from agent import run_agent

app = FastAPI(title="Stock News Agent")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        return run_agent(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
