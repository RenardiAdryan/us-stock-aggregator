from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about a stock or company")


class NewsItem(BaseModel):
    title: str = Field(..., description="Headline of the news article")
    source: str = Field(..., description="Publisher or domain of the article")
    date: str = Field(..., description="Publication date in YYYY-MM-DD format")
    summary: str = Field(..., description="Detailed summary of the article, max 600 chars")


class ToolDecision(BaseModel):
    use_tool: bool = Field(..., description="True if a live web search is needed to answer the query")
    reason: str = Field(..., description="One-sentence justification for the decision")


class CompanyExtract(BaseModel):
    company: str = Field(..., description="Full company name e.g. Nvidia Corporation")
    ticker_symbol: str = Field(..., description="Ticker symbol e.g. NVDA")


class QueryResponse(BaseModel):
    query: str
    tool_used: str
    result: list[NewsItem]
