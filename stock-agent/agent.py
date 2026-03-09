import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool

from schema import NewsItem, QueryResponse

load_dotenv()

_DIRECT_ANSWER_SCHEMA = {
    "title": "answer",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": (
                "Brief, specific title derived from the actual content of the answer. "
                "Include the ticker symbol or company name only when central to the topic. "
                "Focus on the key fact or concept, e.g. 'Apple Q4 2024 Earnings Beat Estimates' "
                "or 'What Is a P/E Ratio?'. Max 100 characters."
            ),
        },
        "summary": {
            "type": "string",
            "description": (
                "Detailed factual answer in 3-5 sentences. Include the ticker symbol, "
                "key numbers (price, % change, revenue, EPS, market cap), the core takeaway, "
                "context about why it matters, and any analyst reactions or forward outlook if available. "
                "Max 600 characters."
            ),
        },
    },
    "required": ["title", "summary"],
}


@tool
def record_evaluation(index: int, relevance_score: float, reason: str, summary: str) -> dict:
    """Record the relevance evaluation for a single news article.

    Args:
        index: Original index of the news article in the input list.
        relevance_score: Relevance score from 0.0 to 1.0. Only call this for articles >= 0.6.
        reason: One-sentence reason why this article is relevant to the query.
        summary: Detailed 3-5 sentence summary of the article. Must include the ticker symbol,
            key figures (price, % change, revenue, EPS, market cap, or other metrics if available),
            the core news event, why it matters for the stock, and any analyst reactions or
            forward outlook mentioned. Max 600 characters.
    """
    return {"index": index, "relevance_score": relevance_score, "reason": reason, "summary": summary}

@tool
def extract_ticker(company: str, ticker_symbol: str) -> dict:
    """Extract and confirm the company name and ticker symbol from the user query.
    Always call this first before any other tool, whenever a company or stock is mentioned."""
    return {"company": company, "ticker_symbol": ticker_symbol}


@tool
def build_search_query(
    company: Annotated[str, "Full company name"],
    ticker_symbol: Annotated[str, "Stock ticker symbol"],
    intent: Annotated[str, "User's query intent, e.g. 'earnings report', 'price movement', 'analyst rating', 'merger', 'IPO', 'recent news'"],
    keywords: Annotated[list[str], "Specific financial keywords relevant to the query, e.g. ['Q4 earnings', 'beat estimates', 'revenue']"],
) -> str:
    """Build an optimized search query for stock news. Always call this before search_stock_news."""
    keyword_part = " ".join(keywords) if keywords else ""
    return " ".join(filter(None, [company, ticker_symbol, intent, keyword_part]))


@tool
def search_stock_news(
    query: Annotated[str, "Optimized search query built by build_search_query"],
) -> list[dict]:
    """Search for recent stock news about a company. Use this for time-sensitive topics:
    recent headlines, price movements, earnings updates, analyst ratings, IPOs, mergers,
    acquisitions, or regulatory events. Always call build_search_query first to build the query."""

    today = date.today()
    start_date = (today - timedelta(days=7)).isoformat()
    end_date = today.isoformat()
    search_tool = TavilySearch(
        max_results=20,
        topic="news",
        start_date=start_date,
        end_date=end_date,
    )
    today = end_date
    raw = search_tool.invoke(query)
    results: list[dict] = raw.get("results", []) if isinstance(raw, dict) else (json.loads(raw) if isinstance(raw, str) else [])

    news_items = []
    for r in results:
        if not isinstance(r, dict):
            continue
        content = r.get("content", "")
        title = r.get("title") or (content.splitlines()[0] if content else "")
        try:
            netloc = urlparse(r.get("url", "")).netloc
            source = netloc.removeprefix("www.") or "unknown"
        except Exception:
            source = "unknown"
        news_items.append(
            {
                "title": title,
                "source": source,
                "date": r.get("published_date") or today,
                "summary": content,
            }
        )
    
    return news_items


class StockNewsAgent:
    def _system_prompt(self) -> str:
        today = date.today().isoformat()
        return f"""You are a stock news assistant with access to a real-time web search tool.
Today's date is {today}. Focus on news and data from the last 7 days when available.

Your goal is to answer user queries about stocks and companies.

Steps to follow:
1. If a company or stock is mentioned, ALWAYS call extract_ticker first to identify the company
   and ticker symbol before doing anything else.
2. If the query involves time-sensitive information (recent news, price movements, earnings,
   analyst ratings, IPOs, mergers, acquisitions, regulatory events) → call build_search_query
   to construct an optimized query, then pass the result to search_stock_news.
3. After search_stock_news returns results, evaluate each article and call record_evaluation for
   every article that scores >= 0.6 relevance.

   Criteria:
    - The article title or summary must explicitly mention the company name (e.g. "Google", "Alphabet")
        OR the ticker symbol (e.g. "GOOGL"). If neither appears, score it 0 and do NOT call record_evaluation.
    - The article must be directly about the company/ticker — not just a passing mention or unrelated company
    - Published within the last 7 days
    - Relevant to the user's query intent (news, price, earnings, analyst rating, etc.)

   When calling record_evaluation, write a detailed 3-5 sentence summary that:
    - Leads with the ticker symbol in parentheses, e.g. "(AAPL)"
    - States the key event or development clearly with specific numbers (price, % move, revenue, EPS, beat/miss margin)
    - Explains why the event matters for the stock or company
    - Includes analyst reactions, price targets, or ratings changes if mentioned
    - Covers any forward outlook, guidance, or next catalyst if available
    - Omits filler like "the article says" or "according to reports"
    Example: "(NVDA) reported Q4 revenue of $39.3B (+78% YoY), beating estimates by $1.2B, driven by surging data center demand. EPS came in at $0.89, above the $0.84 consensus. Management guided Q1 revenue to $43B, ahead of expectations. Analysts widely raised price targets, with several citing AI infrastructure spending as a sustained tailwind."

   Do NOT call record_evaluation for articles that score below 0.6.
   If fewer than 3 articles pass evaluation, call build_search_query again with different keywords
   that are still closely related to the original query intent — rephrase or expand the keywords
   (e.g. if the original intent was "recent news", try synonyms like "latest update", "announcement",
   "press release"; if it was "earnings", try "revenue", "profit", "quarterly results"). Do NOT
   switch to unrelated financial topics. Then call search_stock_news with the new query, evaluate
   the results the same way, and call record_evaluation for any that qualify. You may retry up to
   2 additional times. All passing articles across all searches are combined into the final result.
   
4. If the query is about general knowledge (definitions, historical facts, investing concepts)
   → after extract_ticker, answer the query directly without searching.

Important: Your response must be highly relevant to the specific stock or ticker mentioned in the query.
Only include information that directly relates to that company or its stock. Do not include unrelated news or generic market commentary."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.llm = ChatOpenAI(model=model, temperature=0.5)
        self.tools = [extract_ticker, build_search_query, search_stock_news, record_evaluation]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.llm_structured = self.llm.with_structured_output(_DIRECT_ANSWER_SCHEMA)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        self._log_file = log_dir / f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # ------------------------------------------------------------------ #
    # DEBUG — comment out the body of this method to silence all logging  #
    # ------------------------------------------------------------------ #
    def _debug(self, label: str, data=None) -> None:
        SEP = "=" * 60
        lines = [f"\n{SEP}", f"[DEBUG] {label}"]
        if data is not None:
            if isinstance(data, (dict, list)):
                lines.append(json.dumps(data, indent=2, default=str, ensure_ascii=False))
            else:
                lines.append(str(data))
        lines.append(SEP)
        output = "\n".join(lines)
        # print(output)
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(output + "\n")

    def run(self, query: str) -> QueryResponse:
        self._debug("START", {"query": query})

        messages = [
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content=query),
        ]

        tool_used = "Not Used"
        news_items: list[dict] = []
        kept_indices: set[int] = set()
        article_summaries: dict[int, str] = {}
        search_count = 0
        iteration = 0

        # Agent loop: LLM autonomously decides when to stop calling tools
        while True:
            iteration += 1
            self._debug(f"ITERATION {iteration} — invoking LLM")

            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            # No tool calls means the LLM is done reasoning
            if not response.tool_calls:
                self._debug(f"ITERATION {iteration} — LLM finished (no tool calls)", {
                    "content": response.content,
                })
                break

            self._debug(f"ITERATION {iteration} — tool calls requested", [
                {"name": tc["name"], "args": tc["args"]} for tc in response.tool_calls
            ])

            had_evaluation = any(tc["name"] == "record_evaluation" for tc in response.tool_calls)

            # Execute every tool call the LLM requested
            for tool_call in response.tool_calls:
                name = tool_call["name"]

                if name == "extract_ticker":
                    tool_result = extract_ticker.invoke(tool_call["args"])
                    self._debug("extract_ticker → result", tool_result)

                elif name == "build_search_query":
                    tool_result = build_search_query.invoke(tool_call["args"])
                    self._debug("build_search_query → query", tool_result)
                    

                elif name == "search_stock_news":
                    tool_result = search_stock_news.invoke(tool_call["args"])
                    news_items.extend(tool_result)
                    tool_used = "tavily"
                    search_count += 1
                    self._debug(f"search_stock_news #{search_count} → {len(tool_result)} articles", [
                        {"index": len(news_items) - len(tool_result) + i, "title": a.get("title"), "source": a.get("source"), "date": a.get("date")}
                        for i, a in enumerate(tool_result)
                    ])

                elif name == "record_evaluation":
                    args = tool_call["args"]
                    passed = args.get("relevance_score", 0) >= 0.6
                    if passed:
                        idx = args["index"]
                        kept_indices.add(idx)
                        if args.get("summary"):
                            article_summaries[idx] = args["summary"]
                    tool_result = record_evaluation.invoke(args)
                    self._debug(f"record_evaluation index={args.get('index')} score={args.get('relevance_score')} {'✓ KEPT' if passed else '✗ SKIPPED'}", {
                        "reason": args.get("reason"),
                        "summary": args.get("summary"),
                    })

                else:
                    tool_result = {}

                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

            # After a round of evaluations, tell the LLM how many passed so far
            if had_evaluation:
                feedback = (
                    f"Evaluation complete. {len(kept_indices)} article(s) passed so far (minimum 3 required). "
                    f"You have used {search_count} search(es) and may retry up to {3 - search_count} more time(s) with different keywords related to the original query."
                    if len(kept_indices) < 3 and search_count < 3
                    else f"Evaluation complete. {len(kept_indices)} article(s) passed. Minimum met — you may finalize."
                )
                messages.append(HumanMessage(content=feedback))
                self._debug("evaluation round summary", {
                    "passed_so_far": len(kept_indices),
                    "searches_used": search_count,
                    "feedback_sent": feedback,
                })


        if news_items:
            self._debug("BUILDING RESULT from search", {
                "total_fetched": len(news_items),
                "kept_indices": sorted(kept_indices),
                "llm_summaries_available": sorted(article_summaries.keys()),
            })
            evaluated = []
            for i in sorted(kept_indices):
                if i < len(news_items):
                    item = dict(news_items[i])
                    if i in article_summaries:
                        item["summary"] = article_summaries[i]
                    evaluated.append(NewsItem(**item))
            result = evaluated
        else:
            # LLM answered directly — use structured output to enforce fixed schema
            self._debug("BUILDING RESULT from direct LLM answer (no search)")
            parsed: dict = self.llm_structured.invoke(messages)
            self._debug("structured output", parsed)
            # structured output may nest values under "properties" depending on model behavior
            data = parsed if parsed.get("title") or parsed.get("summary") else parsed.get("properties", parsed)
            result = [
                NewsItem(
                    title=data.get("title", query),
                    source="AI Summary",
                    date=date.today().isoformat(),
                    summary=data.get("summary", ""),
                )
            ]

        response = QueryResponse(query=query, tool_used=tool_used, result=result)
        self._debug("FINAL RESPONSE", response.model_dump())
        return response


def run_agent(query: str) -> QueryResponse:
    return StockNewsAgent().run(query)
