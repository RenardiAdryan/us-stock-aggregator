# US Stock News Aggregator

A FastAPI service that uses an agentic LLM pipeline to fetch, evaluate, and summarize recent stock news. It uses GPT-4o-mini to orchestrate tool calls — extracting tickers, building search queries, searching via Tavily, and scoring article relevance — before returning structured results.

---

## How It Works

1. The agent identifies the company and ticker from the user's query.
2. It builds an optimized search query and fetches up to 20 articles from the last 7 days via Tavily.
3. Each article is scored for relevance (threshold: 0.6). At least 3 must pass to finalize.
4. If fewer than 3 pass, the agent retries with rephrased keywords (up to 3 total searches).
5. Passing articles are returned with LLM-written summaries. If no search was needed (general knowledge queries), the LLM answers directly.

---

## Setup

### Prerequisites

- Python 3.13
- OpenAI API key
- Tavily API key ([tavily.com](https://tavily.com))

### Install

```bash
git clone https://github.com/your-username/us-stock-aggregator.git
cd us-stock-aggregator/stock-agent

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### Run

```bash
uvicorn main:app --reload
```

The server starts at `http://localhost:8000`.

---

## API Reference

### `GET /health`

Health check.

**Response**

```json
{ "status": "ok" }
```

---

### `POST /query`

Query for recent stock news or general stock information.

**Request**

```json
{
  "query": "string"
}
```

**Response**

```json
{
  "query": "string",
  "tool_used": "string",
  "result": [
    {
      "title": "string",
      "source": "string",
      "date": "YYYY-MM-DD",
      "summary": "string (max 600 chars)"
    }
  ]
}
```

| Field | Description |
|---|---|
| `tool_used` | `"tavily"` if web search was performed; `"Not Used"` if answered from model knowledge |
| `result` | Array of relevant news items with LLM-written summaries |

**Error**

Returns HTTP `500` with `{ "detail": "error message" }` on failure.

---

## Examples

### Recent earnings news

**Request**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "NVIDIA earnings results this week"}'
```

**Response**
```json
{
  "query": "NVIDIA earnings results this week",
  "tool_used": "tavily",
  "result": [
    {
      "title": "What's Behind The 60% Rise In Nvidia Stock? - Forbes",
      "source": "forbes.com",
      "date": "Mon, 09 Mar 2026 12:48:32 GMT",
      "summary": "(NVDA) reported fiscal Q4 2026 earnings with a revenue of $68.13 billion, a 73.2% increase year-over-year, driven by strong demand in the data center sector. This performance surpassed Wall Street expectations, which were set at $66.19 billion. The results underscore NVIDIA's dominant position in AI infrastructure, and analysts are optimistic about its future growth potential as they anticipate continued demand from hyperscalers. The stock has seen a 60% rise over the past year, reflecting strong investor confidence."
    },
    {
      "title": "'Cheapest of the Bunch,' Says Investor About Nvidia Stock - TipRanks",
      "source": "tipranks.com",
      "date": "Sun, 08 Mar 2026 14:00:00 GMT",
      "summary": "(NVDA) has adopted a new compensation plan for CEO Jensen Huang, setting a target cash bonus of $4 million tied to specific revenue goals for fiscal 2027. This move follows NVIDIA's recent earnings report, which exceeded expectations, showcasing its robust performance in the AI sector. The company's decision to include stock compensation in its financial reporting could set a precedent for other tech firms, potentially impacting how their earnings are perceived. Analysts remain bullish on NVIDIA's prospects, anticipating ongoing demand for AI processors."
    },
    {
      "title": "Nvidia Swears Off an Earnings Crutch, Putting Pressure on Other Tech Companies - WSJ",
      "source": "wsj.com",
      "date": "Wed, 04 Mar 2026 10:30:00 GMT",
      "summary": "(NVDA) is being viewed favorably by analysts, with a consensus Strong Buy rating based on recent earnings that exceeded expectations. Analysts project a price target of $271.89, indicating potential upside of approximately 48.54%. The company is seen as a leader in AI data center infrastructure, and its recent performance has reaffirmed investor confidence. As demand for AI technologies continues to rise, NVIDIA's stock is positioned for further growth."
    }
  ]
}
```

---

### Analyst rating change

**Request**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Apple analyst upgrade or downgrade recent"}'
```

**Response**
```json
{
  "query": "Apple analyst upgrade or downgrade recent",
  "tool_used": "tavily",
  "result": [
    {
      "title": "Sphere 3D Cuts Costs, Upgrades Fleet as It Prepares Merger With Cathedra Bitcoin - TipRanks",
      "source": "tipranks.com",
      "date": "Sat, 07 Mar 2026 06:13:12 GMT",
      "summary": "(AAPL) Apple has recently received a new 'street-high' price target from Wedbush, reflecting growing confidence in the company following its latest product launches. The specific target was not disclosed, but this upgrade indicates that analysts are optimistic about Apple's potential for growth in the near future. The new product announcements, including updated MacBooks, are seen as pivotal in driving sales and maintaining competitive positioning."
    },
    {
      "title": "Here are Wednesday's biggest analyst calls: Nvidia, Apple, Tesla, Oracle, Target, Blue Owl, Toll Brothers & more - CNBC",
      "source": "cnbc.com",
      "date": "Wed, 04 Mar 2026 13:21:06 GMT",
      "summary": "(AAPL) Oppenheimer has reiterated its 'perform' rating for Apple following the company's recent product announcements, which included new MacBook Pro models. The firm expressed confidence in Apple's strategy, particularly the new MacBook Pro offerings, although no specific price target was mentioned. This stability in analyst ratings reflects a cautious optimism regarding Apple's product lineup amidst a competitive market."
    },
    {
      "title": "Here are Monday's biggest analyst calls: Nvidia, Apple, Netflix, Oracle, Micron, Starbucks, Jefferies & more - CNBC",
      "source": "cnbc.com",
      "date": "Mon, 09 Mar 2026 12:37:51 GMT",
      "summary": "(AAPL) Recent analyst calls highlighted that Oppenheimer reiterated its 'perform' rating for Apple, suggesting that the market remains stable in its outlook for the tech giant. This comes after Apple launched new MacBook models, which analysts believe could bolster sales. However, the absence of a price target indicates a more cautious stance, reflecting ongoing market conditions."
    }
  ]
}
```

---

### General knowledge query (no web search)

**Request**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does Tesla do as a business?"}'
```

**Response**
```json
{
  "query": "What does Tesla do as a business?",
  "tool_used": "Not Used",
  "result": [
    {
      "title": "Overview of Tesla's Business Operations",
      "source": "AI Summary",
      "date": "2026-03-09",
      "summary": "Tesla, Inc. (TSLA) is an American company primarily focused on electric vehicles (EVs) and clean energy solutions. Its business encompasses the design, manufacture, and sale of electric cars, including popular models like the Model S, Model 3, Model X, and Model Y. Additionally, Tesla produces energy products such as solar panels and battery storage systems like the Powerwall. The company is also advancing autonomous driving technologies and maintains a global network of charging stations to support its EVs."
    }
  ]
}
```

---

## Project Structure

```
us-stock-aggregator/
└── stock-agent/
    ├── main.py          # FastAPI app and route definitions
    ├── agent.py         # Agentic loop, tool definitions, LLM orchestration
    ├── schema.py        # Pydantic request/response models
    ├── requirements.txt # Python dependencies
    ├── .env.example     # Environment variable template
    └── logs/            # Per-run debug logs (auto-created)
```

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `langchain` + `langchain-openai` | LLM orchestration and OpenAI integration |
| `langchain-tavily` | Tavily web search tool |
| `tavily-python` | Tavily API client |
| `pydantic` | Request/response validation |
| `python-dotenv` | `.env` file loading |
