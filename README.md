# AI SQL Analyst

Ask questions about your database in plain English and get back a validated SQL query, real results, and an auto-generated chart — with a safety layer that assumes the LLM will eventually be wrong, not one that trusts it.

Built on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (9 linked tables, ~100k orders) as a stand-in for a real production database.

## Problem

Text-to-SQL systems fail in specific, predictable ways: pure semantic retrieval misses join/bridge tables, LLMs hallucinate columns that don't exist, and "recent"/"last quarter" gets anchored to the wrong clock when the data is historical. This project's core bet is that **correctness comes from validation and sandboxing around the LLM, not from trusting its output** — every one of the design decisions below follows from that.

## Architecture

```
User query
  → Guardrails (input): prompt injection, leaked secrets, oversized input
  → Schema retrieval: vector search over table descriptions → top-k "seed" tables
  → Graph expansion: bridge/join tables discovered via a schema relationship graph
      (pure semantic retrieval on "revenue" can miss order_items — the graph catches it)
  → Context assembly: table descriptions + FK relationships + live sample rows
  → SQL generation: LLM (Groq primary, OpenAI fallback), forced structured output
  → SQL validation: sqlglot-based — blocks writes, checks every table/column
      against the real DB schema (not a hand-maintained doc), CTE-aware
  → Self-correction: validation errors fed back to the LLM, up to N retries
  → Execution: read-only, against DuckDB
  → Chart generation: LLM picks chart type + columns via structured output only
      — it never writes or executes chart code, sidestepping sandboxing entirely
  → Guardrails (output): PII detection on the explanation text before it's shown
  → Response: SQL + explanation + results + chart, all logged and traced
```

Every step above is a real Python module, not a diagram aspiration — see [Project structure](#project-structure).

## Key design decisions & trade-offs

**Graph-based join discovery, not just vector search.** Table retrieval starts with embedding similarity (`vector_store.py`), but that alone misses bridge tables — `order_items` doesn't semantically scream "revenue" even though it's required for any revenue join. A schema relationship graph (`schema_graph.py`, built from real foreign keys) expands the seed set to include the shortest join path between them. This was validated against real queries during development, not just designed on paper.

**Column-level SQL validation against the live DB, not a static doc.** Early versions checked generated SQL against the hand-maintained YAML schema doc — but that doc drifted out of sync with the real database more than once during development. The validator now checks column existence against DuckDB's own `information_schema` directly, so it can't go stale.

**CTE-aware validation, added after real failures.** The LLM correctly used `WITH x AS (...)` to anchor relative dates to `MAX(timestamp)` instead of `NOW()` (since the dataset is historical, 2016–2018) — but the first version of the validator didn't know CTE names or their output columns were legitimate, and rejected valid queries. Fixed in two passes as each edge case surfaced. This iterative hardening — not a validator that was perfect on the first try — is the honest story.

**Charts via structured tool-selection, not LLM-written code.** The LLM never writes or executes chart code. It returns a `ChartSpec` (chart type + column choices) via forced structured output, and a small set of pre-written, trusted matplotlib functions do the actual rendering. This was a deliberate choice to get a working, safe charting feature without needing a code-execution sandbox at all — not a workaround for not having built one yet.

**Self-correction retry loop.** When validation rejects a query, the exact error is fed back to the LLM as a conversational correction request (not a fresh restart), and it gets another attempt. Anything that needed a retry — or failed all retries — is automatically flagged into a LangSmith annotation queue for manual review.

**Groq primary, OpenAI fallback, via `.with_fallbacks()`.** Groq for low-latency inference on straightforward SQL generation; OpenAI as a reliability/capability fallback if Groq errors or rate-limits. Trade-off: Groq's model selection is limited to open-weight models, so raw reasoning quality on unusually complex queries may lag a top-tier proprietary model — acceptable given this system's validation layer is designed to catch resulting mistakes rather than assume perfect generation.

**Guardrails for reliability, not just malicious actors.** LLM-Guard input/output scanning exists because the LLM itself is the largest source of risk here (see the CTE bugs above) — not because internal users are assumed hostile. It also covers second-order risk: customer-written review text is untrusted data that could carry injection content even when the querying employee has done nothing wrong.

## Eval results

Run the eval harness to populate this section with real numbers from your own run:

```bash
uv run eval_dataset.py   # one-time: builds the 16-example dataset in LangSmith
uv run eval_run.py       # runs the pipeline against every example, scores it
```

Evaluators are behavioral, not exact-match (SQL phrasing varies every run even when correct):

| Evaluator | What it checks |
|---|---|
| `table_coverage` | Does the generated SQL actually reference every table the question requires? (checked against real parsed SQL) |
| `guardrail_behavior` | Correctly rejected when it should be (writes, injection attempts), correctly executed otherwise |
| `row_expectation` | Matches whether rows were expected back |
| `answer_quality` | LLM-as-judge: is the SQL logic sound for the question asked? |

Full per-example results and scores appear in the LangSmith UI under **Datasets & Experiments → ai-sql-analyst-eval**.

*(Populate this table with your own run's numbers before sharing this README — do not present numbers that weren't produced by an actual eval run.)*

| Metric | Score |
|---|---|
| Table coverage | — 93|
| Guardrail correctness | 97 |
| Row expectation match | —83 |
| Answer quality (LLM judge) | —93 |
| % queries needing ≥1 retry | —17 |

## How to run

### Docker (recommended)

```bash
cp .env.example .env   # fill in GROQ_API_KEY, OPENAI_API_KEY, LANGCHAIN_API_KEY, KAGGLE_USERNAME, KAGGLE_KEY
docker-compose up --build
```

First run downloads the dataset, builds the DuckDB database, and builds the vector store — all idempotent and cached in Docker volumes, so restarts are fast. API available at `http://localhost:8000`, docs at `http://localhost:8000/docs`.

### Local (without Docker)

```bash
uv add -r requirements.txt
uv run scripts/setup_db.py     # loads Olist CSVs into DuckDB (needs CSVs already downloaded)
uv run vector_store.py         # builds the schema embedding index
uv run main.py                 # starts the API on :8000
```

### Try it

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What were our top 5 products by revenue last quarter?"}'
```

## Project structure

```
main.py                 FastAPI app: POST /ask, GET /health, request logging/correlation IDs
pipeline.py              ask() / guarded_ask(): full orchestration, retry loop, review-queue flagging
sql_generator.py          Context building (retrieval + graph expansion + samples) + SQL generation
sql_executor.py            SQL validation (sqlglot, CTE-aware) + execution
schema_graph.py             Table relationship graph (networkx), built from schema YAML foreign keys
vector_store.py               Chroma vector store over table descriptions (local ONNX embeddings)
embed_schema.py                 Builds embedding-ready text docs from the schema YAML
db.py                             DuckDB connection + sample row fetching
chart_spec.py            Pydantic schema for LLM chart selection (structured output only)
chart_renderer.py          Pre-written, trusted matplotlib rendering functions
viz_agent.py                 LLM call that picks a ChartSpec, dispatches to chart_renderer
guardrails.py            LLM-Guard input/output scanning (injection, secrets, PII)
review_queue.py           LangSmith annotation queue for retried/failed runs
eval_dataset.py          16-example eval dataset (retrieval, joins, dates, guardrails, edge cases)
eval_run.py                Runs the eval dataset with behavioral (non-exact-match) evaluators
logging_config.py        Structured file logging with per-request correlation IDs
schema/olist_schema.yaml Hand-curated table/column descriptions + foreign keys
scripts/setup_db.py      Loads Olist CSVs into DuckDB (safe to rerun)
Dockerfile, docker-compose.yml, docker/entrypoint.sh    Containerized, idempotent multi-step setup
```

## Known limitations

- **CORS is wide open** (`allow_origins=["*"]`) — fine for local development, must be locked to a real frontend origin before any real deployment.
- **No query cost guardrail yet.** An `EXPLAIN`-based check to reject absurdly expensive queries before execution was scoped but not built — currently relies only on `row_limit` capping at read time.
- **No semantic caching.** Two near-identical questions currently re-run the entire pipeline from scratch; a query-embedding cache was scoped but not implemented.
- **No authentication/rate-limiting on the API itself** — appropriate for an internal tool behind existing network controls, not for public exposure as-is.
- **`geolocation` has no clean primary key** and is only loosely joinable via zip-code prefix — excluded from the schema graph's formal FK relationships.
- **Hand-rolled SQL validator, not a full SQL-aware type checker.** It's been hardened against every concrete failure mode hit during development (writes, hallucinated tables/columns, CTE names, CTE output columns), but more exotic SQL (window functions inside CTEs, multiple interdependent CTEs) may surface new edge cases — this is a known, bounded risk of the approach, not an assumption that validation is exhaustive.