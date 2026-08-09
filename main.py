"""
FastAPI wrapper around the pipeline. Single main endpoint: POST /ask.

Every request gets a correlation ID (returned in the X-Request-ID header
and logged on every line for that request), and errors map to sensible
HTTP status codes instead of leaking raw Python tracebacks.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from logging_config import setup_logging, get_logger, set_request_id
from pipeline import guarded_ask
from sql_executer import SQLValidationError
from Guardrails import GuardrailRejection

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once at startup -- this is where the embedding model, vector
    # store, and any other "expensive first call" resources should warm up,
    # so the FIRST real user request isn't the one paying that cost.
    setup_logging()
    logger.info("Starting up: warming caches (embedding model, vector store)...")

    from Vector_store import retrieve_relevant_tables
    retrieve_relevant_tables("warmup query", top_k=1)
    logger.info("Vector store / embedding model warmed.")

    from Guardrails import check_input, check_output
    check_input("warmup query")
    check_output("warmup response")
    logger.info("Guardrail scanners warmed.")

    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="AI SQL Analyst", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)

    start = time.perf_counter()
    logger.info(f"--> {request.method} {request.url.path}")

    response = await call_next(request)

    elapsed = time.perf_counter() - start
    response.headers["X-Request-ID"] = request_id
    logger.info(f"<-- {request.method} {request.url.path} | status={response.status_code} | {elapsed:.2f}s")

    return response


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    include_chart: bool = True
    max_retries: int = Field(default=2, ge=0, le=5)


class AskResponse(BaseModel):
    query: str
    sql: str
    explanation: str
    confidence: float
    columns: list[str]
    rows: list[dict]
    row_count: int
    attempt_count: int
    seed_tables: list[str]
    expanded_tables: list[str]
    chart: dict | None = None
    chart_error: str | None = None
    output_guardrail_flag: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def ask(request: AskRequest):
    try:
        result = guarded_ask(
            request.query,
            max_retries=request.max_retries,
            include_chart=request.include_chart,
        )
        return result

    except GuardrailRejection as e:
        logger.warning(f"request rejected by guardrails: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": "guardrail_rejection", "detail": str(e)},
        )

    except SQLValidationError as e:
        logger.warning(f"request failed SQL validation after all retries: {e}")
        return JSONResponse(
            status_code=422,
            content={"error": "sql_validation_failed", "detail": str(e)},
        )

    except Exception as e:
        logger.exception(f"unexpected error handling request: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": "Something went wrong processing your request."},
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)