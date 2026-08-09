"""
The full pipeline: query -> generate SQL -> validate -> execute -> chart.

If validation rejects the generated SQL (hallucinated column/table, or a
write attempt), the specific error is fed back to the LLM as a correction
request, and it gets another attempt -- up to max_retries times -- instead
of just failing outright.
"""

import time
import base64
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from Sql_generator import build_context, get_llm, SQLGenerationResult, SYSTEM_PROMPT_TEXT
from sql_executer import execute_sql, SQLValidationError
from vizagent import generate_chart
from Review_queue import queue_run_for_review
from logging_config import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2

INITIAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_TEXT),
    ("human", "{query}"),
])

RETRY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_TEXT),
    ("human", "{query}"),
    ("ai", "{previous_sql}"),
    ("human",
     "That query was rejected by the validator with this error:\n"
     "{error}\n\n"
     "Fix the query so it only uses real tables/columns from the schema above, "
     "and produce a corrected version."),
])


@traceable(name="generate_and_execute", run_type="chain")
def generate_and_execute(query: str, max_retries: int = MAX_RETRIES) -> dict:
    """Runs the full pipeline with self-correction. Returns a dict with the
    final SQL, explanation, results, and how many attempts it took. Raises
    SQLValidationError if it still fails to produce valid SQL after all
    retries are exhausted."""

    t0 = time.perf_counter()
    context = build_context(query)
    t1 = time.perf_counter()
    logger.info(f"[timing] build_context: {t1 - t0:.2f}s")

    llm = get_llm()
    t2 = time.perf_counter()
    logger.info(f"[timing] get_llm: {t2 - t1:.2f}s")

    logger.info(f"generate_and_execute start | query={query!r} | seed_tables={context['seed_tables']} | expanded_tables={context['expanded_tables']}")

    chain = INITIAL_PROMPT | llm
    result: SQLGenerationResult = chain.invoke({
        "schema_context": context["schema_context"],
        "query": query,
    })
    t3 = time.perf_counter()
    logger.info(f"[timing] initial LLM generation call: {t3 - t2:.2f}s")

    attempts = [{"sql": result.sql, "explanation": result.explanation}]

    for attempt_num in range(1, max_retries + 1):
        try:
            t_exec_start = time.perf_counter()
            exec_result = execute_sql(result.sql)
            logger.info(f"[timing] execute_sql (attempt {attempt_num}): {time.perf_counter() - t_exec_start:.2f}s")
            logger.info(f"generate_and_execute succeeded | attempt={attempt_num}/{max_retries} | rows={exec_result['row_count']}")

            if attempt_num > 1:
                logger.warning(f"query needed {attempt_num} attempts before succeeding | query={query!r}")
                run = get_current_run_tree()
                queue_run_for_review(
                    str(run.id) if run else None,
                    reason=f"succeeded but needed {attempt_num} attempts",
                )

            return {
                "query": query,
                "seed_tables": context["seed_tables"],
                "expanded_tables": context["expanded_tables"],
                "sql": result.sql,
                "explanation": result.explanation,
                "confidence": result.confidence,
                "columns": exec_result["columns"],
                "rows": exec_result["rows"],
                "row_count": exec_result["row_count"],
                "attempts": attempts,
                "attempt_count": attempt_num,
            }
        except SQLValidationError as e:
            logger.warning(f"validation failed | attempt={attempt_num}/{max_retries} | error={e}")

            if attempt_num == max_retries:
                logger.error(f"generate_and_execute FAILED after {max_retries} attempts | query={query!r} | last_error={e}")
                run = get_current_run_tree()
                queue_run_for_review(
                    str(run.id) if run else None,
                    reason=f"failed after {max_retries} attempts: {e}",
                )
                raise  # out of retries, let the caller handle the final failure

            t_retry_start = time.perf_counter()
            retry_chain = RETRY_PROMPT | llm
            result: SQLGenerationResult = retry_chain.invoke({
                "schema_context": context["schema_context"],
                "query": query,
                "previous_sql": result.sql,
                "error": str(e),
            })
            logger.info(f"[timing] retry LLM generation call: {time.perf_counter() - t_retry_start:.2f}s")
            attempts.append({"sql": result.sql, "explanation": result.explanation, "fixed_error": str(e)})

    # first attempt succeeded (loop above returns immediately on success);
    # this line is only reached if max_retries == 0 and the first try failed
    raise SQLValidationError("Failed to produce valid SQL and no retries were allowed.")


@traceable(name="ask_pipeline", run_type="chain")
def ask(query: str, max_retries: int = MAX_RETRIES, include_chart: bool = True) -> dict:
    """The single top-level entrypoint: user query in, SQL + results + chart
    out. This is what your FastAPI route should call.

    If SQL generation/execution fails after all retries, the SQLValidationError
    propagates -- the caller (e.g. the API route) decides how to surface that
    to the user. Chart generation failures do NOT fail the whole request --
    if charting breaks for any reason, we still return the SQL results with
    chart=None, since the SQL answer is the important part.
    """
    result = generate_and_execute(query, max_retries=max_retries)

    result["chart"] = None
    if include_chart and result["rows"]:
        try:
            t_chart_start = time.perf_counter()
            result["chart"] = generate_chart(
                query=query,
                columns=result["columns"],
                rows=result["rows"],
            )
            logger.info(f"[timing] generate_chart: {time.perf_counter() - t_chart_start:.2f}s")
        except Exception as e:
            logger.warning(f"chart generation failed | query={query!r} | error={e}")
            result["chart_error"] = str(e)

    return result


@traceable(name="guarded_ask", run_type="chain")
def guarded_ask(query: str, max_retries: int = MAX_RETRIES, include_chart: bool = True) -> dict:
    """ask() wrapped with input/output guardrails. This is the entrypoint
    your API route should actually call in production -- ask() stays
    guardrail-free so it's easy to test/eval the core pipeline in isolation.

    Raises GuardrailRejection if the input is rejected before ever reaching
    the LLM. If the input passes but the output fails guardrails (e.g. PII
    leaked into the explanation text), the explanation is replaced with a
    safe placeholder rather than failing the whole request -- the SQL
    results/chart are still useful even if the explanation had to be redacted.
    """
    from Guardrails import check_input, check_output, GuardrailRejection

    t_input_start = time.perf_counter()
    try:
        clean_query = check_input(query)  # raises GuardrailRejection if rejected
    except GuardrailRejection as e:
        logger.warning(f"input REJECTED by guardrails | query={query!r} | reason={e}")
        raise
    logger.info(f"[timing] check_input (guardrails): {time.perf_counter() - t_input_start:.2f}s")

    result = ask(clean_query, max_retries=max_retries, include_chart=include_chart)

    t_output_start = time.perf_counter()
    try:
        result["explanation"] = check_output(result["explanation"], prompt=clean_query)
    except GuardrailRejection as e:
        logger.warning(f"output flagged by guardrails, explanation redacted | query={query!r} | reason={e}")
        result["explanation"] = "[Explanation withheld: contained sensitive content]"
        result["output_guardrail_flag"] = str(e)
    logger.info(f"[timing] check_output (guardrails): {time.perf_counter() - t_output_start:.2f}s")

    return result


if __name__ == "__main__":
    from logging_config import setup_logging
    setup_logging(log_to_console=True)  # want to see timing live during this debug run

    try:
        result = guarded_ask("What were our top 5 products by revenue last quarter?")

        print(f"Succeeded on attempt {result['attempt_count']} (of {len(result['attempts'])} total tries)")
        print("\nFinal SQL:\n", result["sql"])
        print("\nExplanation:\n", result["explanation"])
        print(f"\nColumns: {result['columns']}")
        print(f"Row count: {result['row_count']}")
        for row in result["rows"][:5]:
            print(" ", row)

        if result["chart"]:
            chart = result["chart"]
            print("\nChart type:", chart["chart_type"])
            print("Chart spec reasoning:", chart["spec"]["reasoning"])

            if chart["type"] == "image":
                out_path = "chart_output.png"
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(chart["image_base64"]))
                print(f"Saved chart to {out_path}")
            elif chart["type"] == "number_card":
                print(f"Number card: {chart['title']} = {chart['value']}")
            elif chart["type"] == "table":
                print(f"Table with {len(chart['rows'])} rows (see above)")

        elif "chart_error" in result:
            print("\nChart generation failed:", result["chart_error"])
        elif not result["rows"]:
            print("\nNo chart generated: query returned 0 rows.")

    except SQLValidationError as e:
        print("Failed after all retries:", e)