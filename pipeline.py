"""
The full pipeline: query -> generate SQL -> validate -> execute -> chart.

If validation rejects the generated SQL (hallucinated column/table, or a
write attempt), the specific error is fed back to the LLM as a correction
request, and it gets another attempt -- up to max_retries times -- instead
of just failing outright.
"""

import base64
from langchain_core.prompts import ChatPromptTemplate

from Sql_generator import build_context, get_llm, SQLGenerationResult, SYSTEM_PROMPT_TEXT
from sql_executer import execute_sql, SQLValidationError
from vizagent import generate_chart

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


def generate_and_execute(query: str, max_retries: int = MAX_RETRIES) -> dict:
    """Runs the full pipeline with self-correction. Returns a dict with the
    final SQL, explanation, results, and how many attempts it took. Raises
    SQLValidationError if it still fails to produce valid SQL after all
    retries are exhausted."""

    context = build_context(query)
    llm = get_llm()

    chain = INITIAL_PROMPT | llm
    result: SQLGenerationResult = chain.invoke({
        "schema_context": context["schema_context"],
        "query": query,
    })

    attempts = [{"sql": result.sql, "explanation": result.explanation}]

    for attempt_num in range(1, max_retries + 1):
        try:
            exec_result = execute_sql(result.sql)
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
            if attempt_num == max_retries:
                raise  # out of retries, let the caller handle the final failure

            retry_chain = RETRY_PROMPT | llm
            result: SQLGenerationResult = retry_chain.invoke({
                "schema_context": context["schema_context"],
                "query": query,
                "previous_sql": result.sql,
                "error": str(e),
            })
            attempts.append({"sql": result.sql, "explanation": result.explanation, "fixed_error": str(e)})

    # first attempt succeeded (loop above returns immediately on success);
    # this line is only reached if max_retries == 0 and the first try failed
    raise SQLValidationError("Failed to produce valid SQL and no retries were allowed.")


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
            result["chart"] = generate_chart(
                query=query,
                columns=result["columns"],
                rows=result["rows"],
            )
        except Exception as e:
            result["chart_error"] = str(e)

    return result


if __name__ == "__main__":
    try:
        result = ask("What were our top 5 products by revenue last quarter?")

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