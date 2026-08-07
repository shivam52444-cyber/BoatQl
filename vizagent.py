"""
The viz agent. Given the SQL result (columns + a few sample rows), the LLM
picks a chart type and which columns to use -- via structured output only.
It never writes or executes chart code; chart_renderer.py does the actual
rendering using pre-written, trusted functions.
"""

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from chart_spec import ChartSpec
from chart_rendere import render_chart, ChartRenderError


def get_viz_llm():
    primary = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    fallback = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return primary.with_fallbacks([fallback]).with_structured_output(ChartSpec)


VIZ_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You choose the best way to visualize a SQL query result. You do NOT "
     "write any code -- you only pick a chart_type and, if needed, which "
     "columns to use for it. Guidelines:\n"
     "- A single row with one numeric value -> number_card\n"
     "- Categorical labels with one numeric value each (e.g. top N by some metric) -> bar\n"
     "- A sequence over time/order -> line\n"
     "- A small number of categories that sum to a whole (e.g. proportions) -> pie\n"
     "- Many rows, or no clear single metric to chart -> table\n"
     "Only use column names that actually appear in the result columns given."),
    ("human",
     "User's original question: {query}\n\n"
     "Result columns: {columns}\n"
     "Row count: {row_count}\n"
     "Sample rows: {sample_rows}"),
])


def generate_chart(query: str, columns: list[str], rows: list[dict], max_sample_rows: int = 5) -> dict:
    """Picks a chart via the LLM, then renders it with trusted code. Returns
    a dict describing the rendered chart (see chart_renderer.py return shapes)
    plus the ChartSpec the LLM chose, for debugging/eval."""

    llm = get_viz_llm()
    chain = VIZ_PROMPT | llm

    spec: ChartSpec = chain.invoke({
        "query": query,
        "columns": columns,
        "row_count": len(rows),
        "sample_rows": rows[:max_sample_rows],
    })

    try:
        rendered = render_chart(rows, spec)
    except ChartRenderError:
        # LLM picked a chart/columns that don't actually fit the data --
        # fall back to the one chart type that always works: a table.
        fallback_spec = ChartSpec(
            chart_type="table",
            title=spec.title or "Results",
            reasoning="Fell back to table because the chosen chart_type/columns didn't match the data.",
        )
        rendered = render_chart(rows, fallback_spec)
        spec = fallback_spec

    rendered["spec"] = spec.model_dump()
    return rendered





if __name__ == "__main__":
    from chart_spec import ChartSpec as CS

    fake_rows = [
        {"product_id": "abc123", "revenue": 22504.69},
        {"product_id": "def456", "revenue": 18480.0},
        {"product_id": "ghi789", "revenue": 17999.91},
    ]

    spec = CS(chart_type="bar", x_column="product_id", y_column="revenue", title="Top products by revenue", reasoning="Testing renderer directly.")
    result = render_chart(fake_rows, spec)
    print("Renderer smoke test OK. Keys:", list(result.keys()), "chart_type:", result["chart_type"])

    result = generate_chart(
        query="What were our top 5 products by revenue last quarter?",
        columns=["product_id", "revenue"],
        rows=fake_rows,
    )
    import base64

    with open("chart_output.png", "wb") as f:
        f.write(base64.b64decode(result["image_base64"]))

        print("Saved chart_output.png")
    print("\nLLM-picked spec:", result["spec"])
    print("Rendered type:", result["type"])