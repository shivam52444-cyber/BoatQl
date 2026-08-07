"""
The SQL generation agent.

Flow:
  user query
    -> vector store: top-k semantically relevant "seed" tables
    -> graph: expand seeds to include bridge/join tables
    -> for each expanded table: pull description (from schema yaml) + n sample rows (from duckdb)
    -> assemble one context block
    -> LLM (Groq primary, OpenAI fallback) generates {sql, explanation, confidence} as structured output
"""

import yaml
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from Vector_store import retrieve_relevant_tables
from graph.graphset import build_graph_from_yaml, expand_with_graph
from sample import get_sample_rows, get_connection



SCHEMA_PATH = "schema/olist_schema.yaml"


# ---------------------------------------------------------------------------
# Structured output schema -- this IS the "tool" the LLM is forced to call
# ---------------------------------------------------------------------------
class SQLGenerationResult(BaseModel):
    sql: str = Field(description="A single, valid, read-only (SELECT-only) SQL query answering the user's question.")
    explanation: str = Field(description="Plain-English explanation of what the query does and why it's structured this way.")
    confidence: float = Field(description="Model's confidence in the correctness of this query, from 0.0 to 1.0.")


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------
def build_context(query: str, top_k: int = 5, sample_rows_per_table: int = 5) -> dict:
    """Returns the assembled context needed to generate SQL: the expanded
    table set, a formatted schema block, and which tables were seeds vs
    bridge tables (useful for debugging/eval)."""

    graph, schema = build_graph_from_yaml(SCHEMA_PATH)

    seed_tables = retrieve_relevant_tables(query, top_k=top_k)
    expanded_tables = expand_with_graph(seed_tables, graph)

    con = get_connection()
    try:
        table_blocks = []
        for table_name in sorted(expanded_tables):
            table_info = schema["tables"][table_name]

            columns_text = "\n".join(
                f"    - {col}: {desc}" for col, desc in table_info.get("columns", {}).items()
            )

            fk_lines = [
                f"    - {fk['column']} -> {fk['references']}"
                for fk in table_info.get("foreign_keys", [])
            ]
            fk_text = "\n".join(fk_lines) if fk_lines else "    (none)"

            samples = get_sample_rows(table_name, n=sample_rows_per_table, con=con)

            table_blocks.append(
                f"Table: {table_name}\n"
                f"Description: {table_info['description']}\n"
                f"Columns:\n{columns_text}\n"
                f"Foreign keys:\n{fk_text}\n"
                f"Sample rows:\n{samples}"
            )
    finally:
        con.close()

    schema_context = "\n\n---\n\n".join(table_blocks)

    return {
        "seed_tables": seed_tables,
        "expanded_tables": sorted(expanded_tables),
        "schema_context": schema_context,
    }


# ---------------------------------------------------------------------------
# LLM setup -- Groq primary, OpenAI fallback, forced structured output
# ---------------------------------------------------------------------------
def get_llm():
    primary = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    fallback = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    llm_with_fallback = primary.with_fallbacks([fallback])
    return llm_with_fallback.with_structured_output(SQLGenerationResult)


SYSTEM_PROMPT_TEXT = (
    "You are a SQL analyst. You write a single, valid, READ-ONLY (SELECT-only) "
    "SQL query to answer the user's question, using ONLY the tables, columns, "
    "and foreign key relationships given below. Never write INSERT, UPDATE, "
    "DELETE, DROP, or ALTER. Use the foreign key relationships to determine "
    "correct joins -- do not invent joins, tables, or columns that aren't listed.\n\n"
    "IMPORTANT: This is a historical dataset, not live data. Never use NOW() "
    "or CURRENT_DATE for relative time filters like 'last quarter' or 'recent'. "
    "Instead, anchor relative dates to the MAX(timestamp_column) actually present "
    "in the relevant table -- e.g. use a subquery or CTE to find the latest date "
    "in the data first, then filter relative to that.\n\n"
    "Schema:\n{schema_context}"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_TEXT),
    ("human", "{query}")
])


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def generate_sql(query: str) -> dict:
    context = build_context(query)
    llm = get_llm()
    chain = PROMPT | llm

    result: SQLGenerationResult = chain.invoke({
        "schema_context": context["schema_context"],
        "query": query,
    })

    return {
        "query": query,
        "seed_tables": context["seed_tables"],
        "expanded_tables": context["expanded_tables"],
        "sql": result.sql,
        "explanation": result.explanation,
        "confidence": result.confidence,
    }


if __name__ == "__main__":
    result = generate_sql("What were our top 5 products by revenue last quarter?")

    print("Seed tables:", result["seed_tables"])
    print("Expanded tables (with joins):", result["expanded_tables"])
    print("\nGenerated SQL:\n", result["sql"])
    print("\nExplanation:\n", result["explanation"])
    print("\nConfidence:", result["confidence"])