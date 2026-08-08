"""
Runs the LangSmith eval dataset through the real pipeline and scores it
with custom evaluators. SQL/explanations are never exactly the same twice,
so nothing here does string equality -- instead:

  - table_coverage: hard check -- did the query actually touch every table
    named in expected_tables? (checks the sqlglot-parsed SQL, not just the
    retrieval step, so it reflects what really executed)
  - guardrail_behavior: hard check -- for guardrail/out_of_scope examples,
    did the pipeline correctly refuse rather than execute?
  - row_expectation: hard check -- did we get rows back when we expected to
    (or correctly get none for a reject case)?
  - answer_quality: LLM-as-judge -- given the query, generated SQL, and
    explanation, does this look like a sound, non-hallucinated answer?
    (skipped for guardrail/out_of_scope cases, where "quality" isn't the point)
"""

import sqlglot
from sqlglot import exp
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from pipeline import guarded_ask
from sql_executer import SQLValidationError
from Guardrails import GuardrailRejection
from Eval_dataset import DATASET_NAME




MAX_RETRIES_FOR_EVAL = 2


# ---------------------------------------------------------------------------
# Target function: what actually gets run for each example
# ---------------------------------------------------------------------------
def target(inputs: dict) -> dict:
    query = inputs["query"]
    try:
        result = guarded_ask(query, max_retries=MAX_RETRIES_FOR_EVAL)
        return {
            "rejected": False,
            "sql": result["sql"],
            "explanation": result["explanation"],
            "expanded_tables": result["expanded_tables"],
            "row_count": result["row_count"],
            "attempt_count": result["attempt_count"],
        }
    except (SQLValidationError, GuardrailRejection) as e:
        return {
            "rejected": True,
            "rejection_reason": str(e),
            "sql": None,
            "explanation": None,
            "expanded_tables": [],
            "row_count": 0,
            "attempt_count": None,
        }
    except Exception as e:
        # anything else (LLM/API errors, etc.) -- surface as a failure, not
        # a silent pass, so it shows up clearly in the eval results
        return {
            "rejected": True,
            "rejection_reason": f"Unexpected error: {e}",
            "sql": None,
            "explanation": None,
            "expanded_tables": [],
            "row_count": 0,
            "attempt_count": None,
        }


# ---------------------------------------------------------------------------
# Hard-check evaluators
# ---------------------------------------------------------------------------
def table_coverage_evaluator(run, example) -> dict:
    """Did the query actually reference every table named in expected_tables?
    Checked against the real parsed SQL, not just the retrieval step."""
    expected = example.outputs.get("expected_tables")
    if not expected:
        return {"key": "table_coverage", "score": None, "comment": "n/a for this example"}

    sql = run.outputs.get("sql")
    if not sql:
        return {"key": "table_coverage", "score": 0, "comment": "no SQL generated (rejected or errored)"}

    try:
        parsed = sqlglot.parse_one(sql, read="duckdb")
        actual_tables = {t.name.lower() for t in parsed.find_all(exp.Table)}
    except Exception:
        return {"key": "table_coverage", "score": 0, "comment": "SQL failed to parse"}

    expected_lower = {t.lower() for t in expected}
    missing = expected_lower - actual_tables
    covered = len(missing) == 0

    return {
        "key": "table_coverage",
        "score": 1 if covered else 0,
        "comment": "all expected tables present" if covered else f"missing: {missing}",
    }


def guardrail_behavior_evaluator(run, example) -> dict:
    """For guardrail/out_of_scope examples, did the pipeline correctly
    refuse instead of executing? For everything else, did it NOT get
    incorrectly rejected?"""
    example_type = example.outputs.get("type")
    was_rejected = run.outputs.get("rejected", False)

    if example_type not in ("guardrail", "out_of_scope"):
        # for normal queries, an unexpected rejection is a real failure
        correct = not was_rejected
        return {
            "key": "guardrail_behavior",
            "score": 1 if correct else 0,
            "comment": "correctly executed" if correct else f"incorrectly rejected: {run.outputs.get('rejection_reason')}",
        }

    if example_type == "guardrail":
        correct = was_rejected
        return {
            "key": "guardrail_behavior",
            "score": 1 if correct else 0,
            "comment": "correctly rejected" if correct else "SHOULD have been rejected but wasn't",
        }

    # out_of_scope: either a clean rejection OR a zero-row result is acceptable
    correct = was_rejected or run.outputs.get("row_count", 0) == 0
    return {
        "key": "guardrail_behavior",
        "score": 1 if correct else 0,
        "comment": "failed gracefully" if correct else "should have failed gracefully but returned data",
    }


def row_expectation_evaluator(run, example) -> dict:
    expect_rows = example.outputs.get("expect_rows")
    if expect_rows is None:
        return {"key": "row_expectation", "score": None, "comment": "n/a for this example"}

    got_rows = run.outputs.get("row_count", 0) > 0
    correct = got_rows == expect_rows

    return {
        "key": "row_expectation",
        "score": 1 if correct else 0,
        "comment": f"expected rows={expect_rows}, got rows={got_rows}",
    }


# ---------------------------------------------------------------------------
# LLM-as-judge evaluator
# ---------------------------------------------------------------------------
class QualityJudgment(BaseModel):
    is_sound: bool = Field(description="Does the SQL plausibly and correctly answer the user's question, with no obviously hallucinated logic?")
    reasoning: str = Field(description="One sentence justifying the judgment.")


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are grading whether a generated SQL query and explanation correctly "
     "answer a user's question. You are NOT checking exact syntax -- judge "
     "whether the logic is sound and matches what was asked. Be strict about "
     "obvious mismatches (wrong aggregation, wrong filter, ignoring part of "
     "the question) but don't penalize reasonable stylistic choices."),
    ("human",
     "User question: {query}\n\n"
     "Generated SQL: {sql}\n\n"
     "Explanation: {explanation}"),
])


def _get_judge():
    primary = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    fallback = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return primary.with_fallbacks([fallback]).with_structured_output(QualityJudgment)


def answer_quality_evaluator(run, example) -> dict:
    example_type = example.outputs.get("type")
    if example_type in ("guardrail", "out_of_scope"):
        return {"key": "answer_quality", "score": None, "comment": "n/a -- graded by guardrail_behavior instead"}

    sql = run.outputs.get("sql")
    if not sql:
        return {"key": "answer_quality", "score": 0, "comment": "no SQL generated"}

    judge = _get_judge()
    judgment: QualityJudgment = (JUDGE_PROMPT | judge).invoke({
        "query": example.inputs["query"],
        "sql": sql,
        "explanation": run.outputs.get("explanation", ""),
    })

    return {
        "key": "answer_quality",
        "score": 1 if judgment.is_sound else 0,
        "comment": judgment.reasoning,
    }


# ---------------------------------------------------------------------------
# Run the eval
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    setup_logging()  # pipeline logs go to app.log, not the terminal -- keeps eval output readable

    client = Client()

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            table_coverage_evaluator,
            guardrail_behavior_evaluator,
            row_expectation_evaluator,
            answer_quality_evaluator,
        ],
        experiment_prefix="ai-sql-analyst",
        client=client,
    )

    print("\nEval run complete. View full results in the LangSmith UI.")
    print(results)