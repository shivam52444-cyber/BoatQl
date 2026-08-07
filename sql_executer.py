"""
Validates generated SQL before ever running it, then executes it.

Two layers of validation:
  1. sqlglot parse check -- must be a single SELECT statement, no writes
  2. schema check -- every table AND column referenced must actually exist
     in schema/olist_schema.yaml (catches hallucinated columns like the
     LLM inventing order_items.order_purchase_timestamp)

Only if both pass does the query touch the database.
"""

import yaml
import sqlglot
from sqlglot import exp

from sample import get_connection

SCHEMA_PATH = "schema/olist_schema.yaml"

# statement types we never allow, regardless of framing
FORBIDDEN_STATEMENT_TYPES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop,
    exp.Alter, exp.Create, exp.TruncateTable,
)


class SQLValidationError(Exception):
    """Raised when generated SQL fails safety or schema validation."""
    pass


def _load_schema(schema_path: str = SCHEMA_PATH) -> dict:
    with open(schema_path) as f:
        return yaml.safe_load(f)["tables"]


def _build_column_lookup_from_db() -> dict[str, set[str]]:
    """table_name -> set of valid column names (lowercased), read directly
    from DuckDB. More reliable than the YAML, which is hand-maintained and
    can drift out of sync with the real columns (as it already has once)."""
    con = get_connection()
    try:
        rows = con.execute("""
            SELECT table_name, column_name
            FROM information_schema.columns
        """).fetchall()
    finally:
        con.close()

    lookup: dict[str, set[str]] = {}
    for table_name, column_name in rows:
        lookup.setdefault(table_name.lower(), set()).add(column_name.lower())
    return lookup


def validate_sql(sql: str, schema_path: str = SCHEMA_PATH) -> sqlglot.Expression:
    """Raises SQLValidationError if invalid. Returns the parsed expression if OK."""

    # --- parse ---
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except Exception as e:
        raise SQLValidationError(f"SQL failed to parse: {e}")

    if len(statements) != 1:
        raise SQLValidationError("Only a single SQL statement is allowed.")

    parsed = statements[0]
    if parsed is None:
        raise SQLValidationError("SQL failed to parse.")

    # --- must be SELECT only ---
    if not isinstance(parsed, exp.Select):
        raise SQLValidationError(f"Only SELECT statements are allowed, got: {type(parsed).__name__}")

    for forbidden_type in FORBIDDEN_STATEMENT_TYPES:
        if list(parsed.find_all(forbidden_type)):
            raise SQLValidationError(f"Query contains forbidden operation: {forbidden_type.__name__}")

    # --- table whitelist check ---
    schema = _load_schema(schema_path)
    valid_tables = {t.lower() for t in schema}
    column_lookup = _build_column_lookup_from_db()

    # CTE names (WITH x AS (...)) are "virtual tables" defined within the
    # query itself -- they're not in the schema, but they're legitimate.
    # We can't know their exact output columns without evaluating them, so
    # we allow the table reference but skip column-level checks for anything
    # qualified with a CTE alias (handled below).
    cte_names = {cte.alias.lower() for cte in parsed.find_all(exp.CTE) if cte.alias}

    # map alias -> real table name, e.g. "oi" -> "order_items"
    alias_to_table = {}
    referenced_tables = set()

    for table_expr in parsed.find_all(exp.Table):
        table_name = table_expr.name.lower()

        if table_name in cte_names:
            # references the CTE, not a real schema table -- skip whitelist check
            alias = table_expr.alias.lower() if table_expr.alias else table_name
            alias_to_table[alias] = table_name  # marks it as a CTE downstream
            continue

        referenced_tables.add(table_name)
        alias = table_expr.alias.lower() if table_expr.alias else table_name
        alias_to_table[alias] = table_name

        if table_name not in valid_tables:
            raise SQLValidationError(f"Unknown table referenced: '{table_name}'")

    # --- column whitelist check (catches hallucinated columns) ---
    # SELECT-list aliases (e.g. `SUM(price) AS revenue`) are valid targets
    # for unqualified references in GROUP BY / ORDER BY / HAVING -- these
    # aren't real table columns, so they must be excluded from the check.
    select_aliases = {
        alias.output_name.lower()
        for alias in parsed.selects
        if alias.output_name
    }

    for col_expr in parsed.find_all(exp.Column):
        col_name = col_expr.name.lower()
        table_ref = col_expr.table.lower() if col_expr.table else None

        if table_ref:
            real_table = alias_to_table.get(table_ref)
            if real_table is None:
                raise SQLValidationError(f"Column references unknown table alias: '{table_ref}'")
            if real_table in cte_names:
                continue  # CTE output column -- can't validate without evaluating the CTE
            if col_name not in column_lookup.get(real_table, set()):
                raise SQLValidationError(
                    f"Column '{col_name}' does not exist on table '{real_table}' "
                    f"(referenced as '{table_ref}.{col_name}')"
                )
        else:
            if col_name in select_aliases:
                continue  # refers to a computed SELECT-list alias, not a table column
            # no table prefix -- must exist on at least ONE referenced table
            if not any(col_name in column_lookup.get(t, set()) for t in referenced_tables):
                raise SQLValidationError(f"Column '{col_name}' not found on any referenced table")

    return parsed


def execute_sql(sql: str, schema_path: str = SCHEMA_PATH, row_limit: int = 1000) -> dict:
    """Validates then executes SQL. Returns rows + column names, or raises
    SQLValidationError before ever touching the database."""

    validate_sql(sql, schema_path)  # raises if invalid -- nothing below runs otherwise

    con = get_connection()
    try:
        result = con.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchmany(row_limit)
        return {
            "columns": columns,
            "rows": [dict(zip(columns, row)) for row in rows],
            "row_count": len(rows),
        }
    finally:
        con.close()


if __name__ == "__main__":
    # example: the buggy query from the LLM (wrong column on order_items)
    bad_sql = (
        "SELECT p.product_id, SUM(oi.price) AS revenue "
        "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
        "WHERE EXTRACT(QUARTER FROM oi.order_purchase_timestamp) = "
        "EXTRACT(QUARTER FROM CURRENT_DATE) - 1 "
        "GROUP BY p.product_id ORDER BY revenue DESC LIMIT 5"
    )

    try:
        validate_sql(bad_sql)
        print("Validation passed (unexpected).")
    except SQLValidationError as e:
        print("Validation correctly rejected the query:")
        print(" ", e)

    # example: a corrected version, joining through orders for the timestamp
    good_sql = (
        "SELECT p.product_id, SUM(oi.price) AS revenue "
        "FROM order_items oi "
        "JOIN products p ON oi.product_id = p.product_id "
        "JOIN orders o ON oi.order_id = o.order_id "
        "WHERE EXTRACT(QUARTER FROM o.order_purchase_timestamp) = "
        "EXTRACT(QUARTER FROM CURRENT_DATE) - 1 "
        "GROUP BY p.product_id ORDER BY revenue DESC LIMIT 5"
    )

    print("\nExecuting corrected query:")
    result = execute_sql(good_sql)
    print(f"Columns: {result['columns']}")
    print(f"Row count: {result['row_count']}")
    for row in result["rows"][:5]:
        print(" ", row)