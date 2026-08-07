"""
Thin DuckDB helper: connect to the olist.db file and pull sample rows
per table for context-building. Keep this file dumb on purpose -- no
query generation logic here, just raw access.
"""

"""
Thin DuckDB helper: connect to the olist.db file and pull sample rows
per table for context-building. Keep this file dumb on purpose -- no
query generation logic here, just raw access.
"""

import duckdb

DB_PATH = "database.db"


def get_connection(db_path: str = DB_PATH):
    return duckdb.connect(db_path)


def get_sample_rows(table_name: str, n: int = 5, con=None) -> list[dict]:
    """Returns up to n randomly sampled rows from a table as a list of dicts."""
    own_con = con is None
    con = con or get_connection()
    try:
        return con.execute(
            f"SELECT * FROM {table_name} USING SAMPLE {n} ROWS"
        ).fetchdf().to_dict(orient="records")
    finally:
        if own_con:
            con.close()