import os
from pathlib import Path
import duckdb

# Docker supplies DB_PATH and DATA_DIR; defaults keep local development
# compatible with the existing database.db layout.
db_path = Path(os.environ.get("DB_PATH", "database.db"))
data_folder = Path(os.environ.get("DATA_DIR", "Dataset"))
db_path.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(db_path))

# DuckDB table name -> Olist CSV file. CREATE OR REPLACE keeps this script
# safely re-runnable when a database needs to be regenerated.
tables = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

print("Importing CSV files into DuckDB...")
for table_name, filename in tables.items():
    csv_path = data_folder / filename
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {csv_path}")
    escaped_path = str(csv_path).replace("'", "''")
    con.execute(
        f"CREATE OR REPLACE TABLE {table_name} "
        f"AS SELECT * FROM read_csv_auto('{escaped_path}');"
    )

print("\nSuccess! Database tables created:")
print(con.execute("SHOW TABLES").fetchall())
con.close()