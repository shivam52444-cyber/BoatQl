# scripts/setup_duckdb.py
import os
import duckdb

# Connect to database (creates olist.db inside your main root folder)
db_path = r"D:\DATA science projects\BOatQl\DATABASE.db"
con = duckdb.connect(db_path)

# Explicitly use the absolute path to your data folder
data_folder = r"D:\DATA science projects\BOatQl\Dataset"

# Execute queries one by one using the correct path format
queries = [
    f"CREATE TABLE customers AS SELECT * FROM read_csv_auto('{os.path.join(data_folder, 'olist_customers_dataset.csv')}');",
    f"CREATE TABLE orders AS SELECT * FROM read_csv_auto('{os.path.join(data_folder, 'olist_orders_dataset.csv')}');",
    f"CREATE TABLE order_items AS SELECT * FROM read_csv_auto('{os.path.join(data_folder, 'olist_order_items_dataset.csv')}');",
    f"CREATE TABLE order_payments AS SELECT * FROM read_csv_auto('{os.path.join(data_folder, 'olist_order_payments_dataset.csv')}');",
    f"CREATE TABLE order_reviews AS SELECT * FROM read_csv_auto('{os.path.join(data_folder, 'olist_order_reviews_dataset.csv')}');",
    f"CREATE TABLE products AS SELECT * FROM read_csv_auto('{os.path.join(data_folder, 'olist_products_dataset.csv')}');",
    f"CREATE TABLE sellers AS SELECT * FROM read_csv_auto('{os.path.join(data_folder, 'olist_sellers_dataset.csv')}');",
    f"CREATE TABLE geolocation AS SELECT * FROM read_csv_auto('{os.path.join(data_folder, 'olist_geolocation_dataset.csv')}');",
    f"CREATE TABLE category_translation AS SELECT * FROM read_csv_auto('{os.path.join(data_folder, 'product_category_name_translation.csv')}');"
]

print("Importing CSV files into DuckDB...")
for query in queries:
    con.execute(query)

print("\nSuccess! Database tables created:")
print(con.execute("SHOW TABLES").fetchall())
