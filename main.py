import duckdb
con = duckdb.connect(r"D:\DATA science projects\BOatQl\database.db")
print(con.execute("SHOW TABLES").fetchall())
from sample import get_sample_rows
print(get_sample_rows("order_items", n=3))
con.close()