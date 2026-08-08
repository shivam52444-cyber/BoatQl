"""
Builds a graph of table relationships from schema/olist_schema.yaml.

We build this from the YAML config rather than DuckDB's FK introspection,
because DuckDB's constraint support/introspection is inconsistent -- hand
curating relationships in YAML is also closer to what real production
text-to-SQL systems do anyway, since not every real DB has clean FK
constraints either.
"""

import functools
import yaml
import networkx as nx


@functools.lru_cache(maxsize=1)
def build_graph_from_yaml(path: str = "schema/olist_schema.yaml"):
    """Cached: the schema file doesn't change while the app is running, so
    this only actually parses the YAML and builds the graph once per
    process -- every subsequent call (i.e. every query) reuses the same
    graph object instead of rebuilding it from scratch."""
    with open(path) as f:
        schema = yaml.safe_load(f)

    graph = nx.DiGraph()

    for table_name, table_info in schema["tables"].items():
        graph.add_node(table_name, description=table_info["description"])

    for table_name, table_info in schema["tables"].items():
        for fk in table_info.get("foreign_keys", []):
            ref_table = fk["references"].split(".")[0]
            graph.add_edge(table_name, ref_table, via=fk["column"])

    return graph, schema


def find_join_path(graph: nx.DiGraph, table_a: str, table_b: str) -> list[str]:
    """Shortest path between two tables, treating edges as undirected
    (a join doesn't care which side declared the FK)."""
    return nx.shortest_path(graph.to_undirected(), table_a, table_b)


def expand_with_graph(seed_tables: list[str], graph: nx.DiGraph) -> set[str]:
    """Given a set of semantically-relevant 'seed' tables, find the minimum
    set of bridge tables needed to actually join them together.

    e.g. seeds = {orders, products} -> returns {orders, products, order_items}
    because order_items is the bridge table connecting them.
    """
    relevant = set(seed_tables)
    undirected = graph.to_undirected()

    for table in seed_tables:
        for other in seed_tables:
            if table == other:
                continue
            try:
                path = nx.shortest_path(undirected, table, other)
                relevant.update(path)
            except nx.NetworkXNoPath:
                continue

    return relevant


if __name__ == "__main__":
    graph, schema = build_graph_from_yaml()

    print("Tables:", list(graph.nodes()))
    print("\nRelationships:")
    for source, target, data in graph.edges(data=True):
        print(f"  {source} --({data['via']})--> {target}")

    print("\nJoin path orders -> products:", find_join_path(graph, "orders", "products"))

    print(
        "\nExpanded set for seeds {orders, products}:",
        expand_with_graph(["orders", "products"], graph),
    )