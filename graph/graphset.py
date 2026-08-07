import yaml
import networkx as nx

def build_graph_from_yaml(path="Schema/olist_schema.yaml"):
    with open(path) as f:
        schema = yaml.safe_load(f)

    graph = nx.DiGraph()
    for table_name, table_info in schema["tables"].items():
        graph.add_node(table_name, description=table_info["description"])
        for fk in table_info.get("foreign_keys", []):
            ref_table = fk["references"].split(".")[0]
            graph.add_edge(table_name, ref_table, via=fk["column"])

    return graph, schema