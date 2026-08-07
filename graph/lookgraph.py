from graphset import build_graph_from_yaml

graph, schema = build_graph_from_yaml()

# list all tables (nodes)
print("Tables:", list(graph.nodes()))

# list all relationships (edges)
print("\nRelationships:")
for source, target, data in graph.edges(data=True):
    print(f"  {source} --({data['via']})--> {target}")
    
    

