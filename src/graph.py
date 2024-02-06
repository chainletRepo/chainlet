import networkx as nx
from Orbit import Node
from Orbit import WeightedEdge
from Orbit.OrbitAnalyzer import OrbitAnalyzer

G = nx.DiGraph()
G = nx.MultiDiGraph()
nodes = [Node(f"tx{i}") for i in range(1, 7)]
nodes += [Node(f"addr{j}") for j in range(1, 13)]
edges = [    WeightedEdge(nodes[0], nodes[1], 1, 0, 1),
             WeightedEdge(nodes[1], nodes[2], 2, 0, 2),
             WeightedEdge(nodes[2], nodes[3], 3, 0, 3),
             WeightedEdge(nodes[3], nodes[4], 4, 0, 4),
             WeightedEdge(nodes[4], nodes[5], 5, 0, 5),
             WeightedEdge(nodes[5], nodes[6], 6, 0, 6),
             WeightedEdge(nodes[1], nodes[7], 7, 0, 7),
             WeightedEdge(nodes[7], nodes[8], 8, 0, 8),
             WeightedEdge(nodes[8], nodes[9], 9, 0, 9),
             WeightedEdge(nodes[2], nodes[10], 10, 0, 10),
             WeightedEdge(nodes[10], nodes[11], 11, 0, 11),
             WeightedEdge(nodes[11], nodes[9], 12, 0, 12)
             ]

G.add_nodes_from(nodes)

# Add edges to the graph
for edge in edges:
    G.add_edge(edge.from_node, edge.to_node, edge=edge)

nx.draw(G)
analyzer = OrbitAnalyzer()
orbits = analyzer.compute_orbits(G)
print(orbits)