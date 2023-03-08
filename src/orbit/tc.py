import networkx as nx
from Node import Node
from WeightedEdge import WeightedEdge
from OrbitAnalyzer import OrbitAnalyzer

G = nx.DiGraph()
nodes = [Node("tx1"), Node("addr1"), Node("tx2"), Node("addr2")]
edges = [WeightedEdge(nodes[0], nodes[1], 1, 0, 1),
            WeightedEdge(nodes[1], nodes[2], 2, 0, 2),
            WeightedEdge(nodes[2], nodes[3], 3, 0, 3)
            ]

G.add_nodes_from(nodes)

# Add edges to the graph
for edge in edges:
    G.add_edge(edge.from_node, edge.to_node, edge=edge)
    
analyzer = OrbitAnalyzer()
orbits = analyzer.compute_orbits(G)
print(orbits)