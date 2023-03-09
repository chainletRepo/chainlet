Usage
Initialization
To use this project, you'll first need to initialize it with running 

Run graph.py to pass a grapg to the ExperimentDailyOrbits class for computing daily address orbits for Bitcoin transactions. This code requires the following dependencies:
    GraphExtractorBitcoinETL
    OrbitAnalyzer
    NetworkExporter
    Util
    csv
    BlockReader

The BitcoinHeistAnalyzer class has a compute_metrics method that takes a graph object as input and returns a dictionary of features computed for each node in the graph. The graph object is assumed to be a networkx graph. The compute_metrics method first finds the starter transactions by calling the find_starters2 method. If there are no starters or the list of starters is empty, the method prints an error message and returns None. Otherwise, it computes the following features for each address node in the graph:

count: the number of times the address appears in the graph
income: the total amount of Bitcoin that was received by the address
length: the shortest path length from the address to a starter transaction
loop: the number of times a cycle of transactions involving the address occurs
neighbor: the number of neighbors of the address in the graph
weight: the sum of the weights of all edges in the paths from the address to a starter transaction
The walk method is called by compute_metrics to compute the length and weight features. It performs a breadth-first search from each starter transaction, computing the length and weight of the path from each address to the starter transaction. The find_starters2 method is used to find the starter transactions. It returns a set of transactions that are starters or coinbase transactions.

The NetworkExporter class is used to export transaction network data for a given day.

