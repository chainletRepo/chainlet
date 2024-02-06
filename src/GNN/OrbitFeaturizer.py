from typing import Optional, Sequence

import networkx as nx
import torch
import numpy as np

class OrbitFeaturizer():
    def __init__(self):
        pass

    def orbit_featurize(self,G):
        self.compute_node_features(G)
        weight_list = []
        for i, f, data in G.edges(data='weight'):
            weight_list.append(data)
        self.edge_features = np.array(weight_list)
        edges = nx.to_pandas_edgelist(G)
        self.edge_index = [edges.source, edges.target]


    def compute_node_features(self, G):
        in_degrees = G.in_degree()
        sums = []
        for node, degree in in_degrees:
            weight_sum = sum(G.get_edge_data(u, node)['weight'] for u in G.predecessors(node))
            sums.append(weight_sum)
        self.node_features = np.array(sums)

    def to_pyg_graph(self):
        """Convert to PyTorch Geometric graph data instance
        Returns
        -------
        torch_geometric.data.Data
          Graph data for PyTorch Geometric
        Note
        ----
        This method requires PyTorch Geometric to be installed.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ModuleNotFoundError:
            raise ImportError(
                "This function requires PyTorch Geometric to be installed.")

        edge_features = self.edge_features
        if edge_features is not None:
            edge_features = torch.from_numpy(self.edge_features).float().reshape([-1, 1])
        node_features = self.node_features
        if node_features is not None:
            node_features = torch.from_numpy(self.node_features).float().reshape([-1, 1])
        edge_index =  torch.tensor(self.edge_index)

        return Data(x=node_features, edge_index=edge_index,
                    edge_attr=edge_features )