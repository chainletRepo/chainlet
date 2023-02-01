import os
import pandas as pd
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
# !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git


import networkx as nx
import matplotlib.pyplot as plt

allDataModified = '../data/allDataModified.csv'
dataset = pd.read_csv(allDataModified, sep=",", header=0, index_col=0)


def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()


num_features = dataset.shape[1]
num_classes = dataset['label'].nunique()
num_nodes = dataset.shape[0]
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {num_features}')
print(f'Number of classes: {num_classes}')
data = dataset[0]  # Get the first graph object.
print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {num_nodes}')
# print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')