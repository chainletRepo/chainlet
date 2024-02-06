# Import libraries for different Graph Neural Network

import numpy as np
import pandas as pd
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric
from torch.nn import Parameter
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import urllib.request
import tarfile
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx
from ThisDataset import ThisDataset
# Load the PubMMed dataset
addFile = "/home/bcypher/IdeaProjects/Orbit/articleOrbits.csv"
dataset= ThisDataset(root="data/", n_graphs=40000,sampleWhite= 50, filepath="/home/bcypher/PycharmProjects/PytorchTutorial/data/Orbit/graphs/", addressFile=addFile)
data=dataset[0]
class GNN(torch.nn.Module):
    def __init__(self, dataset,hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        # print('gnn output', x)

        return x
model = GNN(dataset,hidden_channels=64)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
dataset = dataset.shuffle()

train_dataset = dataset[:int(0.9* len(dataset))]
test_dataset = dataset[int(0.9* len(dataset)):]
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(test_loader, model):
    model.eval()

    correct = 0
    auc_score = 0
    total_samples = 0

    for data in test_loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with the highest probability.

        correct += int((pred == data.y).sum().item())  # Check against ground-truth labels.
        total_samples += data.y.size(0)

        arr2 = out[:, 1].detach().numpy()
        arr1 = data.y.detach().numpy()
        auc_score += roc_auc_score(y_true=arr1, y_score=arr2, multi_class='ovr', average='weighted')

    accuracy = correct / total_samples
    auc_score /= len(test_loader)

    #print(f"Accuracy: {accuracy:.4f}, AUC Score: {auc_score:.4f}")

    return accuracy, auc_score

for epoch in range(0, 10001):
    train()
    scores_tr = test(train_loader, model)
    train_acc = scores_tr[0]
    train_auc = scores_tr[1]
    scores_te = test(test_loader, model)
    test_acc = scores_te[0]
    test_auc = scores_te[1]
    print(
        f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Train Auc: {train_auc:.4f}, Test Auc: {test_auc:.4f}')
