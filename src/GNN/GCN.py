import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset
import numpy as np
import sys
import os
from sklearn.utils import shuffle
from tqdm import tqdm
import networkx as nx
from OrbitFeaturizer import OrbitFeaturizer
from torchmetrics.classification import BinaryAUROC
from os import listdir
from os.path import isfile, join
from ThisDataset import ThisDataset
from OrbitFeaturizer import OrbitFeaturizer
from torchmetrics.classification import BinaryAUROC
from os import listdir
from os.path import isfile, join

from sklearn.metrics import roc_auc_score

addFile = "/home/bcypher/IdeaProjects/Orbit/articleOrbits.csv"
dataset= ThisDataset(root="data/", n_graphs=5000,sampleWhite= 50, filepath="/home/bcypher/PycharmProjects/PytorchTutorial/data/Orbit/graphs/", addressFile=addFile)

data=dataset[0]
# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Dataset num node features: {dataset.num_node_features:.2f}')
print(f'Dataset num classes: {dataset.num_classes:.2f}')
print(f'Dataset num graphs: {len(dataset)}')
#print(f'Dataset classes: {dataset.vals}')
dataset = dataset.shuffle()

#train_dataset = dataset[:int(0.9* len(dataset))]
#test_dataset = dataset[int(0.9* len(dataset)):]

# print(f'Number of training graphs: {len(train_dataset)}')
# print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.loader import DataLoader

#train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x





def train(train_loader,model,criterion,optimizer):
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


for duplication in range(0,5):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, 128, shuffle=True)
    test_loader = DataLoader(test_dataset, 128)
    model = GCN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(0, 1001):
        train(train_loader,model,criterion,optimizer)
        scores_tr = test(train_loader,model)
        train_acc = scores_tr[0]
        train_auc = scores_tr[1]
        scores_te = test(test_loader,model)
        test_acc = scores_te[0]
        test_auc = scores_te[1]
        if epoch % 10 == 0:
            print(f"Duplicate\t{duplication},Epoch\t {epoch}\t "
                  f"Train Accuracy\t {train_acc:.4f}\t Train AUC Score: {train_auc:.4f}\t "
                  f"Test Accuracy: {test_acc:.4f}\t Test AUC Score: {test_auc:.4f}")


