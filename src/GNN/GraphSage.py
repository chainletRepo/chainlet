#
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import torch
from torch import nn
from torch.nn import functional as F
import yaml
from torch_geometric.nn import SAGEConv, global_max_pool
from ThisDataset import ThisDataset
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import random

random.seed(42)
class GraphSAGE(nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super().__init__()

        num_layers = config['num_layers'][1]
        dim_embedding = config['dim_embedding'][1]
        self.aggregation = config['aggregation'][1]  # can be mean or max

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_features if i == 0 else dim_embedding

            conv = SAGEConv(dim_input, dim_embedding)
            # Overwrite aggregation method (default is set to mean
            conv.aggr = self.aggregation

            self.layers.append(conv)

        # For graph classification
        self.fc1 = nn.Linear(num_layers * dim_embedding, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = global_max_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


with open("config_GraphSAGE.yml", "r") as f:
    config = yaml.load(f)

addFile = "/home/bcypher/IdeaProjects/Orbit/articleOrbits.csv"
dataset= ThisDataset(root="data/", n_graphs=40000,sampleWhite= 50, filepath="/home/bcypher/PycharmProjects/PytorchTutorial/data/Orbit/graphs/", addressFile=addFile)
dataset = dataset.shuffle()
from torch_geometric.loader import DataLoader
#train_dataset = dataset[len(dataset) // 10:]
#train_loader = DataLoader(train_dataset, 512, shuffle=True)

#test_dataset = dataset[:len(dataset) // 10]
#test_loader = DataLoader(test_dataset, 512)


def train(train_loader,model,criterion,optimizer):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data)  # Perform a single forward pass.
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
        out = model(data)
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
    model = GraphSAGE(dim_features=dataset.num_features, dim_target=dataset.num_classes, config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(0, 101):
        train(train_loader,model,criterion,optimizer)
        scores_tr = test(train_loader,model)
        train_acc = scores_tr[0]
        train_auc = scores_tr[1]
        scores_te = test(test_loader,model)
        test_acc = scores_te[0]
        test_auc = scores_te[1]
        print(f'Duplication:{duplication},Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Train Auc: {train_auc:.4f}, Test Auc: {test_auc:.4f}')


