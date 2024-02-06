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
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
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

class GIN(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(GIN, self).__init__()

        self.config = config
        self.dropout = config['dropout'][0]
        self.embeddings_dim = [config['hidden_units'][0][0]] + config['hidden_units'][0]
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []

        train_eps = config['train_eps'][0]
        if config['aggregation'][0] == 'sum':
            self.pooling = global_add_pool
        elif config['aggregation'][0] == 'mean':
            self.pooling = global_mean_pool

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                    Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                      Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2

                self.linears.append(Linear(out_emb_dim, dim_target))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0

        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(self.pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer-1](x, edge_index)
                out += F.dropout(self.linears[layer](self.pooling(x, batch)), p=self.dropout, training=self.training)

        return out


with open("config_GIN.yml", "r") as f:
    config = yaml.load(f)
addFile = "/home/bcypher/IdeaProjects/Orbit/articleOrbits.csv"
dataset= ThisDataset(root="data/", n_graphs=48000,sampleWhite= 50, filepath="/home/bcypher/PycharmProjects/PytorchTutorial/data/Orbit/graphs/", addressFile=addFile)
dataset_test=dataset[40000:48000]
print(f'Dataset_test num graphs: {len(dataset_test)}')
dataset = dataset[:20000].shuffle()
print(f'Dataset num graphs: {len(dataset)}')

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
    unseen_loader = DataLoader(dataset_test, 128)
    model = GIN(dim_features=dataset.num_features, dim_target=dataset.num_classes, config=config)
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
        scores_unseen = test(test_loader, model)
        unseen_acc = scores_unseen[0]
        unseen_auc = scores_unseen[1]
        if epoch % 10 == 0:
            print(f"Duplicate\t{duplication}\tEpoch\t {epoch}\t Train Accuracy\t {train_acc:.4f}\t Train AUC Score\t {train_auc:.4f}\t Test Accuracy: {test_acc:.4f}\t test AUC Score\t {test_auc:.4f}\t unseen AUC Score\t {unseen_auc:.4f}")


