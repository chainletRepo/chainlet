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
from torch_geometric.nn import MessagePassing, global_sort_pool
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import roc_auc_score
from ThisDataset import ThisDataset
import random
import yaml
random.seed(42)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
class DGCNN(nn.Module):
    """
    Uses fixed architecture
    """

    def __init__(self, dim_features, dim_target, config):
        super(DGCNN, self).__init__()

        self.ks = {'NCI1': { '0.6': 30, '0.9': 46 },
                   'PROTEINS_full': { '0.6': 32, '0.9': 81 },
                   'DD': {'0.6': 291, '0.9': 503 },
                   'ENZYMES': { '0.6': 36, '0.9': 48 },
                   'IMDB-BINARY': { '0.6': 18, '0.9': 31 },
                   'IMDB-MULTI': { '0.6': 11, '0.9': 22 },
                   'REDDIT-BINARY': { '0.6': 370, '0.9': 1002 },
                   'REDDIT-MULTI-5K': { '0.6': 469, '0.9': 1081 },
                   'COLLAB': { '0.6': 61, '0.9': 130 },
                   }

        self.k = 30#self.ks[config.dataset.name][str(config['k'])]
        self.embedding_dim = config['embedding_dim'][0]
        self.num_layers = config['num_layers'][0]

        self.convs = []
        for layer in range(self.num_layers):
            input_dim = dim_features if layer == 0 else self.embedding_dim
            self.convs.append(DGCNNConv(input_dim, self.embedding_dim))
        self.total_latent_dim = self.num_layers * self.embedding_dim

        # Add last embedding
        self.convs.append(DGCNNConv(self.embedding_dim, 1))
        self.total_latent_dim += 1

        self.convs = nn.ModuleList(self.convs)

        # should we leave this fixed?
        self.conv1d_params1 = nn.Conv1d(1, 16, self.total_latent_dim, self.total_latent_dim)
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(16, 32, 5, 1)

        dense_dim = int((self.k - 2) / 2 + 1)
        self.input_dense_dim = (dense_dim - 5 + 1) * 32

        self.hidden_dense_dim = config['dense_dim'][0]
        self.dense_layer = nn.Sequential(nn.Linear(self.input_dense_dim, self.hidden_dense_dim),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(self.hidden_dense_dim, dim_target))

    def forward(self, data):
        # Implement Equation 4.2 of the paper i.e. concat all layers' graph representations and apply linear model
        # note: this can be decomposed in one smaller linear model per layer
        x, edge_index, batch = data.x, data.edge_index, data.batch

        hidden_repres = []

        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            hidden_repres.append(x)

        # apply sortpool
        x_to_sortpool = torch.cat(hidden_repres, dim=1)
        x_1d = global_sort_pool(x_to_sortpool, batch, self.k)  # in the code the authors sort the last channel only

        # apply 1D convolutional layers
        x_1d = torch.unsqueeze(x_1d, dim=1)
        conv1d_res = F.relu(self.conv1d_params1(x_1d))
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = F.relu(self.conv1d_params2(conv1d_res))
        conv1d_res = conv1d_res.reshape(conv1d_res.shape[0], -1)

        # apply dense layer
        out_dense = self.dense_layer(conv1d_res)
        return out_dense


class DGCNNConv(MessagePassing):
    """
    Extended from tuorial on GCNs of Pytorch Geometrics
    """

    def __init__(self, in_channels, out_channels):
        super(DGCNNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        src, dst = edge_index  # we assume source_to_target message passing
        deg = degree(src, size[0], dtype=x_j.dtype)
        deg = deg.pow(-1)
        norm = deg[dst]

        return norm.view(-1, 1) * x_j  # broadcasting the normalization term to all out_channels === hidden features

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


with open("config_DGCNN.yml", "r") as f:
    config = yaml.load(f)
addFile = "/home/bcypher/IdeaProjects/Orbit/articleOrbits.csv"
dataset= ThisDataset(root="data/", n_graphs=5000,sampleWhite= 50, filepath="/home/bcypher/PycharmProjects/PytorchTutorial/data/Orbit/graphs/", addressFile=addFile)
dataset = dataset.shuffle()
from torch_geometric.loader import DataLoader
print(f'Dataset num node features: {dataset.num_node_features:.2f}')
print(f'Dataset num classes: {dataset.num_classes:.2f}')
print(f'Dataset num graphs: {len(dataset)}')
#print(f'Dataset classes: {dataset.vals}')

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
    model = DGCNN(dim_features=dataset.num_features, dim_target=dataset.num_classes, config=config)
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
            print(
                f"Duplicate\t{duplication}\tEpoch\t {epoch}\t Train Accuracy\t {train_acc:.4f}\t Train AUC Score\t {train_auc:.4f}\t Test Accuracy: {test_acc:.4f}\t Test AUC Score\t {test_auc:.4f}")


