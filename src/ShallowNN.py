import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch import tensor
import tensorflow as tf
tf.config.run_functions_eagerly(True)

# This line detects if we have a gpu support on our system
device = ("cuda" if torch.cuda.is_available() else "cpu")


merged = "merged.csv"
merged_df = pd.read_csv(merged, sep=",", header=0)
merged_df = merged_df.drop(["Unnamed: 0"],axis=1)
merged_df["label"].replace({"montrealAPT":"virus","princetonLocky":"virus","montrealCryptoLocker": "virus", "montrealNoobCrypt": "virus",'montrealDMALocker':'virus','paduaCryptoWall':'virus','montrealCryptoTorLocker2015':'virus','montrealSamSam':'virus','montrealGlobeImposter':'virus','princetonCerber':'virus','montrealDMALockerv3':'virus','montrealGlobe':'virus'}, inplace=True)
merged_df["label"].replace({"virus":1,"white":0}, inplace=True)

X = merged_df[merged_df.columns[4:52]]  # Features
y = merged_df['label']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# converting the datatypes from numpy array into tensors of type float
X_train = torch.from_numpy(X_train.values).float().to(device)
X_test = torch.from_numpy(X_test.values).float().to(device)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
y_train = torch.from_numpy(y_train.squeeze()).type(torch.FloatTensor).view(-1, 1)
y_test = torch.from_numpy(y_test.squeeze()).type(torch.FloatTensor).view(-1, 1)

# checking the shape
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

class ShallowNeuralNetwork(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(ShallowNeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_num, hidden_num)  # hidden layer
        self.output = nn.Linear(hidden_num, output_num)  # output layer
        self.sigmoid = nn.Sigmoid()  # sigmoid activation function
        self.relu = nn.ReLU()  # relu activation function

    def forward(self, x):
        x = self.relu(self.hidden(x))
        out = self.output(x)
        return self.sigmoid(out)


input_num = 48
hidden_num = 2
output_num = 1  # The output should be the same as the number of classes

model = ShallowNeuralNetwork(input_num, hidden_num, output_num)
model.to(device)  # send our model to gpu if available else cpu.
print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

if torch.cuda.is_available():
    X_train = Variable(X_train).cuda()
    y_train = Variable(y_train).cuda()
    X_test = Variable(X_test).cuda()
    y_test = Variable(y_test).cuda()

num_epochs = 1000

total_acc, total_loss = [], []

for epoch in range(num_epochs):
    # forward propagation
    model.train()

    y_pred = model(X_train)
    pred = np.where(y_pred > 0.5, 1, 0)
    loss = criterion(y_pred, y_train)

    # back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        model.eval()
        y_pred_test = model(X_test)

        test_loss = criterion(y_pred_test, y_test)
        total_loss.append(test_loss.item())

        total = 0
        pred = np.where(y_pred_test > 0.5, 1, 0)
        for i in range(len(y_test)):
            if int(y_test[i]) == int(pred[i]):
                total += 1

        acc = total / len(y_test)
        total_acc.append(acc)

        print('Epoch [{}/{}], Train Loss: {:.5f}, Test Loss: {:.5f}, Accuracy: {:.5f}'.format(epoch, num_epochs,
                                                                                              loss.item(),
                                                                                              test_loss.item(), acc))
print('\nTraining Complete')

model.eval()
model_prediction_prob = model(X_test)
predictions_NN_01 = np.where(model_prediction_prob > 0.5, 1, 0)
acc_NN = accuracy_score(y_test, predictions_NN_01)
print(acc_NN)
false_positive_rate, recall, thresholds = roc_curve(y_test, predictions_NN_01)
roc_auc = auc(false_positive_rate, recall)
print(roc_auc)
