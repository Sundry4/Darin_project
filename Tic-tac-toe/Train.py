import numpy as np
import pandas as pd
import torch
import time

from torch import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from Layer7_Dataset import form_dataset
from Net import *
import warnings
warnings.filterwarnings("ignore")

start = time.clock()


def train(data_loader):
    model.train()
    exp_lr_scheduler.step()

    for k, (data, target) in enumerate(data_loader):
        x = data.to(device)
        y = target.to(device)

        optimizer.zero_grad()
        output_p, _ = model(x)

        loss = criterion(output_p, y)
        loss.backward()
        optimizer.step()

        if k % 100 == 0:
            print(loss.item())

    print()


def test_model(data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0

        for data, target in data_loader:
            x = data.to(device)
            y = target.to(device)

            output_p, _ = model(x)

            pred = output_p.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cuda().sum()

    print('Accuracy: {}/{} ({:.3f}%)\n'.format(correct, len(data_loader.dataset),
                                               100. * correct / len(data_loader.dataset)))


# ------------  white  ------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net()

optimizer = optim.Adam(model.parameters(), lr=0.003)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

data_train, data_test = form_dataset(0)
for i in range(20):
    print("EPOCH:", i + 1)
    train(data_train)

test_model(data_test)
test_model(data_train)

path = 'model7_white.pth'
torch.save(model.state_dict(), path)

del data_train
del data_test

print("White:", time.clock() - start)
start = time.clock()

# ------------  black  ------------

model = Net()

optimizer = optim.Adam(model.parameters(), lr=0.003)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)


data_train, data_test = form_dataset(1)
for i in range(40):
    print("EPOCH:", i + 1)
    train(data_train)
test_model(data_test)
test_model(data_train)

path = 'model7_black.pth'
torch.save(model.state_dict(), path)

print("Black:", time.clock() - start)
