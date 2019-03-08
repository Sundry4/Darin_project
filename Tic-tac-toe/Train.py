import numpy as np
import pandas as pd
import torch
import time
import warnings

warnings.filterwarnings("ignore")

from torch import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

start = time.clock()

from Dataset_shift import white_train_loader, white_test_loader
from Dataset_shift import black_train_loader, black_test_loader
from Net import *

print("Imported:", time.clock() - start)
start = time.clock()


def train(epoch, data_loader):
    model.train()
    exp_lr_scheduler.step()

    k = 0
    for data, target in data_loader:
        k += 1

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        # print(*output)
        loss = criterion(output, target)
        loss.backward()
        if k % 1000 == 0:
            print(loss.item())
        optimizer.step()

    print()


def V_train(epoch, data_loader):
    alpha = 0.8

    model.train()
    exp_lr_scheduler.step()

    k = 0
    for data, target in data_loader:
        k += 1

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output_p, output_V = model(data)

        target_p = target[:, 0]
        target_V = target[:, 1]
        target_V = target_V.float()

        loss_p = criterion_p(output_p, target_p)
        loss_V = criterion_V(output_V, target_V)

        loss = loss_V * alpha
        if target_V[0] == 1:
            loss += loss_p

        loss.backward()
        if k % 10000 == 0:
            print(loss.item())
        optimizer.step()

    print()


def test_model(data_loader):
    model.eval()
    with torch.no_grad():
        correct_p = 0

        for data, target in data_loader:
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            target_p = target[:, 0]

            output = model(data)
            output_p = output[0]

            pred = output_p.data.max(1, keepdim=True)[1]
            correct_p += pred.eq(target_p.data.view_as(pred)).cuda().sum()

    print('Accuracy: {}/{} ({:.3f}%)\n'.format(correct_p, len(data_loader.dataset),
                                               100. * correct_p / len(data_loader.dataset)))


# ------------  white  ------------

model_old = Net()
model_old.load_state_dict(torch.load("model_white11.pth"))
model_old.eval()

model = VNet()
model.features = model_old.features

optimizer = optim.Adam(model.parameters(), lr=0.003)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
criterion_p = nn.CrossEntropyLoss()
criterion_V = nn.MSELoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion_p = criterion_p.cuda()
    criterion_V = criterion_V.cuda()

for i in range(20):
    print("EPOCH:", i + 1)
    V_train(i, white_train_loader)

test_model(white_test_loader)
test_model(white_train_loader)

path = 'model_white_V.pth'
torch.save(model.state_dict(), path)

print("White:", time.clock() - start)
start = time.clock()

# ------------  black  ------------

model_old = Net()
model_old.load_state_dict(torch.load("model_black11.pth"))
model_old.eval()

model = VNet()
model.features = model_old.features

optimizer = optim.Adam(model.parameters(), lr=0.003)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
criterion_p = nn.CrossEntropyLoss()
criterion_V = nn.MSELoss()

if torch.cuda.is_available():
    criterion_p = criterion_p.cuda()
    criterion_V = criterion_V.cuda()

for i in range(20):
    print("EPOCH:", i + 1)
    train_V(i, black_train_loader)
test_model(black_test_loader)
test_model(black_train_loader)

path = 'model_black_V.pth'
torch.save(model.state_dict(), path)

print("Black:", time.clock() - start)
