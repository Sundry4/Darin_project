import numpy as np
import pandas as pd
import torch

from torch import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from Dataset import white_train_loader, white_test_loader
from Dataset import black_train_loader, black_test_loader
from Net import Net


def train(epoch, data_loader):
    # model.train()
    # exp_lr_scheduler.step()

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
        if k % 50 == 0:
            print(loss.item())
        optimizer.step()

    print()


def test_model(data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0

        for data, target in data_loader:
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = model(data)

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('Accuracy: {}/{} ({:.3f}%)\n'.format(correct, len(data_loader.dataset),
                                               100. * correct / len(data_loader.dataset)))


batch_size = 32

model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.003)

criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)

for i in range(40):
    train(i, white_train_loader)
test_model(white_test_loader)
test_model(white_train_loader)


path = 'model.pth'
torch.save(model.state_dict(), path)
