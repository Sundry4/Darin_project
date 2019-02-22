import numpy as np
import pandas as pd
import torch

from torch import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import math
from Dataset import white_data_loader, black_data_loader


class Net(nn.Module):
    size = 15

    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.size * self.size * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 225)
        )

    def forward(self, x):
        x = x.view(-1, 1, 15, 15)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train(epoch, data_loader):
    # model.train()
    # exp_lr_scheduler.step()

    batches = len(data_loader)
    percent = {int(batches * 1 / 5): 20,
               int(batches * 2 / 5): 40,
               int(batches * 3 / 5): 60,
               int(batches * 4 / 5): 80,
               batches - 1: 100}
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx in percent:
            print("{}% ready".format(percent[batch_idx]))

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        print("Nani??")
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

    print("Training finished\n")


batch_size = 50

model = Net()
optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)

for i in range(1):
    train(i, white_data_loader)

