import numpy as np
import pandas as pd
import torch
import time

from torch import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import os

from Layer11_V_Dataset import form_dataset
from New_Net import VNet
import warnings
warnings.filterwarnings("ignore")

start = time.clock()


def gpu_to_common(model_):
    new_state_dict = {}
    for key, value in model_.state_dict().items():
        new_key = key[7:]
        new_state_dict[new_key] = value

    return new_state_dict


def run():
    torch.multiprocessing.freeze_support()


def train(data_loader):
    model.train()
    scheduler.step()

    for k, (data, target) in enumerate(data_loader):
        x = data.to(device)
        y = target.to(device)

        optimizer.zero_grad()
        output = model(x)

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if k % 100000 == 0:
            print(loss.item())

    print()


def test_model(data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0

        for data, target in data_loader:
            x = data.to(device)
            y = target.to(device)

            output = model(x)

            for i in range(y.size(0)):
                if y[i] * output[i] > 0:
                    correct += 1

    print('Accuracy: {}/{} ({:.3f}%)\n'.format(correct, len(data_loader.dataset),
                                               100. * correct / len(data_loader.dataset)))


if __name__ == "__main__":
    run()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    size = 10
    for number in range(1, 2):
        print("CURR BATCH:", number, '\n')

        path = 'model11_V_{}.pth'
        model = VNet()
        # model.load_state_dict(torch.load(path.format(number)))
        # model.eval()

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=0.003)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
        criterion = nn.MSELoss().to(device)

        model = model.to(device)

        data_train, data_test = form_dataset((number - 1) * size, number * size)
        for i in range(40):
            print("EPOCH:", i + 1)
            train(data_train)

        test_model(data_test)
        test_model(data_train)

        torch.save(gpu_to_common(model), path.format(number + 1))

        del data_train
        del data_test

        print("BATCH ENDED:", time.clock() - start)
        start = time.clock()
