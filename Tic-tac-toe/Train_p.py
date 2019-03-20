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

from Layer11_Dataset import form_dataset
from New_Net import PNet
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
    exp_lr_scheduler.step()

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

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cuda().sum()

    print('Accuracy: {}/{} ({:.3f}%)\n'.format(correct, len(data_loader.dataset),
                                               100. * correct / len(data_loader.dataset)))


if __name__ == "__main__":
    run()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    size = 100
    for number in range(1, 2):
        print("CURR BATCH:", number, '\n')

        # ------------  white  ------------

        path = 'model7_white_{}.pth'
        model = PNet()
        # model.load_state_dict(torch.load(path.format(number)))
        # model.eval()

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=0.003)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
        criterion = nn.CrossEntropyLoss().to(device)

        model = model.to(device)

        data_train, data_test = form_dataset(0, (number - 1) * size, number * size)
        for i in range(10):
            print("EPOCH:", i + 1)
            train(data_train)

        test_model(data_test)
        test_model(data_train)

        torch.save(gpu_to_common(model), path.format(number))

        del data_train
        del data_test

        print("White:", time.clock() - start)
        start = time.clock()

        # ------------  black  ------------

        path = 'model7_black_{}.pth'
        model = PNet()
        # model.load_state_dict(torch.load(path.format(number)))
        # model.eval()

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=0.003)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
        criterion = nn.CrossEntropyLoss().to(device)

        model = model.to(device)

        data_train, data_test = form_dataset(1, (number - 1) * size, number * size)
        for i in range(10):
            print("EPOCH:", i + 1)
            train(data_train)
        test_model(data_test)
        test_model(data_train)

        torch.save(gpu_to_common(model), path.format(number))

        print("Black:", time.clock() - start)
