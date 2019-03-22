import numpy as np
import torch
import time

from torch import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from MCTS import *
from Self_play_dataset import Self_play_dataset
from Net import *
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


def train_RL(data_loader):
    epochs = 10
    for i in range(epochs):
        # print("EPOCH:", i + 1)
        model.train()
        exp_lr_scheduler.step()
        for k, (data, target_p, target_v) in enumerate(data_loader):
            data = data.to(device)
            target_v = target_v.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target_v)
            loss.backward()
            optimizer.step()

            # if k % 20:
            #     print(loss.item())


def test_model(data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0

        for data, target_p, target_v in data_loader:
            data = data.to(device)
            target_v = target_v.to(device)

            output = model(data)

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target_v.data.view_as(pred)).cuda().sum()

    print('Accuracy: {}/{} ({:.3f}%)\n'.format(correct, len(data_loader.dataset),
                                               100. * correct / len(data_loader.dataset)))


if __name__ == "__main__":
    run()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = VNet()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    criterion = nn.CrossEntropyLoss().to(device)

    for number in range(2, 40):
        print("BIG EPOCH:", number)

        mcts = MCTS(number, number)
        player_one = AI_Player(1, mcts)  # 1 - for 'X', 0 - for 'O'
        player_two = AI_Player(0, mcts)

        # for j in range(5):
        game = Self_play_dataset(mcts)
        game.play()
        dataset = game.form_dataset()

        model.load_state_dict(torch.load(path.format(number)))
        model.eval().to(device)

        train_RL(dataset)
        test_model(dataset)

        # game.full_clear()

        torch.save(model.state_dict(), path.format(number + 1))
