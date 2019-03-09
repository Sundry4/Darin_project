from Parse import data_white, data_black
import numpy as np
from copy import deepcopy
import torch
from torch import utils
import torch.utils.data
import random

field_size = 15


def find_max_min(game):
    max_x = game[0][0]
    min_x = game[0][0]
    max_y = game[0][1]
    min_y = game[0][1]

    for i in game:
        max_x = max(max_x, i[0])
        min_x = min(min_x, i[0])
        max_y = max(max_y, i[1])
        min_y = min(min_y, i[1])

    shifts = [(field_size - max_x, 0), (-min_x, 0), (0, field_size - max_y), (0, -max_y)]
    return shifts


# returns tensors of boards (1 - black stone, -1 - white stone, 0 - empty cell)
# and labels (stone positions which are numbers from 0 to 224)
def create_dataset_V_no_shift(data, V, player):
    boards = []
    labels = []
    for game in data:
        board = [[0] * field_size for i in range(field_size)]
        is_black = True
        for turn in game:
            if player == is_black:
                boards.append(deepcopy(board))
                labels.append([(turn[0] - 1) * 15 + turn[1] - 1, V])

            if is_black:
                board[turn[0] - 1][turn[1] - 1] = 1
            else:
                board[turn[0] - 1][turn[1] - 1] = -1

            is_black = not is_black

    x = [np.array(boards[i]) for i in range(len(boards))]
    del boards
    data_x = torch.stack([torch.from_numpy(i).cuda().type(torch.FloatTensor) for i in x])
    del x

    y = [labels[i] for i in range(len(labels))]
    del labels
    data_y = torch.stack([torch.tensor(i).cuda() for i in y])
    del y

    dataset = utils.data.TensorDataset(data_x, data_y)
    del data_x
    del data_y

    return dataset


def create_dataset_V(data, V, player):
    boards = []
    labels = []
    for game in data:
        for shift in find_max_min(game):
            board = [[0] * field_size for i in range(field_size)]
            is_black = True
            for turn in game:
                if player == is_black:
                    boards.append(deepcopy(board))
                    labels.append([(turn[0] - 1 + shift[0]) * 15 + turn[1] - 1 + shift[1], V])

                if is_black:
                    board[turn[0] - 1][turn[1] - 1] = 1
                else:
                    board[turn[0] - 1][turn[1] - 1] = -1

                is_black = not is_black

    x = [np.array(boards[i]) for i in range(len(boards))]
    del boards
    data_x = torch.stack([torch.from_numpy(i).cuda().type(torch.FloatTensor) for i in x])
    del x

    y = [labels[i] for i in range(len(labels))]
    del labels
    data_y = torch.stack([torch.tensor(i).cuda() for i in y])
    del y

    dataset = utils.data.TensorDataset(data_x, data_y)
    del data_x
    del data_y

    return dataset


def form_dataset(player):
    V = player * 2 - 1

    dataset_1 = create_dataset_V(data_black, V, player)
    train_1, test_1 = torch.utils.data.random_split(dataset_1, (len(dataset_1) - 1000, 1000))
    del dataset_1

    dataset_2 = create_dataset_V(data_white, -V, player)
    train_2, test_2 = torch.utils.data.random_split(dataset_2, (len(dataset_2) - 1000, 1000))
    del dataset_2

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([train_1,
                                        train_2]),
        batch_size=64,
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([test_1,
                                        test_2])
    )

    return train_loader, test_loader
