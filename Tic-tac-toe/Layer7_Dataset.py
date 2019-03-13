from Parse import parse
import numpy as np
from copy import deepcopy
import torch
from torch import utils
import torch.utils.data
import random
# from keras.datasets import cifar10
import warnings
warnings.filterwarnings("ignore")


path = 'C:/Users/ashab.DESKTOP-4CJ6TE5/Home Work/Darin/train-1.renju'
board_size = 15

start, end = 0, 500
white, black = parse(path, start, end)


def create_dataset_white():
    data = []
    labels = []
    for game in white:
        board = [[0] * board_size for i in range(board_size)]
        is_black = True
        for turn in game:
            if is_black:
                board[turn[0] - 1][turn[1] - 1] = 1
            else:
                data.append(deepcopy(board))
                labels.append((turn[0] - 1) * 15 + turn[1] - 1)
                board[turn[0] - 1][turn[1] - 1] = -1

            is_black = not is_black

    with torch.cuda.device(0):
        x = [np.array(data[i]) for i in range(len(data))]
        data_x = torch.stack([torch.from_numpy(i).cuda().type(torch.FloatTensor) for i in x])

        y = [labels[i] for i in range(len(labels))]
        data_y = torch.stack([torch.tensor(i).cuda() for i in y])

        dataset = utils.data.TensorDataset(data_x, data_y)

    del data
    del labels
    del x
    del y
    del data_x
    del data_y
    return dataset


# 1 layer - black positions
# 2 layer - white positions
# 3 layer - turn
# 4 layer - black positions one turn ago
# 5 layer - white positions one turn ago
# 6 layer - black positions two turns ago
# 7 layer - white positions two turns ago
def create_dataset(player):
    data = white
    if player == 1:
        data = black

    x = []
    y = []
    for game in data:
        black_pos = np.array([[0] * board_size for _ in range(board_size)])
        white_pos = np.array([[0] * board_size for _ in range(board_size)])
        turn = np.array([[1] * board_size for _ in range(board_size)])  # 1 for black, -1 for white
        hist_1_black = np.array([[0] * board_size for _ in range(board_size)])
        hist_1_white = np.array([[0] * board_size for _ in range(board_size)])
        hist_2_black = np.array([[0] * board_size for _ in range(board_size)])
        hist_2_white = np.array([[0] * board_size for _ in range(board_size)])

        is_black = True
        for k, move in enumerate(game):
            if player == is_black:
                x.append(torch.from_numpy(
                    np.stack(
                        (black_pos, white_pos, turn,
                         hist_1_black, hist_1_white,
                         hist_2_black, hist_2_white),
                        axis=0
                    ))
                )
                y.append((move[0] - 1) * 15 + move[1] - 1)

            turn *= -1

            hist_2_black = deepcopy(hist_1_black)
            hist_2_white = deepcopy(hist_1_white)

            hist_1_black = deepcopy(black_pos)
            hist_1_white = deepcopy(white_pos)

            if is_black:
                black_pos[move[0] - 1][move[1] - 1] = 1
            else:
                white_pos[move[0] - 1][move[1] - 1] = 1

            is_black = not is_black

    X = [np.array(x[i]) for i in range(len(x))]
    data_x = torch.stack([torch.from_numpy(i).cuda().type(torch.cuda.FloatTensor) for i in X])
    del X

    Y = [y[i] for i in range(len(y))]
    data_y = torch.stack([torch.tensor(i).cuda() for i in Y])
    del Y

    dataset = utils.data.TensorDataset(data_x, data_y)
    del data_x
    del data_y

    return dataset


def form_dataset(player):
    batch_size = 64

    dataset = create_dataset(player)
    # print(dataset)
    # print(*dataset[0])
    train, test = torch.utils.data.random_split(dataset, (len(dataset) - 100, 100))
    del dataset

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    del train
    del test

    return train_loader, test_loader

