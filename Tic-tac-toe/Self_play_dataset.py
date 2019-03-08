from Parse import *
from Players import *

import numpy as np
from copy import deepcopy
import torch
from torch import utils
import torch.utils.data

N = 15
possible_moves = []
for i in range(N):
    for j in range(N):
        possible_moves.append([i, j])


player_one = OMGPlayer(1)
player_two = OMGPlayer(0)


def check_win_condition(last_move, board):
    n = 5
    x, y = last_move
    winner = board[x][y]

    # vertical
    for i in range(n):
        match_count = 0
        for j in range(i - n + y + 1, i + y + 1):
            if j < 0 or j >= N:
                continue
            if winner != board[x][j]:
                break
            match_count += 1
        if match_count == n:
            return [(x, i - n + y + 1), (x, i + y)]

    # horizontal
    for i in range(n):
        match_count = 0
        for j in range(i - n + x + 1, i + x + 1):
            if j < 0 or j >= N:
                continue
            if winner != board[j][y]:
                break
            match_count += 1
        if match_count == n:
            return [(i - n + x + 1, y), (i + x, y)]

    # diagonals
    for i in range(n):
        match_count = 0
        for j in range(i - n + 1, i + 1):
            if x + j < 0 or x + j >= N or y + j < 0 or y + j >= N:
                continue
            if winner != board[x + j][y + j]:
                break
            match_count += 1
        if match_count == n:
            return [(i - n + 1 + x, i - n + 1 + y), (i + x, i + y)]

    for i in range(n):
        match_count = 0
        for j in range(i - n + 1, i + 1):
            if x - j < 0 or x - j >= N or y + j < 0 or y + j >= N:
                continue
            if winner != board[x - j][y + j]:
                break
            match_count += 1
        if match_count == n:
            return [(n - i - 1 + x, i - n + 1 + y), (-i + x, i + y)]

    return None


def play():
    curr_player = player_one

    data_black = []
    labels_black = []
    data_white = []
    labels_white = []

    board = [[0] * N for i in range(N)]

    while True:
        cell = curr_player.move_(possible_moves, board)

        if curr_player == player_one:
            data_black.append(deepcopy(board))
            labels_black.append(cell[0] * 15 + cell[1])

            board[cell[0]][cell[1]] = 1
            curr_player = player_two
        else:
            data_white.append(deepcopy(board))
            labels_white.append(cell[0] * 15 + cell[1])

            board[cell[0]][cell[1]] = -1
            curr_player = player_one

        # win condition check
        cells = check_win_condition(cell, board)
        if cells:
            if curr_player == player_two:   # reversed because we switched player 4 lines above
                return 1, data_black, labels_black
            else:
                return 0, data_white, labels_white

        # full board condition check
        if len(possible_moves) == 0:
            return None, None, None

        possible_moves.pop(possible_moves.index(cell))


def create_dataset(games):
    data_black = []
    labels_black = []
    data_white = []
    labels_white = []

    for _ in range(games):
        winner, data, labels = play()
        if winner == 1:
            data_black += data
            labels_black += labels
        if winner == 0:
            data_white += data
            labels_white += labels

    # BLACK
    dataset_black = []
    if len(data_black) != 0:
        x = [np.array(data_black[i]) for i in range(len(data_black))]
        data_black = torch.stack([torch.from_numpy(i).type(torch.FloatTensor) for i in x])
        del x

        y = [labels_black[i] for i in range(len(labels_black))]
        labels_black = torch.stack([torch.tensor(i) for i in y])
        del y

        dataset_black = utils.data.TensorDataset(data_black, labels_black)

    del data_black
    del labels_black

    # WHITE
    dataset_white = []
    if len(data_white) != 0:
        x = [np.array(data_white[i]) for i in range(len(data_white))]
        data_white = torch.stack([torch.from_numpy(i).type(torch.FloatTensor) for i in x])
        del x

        y = [labels_white[i] for i in range(len(labels_white))]
        labels_white = torch.stack([torch.tensor(i) for i in y])
        del y

        dataset_white = utils.data.TensorDataset(data_white, labels_white)

    del data_white
    del labels_white

    return dataset_black, dataset_white


black, white = create_dataset(6)
print(len(black), len(white))
