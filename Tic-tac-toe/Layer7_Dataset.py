from Parse import parse
import numpy as np
from copy import deepcopy
import torch
from torch import utils
import torch.utils.data
import random
import warnings
warnings.filterwarnings("ignore")


board_size = 15


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

    shifts = [(board_size - max_x, 0), (-min_x + 1, 0), (0, board_size - max_y), (0, -min_y + 1),
              (board_size - max_x, board_size - max_y), (-min_x + 1, board_size - max_y),
              (-min_x + 1, board_size - max_y), (-min_x + 1, -min_y + 1)]
    return shifts


# 1 layer - black positions
# 2 layer - white positions
# 3 layer - turn
# 4 layer - black positions one turn ago
# 5 layer - white positions one turn ago
# 6 layer - black positions two turns ago
# 7 layer - white positions two turns ago
def create_dataset(player, start, end):
    white, black = parse(start, end)

    data = white
    if player == 1:
        data = black

    x = []
    y = []
    for game in data:
        if len(game) < 2:
            continue

        shifts = [random.choice(find_max_min(game)), (0, 0)]

        for shift in shifts:
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
                             hist_2_black, hist_2_white)
                        ))
                    )
                    y.append((move[0] - 1 + shift[0]) * 15 + move[1] - 1 + shift[1])

                turn *= -1

                hist_2_black = deepcopy(hist_1_black)
                hist_2_white = deepcopy(hist_1_white)

                hist_1_black = deepcopy(black_pos)
                hist_1_white = deepcopy(white_pos)

                if is_black:
                    black_pos[move[0] - 1 + shift[0]][move[1] - 1 + shift[1]] = 1
                else:
                    white_pos[move[0] - 1 + shift[0]][move[1] - 1 + shift[1]] = 1

                is_black = not is_black

    X = [np.array(x[i]) for i in range(len(x))]
    data_x = torch.stack([torch.from_numpy(i).cuda().type(torch.FloatTensor) for i in X])
    del X

    Y = [y[i] for i in range(len(y))]
    data_y = torch.stack([torch.tensor(i) for i in Y])
    del Y

    dataset = utils.data.TensorDataset(data_x, data_y)
    del data_x
    del data_y

    return dataset


def form_dataset(player, start, end):
    batch_size = 2048

    dataset = create_dataset(player, start, end)
    train, test = torch.utils.data.random_split(dataset, (len(dataset) - 1000, 1000))
    del dataset

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True,
                                               num_workers=torch.cuda.device_count() * 4, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, pin_memory=True,
                                              num_workers=torch.cuda.device_count() * 4, drop_last=False)
    del train
    del test

    return train_loader, test_loader
