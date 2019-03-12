from Parse import parse
import numpy as np


path = 'C:/Users/ashab.DESKTOP-4CJ6TE5/Home Work/Darin/train-1.renju'
start, end = 0, 100
white, black = parse(path, start, end)
board_size = 15


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
        black_pos = [[0] * board_size for _ in range(board_size)]
        white_pos = [[0] * board_size for _ in range(board_size)]
        turn = [[1] * board_size for _ in range(board_size)]  # 1 for black, -1 for white
        hist_1_black = [[0] * board_size for _ in range(board_size)]
        hist_1_white = [[0] * board_size for _ in range(board_size)]
        hist_2_black = [[0] * board_size for _ in range(board_size)]
        hist_2_white = [[0] * board_size for _ in range(board_size)]

        is_black = True
        for k, move in enumerate(game):
            if is_black:
                black_pos[move[0] - 1][move[1] - 1] = 1
            else:
                white_pos[move[0] - 1][move[1] - 1] = 1

            x.append(
                np.stack(
                    (black_pos, white_pos, turn,
                     hist_1_black, hist_1_white,
                     hist_2_black, hist_2_white),
                    axis=-1
                )
            )
            y.append((move[0] - 1) * 15 + move[1] - 1)

            turn *= -1

            hist_2_black = hist_1_black
            hist_2_white = hist_1_white

            hist_1_black = black_pos
            hist_1_white = white_pos

            is_black = not is_black

    X = [np.array(x[i]) for i in range(len(x))]
    data_x = torch.stack([torch.from_numpy(i).type(torch.FloatTensor) for i in x])
    del X

    Y = [y[i] for i in range(len(y))]
    data_y = torch.stack([torch.tensor(i) for i in y])
    del Y

    dataset = utils.data.TensorDataset(data_x, data_y)
    del data_x
    del data_y

    return dataset
