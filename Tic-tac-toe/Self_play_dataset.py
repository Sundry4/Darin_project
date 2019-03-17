from Parse import *
from Players import *

import numpy as np
from copy import deepcopy
import torch
from torch import utils
import torch.utils.data


class Self_play_dataset:
    def __init__(self):
        self.dataset = []
        self.labels = []
        self.board_size = 15
        self.player_one = AI_Player(1)
        self.player_two = AI_Player(0)
        self.possible_moves = set()

        self.black_pos = []
        self.white_pos = []
        self.turn = []
        self.hist_1_black = []
        self.hist_1_white = []
        self.hist_2_black = []
        self.hist_2_white = []

        self.simulation = []

    def clear(self):
        self.possible_moves = {*range(self.board_size**2)}

        self.black_pos = np.array([[0] * self.board_size for _ in range(self.board_size)])
        self.white_pos = np.array([[0] * self.board_size for _ in range(self.board_size)])
        self.turn = np.array([[1] * self.board_size for _ in range(self.board_size)])  # 1 for black, -1 for white
        self.hist_1_black = np.array([[0] * self.board_size for _ in range(self.board_size)])
        self.hist_1_white = np.array([[0] * self.board_size for _ in range(self.board_size)])
        self.hist_2_black = np.array([[0] * self.board_size for _ in range(self.board_size)])
        self.hist_2_white = np.array([[0] * self.board_size for _ in range(self.board_size)])

        self.simulation = []


    def check_win_condition(self, last_move):
        N = self.board_size
        n = 5
        x, y = last_move

        board = self.white_pos
        if self.turn[0][0] == 1:
            board = self.black_pos

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

    def put_X(self, cell):
        self.hist_2_black = deepcopy(self.hist_1_black)
        self.hist_1_black = deepcopy(self.black_pos)
        self.black_pos[cell[0]][cell[1]] = 1

    def put_O(self, cell):
        self.hist_2_white = deepcopy(self.hist_1_white)
        self.hist_1_white = deepcopy(self.white_pos)
        self.white_pos[cell[0]][cell[1]] = 1

    def play(self):
        self.clear()

        curr_player = self.player_one
        while True:
            state = torch.from_numpy(
                np.stack(
                    (self.black_pos, self.white_pos, self.turn,
                     self.hist_1_black, self.hist_1_white,
                     self.hist_2_black, self.hist_2_white)
                )
            )
            self.simulation.append(deepcopy(state))
            cell = curr_player.move_(self.possible_moves, state)
            self.possible_moves.remove(cell[0] * self.board_size + cell[1])

            if curr_player == self.player_one:
                self.put_X(cell)
                curr_player = self.player_two
            else:
                self.put_O(cell)
                curr_player = self.player_one

            # win condition check
            cells = self.check_win_condition(cell)
            if cells:
                winner = 1
                if curr_player == self.player_two:
                    winner = -1
                self.end(winner)
                return

            self.turn *= -1

            # full board condition check
            if len(self.possible_moves) == 0:
                self.end(0)
                return


    def end(self, winner):
        print(len(self.simulation))
        curr_player = self.player_one
        for board in self.simulation:
            self.dataset.append(board)
            self.labels.append([curr_player.get_policy(board, self.possible_moves),
                                [winner] * self.board_size**2])

        print("Winner, winner chicken dinner")

    def create_dataset(self):
        x = [np.array(self.dataset[i]) for i in range(len(self.dataset))]
        data_x = torch.stack([torch.from_numpy(i).cuda().type(torch.FloatTensor) for i in x])
        del x

        y = [self.labels[i] for i in range(len(self.labels))]
        data_y = torch.stack([torch.tensor(i) for i in y])
        del y

        dataset = utils.data.TensorDataset(data_x, data_y)
        del data_x
        del data_y

        return dataset

    def form_dataset(self):
        batch_size = 64

        dataset = self.create_dataset()

        train, test = torch.utils.data.random_split(dataset, (len(dataset) - 10, 10))
        del dataset

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
        del train
        del test

        return train_loader, test_loader


game = Self_play_dataset()
for _ in range(5):
    game.play()

