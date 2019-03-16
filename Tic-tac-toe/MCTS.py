import numpy as np
from copy import deepcopy
import time

import torch
from torch import utils
import torch.nn as nn
import torch.nn.functional as F


class MCTS:
    def __init__(self):
        self.model_black = Net()
        self.model_black.load_state_dict(torch.load("model_black.pth"))
        self.model.eval()

        self.model_white = Net()
        self.model_white.load_state_dict(torch.load("model_white.pth"))
        self.model.eval()

        self.board_size = 15
    def get_data(self, board):
        model = self.model_black

        # костыыыыыыль! :)
        # слой хода 3-тий в списке слоев состояния
        if board[2][0][0].item() == -1:
            model = self.model_white

        output, v = model(board)
        return F.softmax(output, dim=1).detach().numpy()[0], v.item()

    def get_policy(self, board, iterations, possible_moves):
        policy, v = self.get_data(board)
        N = [0] * self.board_size**2
        data = {'': [policy, N, Q]}

        # making simulations, every single one starts from root state
        for _ in range(iterations): # it may be better to change it to time
            data = step(data, possible_moves, board)

        return F.softmax(data[''][1], dim=1) # returning distribution adjusted by N

    def update_board(self, board, simulation):
        black_pos, white_pos, turn, hist_1_black,\
        hist_1_white, hist_2_black, hist_2_white = board

        is_black = False
        if turn[0][0] == 1:
            is_black = True

        for move in simulation:
            hist_2_black = deepcopy(hist_1_black)
            hist_2_white = deepcopy(hist_1_white)

            hist_1_black = deepcopy(black_pos)
            hist_1_white = deepcopy(white_pos)

            if is_black:
                black_pos[move[0] - 1][move[1] - 1] = 1
            else:
                white_pos[move[0] - 1][move[1] - 1] = 1
            turn *= -1

        board = np.stack((black_pos, white_pos, turn,
                          hist_1_black, hist_1_white,
                          hist_2_black, hist_2_white)
                         )
        return board

    # function, that converts set to string, because sets aren't suitable for dicts as keys :(
    def convert(self, x):
        string = ''
        for i in x:
            string += str(i) + ','
        return string

    def step(self, data, possible_moves, board):
        made_moves = set()
        simulation = []
        while made_moves in data.keys():
            curr_set = self.convert(made_moves)

            P = data[curr_set][0]  # prior probabilities
            N = data[curr_set][1]  # visit count
            U = [P[i] / (1 + N[i]) for i in range(self.board_size**2)]
            Q = data[curr_set][2]  # action value

            # extruding an index of max elem of U+Q vector
            U_Q = np.array(U) + np.array(Q)
            move = torch.from_numpy(U_Q).data.max(0, keepdim=True)[1].item()
            while move not in possible_moves:
                U_Q[move] -= 1
                move = torch.from_numpy(U_Q).data.max(0, keepdim=True)[1].item()

            possible_moves.remove(move)
            made_moves.add(move)
            simulation.append(move)


        board = self.update_board(board, simulation)
        policy, evaluation = self.get_data(board)
        N = [0] * self.board_size**2
        data[self.convert(made_moves)] = [policy, 0, N, N] # N matches Q at the beginning

        # updating Q and N
        simulation.reverse()
        for move in simulation:
            made_moves.remove(move)
            curr_set = self.convert(made_moves)

            N = data[curr_set][1]
            Q = data[curr_set][2]
            N[move] += 1
            Q[move] = (Q[move] * (N[move] - 1) + evaluation) / N[move]
            data[curr_set][1] = N
            data[curr_set][2] = Q

        return data