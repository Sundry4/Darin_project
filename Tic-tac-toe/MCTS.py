import numpy as np
from copy import deepcopy
import time

import torch
from torch import utils
import torch.nn as nn
import torch.nn.functional as F
from Net import *


class MCTS:
    def __init__(self, path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model_black = Net()
        self.model_black.load_state_dict(torch.load("model7_black_1.pth"))
        self.model_black.eval().to(self.device)

        self.model_white = Net()
        self.model_white.load_state_dict(torch.load("model7_white_1.pth"))
        self.model_white.eval().to(self.device)

        self.model_V = VNet()
        self.model_V.load_state_dict(torch.load(path))
        self.model_V.eval().to(self.device)

        self.board_size = 15
        self.alpha = 1

    def get_data(self, board):
        model = self.model_black

        # костыыыыыыль! :)
        # слой хода 3-тий в списке слоев состояния
        if board[2][0][0].item() == -1:
            model = self.model_white

        data = torch.tensor(board)
        data = data.to(self.device)
        data = data.type(torch.cuda.FloatTensor)
        output, _ = model(data.to(self.device))

        output_v = F.softmax(self.model_V(data), dim=1)[0]
        evaluation = (output_v[0] - output_v[1]).item()

        return F.softmax(output, dim=1).cpu().detach().numpy()[0], evaluation

    def get_policy(self, board, iterations, possible_moves): # possible move is set of numbers from 0 to 224
        temp = 5

        policy, v = self.get_data(board.to(self.device))
        N = [0] * self.board_size**2
        data = {self.conv([set(), set()]): [policy, deepcopy(N), deepcopy(N)]}  # N matches Q at the beginning

        # making simulations, every single one starts from root state
        for _ in range(iterations): # it may be better to change it to time
            data = self.step(data, possible_moves, deepcopy(board))

        N = torch.FloatTensor(data[self.conv([set(), set()])][1])
        # print(np.array(F.softmax(N, dim=0)).reshape((15, 15)))
        return F.softmax(N**(1/temp), dim=0) # returning distribution adjusted by N

    def update_board(self, board, cell):
        black_pos, white_pos, turn, hist_1_black,\
        hist_1_white, hist_2_black, hist_2_white = board

        is_black = False
        if turn[0][0] == 1:
            is_black = True

        hist_2_black = deepcopy(hist_1_black)
        hist_2_white = deepcopy(hist_1_white)

        hist_1_black = deepcopy(black_pos)
        hist_1_white = deepcopy(white_pos)

        if is_black:
            black_pos[cell[0]][cell[1]] = 1
        else:
            white_pos[cell[0]][cell[1]] = 1
        turn *= -1

        board = np.stack((black_pos, white_pos, turn,
                          hist_1_black, hist_1_white,
                          hist_2_black, hist_2_white)
                         )
        return board

    # self.converting set to dict key
    def conv(self, moves):
        return str(sorted(moves[0])) + str(sorted(moves[1]))

    def step(self, data, possible_moves, board):
        made_moves = [set(), set()] # 1st is black, 2nd is white
        simulation = []

        is_white = False
        if board[2][0][0].item() == -1:
            is_white = True

        winner = 0
        while self.conv(made_moves) in data.keys():
            if len(possible_moves) == 0:
                break

            curr_set = self.conv(made_moves)

            P = data[curr_set][0]  # prior probabilities
            N = data[curr_set][1]  # visit count
            U = [P[i] / (self.alpha * (1 + N[i])) for i in range(self.board_size**2)]
            Q = data[curr_set][2]  # action value

            # extruding an index of max elem of U+Q vector
            U_Q = np.array(U) + np.array(Q)
            move = torch.from_numpy(U_Q).data.max(0, keepdim=True)[1].item()

            while move not in possible_moves:
                U_Q[move] -= 10
                move = torch.from_numpy(U_Q).data.max(0, keepdim=True)[1].item()

            possible_moves.remove(move)
            made_moves[is_white].add(move)
            simulation.append(move)
            board = self.update_board(board, [move // self.board_size, move % self.board_size])

            is_white = not is_white

            if self.check_win_condition(board, [move // self.board_size, move % self.board_size]):
                winner = is_white * 2 - 1
                break


        policy, evaluation = self.get_data(board)

        N = [0] * self.board_size**2
        data[self.conv(made_moves)] = [policy, deepcopy(N), deepcopy(N)] # N matches Q at the beginning

        if winner:
            evaluation = winner

        is_white = not is_white

        # updating Q and N
        simulation.reverse()
        for move in simulation:
            possible_moves.add(move)
            made_moves[is_white].remove(move)

            curr_set = self.conv(made_moves)

            N = data[curr_set][1]
            Q = data[curr_set][2]
            N[move] += 1
            Q[move] = (Q[move] * (N[move] - 1) + evaluation) / N[move]
            data[curr_set][1] = N
            data[curr_set][2] = Q

            is_white = not is_white

        return data

    def check_win_condition(self, field, last_move):
        black_pos, white_pos, turn, hist_1_black, \
        hist_1_white, hist_2_black, hist_2_white = field

        N = self.board_size
        n = 5
        x, y = last_move

        # searching for player made last turn
        board = white_pos
        if turn[0][0] == -1:
            board = black_pos

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
                return True

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
                return True

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
                return True

        for i in range(n):
            match_count = 0
            for j in range(i - n + 1, i + 1):
                if x - j < 0 or x - j >= N or y + j < 0 or y + j >= N:
                    continue
                if winner != board[x - j][y + j]:
                    break
                match_count += 1
            if match_count == n:
                return True

        return False