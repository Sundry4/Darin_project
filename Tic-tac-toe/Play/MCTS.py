import numpy as np
from copy import deepcopy
import time

import torch
from torch import utils
import torch.nn as nn
import torch.nn.functional as F
from Net import *
import warnings
warnings.filterwarnings("ignore")


class MCTS:
    def __init__(self, number_p, number_v):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model_black = PNet()
        self.model_black.load_state_dict(
            torch.load("model11_black_{}.pth".format(number_p), map_location=lambda storage, loc: storage)
        )
        self.model_black.eval().to(self.device)

        self.model_white = PNet()
        self.model_white.load_state_dict(
            torch.load("model11_white_{}.pth".format(number_p), map_location=lambda storage, loc: storage)
        )
        self.model_white.eval().to(self.device)

        self.model_V = VNet()
        self.model_V.load_state_dict(
            torch.load('model11_V_{}.pth'.format(number_v), map_location=lambda storage, loc: storage)
        )
        self.model_V.eval().to(self.device)

        self.board_size = 15
        self.is_black = True

    def get_data(self, board):
        model = self.model_black
        alpha = 1
        if not self.is_black:
            alpha = -1

        # костыыыыыыль! :)
        # слой хода 3-тий в списке слоев состояния
        if board[2][0][0].item() == -1:
            model = self.model_white

        data = torch.tensor(board)
        data = data.to(self.device)
        data = data.type(torch.FloatTensor)
        output = model(data.to(self.device))

        output_v = F.softmax(self.model_V(data), dim=1)[0]
        evaluation = (output_v[0] - output_v[1]).item()

        return F.softmax(output, dim=1).cpu().detach().numpy()[0], evaluation * alpha

    def get_policy(self, board, possible_moves): # possible move is set of numbers from 0 to 224
        self.is_black = True
        if board[2][0][0].item() == -1:
            self.is_black = False

        start = time.clock()
        tm = 2.8

        temp = 1
        count = self.board_size**2 - len(possible_moves)
        if count > 30:
            temp = 0.5
        if count > 50:
            temp = 0.1

        policy, v = self.get_data(board.to(self.device))
        N = [0] * self.board_size**2
        data = {self.conv([set(), set()]): [policy, deepcopy(N), deepcopy(N)]}  # N matches Q at the beginning

        # making simulations, every single one starts from root state
        while time.clock() - start < tm: # it may be better to change it to time
            data = self.step(data, possible_moves, deepcopy(board))

        N = torch.FloatTensor(data[self.conv([set(), set()])][1])
        return F.softmax(N**(1/temp), dim=0) # returning distribution adjusted by N

    def update_board(self, board, cell):
        black_pos, white_pos, turn,\
        hist_1_black, hist_1_white, hist_2_black, hist_2_white,\
        hist_3_black, hist_3_white, hist_4_black, hist_4_white = board

        is_black = False
        if turn[0][0] == 1:
            is_black = True

        hist_4_black = deepcopy(hist_3_black)
        hist_4_white = deepcopy(hist_3_white)

        hist_3_black = deepcopy(hist_2_black)
        hist_3_white = deepcopy(hist_2_white)

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
                          hist_2_black, hist_2_white,
                          hist_3_black, hist_3_white,
                          hist_4_black, hist_4_white)
                         )
        return board

    # self.converting set to dict key
    def conv(self, moves):
        return str(sorted(moves[0])) + str(sorted(moves[1]))

    def step(self, data, possible_moves, board):
        made_moves = [set(), set()] # 1st is black, 2nd is white
        simulation = []

        is_white = not self.is_black

        winner = 0
        while self.conv(made_moves) in data.keys():
            if len(possible_moves) == 0:
                break

            curr_set = self.conv(made_moves)

            P = data[curr_set][0]  # prior probabilities
            N = data[curr_set][1]  # visit count
            c = np.sqrt(np.sum(N) + 1)
            U = [P[i] * c / (1 + N[i]) for i in range(self.board_size**2)]
            Q = data[curr_set][2]  # action value

            # extruding an index of max elem of U+Q vector
            U_Q = np.array(U) + np.array(Q)
            move = np.array(U_Q).argmax()

            while move not in possible_moves:
                U_Q[move] -= 10
                move = np.array(U_Q).argmax()

            possible_moves.remove(move)
            made_moves[is_white].add(move)
            simulation.append(move)
            board = self.update_board(board, [move // self.board_size, move % self.board_size])
            is_white = not is_white

            if self.check_win_condition(board, [move // self.board_size, move % self.board_size]):
                winner = 1  # player gets +1 value if root state is his turn and he won, else he gets -1 value
                if self.is_black and not is_white:
                    winner = -1
                if not self.is_black and is_white:
                    winner = -1
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
        black_pos, white_pos, turn,\
        hist_1_black, hist_1_white, hist_2_black, hist_2_white,\
        hist_3_black, hist_3_white, hist_4_black, hist_4_white = field

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
