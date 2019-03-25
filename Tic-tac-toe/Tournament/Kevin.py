import logging
import os
import random

import backend
import numpy
import renju
import torch
from copy import deepcopy

import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch import utils
import math
import warnings
warnings.filterwarnings("ignore")


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(11, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.residual1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.residual2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.policy1 = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )

        self.policy2 = nn.Sequential(
            nn.Linear(15 * 15 * 2, 15 * 15),
            nn.LeakyReLU(inplace=True)
        )

        self.weight_init(self.convolutional)
        self.weight_init(self.residual1)
        self.weight_init(self.residual2)
        self.weight_init(self.policy1)
        self.weight_init(self.policy2)

    def policy(self, x):
        x = self.policy1(x)
        x = x.view(x.size(0), -1)
        x = self.policy2(x)

        return x

    @staticmethod
    def weight_init(elem):
        for m in elem.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.view(-1, 11, 15, 15)
        x = self.convolutional(x)
        out = self.residual1(x)
        x = x + out
        out = self.residual2(x)
        x = x + out
        x = self.policy(x)
        return x


class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(11, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.residual1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.residual2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.value1 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.value2 = nn.Sequential(
            nn.Linear(15 * 15, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Tanh()
        )

        self.weight_init(self.convolutional)
        self.weight_init(self.residual1)
        self.weight_init(self.residual2)
        self.weight_init(self.value1)
        self.weight_init(self.value2)

    def val(self, x):
        x = self.value1(x)
        x = x.view(x.size(0), -1)
        x = self.value2(x)

        return x

    @staticmethod
    def weight_init(elem):
        for m in elem.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.view(-1, 11, 15, 15)
        x = self.convolutional(x)
        out = self.residual1(x)
        x = x + out
        out = self.residual2(x)
        x = x + out

        return self.val(x)


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
        tm = 2.5

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


class No_Tree_Player:
    board_size = 15

    def __init__(self, is_black):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = PNet()
        if is_black:
            self.model.load_state_dict(torch.load('model11_black_64.pth'))
        else:
            self.model.load_state_dict(torch.load("model11_white_64.pth"))
        self.model.eval()

    # returns 2 numbers from 0 to 14
    def move_(self, possible_moves, board):
        data = torch.from_numpy(np.array(board)).to(self.device).type(torch.FloatTensor)

        policy = self.model(data)[0].tolist()
        # print(np.array(policy).reshape((15, 15)), "\n")

        cell = self.four_in_a_row(deepcopy(data))
        if cell:
            policy = [0] * 15 ** 2
            policy[cell[0] * 15 + cell[1]] = 1.0

        while True:
            number = policy.index(np.max(policy))
            if number in possible_moves:
                return [number // self.board_size, number % self.board_size]
            policy[number] -= 1

    def four_in_a_row(self, state):
        is_black = 0
        if state[2][0][0] == 1:
            is_black = 1

        board = state[abs(is_black - 1)] # if it's black's turn, we choose black's (number 0) board

        for x in range(15):
            for y in range(15):
                if state[is_black][x][y] == board[x][y] == 0:
                    board[x][y] = 1
                    if self.find_four(board, x, y):
                        board[x][y] = 0
                        return x, y
                    board[x][y] = 0

        return None

    def find_four(self, board, x, y):
        N = 15
        n = 5

        winner = 1
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


class AI_Player:
    board_size = 15

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mcts = MCTS(64, 42)

    # returns 2 numbers from 0 to 14
    def move_(self, possible_moves, board):
        data = torch.from_numpy(np.array(board)).type(torch.FloatTensor)
        policy = self.mcts.get_policy(data, deepcopy(possible_moves))
        policy = policy.cpu().detach().numpy().tolist()

        cell = self.four_in_a_row(deepcopy(data))
        if cell:
            policy = [0] * 15 ** 2
            policy[cell[0] * 15 + cell[1]] = 1.0

        while True:
            number = policy.index(np.max(policy))
            if number in possible_moves:
                return [number // self.board_size, number % self.board_size]
            policy[number] -= 1

    def four_in_a_row(self, state):
        is_black = 0
        if state[2][0][0] == 1:
            is_black = 1

        board = state[abs(is_black - 1)] # if it's black's turn, we choose black's (number 0) board

        for x in range(15):
            for y in range(15):
                if state[is_black][x][y] == board[x][y] == 0:
                    board[x][y] = 1
                    if self.find_four(board, x, y):
                        board[x][y] = 0
                        return x, y
                    board[x][y] = 0

        return None

    def find_four(self, board, x, y):
        N = 15
        n = 5

        winner = 1
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




POS_TO_LETTER = 'abcdefghjklmnop'
LETTER_TO_POS = {letter: pos for pos, letter in enumerate(POS_TO_LETTER)}


def to_move(pos):
    return POS_TO_LETTER[pos[0]] + str(pos[1] + 1)


def make_board(dumps):
    dic = {'a': 1, 'b': 2,
           'c': 3, 'd': 4,
           'e': 5, 'f': 6,
           'g': 7, 'h': 8,
           'j': 9, 'k': 10,
           'l': 11, 'm': 12,
           'n': 13, 'o': 14,
           'p': 15, 'q': 16,
           'r': 17, 's': 18,
           't': 19, 'u': 20,
           'v': 21, 'w': 22,
           'x': 23, 'y': 24,
           'z': 25}

    empty = np.array([[0] * 15 for _ in range(15)])
    black_pos = deepcopy(empty)
    white_pos = deepcopy(empty)
    turn = np.array([[1] * 15 for _ in range(15)])  # 1 for black, -1 for white
    hist_1_black = deepcopy(empty)
    hist_1_white = deepcopy(empty)
    hist_2_black = deepcopy(empty)
    hist_2_white = deepcopy(empty)
    hist_3_black = deepcopy(empty)
    hist_3_white = deepcopy(empty)
    hist_4_black = deepcopy(empty)
    hist_4_white = deepcopy(empty)

    possible_moves = {*range(15 ** 2)}

    is_black = True
    for move in dumps.split():
        x, y = dic[move[0]] - 1, int(move[1:]) - 1
        possible_moves.remove(x * 15 + y)

        turn *= -1

        hist_4_black = deepcopy(hist_3_black)
        hist_4_white = deepcopy(hist_3_white)

        hist_3_black = deepcopy(hist_2_black)
        hist_3_white = deepcopy(hist_2_white)

        hist_2_black = deepcopy(hist_1_black)
        hist_2_white = deepcopy(hist_1_white)

        hist_1_black = deepcopy(black_pos)
        hist_1_white = deepcopy(white_pos)

        if is_black:
            black_pos[x][y] = 1
        else:
            white_pos[x][y] = 1

        is_black = not is_black

    board = np.stack(
                    (black_pos, white_pos, turn,
                     hist_1_black, hist_1_white,
                     hist_2_black, hist_2_white,
                     hist_3_black, hist_3_white,
                     hist_4_black, hist_4_white)
                    )

    # for i in board:
    #     logging.debug('Game: [%s]', i)

    board = torch.tensor(board).type(torch.FloatTensor)

    return board, possible_moves


def main():
    player = AI_Player()

    pid = os.getpid()
    LOG_FORMAT = str(pid) + ':%(levelname)s:%(asctime)s: %(message)s'

    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    logging.debug("Start dummy backend...")

    try:
        while True:
            logging.debug("Wait for game update...")
            game = backend.wait_for_game_update()

            if not game:
                logging.debug("Game is over!")
                return

            logging.debug('Game: [%s]', game.dumps())

            board, possible_moves = make_board(game.dumps())

            move = player.move_(possible_moves, board)
            move = to_move(move)

            if not backend.set_move(move):
                logging.error("Impossible set move!")
                return

            logging.debug('AI move: %s', move)

    except:
        logging.error('Error!', exc_info=True, stack_info=True)


if __name__ == "__main__":
    main()
