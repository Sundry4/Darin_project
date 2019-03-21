import time
import random
import pygame
import torch
import numpy as np
from Net import Net
from New_Net import *
import torch.nn as nn
import torch.nn.functional as F
from MCTS import *
from copy import deepcopy


class Human_Player:
    cell_width = 50

    def __init__(self, board_size=15):
        self.board_size = board_size
        self.win_width = board_size * self.cell_width

    def move_(self, possible_moves, board):
        while True:
            for i in pygame.event.get():
                if i.type == pygame.QUIT:
                    exit()
                if i.type == pygame.KEYDOWN:
                    if i.key == pygame.K_ESCAPE:
                        exit()

                if i.type == pygame.MOUSEBUTTONDOWN:
                    if i.button == 1:
                        curr_pos = [self.cell_width, self.cell_width]
                        cell = [self.board_size - 1, self.board_size - 1]

                        for j in range(self.board_size):
                            curr_pos[0] += self.cell_width
                            if i.pos[0] <= curr_pos[0]:
                                cell[0] = j
                                break

                        for j in range(self.board_size):
                            curr_pos[1] += self.cell_width
                            if i.pos[1] <= curr_pos[1]:
                                cell[1] = j
                                break

                        if cell[0] * self.board_size + cell[1] not in possible_moves:
                            print("Choose another cell")
                        else:
                            return cell


class Random_Player:
    def __init__(self):
        pass

    def move_(self, possible_moves, board):
        time.sleep(0.1)
        return random.choice(possible_moves)


def normal(x):
    return x / sum(x)


class AI_Player:
    board_size = 15

    def __init__(self, is_black):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = PNet()
        if is_black:
            self.model.load_state_dict(torch.load('model11_black_29.pth'))
        else:
            self.model.load_state_dict(torch.load("model11_white_29.pth"))
        self.model.eval().to(self.device)

        self.mcts = MCTS(29, 4)
    # returns 2 numbers from 0 to 14
    def move_(self, possible_moves, board):

        data = torch.from_numpy(np.array(board)).type(torch.FloatTensor)
        policy = self.mcts.get_policy(data, deepcopy(possible_moves))
        policy = policy.cpu().detach().numpy().tolist()

        cell = self.four_in_a_row(deepcopy(data))
        if cell:
            policy = np.array([0] * 15 ** 2)
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