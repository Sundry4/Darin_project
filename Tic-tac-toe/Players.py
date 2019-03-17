import time
import random
import pygame
import torch
import numpy as np
from Net import *
import torch.nn as nn
import torch.nn.functional as F
from MCTS import *
from copy import deepcopy


class HumanPlayer:
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


class RandomPlayer:
    def __init__(self):
        pass

    def move_(self, possible_moves, board):
        time.sleep(0.1)
        return random.choice(possible_moves)


class ProVis:
    def __init__(self):
        pass

    def move_(self, possible_moves, cell, board):
        time.sleep(1)
        return [cell[0], cell[1]]


def normal(x):
    return x / sum(x)


class AI_Player:
    board_size = 15

    def __init__(self, is_black):
        self.model = Net()
        if is_black:
            self.model.load_state_dict(torch.load("model7_black_21.pth"))
        else:
            self.model.load_state_dict(torch.load("model7_white_21.pth"))
        self.model.eval()

        self.mcts = MCTS()
        self.iterations = 1000
        self.eps = 10**-5

    # returns 2 numbers from 0 to 14
    def move_(self, possible_moves, board):
        # time.sleep(1)  # delay for making you believe, that model needs to think
        data = torch.from_numpy(np.array(board)).type(torch.FloatTensor)

        # output_p, _ = self.model(data)
        # policy = F.softmax(output_p, dim=1).detach().numpy()[0]

        policy = self.mcts.get_policy(data, self.iterations, deepcopy(possible_moves)).detach().numpy()
        while True:
            actions = range(225)
            number = np.random.choice(actions, 1, p=policy)[0]
            if number in possible_moves:
                return [number // self.board_size, number % self.board_size]

            policy[number] = 0
            policy = F.softmax(torch.tensor([policy]), dim=1).detach().numpy()[0]

    def get_policy(self, board, possible_moves):
        data = torch.from_numpy(np.array(board)).type(torch.FloatTensor)
        policy = self.mcts.get_policy(data, self.iterations, possible_moves).tolist()

        # output_p, _ = self.model(data)
        # policy = F.softmax(output_p, dim=1).detach().numpy()[0]
        # print(policy)
        return policy