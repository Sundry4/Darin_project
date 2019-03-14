import time
import random
import pygame
import torch
import numpy as np
from Net import *
import torch.nn as nn
import torch.nn.functional as F
import os


class HumanPlayer:
    cell_width = 50

    def __init__(self, N=15):
        self.N = N
        self.win_width = N * self.cell_width

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
                        cell = [self.N - 1, self.N - 1]

                        for j in range(self.N):
                            curr_pos[0] += self.cell_width
                            if i.pos[0] <= curr_pos[0]:
                                cell[0] = j
                                break

                        for j in range(self.N):
                            curr_pos[1] += self.cell_width
                            if i.pos[1] <= curr_pos[1]:
                                cell[1] = j
                                break

                        if cell not in possible_moves:
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


class OMGVPlayer:
    board_size = 15

    def __init__(self, is_black):
        self.model = Net()
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model)

        if is_black:
            self.model.load_state_dict(torch.load("model7_black_21.pth"))
        else:
            self.model.load_state_dict(torch.load("model7_white_21.pth"))

        self.model.eval()

    def move_(self, possible_moves, board):
        # time.sleep(1)  # delay for making you believe, that model needs to think

        data = torch.from_numpy(np.array(board)).cuda().type(torch.FloatTensor)
        output_p, _ = self.model(data)
        policy = output_p.detach().numpy()[0]

        while True:
            # actions = range(225)
            # number = np.random.choice(actions, 1, p=policy)[0]
            # cell = [number // self.board_size, number % self.board_size]
            # if cell in possible_moves:
            #     return cell
            number = output_p.data.max(1, keepdim=True)[1].item()
            cell = [number // self.board_size, number % self.board_size]
            if cell in possible_moves:
                return cell
            output_p[0][number] -= 100
