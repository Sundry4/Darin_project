import pygame
import random
from pygame import *


class HumanPlayer:
    WIN_WIDTH = 240
    N = 3
    CELL_WIDTH = WIN_WIDTH // N

    def __init__(self):
        pass

    def move_(self, possible_moves):
        while True:
            for i in pygame.event.get():
                if i.type == QUIT:
                    exit()
                if i.type == KEYDOWN:
                    if i.key == K_ESCAPE:
                        exit()

                if i.type == MOUSEBUTTONDOWN:
                    if i.button == 1:
                        curr_pos = [0, 0]
                        cell = [self.N - 1, self.N - 1]

                        for j in range(self.N):
                            curr_pos[0] += self.CELL_WIDTH
                            if i.pos[0] <= curr_pos[0]:
                                cell[0] = j
                                break

                        for j in range(self.N):
                            curr_pos[1] += self.CELL_WIDTH
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

    def move_(self, possible_moves):
        return random.choice(possible_moves)

