import pygame
from pygame import *
from Players import *
from copy import deepcopy
import time


class Game:
    background_color = (240, 240, 240)
    blue = (0, 0, 255)
    light_blue = (103, 155, 239)
    red = (255, 0, 0)
    black = (0, 0, 0)
    violet = (126, 37, 140)
    nice_green = (99, 150, 24)
    gray = (180, 180, 180)
    boarder_width = 2
    cell_width = 50

    def __init__(self, board_size):
        self.empty = np.array([[0] * board_size for _ in range(board_size)])
        self.black_pos = deepcopy(self.empty)
        self.white_pos = deepcopy(self.empty)
        self.turn = np.array([[1] * board_size for _ in range(board_size)])  # 1 for black, -1 for white
        self.hist_1_black = deepcopy(self.empty)
        self.hist_1_white = deepcopy(self.empty)
        self.hist_2_black = deepcopy(self.empty)
        self.hist_2_white = deepcopy(self.empty)
        self.hist_3_black = deepcopy(self.empty)
        self.hist_3_white = deepcopy(self.empty)
        self.hist_4_black = deepcopy(self.empty)
        self.hist_4_white = deepcopy(self.empty)

        self.turns_amount = 0

        self.board_size = board_size
        self.win_width = self.cell_width * (self.board_size + 2)
        self.win_height = self.cell_width * (self.board_size + 2)
        self.display = (self.win_width, self.win_height)

        # creating board
        pygame.init()
        self.screen = pygame.display.set_mode(self.display)
        pygame.display.set_caption("Tic-tac-toe")
        self.surface = pygame.Surface((self.win_width, self.win_height))

        self.surface.fill(self.background_color)

        # drawing lines
        cells = range(50, self.win_width - 50 + 1, self.cell_width)
        for i in cells:
            draw.line(self.surface, self.light_blue, [self.cell_width, i],
                      [self.win_width - self.cell_width, i],
                      self.boarder_width)
            draw.line(self.surface, self.light_blue, [i, self.cell_width],
                      [i, self.win_width - self.cell_width],
                      self.boarder_width)

        # printing numbers
        letters = "abcdefghjklmnopqrstuvwsyz"
        font = pygame.font.Font(None, 30)
        for i in range(1, len(cells)):
            text = font.render(str(i), 2, (0, 0, 0))
            self.surface.blit(text, (20, self.cell_width * i + 17))

            text = font.render(letters[i - 1], 2, (0, 0, 0))
            self.surface.blit(text, (self.cell_width * i + 20, 20))

        self.screen.blit(self.surface, (0, 0))
        pygame.display.update()

        self.possible_moves = {*range(self.board_size ** 2)}

        self.player_one = None
        self.player_two = None

    def get_pos(self, cell):
        pos = [(cell[0] + 1) * self.cell_width, (cell[1] + 1) * self.cell_width]
        pos[0] += self.cell_width // 2
        pos[1] += self.cell_width // 2
        return pos

    def put_X(self, cell):
        self.hist_4_black = deepcopy(self.hist_3_black)
        self.hist_3_black = deepcopy(self.hist_2_black)
        self.hist_2_black = deepcopy(self.hist_1_black)
        self.hist_1_black = deepcopy(self.black_pos)
        self.black_pos[cell[0]][cell[1]] = 1

        pos = self.get_pos(cell)
        draw.line(self.surface, self.black, [pos[0] - 15, pos[1] - 15],
                  [pos[0] + 15, pos[1] + 15], 6)
        draw.line(self.surface, self.black, [pos[0] - 15, pos[1] + 15],
                  [pos[0] + 15, pos[1] - 15], 6)

    def put_O(self, cell):
        self.hist_4_white = deepcopy(self.hist_3_white)
        self.hist_3_white = deepcopy(self.hist_2_white)
        self.hist_2_white = deepcopy(self.hist_1_white)
        self.hist_1_white = deepcopy(self.white_pos)
        self.white_pos[cell[0]][cell[1]] = 1

        pos = self.get_pos(cell)
        draw.circle(self.surface, self.red, pos, 20, 5)

    def check_win_condition(self, last_move, turn):
        board = self.white_pos
        if turn[0][0] == 1:
            board = self.black_pos

        n = 3
        if self.board_size >= 5:
            n = 5

        x, y = last_move
        winner = board[x][y]

        # vertical
        for i in range(n):
            match_count = 0
            for j in range(i - n + y + 1, i + y + 1):
                if j < 0 or j >= self.board_size:
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
                if j < 0 or j >= self.board_size:
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
                if x + j < 0 or x + j >= self.board_size or y + j < 0 or y + j >= self.board_size:
                    continue
                if winner != board[x + j][y + j]:
                    break
                match_count += 1
            if match_count == n:
                return [(i - n + 1 + x, i - n + 1 + y), (i + x, i + y)]

        for i in range(n):
            match_count = 0
            for j in range(i - n + 1, i + 1):
                if x - j < 0 or x - j >= self.board_size or y + j < 0 or y + j >= self.board_size:
                    continue
                if winner != board[x - j][y + j]:
                    break
                match_count += 1
            if match_count == n:
                return [(n - i - 1 + x, i - n + 1 + y), (-i + x, i + y)]

        return None

    def restart(self):
        self.surface = pygame.Surface((self.win_width, self.win_height))

        self.surface.fill(self.background_color)

        cells = range(50, self.win_width - 50 + 1, self.cell_width)
        for i in cells:
            draw.line(self.surface, self.light_blue, [self.cell_width, i],
                      [self.win_width - self.cell_width, i],
                      self.boarder_width)
            draw.line(self.surface, self.light_blue, [i, self.cell_width],
                      [i, self.win_width - self.cell_width],
                      self.boarder_width)

        letters = "abcdefghjklmnopqrstuvwsyz"
        font = pygame.font.Font(None, 30)
        for i in range(1, len(cells)):
            text = font.render(str(i), 2, (0, 0, 0))
            self.surface.blit(text, (25, self.cell_width * i + 17))

            text = font.render(letters[i - 1], 2, (0, 0, 0))
            self.surface.blit(text, (self.cell_width * i + 20, 20))

        self.screen.blit(self.surface, (0, 0))
        pygame.display.update()

        self.turns_amount = 0

        self.empty = np.array([[0] * self.board_size for _ in range(self.board_size)])
        self.black_pos = deepcopy(self.empty)
        self.white_pos = deepcopy(self.empty)
        self.turn = np.array([[1] * self.board_size for _ in range(self.board_size)])  # 1 for black, -1 for white
        self.hist_1_black = deepcopy(self.empty)
        self.hist_1_white = deepcopy(self.empty)
        self.hist_2_black = deepcopy(self.empty)
        self.hist_2_white = deepcopy(self.empty)
        self.hist_3_black = deepcopy(self.empty)
        self.hist_3_white = deepcopy(self.empty)
        self.hist_4_black = deepcopy(self.empty)
        self.hist_4_white = deepcopy(self.empty)

        self.possible_moves = {*range(self.board_size ** 2)}

        self.start_game(self.player_one, self.player_two)

    def start_game(self, player_one, player_two):
        self.player_one = player_one
        self.player_two = player_two

        start = time.clock()

        curr_player = player_one
        while True:
            state = torch.from_numpy(
                np.stack(
                    (self.black_pos, self.white_pos, self.turn,
                     self.hist_1_black, self.hist_1_white,
                     self.hist_2_black, self.hist_2_white,
                     self.hist_3_black, self.hist_3_white,
                     self.hist_4_black, self.hist_4_white)
                )
            )
            cell = curr_player.move_(self.possible_moves, state)

            self.turns_amount += 1
            if curr_player == player_one:
                self.put_X(cell)
                curr_player = player_two
            else:
                self.put_O(cell)
                curr_player = player_one

            # win condition check
            cells = self.check_win_condition(cell, self.turn)
            if cells:
                print("Amount of turns:", self.turns_amount)
                pos1 = self.get_pos(cells[0])
                pos2 = self.get_pos(cells[1])
                draw.line(self.surface, self.gray, pos1, pos2, 5)
                self.screen.blit(self.surface, (0, 0))
                pygame.display.update()

                winner = 'O'
                if curr_player == player_two:
                    winner = 'X'
                print("Time:", time.clock() - start, "\n")
                self.end(winner)

            self.turn *= -1

            # full board condition check
            if len(self.possible_moves) == 0:
                self.end()

            self.screen.blit(self.surface, (0, 0))
            pygame.display.update()
            self.possible_moves.remove(cell[0] * self.board_size + cell[1])

    def end(self, winner=None):
        font = pygame.font.Font(None, 50)

        text = font.render('DRAW! PRESS SPACE TO RESTART', 2, (0, 0, 0))
        if winner:
            text = font.render('WINNER: {}! PRESS SPACE TO RESTART'.format(winner), 2, (0, 0, 0))

        surf = pygame.Surface((self.win_width, 50))
        surf.fill(self.background_color)
        surf.blit(text, (0, 0))
        self.screen.blit(surf, (10, self.win_height - 40))
        pygame.display.update()

        while True:
            for i in pygame.event.get():
                if i.type == pygame.QUIT:  # if you wanna quit - you're welcome
                    pygame.display.quit()
                    pygame.quit()
                    exit()
                if i.type == KEYDOWN:
                    if i.key == K_ESCAPE:
                        pygame.display.quit()
                        pygame.quit()
                        exit()
                    if i.key == K_SPACE:
                        self.restart()
