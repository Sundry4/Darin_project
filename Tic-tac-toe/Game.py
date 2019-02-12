import pygame
from pygame import *


class Game:
    background_color = (240, 240, 240)
    blue = (0, 0, 255)
    red = (255, 0, 0)
    black = (0, 0, 0)
    boarder_width = 3
    cell_width = 50

    def __init__(self, N):
        self.N = N
        self.win_width = self.cell_width * self.N
        self.win_height = self.cell_width * self.N + 100
        display = (self.win_width, self.win_height)

        # creating board
        pygame.init()
        self.screen = pygame.display.set_mode(display)
        pygame.display.set_caption("Tic-tac-toe")
        self.surface = pygame.Surface((self.win_width, self.win_height))

        self.surface.fill(self.background_color)

        cells = range(0, self.win_width + 1, self.win_width // self.N)
        for i in cells:
            draw.line(self.surface, self.black, [0, i], [self.win_width, i], self.boarder_width)
            draw.line(self.surface, self.black, [i, 0], [i, self.win_width], self.boarder_width)

        self.screen.blit(self.surface, (0, 0))
        pygame.display.update()

        self.board = [[''] * self.N for i in range(self.N)]
        self.possible_moves = []
        for i in range(self.N):
            for j in range(self.N):
                self.possible_moves.append([i, j])

    def get_pos(self, cell):
        pos = [cell[0] * self.cell_width, cell[1] * self.cell_width]
        pos[0] += self.cell_width // 2
        pos[1] += self.cell_width // 2
        return pos

    def put_X(self, cell):
        self.board[cell[0]][cell[1]] = 'X'

        pos = self.get_pos(cell)
        draw.line(self.surface, self.blue, [pos[0] - 15, pos[1] - 15],
                  [pos[0] + 15, pos[1] + 15], 5)
        draw.line(self.surface, self.blue, [pos[0] - 15, pos[1] + 15],
                  [pos[0] + 15, pos[1] - 15], 5)

    def put_O(self, cell):
        self.board[cell[0]][cell[1]] = 'O'

        pos = self.get_pos(cell)
        draw.circle(self.surface, self.red, pos, 20, 4)

    def check_win_condition(self, last_move):
        n = 3
        if self.N >= 5:
            n = 5

        x, y = last_move

        winner = self.board[x][y]

        start_pos = 0

        # vertical victory
        for i in range(n):
            match_count = 0
            for j in range(i - n + y + 1, i + y + 1):
                if j < 0 or j >= self.N:
                    continue
                if winner != self.board[x][j]:
                    break
                match_count += 1
            if match_count == n:
                return [(x, i - n + y + 1), (x, i + y)]

        # horizontal victory
        for i in range(n):
            match_count = 0
            for j in range(i - n + x + 1, i + x + 1):
                if j < 0 or j >= self.N:
                    continue
                if winner != self.board[j][y]:
                    break
                match_count += 1
            if match_count == n:
                return [(i - n + x + 1, y), (i + x, y)]

        # diagonals
        for i in range(n):
            match_count = 0
            for j in range(i - n + 1, i + 1):
                if x + j < 0 or x + j >= self.N or y + j < 0 or y + j >= self.N:
                    continue
                if winner != self.board[x + j][y + j]:
                    break
                match_count += 1
            if match_count == n:
                return [(i - n + 1 + x, i - n + 1 + y), (i + x, i + y)]

        for i in range(n):
            match_count = 0
            for j in range(i - n + 1, i + 1):
                if x - j < 0 or x - j >= self.N or y + j < 0 or y + j >= self.N:
                    continue
                if winner != self.board[x - j][y + j]:
                    break
                match_count += 1
            if match_count == n:
                return [(n - i - 1 + x, i - n + 1 + y), (-i + x, i + y)]

        return None

    def start_game(self, player_one, player_two):
        curr_player = player_one
        while True:

            cell = curr_player.move_(self.possible_moves)

            if curr_player == player_one:
                self.put_X(cell)
                self.board[cell[0]][cell[1]] = 'X'
                curr_player = player_two
            else:
                self.put_O(cell)
                self.board[cell[0]][cell[1]] = 'O'
                curr_player = player_one

            # win condition check
            cells = self.check_win_condition(cell)
            if cells:
                pos1 = self.get_pos(cells[0])
                pos2 = self.get_pos(cells[1])
                draw.line(self.surface, self.black, pos1, pos2, 5)
                self.screen.blit(self.surface, (0, 0))

                winner = 'O'
                if curr_player == player_two:
                    winner = 'X'
                self.end(winner)

            # full board condition check
            if len(self.possible_moves) == 0:
                self.end()

            self.screen.blit(self.surface, (0, 0))
            pygame.display.update()
            self.possible_moves.pop(self.possible_moves.index(cell))

    def end(self, winner=None):
        font = pygame.font.Font(None, 50)

        text = font.render('DRAW!', 2, (0, 0, 0))
        if winner:
            text = font.render('WINNER: {}'.format(winner), 2, (0, 0, 0))

        surf = pygame.Surface((self.win_width, 50))
        surf.fill(self.background_color)
        surf.blit(text, (0, 0))
        self.screen.blit(surf, (20, self.win_height - 45))
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
