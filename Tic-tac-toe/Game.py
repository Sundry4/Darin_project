import pygame
from pygame import *


class Game:
    WIN_WIDTH = 240
    WIN_HEIGHT = 300
    DISPLAY = (WIN_WIDTH, WIN_HEIGHT)
    BACKGROUND_COLOR = (200, 200, 200)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)
    BOARDER_WIDTH = 6
    N = 3
    CELL_WIDTH = WIN_WIDTH // N

    def __init__(self):

        # creating board
        pygame.init()
        self.screen = pygame.display.set_mode(self.DISPLAY)
        pygame.display.set_caption("Tic-tac-toe")
        self.surface = pygame.Surface((self.WIN_WIDTH, self.WIN_HEIGHT))

        self.surface.fill(self.BACKGROUND_COLOR)

        cells = range(self.WIN_WIDTH // self.N, self.WIN_WIDTH, self.WIN_WIDTH // self.N)
        for i in cells:
            draw.line(self.surface, self.BLACK, [0, i], [self.WIN_WIDTH, i], self.BOARDER_WIDTH)
            draw.line(self.surface, self.BLACK, [i, 0], [i, self.WIN_WIDTH], self.BOARDER_WIDTH)

        self.screen.blit(self.surface, (0, 0))
        pygame.display.update()

        self.board = [[''] * self.N for i in range(self.N)]
        self.possible_moves = []
        for i in range(self.N):
            for j in range(self.N):
                self.possible_moves.append([i, j])

    def get_pos(self, cell):
        pos = [cell[0] * self.CELL_WIDTH, cell[1] * self.CELL_WIDTH]
        pos[0] += self.CELL_WIDTH // 2
        pos[1] += self.CELL_WIDTH // 2
        return pos

    def put_X(self, cell):
        self.board[cell[0]][cell[1]] = 'X'

        pos = self.get_pos(cell)
        draw.line(self.surface, self.BLUE, [pos[0] - 15, pos[1] - 15],
                  [pos[0] + 15, pos[1] + 15], 5)
        draw.line(self.surface, self.BLUE, [pos[0] - 15, pos[1] + 15],
                  [pos[0] + 15, pos[1] - 15], 5)

    def put_O(self, cell):
        self.board[cell[0]][cell[1]] = 'O'

        pos = self.get_pos(cell)
        draw.circle(self.surface, self.BLUE, pos, 20, 4)

    def check_win_condition(self):
        cell = None
        winner = None
        for i in range(self.N):
            winner = self.board[i][0]

            for j in range(self.N):
                if winner != self.board[i][j]:
                    winner = None
                    break
            if winner:
                cell = ([i, 0], [i, 2])
                break

        if not winner:
            for i in range(self.N):
                winner = self.board[0][i]

                for j in range(self.N):
                    if winner != self.board[j][i]:
                        winner = None
                        break
                if winner:
                    cell = ([0, i], [2, i])
                    break

        if not winner:
            winner = self.board[0][0]
            for i in range(self.N):
                if winner != self.board[i][i]:
                    winner = None
                    break
            if winner:
                cell = ([0, 0], [2, 2])

        if not winner:
            winner = self.board[-1][0]
            for i in range(self.N):
                if winner != self.board[-i - 1][i]:
                    winner = None
                    break
            if winner:
                cell = ([2, 0], [0, 2])

        return winner, cell

    def start_game(self, player_one, player_two):
        curr_player = player_one
        while True:
            winner, cell = self.check_win_condition()
            if winner:
                pos1 = self.get_pos(cell[0])
                pos2 = self.get_pos(cell[1])
                draw.line(self.surface, self.RED, pos1, pos2, 5)
                self.screen.blit(self.surface, (0, 0))
                self.end(winner)

            if len(self.possible_moves) == 0:
                self.end()

            cell = curr_player.move_(self.possible_moves)

            if curr_player == player_one:
                self.put_X(cell)
                self.board[cell[0]][cell[1]] = 'X'
                curr_player = player_two
            else:
                self.put_O(cell)
                self.board[cell[0]][cell[1]] = 'O'
                curr_player = player_one

            self.screen.blit(self.surface, (0, 0))
            pygame.display.update()
            self.possible_moves.pop(self.possible_moves.index(cell))

    def end(self, winner=None):
        font = pygame.font.Font(None, 50)

        text = font.render('DRAW!', 2, (0, 0, 0))
        if winner:
            text = font.render('WINNER: {}'.format(winner), 2, (0, 0, 0))

        surf = pygame.Surface((self.WIN_WIDTH, 50))
        surf.fill(self.BACKGROUND_COLOR)
        surf.blit(text, (0, 0))
        self.screen.blit(surf, (20, self.WIN_HEIGHT - 45))
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
