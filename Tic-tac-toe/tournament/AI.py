import logging
import os
import random

import backend
import numpy
import renju
import torch
from Players import *
from copy import deepcopy


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
