from Game import *
from Players import *
import time

game = Game(15)

player_two = AI_Player()
player_one = Human_Player()   # 1 - for 'X', 0 - for 'O'

game.start_game(player_one, player_two)
