from Game import *
from Players import *
import time

game = Game(15)

player_one = No_Tree_Player(1)
player_two = Human_Player()   # 1 - for 'X', 0 - for 'O'

game.start_game(player_one, player_two)
