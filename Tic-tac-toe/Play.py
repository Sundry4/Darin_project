from Game import *
from Players import *

game = Game(15)
player_one = HumanPlayer(15)
player_two = OMGVPlayer(0)   # 1 - for 'X', 0 - for 'O'

# player_one = OMGVPlayer(1)   # 1 - for 'X', 0 - for 'O'
# player_two = HumanPlayer(15)

game.start_game(player_one, player_two)
