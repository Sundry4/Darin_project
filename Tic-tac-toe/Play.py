from Game import *
from Players import *

game = Game(15)
player_one = HumanPlayer(15)
player_two = HumanPlayer(15)

game.start_game(player_one, player_two)
