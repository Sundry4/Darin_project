from Game import *
from Players import *

game = Game(15)
player_one = HumanPlayer(15)
player_two = RandomPlayer()

game.start_game(player_one, player_two)
