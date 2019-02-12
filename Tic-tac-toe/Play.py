from Game import *
from Players import *

game = Game()
player_one = HumanPlayer()
player_two = RandomPlayer()

game.start_game(player_one, player_two)
