from Game import *
from Players import *
import time

player_one = Human_Player()
player_two = AI_Player()

print("If you want to play for X type 1, else type 0")
while True:
    num = input()
    if num == '1':
        break
    if num == '0':
        player_one = AI_Player()
        player_two = Human_Player()
        break
    print("Type 1 or 0, pleease")

game = Game(15)
game.start_game(player_one, player_two)
