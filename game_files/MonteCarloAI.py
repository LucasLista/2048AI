from modules.Game2048 import Game2048
from modules.simulate import simulate2048
import numpy as np
import pygame

env = Game2048()
env.reset()
sim = simulate2048(sim_depth=20, batch_size=100000)
actions = ['left', 'up', 'right', 'down']
exit_program = False
action_taken = False
while not exit_program:
    env.render()

    # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True

    action=actions[sim.bestMove(env.score, env.board)]
    (board, score), reward, done = env.step(action)
    action_taken = False

env.close()
