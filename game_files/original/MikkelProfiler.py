from Game2048 import Game2048
import numpy as np
import pygame
import cProfile
from tqdm import tqdm
import pstats

def test_code():
    env = Game2048()
    actions = ['left', 'right', 'up', 'down']
    for _ in tqdm(range(10**4)):
        env.reset()
        for _ in range(10):
            action = np.random.choice(actions)
            env.step(action)
    env.close()

cProfile.run('test_code()','game_files/original/profile_output')
p = pstats.Stats('game_files/original/profile_output')
p.sort_stats('cumulative').print_stats(10)
    
