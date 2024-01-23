from TorchGame import Game2048
import numpy as np
import pygame
import cProfile
from tqdm import tqdm
import pstats

def test_code():
    actions = ['left', 'right', 'up', 'down']
    for _ in tqdm(range(10**4)):
        env = Game2048()
        for _ in range(10):
            action = np.random.choice(actions)
            env.move(action)

cProfile.run('test_code()','game_files/profile_output')
p = pstats.Stats('game_files/profile_output')
p.sort_stats('cumulative').print_stats(10)