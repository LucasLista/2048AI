# What is this?
This is a personal project to optimize an algorithm for playing the game [2048](https://play2048.co/). So far, I've implemented a Monte Carlo search algorithm that utilizes the GPU of the user's computer to parallelize the simulations and thus run much faster. This algorithm is found at `2048AI/game_files/MonteCarloAI.py`. 

# Requirements
- Nvidia GPU and [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) installed.
- Python 3.x

## Necessary Packages
To run this project, you need to install the following Python packages:

- `pycuda`: Access to Nvidia's CUDA parallel computation API from within Python.
- `numpy`: Package for scientific computing with Python.
- `pygame`: Set of Python modules designed for writing video games.

You can install the necessary Python packages using pip:

```bash
pip install pycuda numpy pygame
```
**Make sure the version of PyCUDA matches the version of your CUDA toolkit.**

# Credit
The code for the game itself `Game2048.py` was developed by DTU lecturers for the course "Introduction to Intelligent Systems" (02461), and has only been slightly modified by me. The rest is developed by me. 
