import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
import timeit
import pygame
import random
import time
np.set_printoptions(suppress=True, threshold=np.inf)

# Cuda code
kernel_code = r"""
#include <curand_kernel.h>

extern "C" {
__global__ void setupCurand(curandState *state, unsigned long seed) {
    int idx = blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__device__ void addTile(int8_t *current_row, curandState *state) {
    int idx = blockIdx.x;
    curandState localState = state[idx];
    
    int numZeros = 0;
    for (int i = 0; i < 16; i++) {
        if (current_row[i] == 0) {
            numZeros++;
        }  
    }
    if (numZeros == 0) {
        return;
    }
    int randIndex = curand(&localState) % numZeros;
    int zeroCount = 0;
    for (int i = 0; i < 16; i++) {
        if (current_row[i] == 0) {
            if (zeroCount == randIndex) {
                current_row[i] = (curand_uniform(&localState) > 0.9) ? 2 : 1;
                break;
            }
            zeroCount++;
        }
    }
    state[idx] = localState;
}

__device__ bool moveLeft(int8_t *current_row, int *current_score, curandState *state, bool addTileBool = true) {
    __shared__ bool didSomethingHappen;
    if (threadIdx.x == 0) {
        didSomethingHappen = false;
    }
    __syncthreads();
    for (int i = 0; i < 3; i++) {
        if (current_row[i] == 0) {
            int j = i + 1;
            while (j < 4) {
                if (current_row[j] != 0) {
                    current_row[i] = current_row[j];
                    current_row[j] = 0;
                    didSomethingHappen = true;
                    break;
                }
                j++;
            }
        }
    }

    for (int i = 0; i < 3; i++) {
        if (current_row[i] != 0 && current_row[i] == current_row[i+1]) {
            current_row[i] += 1;
            current_row[i+1] = 0;
            atomicAdd(current_score, (1 << current_row[i]));
            didSomethingHappen = true;
        }
    }

    for (int i = 0; i < 3; i++) {
        if (current_row[i] == 0) {
            int j = i + 1;
            while (j < 4) {
                if (current_row[j] != 0) {
                    current_row[i] = current_row[j];
                    current_row[j] = 0;
                    didSomethingHappen = true;
                    break;
                }
                j++;
            }
        }
    }
    __syncthreads();
    if (didSomethingHappen && threadIdx.x == 0 && addTileBool) {
        addTile(current_row, state);
    }
    return didSomethingHappen;
}

__device__ bool moveRight(int8_t *current_row, int *current_score, curandState *state, bool addTileBool = true) {
    __shared__ bool didSomethingHappen;
    if (threadIdx.x == 0) {
        didSomethingHappen = false;
    }
    __syncthreads();
    for (int i = 3; i > 0; i--) {
        if (current_row[i] == 0) {
            int j = i - 1;
            while (j >= 0) {
                if (current_row[j] != 0) {
                    current_row[i] = current_row[j];
                    current_row[j] = 0;
                    didSomethingHappen = true;
                    break;
                }
                j--;
            }
        }
    }

    for (int i = 3; i > 0; i--) {
        if (current_row[i] != 0 && current_row[i] == current_row[i-1]) {
            current_row[i] += 1;
            current_row[i-1] = 0;
            atomicAdd(current_score, (1 << current_row[i]));
            didSomethingHappen = true;
        }
    }

    for (int i = 3; i > 0; i--) {
        if (current_row[i] == 0) {
            int j = i - 1;
            while (j >= 0) {
                if (current_row[j] != 0) {
                    current_row[i] = current_row[j];
                    current_row[j] = 0;
                    didSomethingHappen = true;
                    break;
                }
                j--;
            }
        }
    }
    __syncthreads();
    if (didSomethingHappen && threadIdx.x == 0 && addTileBool) {
        addTile(current_row, state);
    }
    return didSomethingHappen;
}

__device__ bool moveUp(int8_t *current_row, int *current_score, curandState *state, bool addTileBool = true) {
    __shared__ bool didSomethingHappen;
    if (threadIdx.x == 0) {
        didSomethingHappen = false;
    }
    __syncthreads();
    for (int i = 0; i < 12; i += 4) {
        if (current_row[i] == 0) {
            int j = i + 4;
            while (j < 16) {
                if (current_row[j] != 0) {
                    current_row[i] = current_row[j];
                    current_row[j] = 0;
                    didSomethingHappen = true;
                    break;
                }
                j += 4;
            }
        }
    }

    for (int i = 0; i < 12; i += 4) {
        if (current_row[i] != 0 && current_row[i] == current_row[i+4]) {
            current_row[i] += 1;
            current_row[i+4] = 0;
            atomicAdd(current_score, (1 << current_row[i]));
            didSomethingHappen = true;
        }
    }

    for (int i = 0; i < 12; i += 4) {
        if (current_row[i] == 0) {
            int j = i + 4;
            while (j < 16) {
                if (current_row[j] != 0) {
                    current_row[i] = current_row[j];
                    current_row[j] = 0;
                    didSomethingHappen = true;
                    break;
                }
                j += 4;
            }
        }
    }
    __syncthreads();
    if (didSomethingHappen && threadIdx.x == 0 && addTileBool) {
        addTile(current_row, state);
    }
    return didSomethingHappen;
}

__device__ bool moveDown(int8_t *current_row, int *current_score, curandState *state, bool addTileBool = true) {
    __shared__ bool didSomethingHappen;
    if (threadIdx.x == 0) {
        didSomethingHappen = false;
    }
    __syncthreads();
    for (int i = 12; i > 0; i -= 4) {
        if (current_row[i] == 0) {
            int j = i - 4;
            while (j >= 0) {
                if (current_row[j] != 0) {
                    current_row[i] = current_row[j];
                    current_row[j] = 0;
                    didSomethingHappen = true;
                    break;
                }
                j -= 4;
            }
        }
    }

    for (int i = 12; i > 0; i -= 4) {
        if (current_row[i] != 0 && current_row[i] == current_row[i-4]) {
            current_row[i] += 1;
            current_row[i-4] = 0;
            atomicAdd(current_score, (1 << current_row[i]));
            didSomethingHappen = true;
        }
    }

    for (int i = 12; i > 0; i -= 4) {
        if (current_row[i] == 0) {
            int j = i - 4;
            while (j >= 0) {
                if (current_row[j] != 0) {
                    current_row[i] = current_row[j];
                    current_row[j] = 0;
                    didSomethingHappen = true;
                    break;
                }
                j -= 4;
            }
        }
    }
    __syncthreads();
    if (didSomethingHappen && threadIdx.x == 0 && addTileBool) {
        addTile(current_row, state);
    }
    return didSomethingHappen;
}

__global__ void fourplicate(int8_t *g_initial_board, int8_t *g_final_board, int g_initial_score, int *g_final_scores, int batchSize) {
    int boardno = blockIdx.x;
    int threadno = threadIdx.x;
    g_final_board[batchSize*boardno*16 + threadno] = g_initial_board[threadno];
    g_final_scores[batchSize*boardno] = g_initial_score;
}

__global__ void firstMoves(int8_t *g_final_board, int *g_final_scores, curandState *states, int batchSize, bool *addTiles) {
    int boardno = blockIdx.x;
    int threadno = threadIdx.x;
    switch (boardno) {
        case 0:
            addTiles[boardno] = moveLeft(&g_final_board[4*threadno], &g_final_scores[0], states, false);
            break;
        case 1:
            addTiles[boardno] = moveUp(&g_final_board[batchSize*16+threadno], &g_final_scores[batchSize], states, false);
            break;
        case 2:
            addTiles[boardno] = moveRight(&g_final_board[2*batchSize*16+4*threadno], &g_final_scores[2*batchSize], states, false);
            break;
        case 3:
            addTiles[boardno] = moveDown(&g_final_board[3*batchSize*16+threadno], &g_final_scores[3*batchSize], states, false);
            break;   
    }   
}

__global__ void duplicate(int8_t *g_final_board, int *g_final_scores, int batchSize, bool *addTiles) {
    int boardno = blockIdx.x+1;
    int threadno = threadIdx.x;
    if (addTiles[(boardno-1+3*batchSize)/batchSize]) {g_final_board[16*(boardno+3*batchSize) + threadno] = g_final_board[3*batchSize*16 + threadno];}
    if (addTiles[(boardno-1+2*batchSize)/batchSize]) {g_final_board[16*(boardno+2*batchSize) + threadno] = g_final_board[2*batchSize*16 + threadno];}
    if (addTiles[(boardno-1+batchSize)/batchSize]) {g_final_board[16*(boardno+batchSize) + threadno] = g_final_board[batchSize*16 + threadno];}
    if (addTiles[(boardno-1)/batchSize]) {g_final_board[16*boardno + threadno] = g_final_board[threadno];}

    if (threadno == 0) {
        if (addTiles[(boardno-1+3*batchSize)/batchSize]) {g_final_scores[boardno+3*batchSize] = g_final_scores[3*batchSize];}
        if (addTiles[(boardno-1+2*batchSize)/batchSize]) {g_final_scores[boardno+2*batchSize] = g_final_scores[2*batchSize];}
        if (addTiles[(boardno-1+batchSize)/batchSize]) {g_final_scores[boardno+1*batchSize] = g_final_scores[batchSize];}
        if (addTiles[(boardno-1)/batchSize]) {g_final_scores[boardno] = g_final_scores[0];}
    }  
}

__global__ void firstRandoms(int8_t *current_row, curandState *state, bool *addTiles, int batchSize) {
    int boardno = blockIdx.x;
    if (addTiles[boardno/batchSize]) {
        addTile(&current_row[16*boardno], state);
    }
}

__global__ void simulate(int8_t *g_final_board, int *g_final_scores, curandState *states, int simDepth, int batchSize, bool *addTiles) {
    int boardno = blockIdx.x;
    if (!addTiles[boardno/batchSize]) {
        return;
    }
    int threadno = threadIdx.x; 
    __shared__ int direction;
    __shared__ int size;
    bool didSomethingHappen = false;
    int availableMoves[4] = {0, 1, 2, 3};
    int dirindex = 0;
    for (int i = 0; i < simDepth; i++) {
        if (threadno == 0) {
            for (int j = 0; j < 4; j++) {
                availableMoves[j] = j;
            }
            size = 4;
        }
        __syncthreads();
        while (size > 0) {
            
            if (threadno == 0) {
                dirindex = curand(&states[boardno]) % size;
                direction = availableMoves[dirindex];
            }
            __syncthreads();
            switch (direction) {
                case 0:
                    didSomethingHappen = moveLeft(&g_final_board[16*boardno + 4*threadno], &g_final_scores[boardno], states);
                    break;
                case 1:
                    didSomethingHappen = moveUp(&g_final_board[16*boardno + threadno], &g_final_scores[boardno], states);
                    break;
                case 2:
                    didSomethingHappen = moveRight(&g_final_board[16*boardno + 4*threadno], &g_final_scores[boardno], states);
                    break;
                case 3:
                    didSomethingHappen = moveDown(&g_final_board[16*boardno + threadno], &g_final_scores[boardno], states);
                    break;
            }
            if (threadno == 0 && !didSomethingHappen) {
                for (int j = dirindex; j < size - 1; j++) {
                    availableMoves[j] = availableMoves[j+1];
                }
                size--;
            }
            __syncthreads();
            if (didSomethingHappen) {
                break;
            }
        }
        if (!didSomethingHappen) {
            g_final_scores[boardno] = -1;
            break;
        }
    }
}
}
"""


# Initialize board and score 
batch_size = 1000000
blocks = batch_size * 4
sim_depth = 20

initial_board = np.array([[1, 1, 0, 0],
                          [2, 2, 0, 0],
                          [0, 3, 0, 0],
                          [0, 1, 0, 0]], dtype=np.int8)
initial_score = np.int32(0)
g_initial_board = cuda.mem_alloc(initial_board.nbytes)
cuda.memcpy_htod(g_initial_board, initial_board)

# Make room for and format outputs
final_boards = np.zeros((4, batch_size, 4, 4), dtype=np.int8)
g_final_boards = cuda.mem_alloc(final_boards.nbytes)
final_scores = np.zeros((4, batch_size), dtype=np.int32)
g_final_scores = cuda.mem_alloc(final_scores.nbytes)

# Load kernel code
mod = SourceModule(kernel_code, no_extern_c=True)

# Setup curand
curand_states = cuda.mem_alloc(blocks * 48)
mod.get_function("setupCurand")(curand_states, np.uint64(time.time()), block=(1, 1, 1), grid=(blocks, 1, 1))

# Setup boards with first moves (left, up, right, down)
firstMoveLegal = np.zeros((4), dtype=np.bool_)
g_firstMoveLegal = cuda.mem_alloc(firstMoveLegal.nbytes)
fourplicate = mod.get_function("fourplicate")
firstMoves = mod.get_function("firstMoves")
fourplicate(g_initial_board, g_final_boards, initial_score, g_final_scores, np.int32(batch_size), block=(16, 1, 1), grid=(4, 1, 1))
firstMoves(g_final_boards, g_final_scores, curand_states, np.int32(batch_size), g_firstMoveLegal, block=(4, 1, 1), grid=(4, 1, 1))
cuda.Context.synchronize()

# For testing
"""four_boards = np.zeros((4, 4, 4), dtype=np.int8)
four_scores = np.zeros((4), dtype=np.int32)
cuda.memcpy_dtoh(final_boards, g_final_boards)
cuda.memcpy_dtoh(final_scores, g_final_scores)
print(initial_board)
for x1,i in enumerate(final_boards):
    for x2,j in enumerate(i):
        if np.any(j != np.zeros((4,4),dtype=np.int8)):
            print(x1,x2)
            print(j)
            print()
for i in range(len(final_scores)):
    for k in range(len(final_scores[i])):
        if final_scores[i][k] != 0:
            print(f"{i} {k}: {final_scores[i][k]}")"""

# Copy boards batch_size times
duplicate = mod.get_function("duplicate")
firstRandoms = mod.get_function("firstRandoms")
duplicate(g_final_boards, g_final_scores, np.int32(batch_size), g_firstMoveLegal, block=(16,1, 1), grid=(batch_size-1, 1, 1))
firstRandoms(g_final_boards, curand_states, g_firstMoveLegal, np.int32(batch_size), block=(1, 1, 1), grid=(blocks, 1, 1))
cuda.Context.synchronize()

# For testing
"""cuda.memcpy_dtoh(final_boards, g_final_boards)
cuda.memcpy_dtoh(final_scores, g_final_scores)
print(initial_board)
# count=0
# for i in [final_boards[0]]:
#     for j in i:
#         if np.any(j != i[0]):
            # print("SOMETHING WENT WRONG: PANIK", i[0],j)
            # count+=1
# for i in final_scores:
#     for j in i:
#         if j != i[0]:
#             print("SOMETHING WENT WRONG: PANIK", i[0], j)
# print(count)
print(final_boards[:,4])
print(final_scores[:,4])"""

# SIMULATE!!!!1!
print("start")
simulate = mod.get_function("simulate")
simulate(g_final_boards, g_final_scores,curand_states, np.int32(sim_depth), np.int32(batch_size), g_firstMoveLegal, block=(4, 1, 1), grid=(blocks, 1, 1))
print("plz don't be slow men")
cuda.Context.synchronize()
cuda.memcpy_dtoh(final_boards, g_final_boards)
cuda.memcpy_dtoh(final_scores, g_final_scores)
# print(final_boards[:,999999])
# print(final_scores[:,999999])
# print(final_boards[:,3])
# print(final_scores[:,3])
mean_scores = np.mean(final_scores, axis=1)
std_scores = np.std(final_scores, axis=1,ddof=1)
bounds = [(mean_scores[i] - 1.96*std_scores[i]/np.sqrt(batch_size), mean_scores[i] + 1.96*std_scores[i]/np.sqrt(batch_size)) for i in range(4)]
print(bounds, mean_scores)