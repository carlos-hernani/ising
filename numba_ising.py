"""2D Ising Model simulation using Monte Carlo Techniques.

@github: carlos-hernani

Rules:
    Only Functions

Python Version 3.9, waiting for numba to update for 3.10.

Added todo's to know where should I update once numba supports
python3.10. Priority of updates should follow the next rule:

    - Lv 0: Heavily used function, optimize with highest priority.
    - ...
    - Lv 5: Only used once, optimize if nothing else can be.


First priority is to get a working ising simulation with numba.

Later we can build upon it & make tests and data-validation.

Algorithm:
    1. Given a spin 2d-configuration, randomly select cells.
    2. Check if it should flip given the acceptance probability.
    3. Repeat 1
"""

__version__ = "0.1.0"

from numba.misc.special import prange
import numpy as np
import numba as nb


#________Simulation Parameters___________
k = 8.617e-5 # eV/K
temp = 255 # K
beta = 1/(k*temp)
interaction = 1
size = 256
simulation_random_state = 0 # controls the selection of cells in the grid
start_iteration = 0
"""For all possible configurations of neighboring spins there are only 5
options:
    2 configs where dE is negative
    1 config where dE is 0
    2 configs where dE is positive
The first 3 configurations conserve the flipped spin.
The last 2 configurations only conserve the flipped spin with an acceptance
probability A.

`acc2` and `acc4` are the corresponding probabilites to conserve the flip.
`flip` is a smart way to access the values without calculating them every
iteration.
"""
acc2 = np.exp(-2*np.abs(interaction)*2*beta)
acc4 = np.exp(-2*np.abs(interaction)*4*beta)
flip = (True, True, True, acc2, acc4)

#_____________Functions__________________

# --------- Data Generation -------------


def generate_grid(size, up_prob, random_state):
    """Generates a valid 2-Dimensional grid of spins.

    Returns a 2D grid of +1/-1 with a probability of
    up_prob of +1.

    Args:
        size (int): Width of the square lattice.
        up_prob (float): Probability of spin up. [0,1]
        random_state (int): Reproducible code.
            Don't confuse this rs with simulation random_state

    Returns:
        np.ndarray: 2d lattice grid
    """    
    rng = np.random.default_rng(random_state)
    return 2*rng.binomial(1, p=up_prob, size = [size]*2)-1


# ------ Spin Query Strategies ----------


def single(grid_size, random_state):
    """
    single Single flip dynamics strategy

    Each iteration it will deliver a tuple of only one pair of (i,j).

    Returns:
        Tuple[np.ndarray]: List of 1 positions.
    """
    rng = np.random.default_rng(random_state)
    return (rng.integers(low=0, high=grid_size, size=2))


cb = (np.indices((size,size)).sum(axis=0) % 2)
checkerboards = (
    np.transpose(np.where(cb == 0)),
    np.transpose(np.where(cb == 1))
)


def checkerboard(random_state, checkerboards: tuple):
    """
    checkerboard Checkerboard flip strategy.
    Each iteration it will deliver a list of multiple pairs of (i,j).

    If the random state is even it will choose the first of checkerboards.
    Else, it will choose the second one.

    This way all the cells in the Ising Simulation are flipped at least
    once.

    Args:

        checkerboards (tuple): Tuple with 2 different checkerboards.

    Returns:

        list[np.ndarray]: List of pair positions.
    """
    idx = np.mod(random_state, 2)
    # idx = 0 if even , 1 if odd
    return checkerboards[idx]


# ------- Spin Flip Acceptance ----------


@nb.jit(nopython=True, parallel=True, cache=True, boundscheck=False)
def accept_flip(grid, size, flip, i, j, random_state):
    rng = np.random.default_rng(random_state)
    # each (i, j) pair has 4 nearest neighbors.
    nidx = (
        np.mod((i+1, j), size),
        np.mod((i-1, j), size),
        np.mod((i, j-1), size),
        np.mod((i, j+1), size)
    )
    neighbors_contrib = 0
    for i in prange(4):
        neighbors_contrib += grid[nidx[i][0], nidx[i][1]]
    index_flip = -0.25*grid[i,j]*neighbors_contrib + 2
    return rng.random() < flip[int(index_flip)]


if __name__ == '__main__':
    grid = generate_grid(size, up_prob=0.7, random_state=0)
    print(grid)
    print(accept_flip(grid, size, flip, 1,1, 1))