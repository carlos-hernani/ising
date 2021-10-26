"""2D Ising Model simulation using Monte Carlo Techniques.

@github: carlos-hernani

Rules:

    Classes are only called once.
    Functions can be called more than once.

Python Version 3.10 waiting for numba to update.

Added todo's to know where should I update once numba supports
python3.10. Priority of updates should follow the next rule:
    Lv 0: Heavily used function, optimize with highest priority.
    ...
    Lv 5: Only used once, optimize if nothing else can be.
"""

__version__ = "0.1.0"

import sys
import attr
import json
import numpy as np

from typing import Callable
from collections import deque
from checks import instance_of
from rich import print

# num is a type that is either int or float
num = int | float
# strat is the type of strategy, a callable whose input is a np.ndarray
# and returns a list of (i,j) pairs
strat = Callable[[np.ndarray, int], list[np.ndarray]]

# Validators: They check the inputs.

def gtzero(instance, attribute, value):
    """
    gtzero Validator: checks greather than zero
    """    
    if value <= 0:
        raise ValueError(f'{attribute.file} must be positive & non-zero.')

def gele(instance, attribute, value):
    """
    gele Validator: checks geq than zero or leq than one
    """    
    if value < 0 or value > 1:
        raise ValueError(f'{attribute.file} must be between [0,1].')

def opt_type(type, cond=None, default_value=None):
    """
    opt_type Enforces Optional Type and validates conditions.

    Args:
        type ([type]): The desired type
        cond (callable, optional): Condition function. Defaults to None.
        default_value ([type], optional): The default value. Defaults to None.

    Returns:
        dict: unpack it in attr.ib
    """    
    ret_value = {
            'validator': [attr.validators.optional(
                instance_of(type))
                ],
            'default': default_value}
    if cond is not None:
        ret_value['validator'] = [
            attr.validators.optional(
                instance_of(type)
                ),
            cond
        ]
    
    return ret_value


# Numba Functions: Optimized Functions

# TODO: Optimize when Numba works in python3.10, Lv5

def validate_grid(arr: np.ndarray) -> bool:
    """
    validate_grid Checks that the given grid is correct.
    Args:
        arr (np.ndarray): grid of values.
    Returns:
        bool: True if correct.
    """
    for x in arr.ravel():
        if np.abs(x) != 1:
            return False
    return True


# NOTE: I don't know the inner workings of Numba but I would guess that
# the performance will be affected by a function calling other functions
# and so on.
# HACK: For this reason, I'll do one f to get the end result
# TODO: Optimize when Numba works in python3.10, Lv0

def get_flip_nmb(lattice: 'Lattice', flip: tuple, i:int, j:int) -> bool:
    """
    get_flip_nmb Gives us the spin-flip.
    Args:
        lattice (Lattice): Lattice Object.
        flip (tuple): Tuple of possible outcomes.
            flip[:3] -> dE <= 0 => Flip.
            flip[4:] -> dE > 0 => Compare against rng.
        i (int): Row Index of 2D 'np.ndarray'.
        j (int): Column Index of 2D 'np.ndarray'.
    Returns:
        bool: If 'True' the spin flips, else it doesn't.
    """
    rs = lattice.random_state
    rng = np.random.default_rng(rs)
    # each (i, j) pair has 4 nearest neighbors.
    neighbors_indices = (
        np.mod((i+1, j), lattice.size),
        np.mod((i-1, j), lattice.size),
        np.mod((i, j-1), lattice.size),
        np.mod((i, j+1), lattice.size)
    )
    # the difference in energy between one state and the other is given by
    # s_i := spin to be flipped, s_j := neighbor spins
    # dE = (sum_<ij> J*(-s_i)*s_j) - (sum_<ij> J*s_i*s_j)
    # dE = -2*J*(sum_<ij> s_i*s_j) = -2*J*s_i*sum(neighbors_of_s_i())
    # dE = -2*J*s_i*neigbors_contrib
    neighbors_contrib = 0
    for ids in neighbors_indices:
        # each neighbor can be accessed in parallel.
        neighbors_contrib += lattice.grid[ids[0], ids[1]]
    # print(neighbors_contrib)
    # dE = -2*1*lattice.grid[i,j]*neighbors_contrib
    # print(f'Diff Energy: {dE}')
    # TRICK: the possible values of dE \in -2*J*[-4,-2,0,+2,+4] so
    # dE/(8*J) -> [-2,-1,0,+1,+2] and
    # .+2 -> [0,1,2,3,4] that can index a list of results.
    # For dE <= 0, we FLIP. For dE > 0 we FLIP only if rng() < exp(-dE*b)
    index_flip = -0.25*lattice.grid[i,j]*neighbors_contrib + 2
    # print(f'index_flip is : {index_flip} (int) {int(index_flip)}')
    # index_flip = dE/(8*1) + 2
    # print(f'index_flip is : {index_flip} (int) {int(index_flip)}')
    return rng.random() < flip[int(index_flip)]



@attr.s
class Lattice:
    """
    Lattice Object.

    Creates a square lattice of dimensions size x size filled with
    +1 or -1 with a given probability, p, (+1:p, -1:1-p) with a
    random seed.
    
    Args:
        size (int): number of cells in each direction of the sqaure lattice.
        p (num, optional): probability of value +1. Defaults to 0.5.
        grid (np.ndarray, optional): grid of values. Defaults to None.
        random_state (int, optional): random seed for deterministic behavior.
                    Defaults to None.
    """    
    size: int = attr.ib(validator=[
        attr.validators.instance_of(int),
        gtzero])
    p: num = attr.ib(validator=[
        attr.validators.instance_of(num),
        gele], default = 0.5)
    grid: np.ndarray | None = attr.ib(**opt_type(np.ndarray))
    random_state: int | None = attr.ib(**opt_type(int))

    def generate_grid(self) -> None:
        """
        generate_grid
            Creates a square lattice of dimensions size x size filled with
            +1 or -1 with a given probability, p, (+1:p, -1:1-p) with a
            random seed.
        """        
        rng = np.random.default_rng(self.random_state)
        self.grid = 2*rng.binomial(1, p=self.p, size=[self.size]*2)-1

    def __attrs_post_init__(self) -> None:
        if self.grid is None:
            self.generate_grid()
        else:
            assert validate_grid(self.grid), \
                "Values of spin must be +1 or -1 ."



default_params = {
    'lattice': Lattice(32, random_state=0),
    'temp': 255,
    'inter': 1
}


@attr.s
class Ising:
    """
    Ising Object

    Implements Markov Chain Monte Carlo method using the Metropolis algorithm
    to solve the 2D Ising Model.

    Args:
        lattice (Lattice, optional): Lattice Object. 
            Defaults to default_params['lattice']
        temp (num, optional): Temperature in Kelvin > 0.
            Defaults to default_params['temp']
        inter (num, optional): Interaction J.
            Defaults to default_params['inter']
        warmstart (str, optional): filepath to previous runs.
            Defaults to None
    """    
    lattice: Lattice | None = attr.ib(**opt_type(Lattice, None, default_params['lattice']))
    temp: num | None = attr.ib(**opt_type(num, gtzero, default_params['temp']))
    inter: num | None = attr.ib(**opt_type(num, None, default_params['inter']))
    warmstart: str | None = attr.ib(**opt_type(str))

    def __attrs_post_init__(self):
        k = 8.617e-5 # eV/K
        self.beta = 1/(k*self.temp)
        # That's a surprise tool that will help us later:
        exp2 = np.exp(-2*np.abs(self.inter)*2*self.beta)
        exp4 = np.exp(-2*np.abs(self.inter)*4*self.beta)
        self.flip = (True, True, True, exp2, exp4)
        self.start_iteration = 0
        self.name = 'ising'
        if self.warmstart:
            self.load()
    
    # TODO: Optimize when Numba works in python3.10, Lv1
    @staticmethod
    def update_grid_nmb(lattice: Lattice, flip: tuple, points:list) -> None:
        for point in points:
            if get_flip_nmb(lattice, flip, point[0], point[1]):
                print(f"[green]FLIPPED[/green] at ({point[0], point[1]})")
                lattice.grid[point[0], point[1]] *= -1

    def update_grid(self, points:list) -> None:
        self.update_grid_nmb(self.lattice, self.flip, points)

    # -------------Spin Query Strategies----------------
    def single(self) -> list[np.ndarray]:
        """
        single Single flip dynamics strategy

        Each iteration it will deliver a list of only one pair of (i,j).

        Returns:
            list[np.ndarray]: List of 1 positions.
        """
        rng = np.random.default_rng(self.lattice.random_state)
        return [rng.integers(low=0, high=self.lattice.grid.shape[0], size=2)]

    def checkerboard(self, checkerboards:tuple) -> list[np.ndarray]:
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
        idx = np.mod(self.lattice.random_state,2)
        # idx = 0 if even , 1 if odd
        return checkerboards[idx]

    def set_strategy(self, strategy:str) -> None:
        """
        set_strategy Sets the strategy at the beggining of the execution.

        Args:
            strategy (str): file of the strategy
                Possible strategies are: 'single', 'checkerboard'
        """        
        # when numba works in 3.10 change this to match
        if strategy == 'single':
            self.strategy = self.single
        elif strategy == 'checkerboard':
            size = self.lattice.size
            cb = (np.indices((size,size)).sum(axis=0) % 2)
            # Generates a checkerboard pattern, only once
            checkerboards = (
                np.transpose(np.where(cb == 0)),
                np.transpose(np.where(cb == 1))
            )
            self.strategy = self.checkerboard(checkerboards)
        else:
            raise NotImplementedError(
                "Possible strategies are: 'single', 'checkerboard'")

    # -------------Save and loading in Las Vegas--------
    # FIXME: size out of bounds when loading previous, need to override the default grid
    def load(self):
        """
        load Make sure to load the last model.

        Args:
            field (str): [description]
        """        
        file = self.warmstart
        self.name = file.split('_')[0]
        with open(self.name+'.json') as read_file:
            data = json.load(read_file)
            self.temp = data['temp']
            self.inter = data['inter']
            self.beta = data['beta']
            self.flip = data['flip']
        self.start_iteration = int(file.split('_')[1])
        self.lattice.random_state = int(file.split('_')[2])
        self.lattice.grid = np.load(file+'.npy')

    def save(self, history: deque):
        # saves the state of the simulation for later.

        parameters = {
            'temp': self.temp,
            'inter': self.inter,
            'beta': self.beta,
            'flip': self.flip
        }
        with open(self.name+'.json', 'w') as outfile:
            json.dump(parameters, outfile)
        
        for h in history:
            np.save('_'.join([
                self.name,
                f'{h[0]}',
                f'{h[1]}']), 
                h[2], allow_pickle=False)

    # -------------Execution Phase----------------------
    def _run(self, strategy:str, max_iter:int, max_len:int):
        history = deque(maxlen=max_len)
        self.set_strategy(strategy)

        try:
            print("[red]START LOOP[/red]")
            for i in range(self.start_iteration, max_iter+1):
                print(f"[yellow]GRID[/yellow] at i: {i}")
                print(self.lattice.grid)
                history.append((i, self.lattice.random_state, self.lattice.grid))
                self.update_grid(self.strategy())
                self.lattice.random_state += 1
            self.save(history)
        except KeyboardInterrupt:
            self.save(history)
            print("[red]Saved!!![/red]")
        finally:
            sys.exit(0)

    


    # TODO: Optimize when Numba works in python3.10, Lv1
    # NOTE: Not bad, just not going to use them as they are rn.
    # @staticmethod
    # def get_neighbors(lattice: Lattice, i: int, j: int):
    #     above_point = np.mod((i+1, j), lattice.size)
    #     below_point = np.mod((i-1, j), lattice.size)
    #     left_point = np.mod((i, j-1), lattice.size)
    #     right_point = np.mod((i, j+1), lattice.size)
    #     return (above_point, below_point, left_point, right_point)
    # TODO: Optimize when Numba works in python3.10, Lv1
    # NOTE: Not bad, just not going to use them as they are rn.
    # @staticmethod
    # def get_energy(lattice: Lattice, i:int, j:int, inter: num):
    #     spin_flip_contrib = -2*lattice.grid[i,j]
    #     neighbors_contrib = 0
    #     neighbors = get_neighbors()
    #     for point in :
    #         neighbors_contrib += lattice.grid[*point]
    #     diff_energy = spin_flip_contrib*neighbors_contrib
    #     pass

def main() -> None:
    pass


