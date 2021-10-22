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

import attr
from checks import instance_of
import numpy as np

from rich import print

num = int | float


# Validators: They check the inputs.

def gtzero(instance, attribute, value):
    """
    gtzero Validator: checks greather than zero
    """    
    if value <= 0:
        raise ValueError(f'{attribute.name} must be positive & non-zero.')

def gele(instance, attribute, value):
    """
    gele Validator: checks geq than zero or leq than one
    """    
    if value < 0 or value > 1:
        raise ValueError(f'{attribute.name} must be between [0,1].')

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
    
    # TODO: Optimize when Numba works in python3.10, Lv5
    @staticmethod
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

    def __attrs_post_init__(self) -> None:
        if self.grid is None:
            self.generate_grid()
        else:
            assert self.validate_grid(self.grid), \
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
    """    
    lattice: Lattice | None = attr.ib(**opt_type(Lattice, None, default_params['lattice']))
    temp: num | None = attr.ib(**opt_type(num, gtzero, default_params['temp']))
    inter: num | None = attr.ib(**opt_type(num, None, default_params['inter']))

    
    # TODO: Optimize when Numba works in python3.10, Lv5
    @staticmethod
    def get_neighbors(lattice: Lattice, i: int, j: int):
        above_point = np.mod((i+1, j), lattice.size)
        below_point = np.mod((i-1, j), lattice.size)
        left_point = np.mod((i, j-1), lattice.size)
        right_point = np.mod((i, j+1), lattice.size)
        return (above_point, below_point, left_point, right_point)
        
    @staticmethod
    def get_energy(lattice: Lattice, i:int, j:int, inter: num):
        spin_flip_contrib = -2*lattice.grid[i,j]
        neighbors_contrib = 0
        neighbors = get_neighbors()
        for point in :
            neighbors_contrib += lattice.grid[*point]
        diff_energy = spin_flip_contrib*neighbors_contrib
        pass

def main() -> None:
    pass


