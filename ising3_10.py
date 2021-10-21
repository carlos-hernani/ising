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


import numpy as np

from dataclasses import dataclass, fields
from rich import print



# Useful Functions

# Only called Once: C1T



@dataclass
class Lattice:
    """
    Lattice Object.

    Creates a square lattice of dimensions size x size filled with
    +1 or -1 with a given probability, p, (+1:p, -1:1-p) with a
    random seed.
    
    Args:
        size (int): number of cells in each direction of the sqaure lattice.
        p (float, optional): probability of value +1. Defaults to 0.5.
        grid (np.ndarray, optional): grid of values. Defaults to None.
        random_state (int, optional): random seed for deterministic behavior.
                    Defaults to None.
    """    
    size: int
    p: float = 0.5
    grid: np.ndarray | None = None
    random_state: int | None = None

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

    def __post_init__(self) -> None:
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


@dataclass
class Ising:
    lattice: Lattice | None = None
    temp: float | None = None
    inter: float | None = None

    def __post_init__(self) -> None:
        if self.lattice:
            assert isinstance(self.lattice, Lattice), "Provide a valid Lattice Object."
        if self.temp:
            assert isinstance(self.temp, float) and self.temp > 0, "Provide a valid temperature."
        if self.inter:
            assert isinstance(self.inter, float), "Provide a valid interaction value."
        d = self.__dict__
        self.__dict__ = {
            k:(
                p if d[k] is None else d[k]) \
                for (k,p) in default_params.items() 
        }

    
    # TODO: Optimize when Numba works in python3.10, Lv5
    @staticmethod
    def p():
        pass

def main() -> None:
    pass


