"""2D Ising Model simulation using Monte Carlo Techniques.

@github: carlos-hernani

Rules:

    Classes are only called once.
    Functions can be called more than once.

Python Version 3.9

Not using attr this time, only dataclasses and post_init checks.
"""

__version__ = "0.1.0"

import dataclasses
import json
import sys
from numba.np.ufunc import parallel
import numpy as np

from rich import print as pprint
from numba import jit
from collections import deque, namedtuple
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import Union, Optional, Tuple, \
    get_type_hints, get_args, get_origin


# Type Aliases
Num = Union[int, float]
Grid = np.ndarray
ValidationHandler = Tuple[bool, str]
Points = Tuple[np.ndarray, ...]

# dataclass parameters / validation

def validtype(instance):
    """Checks that each field in Class Object has valid type.

    Used in __post_init__ can raise TypeError when the value of the field
    does not correspond with the statically-typed type.


    Args:
        instance: Class Instance with statically-typed fields.

    Raises:
        TypeError: Field type must correspond with the statically-typed type.
    """    
    instance_dict = instance.__dict__
    for field, type_hint in get_type_hints(instance).items():
        if get_origin(type_hint):
            if not isinstance(instance_dict[field], get_args(type_hint)):
                raise TypeError(f'{field} must be type {type_hint}')
        else:
            if not isinstance(instance_dict[field], type_hint):
                raise TypeError(f'{field} must be type {type_hint}')

def cast_to(type_hint, value) -> Union[str, float, int]:
    """Cast a value to the corresponding type_hint.

    If it fails, it will give a ValueError.

    Args:
        type_hint ([type]): The desired final type.
        value : Value to be casted into the desired final type.

    Returns:
        Union[str, float, int]: The value in the corresponding type.
    """    
    if type_hint == 'float':
        value = float(value)
    elif type_hint == 'int':
        value = int(value)
    return value

def validvalue(instance, valid_conditions:dict):
    """Function to validate that values pass certain conditions.

    Used in __post_init__. Some variables are required to 
    obey certain conditions, if these are not met then 
    a ValueError is raised.

    Args:
        instance : Class Instance with restricted fields.
        valid_conditions (dict): Dictionary of fields & conditions(functions)

    Raises:
        ValueError: [description]
    """    
    instance_dict = instance.__dict__
   
    for field, value in instance_dict.items():
        is_valid, error_message = \
            valid_conditions.get(field, no_validation)(value)
        if not is_valid:
            raise ValueError(f'{field}: {error_message}')

def no_validation(value) -> ValidationHandler:
    """
    no_validation Validator: self-explanatory
    """    
    return True, 'No validation needed'

def gtzero(value) -> ValidationHandler:
    """
    gtzero Validator: checks greather than zero
    """    
    if value <= 0:
        is_valid = False
    else:
        is_valid = True
    return is_valid, 'must be positive & non-zero.'

def gele(value) -> ValidationHandler:
    """
    gele Validator: checks geq than zero or leq than one
    """    
    if value < 0 or value > 1:
        is_valid = False
    else:
        is_valid = True
    return is_valid, 'must be between [0,1].'

def validate_grid(value) -> ValidationHandler:
    """
    validate_grid Checks that the given grid is correct.
    Args:
        arr (np.ndarray): grid of values.
    Returns:
        bool: True if correct.
    """
    is_valid = True
    for x in value.ravel():
        if np.abs(x) != 1:
            is_valid = False
            break
    return is_valid, 'Values of spin must be +1 or -1 .'

lattice_validation = {
    'size': gtzero,
    'up_prob': gele,
    'grid': validate_grid
}

ising_validation = {
    'temp': gtzero
}


@dataclass(order=True)
class Lattice:
    size: int
    up_prob: Num = field(compare=False)
    grid: Grid = field(compare=False, init=False, repr=False)
    random_state: int = field(compare=False)

    def generate_grid(self):
        """
        generate_grid
            Creates a square lattice of dimensions size x size filled with
            +1 or -1 with a given probability, up_prob, 
            (+1:up_prob, -1:1-up_prob) with a random seed.
        """        
        rng = np.random.default_rng(self.random_state)
        self.grid = 2*rng.binomial(1, p=self.up_prob, size=[self.size]*2)-1

    def __len__(self):
        return self.size

    def __post_init__(self):
        self.generate_grid()
        validtype(self)
        validvalue(self, lattice_validation)
    
    def snapshot(self):
        pass

# Ising class is not a type of lattice, is a completely different
# thing. So, Ising class shouldn't inherit from Lattice.
# The lattice object will be update as it is a piece of Ising.


@jit(nopython=True, parallel=True)
def _flip(grid:Grid, flip:tuple, i:int, j:int, rs:int) -> bool:
    rng = np.random.default_rng(rs)
    # each (i, j) pair has 4 nearest neighbors.
    neighbors_indices = (
        np.mod((i+1, j), grid.shape[0]),
        np.mod((i-1, j), grid.shape[0]),
        np.mod((i, j-1), grid.shape[0]),
        np.mod((i, j+1), grid.shape[0])
    )
    neighbors_contrib = 0
    #TODO: parallel here
    for ids in neighbors_indices:
        # each neighbor can be accessed in parallel.
        neighbors_contrib += grid[ids[0], ids[1]]
    index_flip = -0.25*grid[i,j]*neighbors_contrib + 2
    return rng.random() < flip[int(index_flip)]


@dataclass
class Ising2D:
    lattice: Lattice = field(repr=False)
    temp: Num
    inter: Num = 1
    name: str = 'ising'
    random_state: int = 0
    beta: float = field(init=False, repr=False)
    flip: tuple = field(init=False, repr=False)
    start_iteration: int = field(init=False, repr=False)

    
    # Loading and saving the state of the simulation

    def load(self, warmstart:Optional[str] = None):
        """Load method

        Called after init of Ising2D, loads data from previous runs.

        Args:
            warmstart (Optional[str], optional): 
                | The file path of the .npy to use.
                | Defaults to None.
        """        
        if warmstart:
            self.name, self.start_iteration, self.random_state = warmstart.split('_')
            with open(self.name+'.json') as read_file:
                data = json.load(read_file)
                idict = data['ising']
                ldict = data['lattice']

                for key, value in idict.items():
                    self.__dict__[key] = cast_to(*value)
                for key, value in ldict.items():
                    self.lattice.__dict__[key] = cast_to(*value)
            self.lattice.grid = np.load(warmstart+'.npy')
            exp2 = np.exp(-2*np.abs(self.inter)*2*self.beta)
            exp4 = np.exp(-2*np.abs(self.inter)*4*self.beta)
            self.flip = (True, True, True, exp2, exp4)

        pprint("[red]Loaded!!![/red]")

    def save(self, history:deque):
        """Save method

        Called from within the simulation phase.
        Stores the information about the experiment inside two files:
            json: 
                parameters of ising (last iteration)
                parameters of lattice (original)
            npy: stores the grids used in the simulation
                one .npy file per grid, name of file is:
                    Name_Iteration_randomSeed.npy
            This way one can tail the files and use the last grid to
            continue the experiment.

        Args:
            history (deque): An optimized version of a list.
        """        
        parameters = {
            'ising': {
                'temp': ('float', self.temp),
                'inter': ('float', self.inter),
                'random_state': ('int', self.random_state),
                'beta': ('float', self.beta),
            },
            'lattice': {
                'size': ('int', self.lattice.size),
                'up_prob': ('float', self.lattice.up_prob),
                'random_state': ('int', self.lattice.random_state)
            } 
        }
        with open(self.name+'.json', 'w') as outfile:
            json.dump(parameters, outfile)
        
        for h in history:
            np.save(
                '_'.join([
                    self.name, f'{h[0]}', f'{h[1]}'
                ]), h[2], allow_pickle=False
            )
        pprint("[red]Saved!!![/red]")

    def __post_init__(self):
        """After calling Ising2D this is automatically called.

        Not initialized fields are initialized.

        acc2 & acc4 are the acceptance probabilites associated
        with the cases where we have 3/4 neighbor spins & 4/4
        neighbor spins with the same sign.

        The tuple `flip` is used to know when to flip a spin.
        More details later.
        """        
        k = 8.617e-5 # eV/K
        self.beta = 1/(k*self.temp)
        # That's a surprise tool that will help us later:
        acc2 = np.exp(-2*np.abs(self.inter)*2*self.beta)
        acc4 = np.exp(-2*np.abs(self.inter)*4*self.beta)
        self.flip = (True, True, True, acc2, acc4)
        self.start_iteration = 0
        validtype(self)
        validvalue(self, ising_validation)
    

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _update_grid(grid:Grid, flip:tuple, points:Points, rs:int):
        print("hola1")
        for point in points:
            print(point)
            # rs is accessed but not changed
            # grid is accessed at points & changed
            if _flip(grid, flip, point[0], point[1], rs):
                grid[point[0], point[1]] *= -1
    
    def update_grid(self, points:Points):
        self._update_grid(self.lattice.grid, self.flip, points, self.random_state)
    
    # -------------Spin Query Strategies----------------
    def single(self) -> Points:
        """
        single Single flip dynamics strategy

        Each iteration it will deliver a list of only one pair of (i,j).

        Returns:
            list[np.ndarray]: List of 1 positions.
        """
        rng = np.random.default_rng(self.random_state)
        return (rng.integers(low=0, high=self.lattice.grid.shape[0], size=2))
    
    def checkerboard(self, checkerboards:tuple) -> Points:
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
        idx = np.mod(self.random_state,2)
        # idx = 0 if even , 1 if odd
        return checkerboards[idx]
    
    def set_strategy(self, strategy:str):
        # when numba works in 3.10 change this to match
        if strategy == 'single':
            self.get_points = self.single
        elif strategy == 'checkerboard':
            size = self.lattice.size
            cb = (np.indices((size,size)).sum(axis=0) % 2)
            # Generates a checkerboard pattern, only once
            checkerboards = (
                np.transpose(np.where(cb == 0)),
                np.transpose(np.where(cb == 1))
            )
            self.get_points = self.checkerboard(checkerboards)
        else:
            raise NotImplementedError(
                "Possible strategies are: 'single', 'checkerboard'")
    
    @jit(nopython=True)
    def _run(self, strategy:str, max_iter:int, max_len:int):
        history = deque(maxlen=max_len)
        self.set_strategy(strategy)

        try:
            pprint("[red]START LOOP[/red]")
            for i in range(self.start_iteration, max_iter+1):
                history.append((i, self.random_state, self.lattice.grid))
                self.update_grid(self.get_points())
                self.random_state += 1
            self.save(history)
        except KeyboardInterrupt:
            self.save(history)
        finally:
            sys.exit(0)