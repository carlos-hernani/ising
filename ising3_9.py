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
import numpy as np

from collections import deque, namedtuple
from dataclasses import dataclass, field, asdict
from typing import Union, Optional, Tuple, \
    get_type_hints, get_args, get_origin


# Type Aliases
Num = Union[int, float]
Grid = np.ndarray
ValidationHandler = Tuple[bool, str]

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
    instance_dict = asdict(instance)
    print(instance_dict)
    for field, type_hint in get_type_hints(instance).items():
        if get_origin(type_hint):
            if not isinstance(instance_dict[field], get_args(type_hint)):
                raise TypeError(f'{field} must be type {type_hint}')
        else:
            if not isinstance(instance_dict[field], type_hint):
                raise TypeError(f'{instance_dict[field]}{field} must be type {type_hint}')

def cast_to(type_hint, value) -> Union[str, float, int]:
    if type_hint == 'float':
        value = float(value)
    elif type_hint == 'int':
        value = int(value)
    return value

def validvalue(instance, valid_conditions:dict):
    instance_dict = asdict(instance)
    for field, value in instance_dict.items():
        is_valid, error_message = \
            valid_conditions.get(field, no_validation)(value)
        if not is_valid:
            raise ValueError(f'{field}: {error_message}')

def no_validation(value) -> ValidationHandler:
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
    'lattice': lambda x: validvalue(x, lattice_validation),
    'temp': gtzero
}


@dataclass(order=True)
class Lattice:
    size: int
    up_prob: Num = field(compare=False)
    grid: Grid = field(compare=False, init=False, repr=False)
    random_state: int = field(compare=False)

    def generate_grid(self) -> None:
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

    def load(self, warmstart: Optional[str] = None) -> None:
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
        else:
            self.start_iteration = 0

    def save(self, history:deque) -> None:
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


    def __post_init__(self):
        k = 8.617e-5 # eV/K
        self.beta = 1/(k*self.temp)
        # That's a surprise tool that will help us later:
        exp2 = np.exp(-2*np.abs(self.inter)*2*self.beta)
        exp4 = np.exp(-2*np.abs(self.inter)*4*self.beta)
        self.flip = (True, True, True, exp2, exp4)
        self.start_iteration = 0
        validtype(self)
        validvalue(self, ising_validation)