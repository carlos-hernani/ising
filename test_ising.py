"""2D Ising Model Profiling & Testing

@github: carlos-hernani
"""

import sys
import pytest
import ising3_10 as isi
import checks as ch
import numpy as np

from collections import namedtuple

field = namedtuple('Field', ['instance', 'attribute', 'value'])
attribute = namedtuple('Attribute', ['name'])

grid_correct = np.array([[1,-1,1],[1,-1,1],[1,-1,1]])
grid_incorrect = np.array([[1,-1,1],[1,-1,1],[1,-1,2]])


def test_always_true():
    assert True

# ------------------------- checks.py ----------------------------

def test_gtzero_negative():
    with pytest.raises(ValueError):
        value = -sys.float_info.min
        f = field('nothing', attribute('var_name'), value)
        ch.gtzero(*f)

def test_gele_lesszero():
    with pytest.raises(ValueError):
        value = -sys.float_info.min
        f = field('nothing', attribute('var_name'), value)
        ch.gele(*f)

def test_gele_moreone():
    with pytest.raises(ValueError):
        value = 1.000000000000001
        print(value)
        f = field('nothing', attribute('var_name'), value)
        ch.gele(*f)


# ------------------------- ising3_10.py ----------------------------

def test_validate_grid_correct():
    grid = grid_correct
    assert isi.validate_grid(grid)

def test_validate_grid_incorrect():
    grid = grid_incorrect
    assert not isi.validate_grid(grid)


def test_lattice_wrongtype_size():
    with pytest.raises(TypeError):
        isi.Lattice(size='error')

def test_lattice_wrongvalue_size():
    with pytest.raises(ValueError):
        isi.Lattice(size=-25)

def test_lattice_wrongtype_p():
    with pytest.raises(TypeError):
        isi.Lattice(size=3, p='error')

def test_lattice_wrongvalue_p():
    with pytest.raises(ValueError):
        isi.Lattice(size=3, p=-25)

def test_lattice_wrongvalue_grid():
    with pytest.raises(AssertionError):
        isi.Lattice(size=3, grid=grid_incorrect)

