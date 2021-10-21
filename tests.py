"""2D Ising Model Profiling

@github: carlos-hernani
"""

import cProfile

from ising import main

if '__name__' == '__main__':
    parameters = {}
    cProfile.run(main(*parameters))