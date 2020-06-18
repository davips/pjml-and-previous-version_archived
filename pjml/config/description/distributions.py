import numpy as np
from numpy.random import randint


def choice(items):
    if len(items) == 0:
        raise Exception("No choice from Empty list!")
    idx = randint(0, len(items))
    return items[idx]


def uniform(low=0.0, high=1.0, size=None):
    return np.random.uniform(low, high, size)
