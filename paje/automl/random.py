import random

import numpy as np

from paje.automl.automl import AutoML
from paje.base.hps import HPTree


class RandomAutoML(AutoML):

    def next_hyperpar_dicts(self):
        dicts = [tree_to_dict(tree) for tree in self.forest]
        return dicts


def tree_to_dict(tree: HPTree):
    """
    Prepare arguments to a Component, sampling randomly from the intervals in the given tree.
    The traversal is stopped at the first leaf found.
    :param tree:
    :return: kwargs for Component constructor
    """
    dict = {}
    while True:
        tree_dic, tree_children = tree.expand()
        for k, v in tree_dic.items():
            dict[k] = sample(*v)
        if len(tree_children) == 0:
            break
        tree = random.choice(tree_children)  # TODO: Parameter 'p' can be set to weight probabilities to each child according to the size of its subtree.
    return dict


def categoric_sample(values):
    return random.choice(values)


def integer_sample(min, max):
    return np.random.randint(min, max + 1)


def ordinal_sample(values):
    return random.choice(values)


def real_sample(min, max):
    return ((max - min) * np.random.ranf()) + min


def sample(kind, interval):
    """
    Handles sampling according to the given type.
    :param kind:
    :param interval:
    :return:
    """
    if kind == 'c': return categoric_sample(interval)
    if kind == 'o': return ordinal_sample(interval)
    if kind == 'r': return real_sample(interval[0], interval[1])
    if kind == 'z': return integer_sample(interval[0], interval[1])
    raise Exception('Unknown kind of interval: ', kind, ' Interval: ', interval)