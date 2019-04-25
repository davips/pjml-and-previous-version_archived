import random

from paje.automl.automl import AutoML
from paje.base.hps import HPTree
from paje.util.distributions import sample


class RandomAutoML(AutoML):

    def next_hyperpar_dicts(self, forest):
        dics = []
        if isinstance(forest, list):
            for item in forest:
                dics.append(self.next_hyperpar_dicts(item))
            return dics
        else:
            return tree_to_dict(forest)


def tree_to_dict(tree: HPTree):
    """
    Prepare arguments to a Component,
    sampling randomly from the intervals in the given tree.
    The traversal is stopped at the first leaf found.
    :param tree:
    :return: kwargs for Component constructor
    """
    dict = {}
    while True:
        tree_dic, tree_children = tree.expand()
        for k, kind_interval in tree_dic.items():
            dict[k] = sample(*kind_interval)
        if len(tree_children) == 0:
            break
        tree = random.choice(tree_children)
        # TODO: Parameter 'p' can be set to weight probabilities
        #  to each child according to the size of its subtree.
    return dict
