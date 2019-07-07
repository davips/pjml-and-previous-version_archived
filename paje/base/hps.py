import random
import traceback
from typing import Dict, List

import numpy

from paje.util.distributions import sample
from functools import partial


class HyperParameter():
    def __init__(self, name, func, *args, **kwargs):
        self.name = name
        self.func = partial(func, *args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def sample(self):
        return self.func()


class CatHP(HyperParameter):
    pass


class RealHP(HyperParameter):
    pass


class IntHP(HyperParameter):
    def sample(self):
        return numpy.round(self.func())


class Node():

    def add_child(self, child):
        self.children.append(child)

    def add_hp(self, hp):
        self.node[hp.name] = hp

class ConfigSpace(object):
    def __init__(self, name='', node=None, children=None, tmp_uuid=None):

        if node is None:
            self.node = {}

        if children is None:
            self.children = []

        self.name = name
        self.start = Node('Start')
        self.end = Node('End')

        # TODO: remove?
        self.tmp_uuid = tmp_uuid


    def expand(self) -> (Dict, List):
        return self.node, self.children

    def tree_to_config(self, random_state=None):
        """Prepare arguments to a Component,
        sampling randomly from the intervals in the given tree.
        The traversal is stopped at the first leaf found.

        Parameters
        ----------
            random_state: int

        Returns
        -------
            kwargs for Component constructor
        """
        if random_state is not None:
            random.seed(random_state)
        return self.pipeline_to_config(self)[0]

    def moduletree_to_config(self, tree):
        args = {}

        # This is needed here because composers
        # when building forget subcomponent names.
        if tree.name is not None:
            args['name'] = tree.name

        for k, kind_interval in tree.node.items():
            try:
                args[k] = sample(*kind_interval)
            except Exception as e:
                traceback.print_exc()
                print(e)
                print('Problems sampling: ', k, kind_interval)
                exit(0)

        if tree.children:
            child = random.choice(tree.children)

            # if child is not a component (it is a, e.g., a kernel)
            if child.name is None:
                node, tree = self.moduletree_to_config(child)
                args.update(node)

        return args, tree

    # TODO: A hyperParameter (?) 'p' could be used to define the probabilities
    #  to weight each node.
    #  It could be defined during the construction of the tree according to
    #  the size of its subtrees.

    def pipeline_to_config(self, tree):
        configs = []
        config, children = tree.expand()
        if len(children) != 1:
            raise Exception(
                "Each pipeline should have only one child. Not " +
                str(children))
        while children:
            tree = random.choice(children)
            if tree.name.startswith('End'):
                break

            # TODO: this IF should be more general:
            if tree.name == 'Pipeline' or tree.name == 'Concat':
                config, children = self.pipeline_to_config(tree)
            else:
                config, last = self.moduletree_to_config(tree)
                children = last.children

            configs.append(config)
        return {'name': tree.name[3:], 'configs': configs}, tree.children

    def __str__(self, depth=''):
        rows = [str(self.node)]
        for child in self.children:
            rows.append(child.__str__(depth + '   '))
        return depth + self.name + '\n'.join(rows)

    __repr__ = __str__
