import random
import traceback
from typing import Dict, List

import numpy

from functools import partial


class HyperParameter():
    def __init__(self, name, func, **kwargs):
        self.name = name
        self.func = partial(func, **kwargs)
        self.kwargs = kwargs

    def sample(self):
        return self.func()

    def __str__(self):
        return str(self.kwargs)
        # return '\n'.join([str(x) for x in self.kwargs.items()])

    __repr__ = __str__


class CatHP(HyperParameter):
    pass


class RealHP(HyperParameter):
    pass


class IntHP(HyperParameter):
    def sample(self):
        return numpy.round(self.func())


class Node:
    def __init__(self, children):
        self.children = children

    def __str__(self, depth=''):
        rows = []
        for child in self.children:
            rows.append(child.__str__(depth + '   '))
        return depth + self.__class__.__name__ + '\n'.join(rows)

    __repr__ = __str__


class SNode(Node):
    def __init__(self, name, children):
        super().__init__(children=children)
        self.name = name
        self.hps = []


class ENode(Node):
    def __init__(self):
        super().__init__(children=[])
        self.hps = []


class INode(Node):
    def __init__(self, hps, children=None):
        children = [] if children is None else children
        super().__init__(children=children)
        self.hps = hps

    def updated(self, children):
        return INode(self.hps, children=children)

    def __str__(self, depth=''):
        rows = [str(self.hps)]
        for child in self.children:
            rows.append(child.__str__(depth + '   '))
        return depth + self.__class__.__name__ + '\n'.join(rows)

    __repr__ = __str__


class ConfigSpace(Node):
    def __init__(self, start, end, children=None):
        children = [] if children is None else children
        super().__init__(children=children)
        self._start = start
        self._end = end

    def updated(self, children):
        return ConfigSpace(self.start(), self.end(), children=children)

    @staticmethod
    def node(hps, children):
        return INode(hps, children)

    @staticmethod
    def top(name, children):
        return SNode(name=name, children=children)

    @staticmethod
    def bottom():
        return ENode()

    def start(self):
        return self._start

    def end(self):
        return self._end

    def sample(self):
        """TODO:
        """
        if self.iselement_hps(self._start):
            config, _ = self._elem_hps_to_config(self._start)
        else:
            config, _ = self._compr_hps_to_config(self._start)

        return config

    @staticmethod
    def iselement_hps(node):
        return not any(
            [not isinstance(child, INode) for child in node.children])

    def _elem_hps_to_config(self, node):
        args = {}

        for hp in node.hps:
            try:
                args[hp.name] = hp.sample()
            except Exception as e:
                traceback.print_exc()
                print(e)
                print('Problems sampling: ', hp.name, hp)
                exit(0)

        if node.children:
            child = random.choice(node.children)

            if not isinstance(child, (SNode, ENode)):
                config, node = self._elem_hps_to_config(child)
                args.update(config)

        return args, node

    def _compr_hps_to_config(self, node):
        res = []

        while node.children:
            child = random.choice(node.children)
            if isinstance(child, ENode):
                break
            if self.iselement_hps(child):
                config, node = self._elem_hps_to_config(child)
            else:
                config, node = self._compr_hps_to_config(node)

            res.append(config)

        return {'configs': res}, child

    def __str__(self):
        return f'{self.name if not isinstance(self, ConfigSpace) else ""}' \
            f' {str(self.start())}'

    __repr__ = __str__

# class HPTree():
#     def __init__(self):
#         # TODO: remove?
#         self.tmp_uuid = tmp_uuid
#
#
#     def expand(self) -> (Dict, List):
#         return self.node, self.children
#
#     def tree_to_config(self, random_state=None):
#         """Prepare arguments to a Component,
#         sampling randomly from the intervals in the given tree.
#         The traversal is stopped at the first leaf found.
#
#         Parameters
#         ----------
#             random_state: int
#
#         Returns
#         -------
#             kwargs for Component constructor
#         """
#         if random_state is not None:
#             random.seed(random_state)
#         return self.pipeline_to_config(self)[0]
#
#     def moduletree_to_config(self, node):
#         args = {}
#
#         for k, hp in node.hps.items():
#             try:
#                 args[k] = hp.sample()
#             except Exception as e:
#                 traceback.print_exc()
#                 print(e)
#                 print('Problems sampling: ', k, hp)
#                 exit(0)
#
#         if node.children:
#             child = random.choice(node.children)
#
#             if not isinstance(child, (SNode, ENode)):
#                 config, node = self.moduletree_to_config(child)
#                 args.update(config)
#
#         return args, node
#
#
#     def pipeline_to_config(self, cs):
#
#         # This is needed here because composers
#         # when building forget subcomponent names.
#         if tree.name is not None:
#             args['name'] = tree.name
#
#
#         configs = []
#         config, children = cs.start()
#
#         while children:
#             tree = random.choice(children)
#             if tree.name.startswith('End'):
#                 break
#
#             # TODO: this IF should be more general:
#             if tree.name == 'Pipeline' or tree.name == 'Concat':
#                 config, children = self.pipeline_to_config(tree)
#             else:
#                 config, last = self.moduletree_to_config(tree)
#                 children = last.children
#
#             configs.append(config)
#         return {'name': tree.name[3:], 'configs': configs}, tree.children
#
#     def __str__(self, depth=''):
#         rows = [str(self.node)]
#         for child in self.children:
#             rows.append(child.__str__(depth + '   '))
#         return depth + self.name + '\n'.join(rows)
#
#     __repr__ = __str__
