import random
from typing import Dict, List

from paje.util.distributions import sample


class HPTree(object):
    def __init__(self, dic, children):
        self.dic = dic
        self.children = children
        self.name = None

    def expand(self) -> (Dict, List):
        return self.dic, self.children

    # def __str__(self, depth=''):
    #     rows = [depth + str(self.dic) + '\n']
    #     depth += '    '
    #     for child in self.children:
    #         # if isinstance(child, list):
    #         #     for child_list in child:
    #         #         rows.append(child_list.__str__())
    #         # else:
    #         rows.append(child.__str__(depth))
    #     return ''.join(rows)
    #
    # __repr__ = __str__

    def tree_to_dict(self):
        """
        Prepare arguments to a Component,
        sampling randomly from the intervals in the given tree.
        The traversal is stopped at the first leaf found.
        :param tree:
        :return: kwargs for Component constructor
        """
        final_dic = {}
        tree = self
        # print(tree.children)
        while True:
            if tree.name is not None:
                dic = {}
                final_dic[tree.name] = dic
            else:
                dic = final_dic
            tree_dic, tree_children = tree.expand()
            for k, kind_interval in tree_dic.items():
                dic[k] = sample(*kind_interval)
            if len(tree_children) == 0:
                break
            tree = random.choice(tree_children)
            # TODO: Parameter 'p' can be set to weight probabilities
            #  to each child according to the size of its subtree.
        return final_dic
