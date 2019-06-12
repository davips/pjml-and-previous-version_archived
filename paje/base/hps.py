import random
import traceback
from typing import Dict, List

from paje.util.distributions import sample


class HPTree(object):
    def __init__(self, dic, children, name=None, tmp_uuid=None):
        self.dic = dic
        self.children = children
        self.name = name
        self.tmp_uuid = tmp_uuid

    def expand(self) -> (Dict, List):
        return self.dic, self.children

    def tree_to_dict(self):
        """
        Prepare arguments to a Component,
        sampling randomly from the intervals in the given tree.
        The traversal is stopped at the first leaf found.
        :param tree:
        :return: kwargs for Component constructor
        """
        return self.pipeline_to_dic(self)[0]

    def moduletree_to_dic(self, tree):
        args = {}

        # This is needed here because composers
        # when building forget subcomponent names.
        if tree.name is not None:
            args['name'] = tree.name

        for k, kind_interval in tree.dic.items():
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
                dic, tree = self.moduletree_to_dic(child)
                args.update(dic)

        return args, tree

    # TODO: A hyperParameter (?) 'p' could be used to define the probabilities
    #  to weight each node.
    #  It could be defined during the construction of the tree according to
    #  the size of its subtrees.

    def pipeline_to_dic(self, tree):
        argss = []
        args, children = tree.expand()
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
                args, children = self.pipeline_to_dic(tree)
            else:
                args, last = self.moduletree_to_dic(tree)
                children = last.children

            argss.append(args)
        return {'name': tree.name[3:], 'dics': argss}, tree.children

    def __str__(self, depth=''):
        rows = [str(self.dic)]
        for child in self.children:
            rows.append(child.__str__(depth + '   '))
        return depth + self.name + '\n'.join(rows)

    __repr__ = __str__
