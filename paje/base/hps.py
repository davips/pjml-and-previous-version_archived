import random
from typing import Dict, List

from paje.util.distributions import sample


class HPTree(object):
    def __init__(self, dic, children, name=''):
        self.dic = dic
        self.children = children
        self.name = name

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
        return self.pipeline_to_dic_rec(self)[0]

    def moduletree_to_dic(self, tree):
        args = {'name': tree.name}
        for k, kind_interval in tree.dic.items():
            args[k] = sample(*kind_interval)
        if tree.children:
            child = random.choice(tree.children)
            # if tree.name is 'EndPipeline':
            #     return {'name': ''}, tree
            if child.name == '':
                dic, tree = self.moduletree_to_dic(child)
                del dic['name']
                args.update(dic)
        return args, tree

    # TODO: A hyperParameter (?) 'p' can be used to define the probabilities
    #  to weight each node.
    #  It could be defined during the construction of the tree according to
    #  the size of its subtrees.

    def pipeline_to_dic_rec(self, tree):
        # ignoring dic of pipline, assumes it is empty
        argss = []
        args, children = tree.expand()
        if len(children) != 1:
            raise Exception(
                "Each pipeline should have only one child. Not " +
                str(children))
        while children:
            tree = random.choice(children)
            if tree.name == 'EndPipeline':
                break
            if tree.name == 'Pipeline':
                args, children = self.pipeline_to_dic_rec(tree)
            else:
                args, last = self.moduletree_to_dic(tree)
                children = last.children
            argss.append(args)
        return {'dics': argss}, tree.children

    def __str__(self, depth=''):
        rows = [str(self.dic)]
        for child in self.children:
            # print(child.name, ' <- child')
            # if isinstance(child, list):
            #     for child_list in child:
            #         rows.append(child_list.__str__())
            # else:
            if child.name != 'EndPipeline':
                rows.append(child.__str__(depth + '   '))
        return depth + self.name + '\n'.join(rows)

    __repr__ = __str__
